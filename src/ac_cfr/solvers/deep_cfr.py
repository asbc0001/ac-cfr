"""Batched external-sampling Deep CFR for Leduc poker."""

from collections.abc import Generator
from contextlib import suppress
from dataclasses import dataclass
from math import fsum
from random import Random

import numpy as np
import torch
from numba import njit
from numpy.typing import NDArray

from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import NodeType
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.models import DeepCFRNetwork
from ac_cfr.solvers.naive_deep_cfr import (
    NaiveDeepCFR,
    NetworkTrainingMetrics,
    _LossMetrics,
    _network_seed_index,
    _train_network_tensors,
)
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import PackedAdvantageReservoir, PackedStrategyReservoir


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    """One paused traversal waiting for a current-strategy prediction."""

    information_set_id: int
    player: int


@dataclass(frozen=True, slots=True)
class _TraversalRandomness:
    """Independent chance and policy streams for one sampled traversal."""

    chance: Random
    policy: Random


_Traversal = Generator[_InferenceRequest, NDArray[np.float64], float]


class DeepCFR(NaiveDeepCFR):
    """Deep CFR with concurrent trajectories, batched inference, and packed memory."""

    def __init__(self, tree: IndexedGameTree, config: DeepCFRTrainingConfig) -> None:
        super().__init__(tree, config)
        warm_up_probabilities = np.asarray((0.5, 0.5), dtype=np.float64)
        _single_positive_position(warm_up_probabilities)
        _sample_position_from_draw(warm_up_probabilities, 0.0)
        seed_deriver = SeedDeriver(config.seed)
        self._packed_advantage_reservoirs = (
            PackedAdvantageReservoir(
                config.advantage_reservoir_capacity,
                seed_deriver.python_rng(RngStream.RESERVOIR, 0),
            ),
            PackedAdvantageReservoir(
                config.advantage_reservoir_capacity,
                seed_deriver.python_rng(RngStream.RESERVOIR, 1),
            ),
        )
        self._packed_strategy_reservoir = PackedStrategyReservoir(
            config.strategy_reservoir_capacity,
            seed_deriver.python_rng(RngStream.RESERVOIR, 2),
        )

    @property
    def advantage_reservoirs(
        self,
    ) -> tuple[PackedAdvantageReservoir, PackedAdvantageReservoir]:
        """Return the two player-specific packed advantage reservoirs."""
        return self._packed_advantage_reservoirs

    @property
    def strategy_reservoir(self) -> PackedStrategyReservoir:
        """Return the shared packed average-strategy reservoir."""
        return self._packed_strategy_reservoir

    def training_rng_state(self) -> dict[str, object]:
        """Return mutable reservoir RNGs; trajectory streams derive from iteration IDs."""
        return {
            "chance": self._chance_rng.getstate(),
            "policy": self._policy_rng.getstate(),
            "advantage_reservoir_0": self._packed_advantage_reservoirs[0].training_state()[
                "rng_state"
            ],
            "advantage_reservoir_1": self._packed_advantage_reservoirs[1].training_state()[
                "rng_state"
            ],
            "strategy_reservoir": self._packed_strategy_reservoir.training_state()["rng_state"],
        }

    def _collect_player_traversals(self, player: int, iteration: int) -> None:
        """Advance K independent sampled traversals in inference batches."""
        for start in range(0, self.config.traversals_per_player, self.config.batch_size):
            stop = min(start + self.config.batch_size, self.config.traversals_per_player)
            traversals = [
                self._traverse_batched(
                    node_id=0,
                    traverser=player,
                    iteration=iteration,
                    sampled_opponent_actions={},
                    randomness=self._trajectory_randomness(iteration, player, traversal_index),
                )
                for traversal_index in range(start, stop)
            ]
            self._resolve_traversals(traversals)

    def _resolve_traversals(self, traversals: list[_Traversal]) -> None:
        """Resume all active traversals one batched inference boundary at a time."""
        active: list[tuple[_Traversal, _InferenceRequest]] = [
            (traversal, next(traversal)) for traversal in traversals
        ]
        while active:
            strategies = self._batched_strategies([request for _, request in active])
            pending: list[tuple[_Traversal, _InferenceRequest]] = []
            for (traversal, _), strategy in zip(active, strategies, strict=True):
                with suppress(StopIteration):
                    pending.append((traversal, traversal.send(strategy)))
            active = pending

    def _batched_strategies(
        self,
        requests: list[_InferenceRequest],
    ) -> NDArray[np.float64]:
        """Evaluate each player's requested states in one inference call."""
        predictions = np.zeros((len(requests), LEDUC_ACTION_COUNT), dtype=np.float32)
        information_set_ids = np.fromiter(
            (request.information_set_id for request in requests),
            dtype=np.int32,
            count=len(requests),
        )
        for player in (0, 1):
            positions = np.fromiter(
                (index for index, request in enumerate(requests) if request.player == player),
                dtype=np.int32,
            )
            network = self._advantage_networks[player]
            if network is None or len(positions) == 0:
                continue
            states = self._neural_data.states[information_set_ids[positions]]
            with torch.inference_mode():
                output = network(torch.from_numpy(states))
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("predicted advantages must be finite")
            predictions[positions] = output.numpy()
        masks = self._neural_data.action_masks[information_set_ids]
        return _regret_match_batch(predictions, masks)

    def _traverse_batched(
        self,
        *,
        node_id: int,
        traverser: int,
        iteration: int,
        sampled_opponent_actions: dict[int, int],
        randomness: _TraversalRandomness,
    ) -> _Traversal:
        """Yield inference requests while applying external-sampling traversal rules."""
        tree = self.tree
        node_type = NodeType(tree.node_types[node_id])
        if node_type is NodeType.TERMINAL:
            player_zero_utility = float(tree.terminal_utilities[node_id])
            return player_zero_utility if traverser == 0 else -player_zero_utility

        edge_start = int(tree.child_offsets[node_id])
        edge_count = int(tree.child_counts[node_id])
        if node_type is NodeType.CHANCE:
            probabilities = tree.chance_probabilities[edge_start : edge_start + edge_count]
            action_position = _sample_position(probabilities, randomness.chance)
            return (
                yield from self._traverse_batched(
                    node_id=int(tree.children[edge_start + action_position]),
                    traverser=traverser,
                    iteration=iteration,
                    sampled_opponent_actions=sampled_opponent_actions,
                    randomness=randomness,
                )
            )

        acting_player = int(tree.current_players[node_id])
        information_set_id = int(tree.information_set_ids[node_id])
        strategy = yield _InferenceRequest(information_set_id, acting_player)
        if acting_player != traverser:
            self._packed_strategy_reservoir.add_values(
                acting_player,
                self._neural_data.states[information_set_id],
                self._neural_data.action_masks[information_set_id],
                iteration,
                strategy,
            )
            action_position = sampled_opponent_actions.get(information_set_id)
            if action_position is None:
                local_probabilities = strategy[
                    tree.edge_labels[edge_start : edge_start + edge_count]
                ]
                action_position = _sample_position(local_probabilities, randomness.policy)
                sampled_opponent_actions[information_set_id] = action_position
            return (
                yield from self._traverse_batched(
                    node_id=int(tree.children[edge_start + action_position]),
                    traverser=traverser,
                    iteration=iteration,
                    sampled_opponent_actions=sampled_opponent_actions,
                    randomness=randomness,
                )
            )

        action_values: list[float] = []
        for action_position in range(edge_count):
            action_value = yield from self._traverse_batched(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                iteration=iteration,
                sampled_opponent_actions=sampled_opponent_actions,
                randomness=randomness,
            )
            action_values.append(action_value)
        local_strategy = strategy[tree.edge_labels[edge_start : edge_start + edge_count]]
        node_value = fsum(
            float(probability) * action_value
            for probability, action_value in zip(local_strategy, action_values, strict=True)
        )
        advantages = np.zeros(LEDUC_ACTION_COUNT, dtype=np.float32)
        for action_position, action_value in enumerate(action_values):
            action = int(tree.edge_labels[edge_start + action_position])
            advantages[action] = action_value - node_value
        self._packed_advantage_reservoirs[traverser].add_values(
            self._neural_data.states[information_set_id],
            self._neural_data.action_masks[information_set_id],
            iteration,
            advantages,
        )
        return node_value

    def _predict_advantages(self, information_set_id: int, player: int) -> tuple[float, ...]:
        """Query one network for the inherited single-information-set interface."""
        network = self._advantage_networks[player]
        if network is None:
            return (0.0,) * LEDUC_ACTION_COUNT
        state = self._neural_data.states[[information_set_id]]
        with torch.inference_mode():
            prediction = network(torch.from_numpy(state))[0]
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("predicted advantages must be finite")
        return tuple(float(value) for value in prediction)

    def _train_advantage_network(self, player: int, iteration: int) -> DeepCFRNetwork:
        """Train a fresh advantage network directly from packed array views."""
        seed_index = _network_seed_index(player, iteration)
        network = self._new_network(seed_index)
        states, masks, sample_iterations, advantages = self._packed_advantage_reservoirs[
            player
        ].arrays
        losses = self._train_packed_network(
            network,
            states,
            masks,
            sample_iterations,
            advantages,
            iteration,
            seed_index,
            training_steps=self.config.advantage_training_steps,
            strategy_targets=False,
        )
        self._training_metrics.append(
            NetworkTrainingMetrics(
                iteration=iteration,
                network_role="advantage",
                player=player,
                training_samples=losses.training_samples,
                validation_samples=losses.validation_samples,
                training_loss=losses.training_loss,
                validation_loss=losses.validation_loss,
            )
        )
        return network

    def _train_strategy_network(self, iteration: int) -> DeepCFRNetwork:
        """Train and freeze one playable network directly from packed array views."""
        seed_index = _network_seed_index(2, iteration)
        network = self._new_network(seed_index)
        _, states, masks, sample_iterations, strategies = self._packed_strategy_reservoir.arrays
        losses = self._train_packed_network(
            network,
            states,
            masks,
            sample_iterations,
            strategies,
            iteration,
            seed_index,
            training_steps=self.config.strategy_training_steps,
            strategy_targets=True,
        )
        self._training_metrics.append(
            NetworkTrainingMetrics(
                iteration=iteration,
                network_role="strategy",
                player=None,
                training_samples=losses.training_samples,
                validation_samples=losses.validation_samples,
                training_loss=losses.training_loss,
                validation_loss=losses.validation_loss,
            )
        )
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        network.eval()
        return network

    def _train_packed_network(
        self,
        network: DeepCFRNetwork,
        states: NDArray[np.float32],
        masks: NDArray[np.bool],
        sample_iterations: NDArray[np.uint32],
        targets: NDArray[np.float32],
        iteration: int,
        seed_index: int,
        *,
        training_steps: int,
        strategy_targets: bool,
    ) -> _LossMetrics:
        """Apply the shared reference training semantics without sample conversion."""
        return _train_network_tensors(
            network=network,
            states=torch.from_numpy(states),
            action_masks=torch.from_numpy(masks),
            targets=torch.from_numpy(targets),
            sample_iterations=torch.from_numpy(sample_iterations.astype(np.float32)),
            current_iteration=iteration,
            training_steps=training_steps,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            data_seed=self._seed(RngStream.DATA_LOADER, seed_index),
            training_seed=self._seed(RngStream.NETWORK_TRAINING, seed_index),
            strategy_targets=strategy_targets,
            validation_fraction=self.config.validation_fraction,
            max_gradient_norm=self.config.max_gradient_norm,
        )

    def _trajectory_randomness(
        self,
        iteration: int,
        player: int,
        traversal_index: int,
    ) -> _TraversalRandomness:
        """Derive stable independent streams without depending on process scheduling."""
        stream_index = (
            (iteration - 1) * 2 + player
        ) * self.config.traversals_per_player + traversal_index
        seed_deriver = SeedDeriver(self.config.seed)
        return _TraversalRandomness(
            chance=seed_deriver.python_rng(RngStream.CHANCE, stream_index),
            policy=seed_deriver.python_rng(RngStream.POLICY, stream_index),
        )


def _regret_match_batch(
    predicted_advantages: NDArray[np.float32],
    action_masks: NDArray[np.bool],
) -> NDArray[np.float64]:
    """Apply Deep CFR regret matching to a batch with stable first-action ties."""
    if predicted_advantages.shape != action_masks.shape or predicted_advantages.ndim != 2:
        raise ValueError("predicted advantages and masks must have matching matrix shapes")
    if predicted_advantages.shape[1] != LEDUC_ACTION_COUNT:
        raise ValueError("action matrices have an incompatible action count")
    if not np.all(np.isfinite(predicted_advantages)):
        raise FloatingPointError("predicted advantages must be finite")
    if not np.all(np.any(action_masks, axis=1)):
        raise ValueError("every action mask must contain a legal action")

    positive = np.where(action_masks, np.maximum(predicted_advantages, 0.0), 0.0).astype(np.float64)
    positive_totals = positive.sum(axis=1)
    strategies = np.zeros_like(positive)
    positive_rows = positive_totals > 0.0
    strategies[positive_rows] = positive[positive_rows] / positive_totals[positive_rows, None]
    fallback_rows = np.flatnonzero(~positive_rows)
    if len(fallback_rows) > 0:
        legal_predictions = np.where(
            action_masks[fallback_rows], predicted_advantages[fallback_rows], -np.inf
        )
        strategies[fallback_rows, np.argmax(legal_predictions, axis=1)] = 1.0
    return strategies


def _sample_position(probabilities: NDArray[np.floating], rng: Random) -> int:
    """Sample one local edge from a normalised probability vector."""
    single_position = _single_positive_position(probabilities)
    if single_position >= 0:
        return single_position
    return _sample_position_from_draw(probabilities, rng.random())


@njit(cache=True)
def _single_positive_position(probabilities: NDArray[np.floating]) -> int:
    """Return the only positive edge, or -1 when sampling is required."""
    positive_count = 0
    positive_position = -1
    for position, probability in enumerate(probabilities):
        if probability > 0.0:
            positive_count += 1
            positive_position = position
    return positive_position if positive_count == 1 else -1


@njit(cache=True)
def _sample_position_from_draw(
    probabilities: NDArray[np.floating],
    draw: float,
) -> int:
    """Select one edge from a supplied uniform draw inside compiled code."""
    cumulative_probability = 0.0
    for position, probability in enumerate(probabilities):
        cumulative_probability += float(probability)
        if draw < cumulative_probability:
            return position
    return len(probabilities) - 1
