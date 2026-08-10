"""Reference external-sampling Deep CFR for Leduc poker."""

from math import fsum, isfinite
from random import Random

import numpy as np
import torch
from torch import Tensor

from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import GameId, NodeType, validate_player
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, build_leduc_neural_data
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.models import DeepCFRNetwork, build_deep_cfr_network
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import AdvantageSample, StrategySample, UniformReservoir


class NaiveDeepCFR:
    """Faithful one-trajectory-at-a-time Deep CFR correctness reference."""

    def __init__(self, tree: IndexedGameTree, config: DeepCFRTrainingConfig) -> None:
        if not isinstance(tree, IndexedGameTree):
            raise TypeError("tree must be an IndexedGameTree")
        if tree.game_id is not GameId.LEDUC:
            raise ValueError("Deep CFR currently supports only Leduc")
        if not isinstance(config, DeepCFRTrainingConfig):
            raise TypeError("config must be a DeepCFRTrainingConfig")

        seed_deriver = SeedDeriver(config.seed)
        self._tree = tree
        self._config = config
        self._neural_data = build_leduc_neural_data(tree)
        self._chance_rng = seed_deriver.python_rng(RngStream.CHANCE)
        self._policy_rng = seed_deriver.python_rng(RngStream.POLICY)
        self._advantage_reservoirs = (
            UniformReservoir[AdvantageSample](
                config.advantage_reservoir_capacity,
                seed_deriver.python_rng(RngStream.RESERVOIR, 0),
            ),
            UniformReservoir[AdvantageSample](
                config.advantage_reservoir_capacity,
                seed_deriver.python_rng(RngStream.RESERVOIR, 1),
            ),
        )
        self._strategy_reservoir = UniformReservoir[StrategySample](
            config.strategy_reservoir_capacity,
            seed_deriver.python_rng(RngStream.RESERVOIR, 2),
        )
        # None is the explicit zero-output predictor before a player's first update.
        self._advantage_networks: list[DeepCFRNetwork | None] = [None, None]
        self._snapshot_networks: dict[int, DeepCFRNetwork] = {}
        self._final_strategy_network: DeepCFRNetwork | None = None
        self._iteration = 0

    @property
    def iteration(self) -> int:
        """Return the number of completed outer iterations."""
        return self._iteration

    @property
    def advantage_reservoirs(
        self,
    ) -> tuple[UniformReservoir[AdvantageSample], UniformReservoir[AdvantageSample]]:
        """Return the two player-specific advantage reservoirs."""
        return self._advantage_reservoirs

    @property
    def strategy_reservoir(self) -> UniformReservoir[StrategySample]:
        """Return the shared average-strategy reservoir."""
        return self._strategy_reservoir

    @property
    def advantage_networks(self) -> tuple[DeepCFRNetwork | None, DeepCFRNetwork | None]:
        """Return the current player advantage predictors."""
        return self._advantage_networks[0], self._advantage_networks[1]

    @property
    def snapshot_networks(self) -> dict[int, DeepCFRNetwork]:
        """Return milestone average-strategy networks keyed by iteration."""
        return self._snapshot_networks.copy()

    @property
    def final_strategy_network(self) -> DeepCFRNetwork | None:
        """Return the final playable network once the configured run completes."""
        return self._final_strategy_network

    def train(self, iterations: int) -> None:
        """Run complete alternating outer iterations up to the configured budget."""
        _validate_non_negative_integer("iterations", iterations)
        if self._iteration + iterations > self._config.iterations:
            raise ValueError("training would exceed the configured iteration budget")

        for _ in range(iterations):
            current_iteration = self._iteration + 1
            self._run_player_update(player=0, iteration=current_iteration)
            self._run_player_update(player=1, iteration=current_iteration)
            self._iteration = current_iteration

            if current_iteration in self._config.snapshot_iterations:
                self._snapshot_networks[current_iteration] = self._train_strategy_network(
                    current_iteration
                )

        if self._iteration == self._config.iterations and self._final_strategy_network is None:
            self._final_strategy_network = self._train_strategy_network(self._iteration)

    def strategy_for_information_set(
        self,
        information_set_id: int,
        player: int,
    ) -> tuple[float, ...]:
        """Return the current traversal strategy for one player decision."""
        if isinstance(information_set_id, bool) or not isinstance(information_set_id, int):
            raise TypeError("information_set_id must be an integer")
        if not 0 <= information_set_id < self._tree.information_set_count:
            raise ValueError("information_set_id is out of range")
        validate_player(player)
        if int(self._tree.information_set_players[information_set_id]) != player:
            raise ValueError("information set does not belong to player")
        advantages = self._predict_advantages(information_set_id, player)
        mask = tuple(bool(value) for value in self._neural_data.action_masks[information_set_id])
        return deep_cfr_regret_matching(advantages, mask)

    def _run_player_update(self, player: int, iteration: int) -> None:
        """Collect K traversals, then replace and train one advantage network."""
        for _ in range(self._config.traversals_per_player):
            self._traverse(
                node_id=0,
                traverser=player,
                iteration=iteration,
                sampled_opponent_actions={},
            )
        self._advantage_networks[player] = self._train_advantage_network(player, iteration)

    def _traverse(
        self,
        *,
        node_id: int,
        traverser: int,
        iteration: int,
        sampled_opponent_actions: dict[int, int],
    ) -> float:
        """Sample chance/opponent nodes and fully explore traverser actions."""
        tree = self._tree
        node_type = NodeType(tree.node_types[node_id])
        if node_type is NodeType.TERMINAL:
            player_zero_utility = float(tree.terminal_utilities[node_id])
            return player_zero_utility if traverser == 0 else -player_zero_utility

        edge_start = int(tree.child_offsets[node_id])
        edge_count = int(tree.child_counts[node_id])
        if node_type is NodeType.CHANCE:
            probabilities = tree.chance_probabilities[edge_start : edge_start + edge_count]
            action_position = _sample_position(probabilities, self._chance_rng)
            return self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                iteration=iteration,
                sampled_opponent_actions=sampled_opponent_actions,
            )

        acting_player = int(tree.current_players[node_id])
        information_set_id = int(tree.information_set_ids[node_id])
        strategy = self.strategy_for_information_set(information_set_id, acting_player)
        if acting_player != traverser:
            self._strategy_reservoir.add(
                StrategySample(
                    player=acting_player,
                    state=self._state_tuple(information_set_id),
                    action_mask=self._mask_tuple(information_set_id),
                    iteration=iteration,
                    strategy=strategy,
                )
            )
            action_position = sampled_opponent_actions.get(information_set_id)
            if action_position is None:
                local_probabilities = _local_action_values(tree, node_id, strategy)
                action_position = _sample_position(local_probabilities, self._policy_rng)
                sampled_opponent_actions[information_set_id] = action_position
            return self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                iteration=iteration,
                sampled_opponent_actions=sampled_opponent_actions,
            )

        action_values = [
            self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                iteration=iteration,
                sampled_opponent_actions=sampled_opponent_actions,
            )
            for action_position in range(edge_count)
        ]
        local_strategy = _local_action_values(tree, node_id, strategy)
        node_value = fsum(
            probability * action_value
            for probability, action_value in zip(local_strategy, action_values, strict=True)
        )
        advantages = [0.0] * LEDUC_ACTION_COUNT
        for action_position, action_value in enumerate(action_values):
            action = int(tree.edge_labels[edge_start + action_position])
            advantages[action] = action_value - node_value
        self._advantage_reservoirs[traverser].add(
            AdvantageSample(
                state=self._state_tuple(information_set_id),
                action_mask=self._mask_tuple(information_set_id),
                iteration=iteration,
                advantages=tuple(advantages),
            )
        )
        return node_value

    def _predict_advantages(self, information_set_id: int, player: int) -> tuple[float, ...]:
        """Query one network once, or return the explicit initial zero prediction."""
        network = self._advantage_networks[player]
        if network is None:
            return (0.0,) * LEDUC_ACTION_COUNT
        state = torch.from_numpy(self._neural_data.states[information_set_id].copy()).unsqueeze(0)
        with torch.no_grad():
            predictions = network(state)[0]
        return tuple(float(value) for value in predictions)

    def _train_advantage_network(self, player: int, iteration: int) -> DeepCFRNetwork:
        """Train a freshly initialised network over one player's full reservoir."""
        seed_index = _network_seed_index(player, iteration)
        network = self._new_network(seed_index)
        _train_network(
            network=network,
            samples=self._advantage_reservoirs[player].samples,
            current_iteration=iteration,
            epochs=self._config.advantage_training_epochs,
            batch_size=self._config.batch_size,
            learning_rate=self._config.learning_rate,
            data_seed=self._seed(RngStream.DATA_LOADER, seed_index),
            strategy_targets=False,
        )
        return network

    def _train_strategy_network(self, iteration: int) -> DeepCFRNetwork:
        """Train and freeze one playable average-strategy network."""
        seed_index = _network_seed_index(2, iteration)
        network = self._new_network(seed_index)
        _train_network(
            network=network,
            samples=self._strategy_reservoir.samples,
            current_iteration=iteration,
            epochs=self._config.strategy_training_epochs,
            batch_size=self._config.batch_size,
            learning_rate=self._config.learning_rate,
            data_seed=self._seed(RngStream.DATA_LOADER, seed_index),
            strategy_targets=True,
        )
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        network.eval()
        return network

    def _new_network(self, seed_index: int) -> DeepCFRNetwork:
        """Construct a normally initialised network without changing global RNG state."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed(RngStream.NETWORK, seed_index))
            return build_deep_cfr_network(self._config.model_config_id)

    def _seed(self, stream: RngStream, index: int) -> int:
        return SeedDeriver(self._config.seed).derive(stream, index)

    def _state_tuple(self, information_set_id: int) -> tuple[float, ...]:
        return tuple(float(value) for value in self._neural_data.states[information_set_id])

    def _mask_tuple(self, information_set_id: int) -> tuple[bool, ...]:
        return tuple(bool(value) for value in self._neural_data.action_masks[information_set_id])


def deep_cfr_regret_matching(
    predicted_advantages: tuple[float, ...],
    action_mask: tuple[bool, ...],
) -> tuple[float, ...]:
    """Convert predicted advantages into a legal Deep CFR strategy."""
    if len(predicted_advantages) != LEDUC_ACTION_COUNT or len(action_mask) != LEDUC_ACTION_COUNT:
        raise ValueError(f"advantages and mask must contain {LEDUC_ACTION_COUNT} values")
    if any(not isfinite(value) for value in predicted_advantages):
        raise ValueError("predicted advantages must be finite")
    legal_actions = [action for action, is_legal in enumerate(action_mask) if is_legal]
    if not legal_actions:
        raise ValueError("action_mask must contain a legal action")

    positive_total = fsum(max(predicted_advantages[action], 0.0) for action in legal_actions)
    strategy = [0.0] * LEDUC_ACTION_COUNT
    if positive_total > 0.0:
        for action in legal_actions:
            strategy[action] = max(predicted_advantages[action], 0.0) / positive_total
    else:
        best_action = max(legal_actions, key=lambda action: predicted_advantages[action])
        strategy[best_action] = 1.0
    return tuple(strategy)


def linear_cfr_loss(
    predictions: Tensor,
    targets: Tensor,
    action_masks: Tensor,
    sample_iterations: Tensor,
    current_iteration: int,
    *,
    strategy_targets: bool,
) -> Tensor:
    """Return the specified iteration-weighted, legal-action-summed loss."""
    if current_iteration < 1:
        raise ValueError("current_iteration must be positive")
    if predictions.shape != targets.shape or predictions.shape != action_masks.shape:
        raise ValueError("predictions, targets and action_masks must have matching shapes")
    if predictions.ndim != 2 or predictions.shape[1] != LEDUC_ACTION_COUNT:
        raise ValueError("action tensors have incompatible dimensions")
    if sample_iterations.shape != (predictions.shape[0],):
        raise ValueError("sample_iterations has an incompatible shape")

    transformed_predictions = (
        _masked_softmax(predictions, action_masks) if strategy_targets else predictions
    )
    squared_errors = (transformed_predictions - targets).square()
    per_sample_errors = (squared_errors * action_masks).sum(dim=1)
    weights = 2.0 * sample_iterations / current_iteration
    return (weights * per_sample_errors).mean()


def _masked_softmax(logits: Tensor, action_masks: Tensor) -> Tensor:
    masked_logits = logits.masked_fill(~action_masks, -torch.inf)
    return torch.softmax(masked_logits, dim=1)


def _train_network(
    *,
    network: DeepCFRNetwork,
    samples: tuple[AdvantageSample, ...] | tuple[StrategySample, ...],
    current_iteration: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    data_seed: int,
    strategy_targets: bool,
) -> None:
    """Train one network with deterministic shuffled PyTorch minibatches."""
    if not samples:
        raise ValueError("cannot train a network from an empty reservoir")
    states = torch.tensor([sample.state for sample in samples], dtype=torch.float32)
    action_masks = torch.tensor([sample.action_mask for sample in samples], dtype=torch.bool)
    targets = torch.tensor(
        [
            sample.strategy if isinstance(sample, StrategySample) else sample.advantages
            for sample in samples
        ],
        dtype=torch.float32,
    )
    sample_iterations = torch.tensor([sample.iteration for sample in samples], dtype=torch.float32)
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(data_seed)
    network.train()

    for _ in range(epochs):
        order = torch.randperm(len(samples), generator=generator)
        for start in range(0, len(samples), batch_size):
            indices = order[start : start + batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = linear_cfr_loss(
                network(states[indices]),
                targets[indices],
                action_masks[indices],
                sample_iterations[indices],
                current_iteration,
                strategy_targets=strategy_targets,
            )
            loss.backward()
            optimiser.step()
    network.eval()


def _local_action_values(
    tree: IndexedGameTree,
    node_id: int,
    canonical_values: tuple[float, ...],
) -> tuple[float, ...]:
    edge_start = int(tree.child_offsets[node_id])
    edge_count = int(tree.child_counts[node_id])
    return tuple(
        canonical_values[int(tree.edge_labels[edge_start + position])]
        for position in range(edge_count)
    )


def _sample_position(probabilities: tuple[float, ...] | np.ndarray, rng: Random) -> int:
    positive_positions = [
        position for position, probability in enumerate(probabilities) if probability > 0.0
    ]
    if len(positive_positions) == 1:
        return positive_positions[0]
    draw = rng.random()
    cumulative_probability = 0.0
    for position, probability in enumerate(probabilities):
        cumulative_probability += float(probability)
        if draw < cumulative_probability:
            return position
    return len(probabilities) - 1


def _network_seed_index(player_or_strategy: int, iteration: int) -> int:
    return iteration * 3 + player_or_strategy


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
