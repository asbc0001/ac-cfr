"""Reference external-sampling Deep CFR for Leduc poker."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from hashlib import blake2b
from math import fsum, isfinite
from random import Random
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ac_cfr.common.config import GameConfigurationId
from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import GameId, NodeType, validate_player
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, build_leduc_neural_data
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.models import DeepCFRNetwork, build_deep_cfr_network
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import (
    AdvantageSample,
    PackedAdvantageReservoir,
    PackedStrategyReservoir,
    StrategySample,
    UniformReservoir,
)


@dataclass(frozen=True, slots=True)
class NetworkTrainingMetrics:
    """Final held-out losses and sample counts for one freshly trained network."""

    iteration: int
    network_role: str
    player: int | None
    training_samples: int
    validation_samples: int
    training_loss: float
    validation_loss: float | None

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int):
            raise TypeError("metric iteration must be an integer")
        if self.iteration < 1:
            raise ValueError("metric iteration must be positive")
        if self.network_role not in ("advantage", "strategy"):
            raise ValueError("network_role must be advantage or strategy")
        if self.network_role == "advantage" and self.player not in (0, 1):
            raise ValueError("advantage metrics must identify a player")
        if self.network_role == "strategy" and self.player is not None:
            raise ValueError("strategy metrics must not identify one player")
        for name, count in (
            ("training_samples", self.training_samples),
            ("validation_samples", self.validation_samples),
        ):
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{name} must be an integer")
            if count < (1 if name == "training_samples" else 0):
                raise ValueError(f"{name} is invalid")
        if not isfinite(self.training_loss):
            raise ValueError("training_loss must be finite")
        if self.validation_loss is not None and not isfinite(self.validation_loss):
            raise ValueError("validation_loss must be finite when present")

    def to_dict(self) -> dict[str, object]:
        """Return checkpoint-safe metric values."""
        return {
            "iteration": self.iteration,
            "network_role": self.network_role,
            "player": self.player,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
        }

    @classmethod
    def from_dict(cls, values: object) -> "NetworkTrainingMetrics":
        """Parse one strictly shaped checkpoint metric record."""
        if not isinstance(values, dict) or set(values) != {
            "iteration",
            "network_role",
            "player",
            "training_samples",
            "validation_samples",
            "training_loss",
            "validation_loss",
        }:
            raise ValueError("network training metric fields are incompatible")
        try:
            return cls(**values)
        except (TypeError, ValueError) as error:
            raise ValueError("network training metric is invalid") from error


@dataclass(frozen=True, slots=True)
class _LossMetrics:
    """Internal loss result before network context is attached."""

    training_samples: int
    validation_samples: int
    training_loss: float
    validation_loss: float | None


@dataclass(frozen=True, slots=True)
class NetworkFitPoint:
    """Loss and cumulative optimiser time after a declared update milestone."""

    update_steps: int
    training_samples: int
    validation_samples: int
    training_loss: float
    validation_loss: float | None
    training_seconds: float


@dataclass(frozen=True, slots=True)
class DeepCFRTrainingTimes:
    """Coarse timings for the most recent call to ``train``."""

    traversal_seconds: float
    advantage_training_seconds: float
    strategy_training_seconds: float


@dataclass(frozen=True, slots=True)
class ExplorationSamplingDiagnostics:
    """Importance-weight health and raw/effective coverage for recent traversals."""

    opponent_actions_sampled: int
    zero_ratio_actions: int
    ratio_p50: float
    ratio_p95: float
    ratio_p99: float
    ratio_max: float
    samples: int
    positive_weight_samples: int
    raw_information_sets: int
    weighted_information_sets: int
    sampling_weight_max: float
    effective_sample_size: float


_LOSS_EVALUATION_BATCHES = 4


class NaiveDeepCFR:
    """Faithful one-trajectory-at-a-time Deep CFR correctness reference."""

    def __init__(
        self,
        tree: IndexedGameTree,
        config: DeepCFRTrainingConfig,
        runtime: DeepCFRRuntimeConfig,
    ) -> None:
        if not isinstance(tree, IndexedGameTree):
            raise TypeError("tree must be an IndexedGameTree")
        if tree.game_id is not GameId.LEDUC:
            raise ValueError("Deep CFR currently supports only Leduc")
        if not isinstance(config, DeepCFRTrainingConfig):
            raise TypeError("config must be a DeepCFRTrainingConfig")
        if config.game_configuration_id is not GameConfigurationId.LEDUC:
            raise ValueError("reference Deep CFR supports only Leduc")
        if not isinstance(runtime, DeepCFRRuntimeConfig):
            raise TypeError("runtime must be a DeepCFRRuntimeConfig")

        seed_deriver = SeedDeriver(config.seed)
        self._tree = tree
        self._config = config
        self._runtime = runtime
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
        self._training_metrics: list[NetworkTrainingMetrics] = []
        self._iteration = 0
        self._traversal_seconds = 0.0
        self._advantage_training_seconds = 0.0
        self._strategy_training_seconds = 0.0
        self._importance_ratios: list[float] = []
        self._sample_weights: list[float] = []
        self._raw_information_sets: set[Hashable] = set()
        self._weighted_information_sets: set[Hashable] = set()

    @property
    def iteration(self) -> int:
        """Return the number of completed outer iterations."""
        return self._iteration

    @property
    def config(self) -> DeepCFRTrainingConfig:
        """Return the immutable training configuration."""
        return self._config

    @property
    def runtime(self) -> DeepCFRRuntimeConfig:
        """Return the solver's immutable execution settings."""
        return self._runtime

    @property
    def tree(self) -> IndexedGameTree:
        """Return the indexed Leduc tree interpreted by this solver."""
        return self._tree

    @property
    def advantage_reservoirs(
        self,
    ) -> tuple[
        UniformReservoir[AdvantageSample] | PackedAdvantageReservoir,
        UniformReservoir[AdvantageSample] | PackedAdvantageReservoir,
    ]:
        """Return the two player-specific advantage reservoirs."""
        return self._advantage_reservoirs

    @property
    def strategy_reservoir(self) -> UniformReservoir[StrategySample] | PackedStrategyReservoir:
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

    @property
    def training_metrics(self) -> tuple[NetworkTrainingMetrics, ...]:
        """Return completed advantage and strategy network training measurements."""
        return tuple(self._training_metrics)

    @property
    def recent_training_times(self) -> DeepCFRTrainingTimes:
        """Return phase timings for the most recent call to ``train``."""
        return DeepCFRTrainingTimes(
            traversal_seconds=self._traversal_seconds,
            advantage_training_seconds=self._advantage_training_seconds,
            strategy_training_seconds=self._strategy_training_seconds,
        )

    @property
    def recent_exploration_diagnostics(self) -> ExplorationSamplingDiagnostics:
        """Summarise exploratory samples collected by the most recent train call."""
        ratios = np.asarray(self._importance_ratios, dtype=np.float64)
        weights = np.asarray(self._sample_weights, dtype=np.float64)
        weight_sum = float(weights.sum())
        squared_weight_sum = float(np.square(weights).sum())
        return ExplorationSamplingDiagnostics(
            opponent_actions_sampled=len(ratios),
            zero_ratio_actions=int(np.count_nonzero(ratios == 0.0)),
            ratio_p50=_quantile_or_default(ratios, 0.50, 1.0),
            ratio_p95=_quantile_or_default(ratios, 0.95, 1.0),
            ratio_p99=_quantile_or_default(ratios, 0.99, 1.0),
            ratio_max=float(ratios.max()) if len(ratios) else 1.0,
            samples=len(weights),
            positive_weight_samples=int(np.count_nonzero(weights > 0.0)),
            raw_information_sets=len(self._raw_information_sets),
            weighted_information_sets=len(self._weighted_information_sets),
            sampling_weight_max=float(weights.max()) if len(weights) else 1.0,
            effective_sample_size=(
                weight_sum * weight_sum / squared_weight_sum if squared_weight_sum > 0.0 else 0.0
            ),
        )

    def training_rng_state(self) -> dict[str, object]:
        """Return every mutable random stream used by traversal and reservoirs."""
        return {
            "chance": self._chance_rng.getstate(),
            "policy": self._policy_rng.getstate(),
            "advantage_reservoir_0": self._advantage_reservoirs[0].training_state()["rng_state"],
            "advantage_reservoir_1": self._advantage_reservoirs[1].training_state()["rng_state"],
            "strategy_reservoir": self._strategy_reservoir.training_state()["rng_state"],
        }

    def restore_training_state(
        self,
        *,
        iteration: int,
        advantage_networks: tuple[DeepCFRNetwork | None, DeepCFRNetwork | None],
        snapshot_networks: dict[int, DeepCFRNetwork],
        final_strategy_network: DeepCFRNetwork | None,
        training_metrics: tuple[NetworkTrainingMetrics, ...],
        chance_rng_state: object,
        policy_rng_state: object,
    ) -> None:
        """Restore validated iteration, model, metric, and traversal RNG state."""
        _validate_non_negative_integer("iteration", iteration)
        if iteration > self._config.iterations:
            raise ValueError("checkpoint iteration exceeds the configured budget")
        if any(network is None for network in advantage_networks) != (iteration == 0):
            raise ValueError("checkpoint advantage-network state is inconsistent")
        expected_snapshots = {
            snapshot_iteration
            for snapshot_iteration in self._config.snapshot_iterations
            if snapshot_iteration <= iteration and snapshot_iteration < self._config.iterations
        }
        if set(snapshot_networks) != expected_snapshots:
            raise ValueError("checkpoint milestone-network state is inconsistent")
        if (final_strategy_network is not None) != (iteration == self._config.iterations):
            raise ValueError("checkpoint final-network state is inconsistent")
        if _metric_schedule(training_metrics) != _expected_metric_schedule(self._config, iteration):
            raise ValueError("checkpoint training metrics are incomplete or misaligned")
        chance_rng = _restored_random(chance_rng_state, "chance")
        policy_rng = _restored_random(policy_rng_state, "policy")

        self._iteration = iteration
        self._advantage_networks = list(advantage_networks)
        self._snapshot_networks = snapshot_networks.copy()
        self._final_strategy_network = final_strategy_network
        self._training_metrics = list(training_metrics)
        self._chance_rng = chance_rng
        self._policy_rng = policy_rng

    def train(self, iterations: int) -> None:
        """Run complete alternating outer iterations up to the configured budget."""
        _validate_non_negative_integer("iterations", iterations)
        if self._iteration + iterations > self._config.iterations:
            raise ValueError("training would exceed the configured iteration budget")
        self._traversal_seconds = 0.0
        self._advantage_training_seconds = 0.0
        self._strategy_training_seconds = 0.0
        self._importance_ratios.clear()
        self._sample_weights.clear()
        self._raw_information_sets.clear()
        self._weighted_information_sets.clear()

        for _ in range(iterations):
            current_iteration = self._iteration + 1
            self._run_player_update(player=0, iteration=current_iteration)
            self._run_player_update(player=1, iteration=current_iteration)
            self._iteration = current_iteration

            if (
                current_iteration in self._config.snapshot_iterations
                and current_iteration < self._config.iterations
            ):
                self._snapshot_networks[current_iteration] = self._timed_strategy_network(
                    current_iteration
                )

        if self._iteration == self._config.iterations and self._final_strategy_network is None:
            self._final_strategy_network = self._timed_strategy_network(self._iteration)

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
        started = perf_counter()
        self._collect_player_traversals(player, iteration)
        self._traversal_seconds += perf_counter() - started
        started = perf_counter()
        self._advantage_networks[player] = self._train_advantage_network(player, iteration)
        self._advantage_training_seconds += perf_counter() - started

    def _collect_player_traversals(self, player: int, iteration: int) -> None:
        """Collect one reference batch of independent sampled traversals."""
        for _ in range(self._config.traversals_per_player):
            self._traverse(
                node_id=0,
                traverser=player,
                iteration=iteration,
                sampled_opponent_actions={},
                sampling_weight=1.0,
            )

    def _timed_strategy_network(self, iteration: int) -> DeepCFRNetwork:
        """Train one strategy network and include it in the recent phase timing."""
        started = perf_counter()
        network = self._train_strategy_network(iteration)
        self._strategy_training_seconds += perf_counter() - started
        return network

    def _traverse(
        self,
        *,
        node_id: int,
        traverser: int,
        iteration: int,
        sampled_opponent_actions: dict[int, int],
        sampling_weight: float,
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
                sampling_weight=sampling_weight,
            )

        acting_player = int(tree.current_players[node_id])
        information_set_id = int(tree.information_set_ids[node_id])
        strategy = self.strategy_for_information_set(information_set_id, acting_player)
        if acting_player != traverser:
            # The current action has not been sampled yet, so this loss weight
            # contains only ratios accumulated along the sampled prefix.
            self._strategy_reservoir.add(
                StrategySample(
                    player=acting_player,
                    state=self._state_tuple(information_set_id),
                    action_mask=self._mask_tuple(information_set_id),
                    iteration=iteration,
                    strategy=strategy,
                    sampling_weight=sampling_weight,
                )
            )
            self._record_sample(information_set_id, sampling_weight)
            local_probabilities = _local_action_values(tree, node_id, strategy)
            action_position = sampled_opponent_actions.get(information_set_id)
            if action_position is None:
                behaviour_probabilities = exploratory_opponent_probabilities(
                    local_probabilities,
                    self._config.opponent_exploration_epsilon,
                )
                action_position = _sample_position(behaviour_probabilities, self._policy_rng)
                sampled_opponent_actions[information_set_id] = action_position
            ratio = opponent_importance_ratio(
                local_probabilities,
                action_position,
                self._config.opponent_exploration_epsilon,
            )
            self._record_importance_ratio(ratio)
            # Carry the ratio down to correct descendant visitation and back up to
            # correct the return sampled through this opponent action.
            child_value = self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                iteration=iteration,
                sampled_opponent_actions=sampled_opponent_actions,
                sampling_weight=sampling_weight * ratio,
            )
            return (
                child_value
                if self._config.opponent_exploration_epsilon == 0.0
                else ratio * child_value
            )

        action_values = [
            self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                iteration=iteration,
                sampled_opponent_actions=sampled_opponent_actions,
                sampling_weight=sampling_weight,
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
        # Suffix ratios are already present in the targets; the stored weight
        # corrects only the probability of reaching this information set.
        self._advantage_reservoirs[traverser].add(
            AdvantageSample(
                state=self._state_tuple(information_set_id),
                action_mask=self._mask_tuple(information_set_id),
                iteration=iteration,
                advantages=tuple(advantages),
                sampling_weight=sampling_weight,
            )
        )
        self._record_sample(information_set_id, sampling_weight)
        return node_value

    def _record_sample(self, information_set_id: Hashable, sampling_weight: float) -> None:
        if self._config.opponent_exploration_epsilon == 0.0:
            return
        self._sample_weights.append(sampling_weight)
        self._raw_information_sets.add(information_set_id)
        if sampling_weight > 0.0:
            self._weighted_information_sets.add(information_set_id)

    def _record_importance_ratio(self, ratio: float) -> None:
        if self._config.opponent_exploration_epsilon == 0.0:
            return
        self._importance_ratios.append(ratio)

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
        """Train a fresh network from uniformly sampled reservoir minibatches."""
        seed_index = _network_seed_index(player, iteration)
        network = self._new_network(seed_index)
        losses = _train_network(
            network=network,
            samples=self._advantage_reservoirs[player].samples,
            current_iteration=iteration,
            training_steps=self._config.advantage_training_steps,
            batch_size=self._config.advantage_batch_size,
            learning_rate=self._config.learning_rate,
            data_seed=self._seed(RngStream.DATA_LOADER, seed_index),
            training_seed=self._seed(RngStream.NETWORK_TRAINING, seed_index),
            strategy_targets=False,
            validation_fraction=self._config.validation_fraction,
            validation_split_id=self._config.validation_split_id,
            max_gradient_norm=self._config.max_gradient_norm,
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
        """Train and freeze one playable average-strategy network."""
        seed_index = _network_seed_index(2, iteration)
        network = self._new_network(seed_index)
        losses = _train_network(
            network=network,
            samples=self._strategy_reservoir.samples,
            current_iteration=iteration,
            training_steps=self._config.strategy_training_steps,
            batch_size=self._config.strategy_batch_size,
            learning_rate=self._config.learning_rate,
            data_seed=self._seed(RngStream.DATA_LOADER, seed_index),
            training_seed=self._seed(RngStream.NETWORK_TRAINING, seed_index),
            strategy_targets=True,
            validation_fraction=self._config.validation_fraction,
            validation_split_id=self._config.validation_split_id,
            max_gradient_norm=self._config.max_gradient_norm,
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

    def _new_network(self, seed_index: int) -> DeepCFRNetwork:
        """Construct a normally initialised network without changing global RNG state."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed(RngStream.NETWORK, seed_index))
            return build_deep_cfr_network(
                self._config.model_config_id,
                dropout_probability=self._config.dropout_probability,
            ).to(self._runtime.device)

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
    if not predicted_advantages or len(predicted_advantages) != len(action_mask):
        raise ValueError("advantages and mask must have the same non-zero length")
    if any(not isfinite(value) for value in predicted_advantages):
        raise ValueError("predicted advantages must be finite")
    legal_actions = [action for action, is_legal in enumerate(action_mask) if is_legal]
    if not legal_actions:
        raise ValueError("action_mask must contain a legal action")

    positive_total = fsum(max(predicted_advantages[action], 0.0) for action in legal_actions)
    strategy = [0.0] * len(action_mask)
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
    sampling_weights: Tensor | None = None,
) -> Tensor:
    """Return the specified iteration-weighted, legal-action-summed loss."""
    if current_iteration < 1:
        raise ValueError("current_iteration must be positive")
    if predictions.shape != targets.shape or predictions.shape != action_masks.shape:
        raise ValueError("predictions, targets and action_masks must have matching shapes")
    if predictions.ndim != 2 or predictions.shape[1] < 1:
        raise ValueError("action tensors have incompatible dimensions")
    if sample_iterations.shape != (predictions.shape[0],):
        raise ValueError("sample_iterations has an incompatible shape")
    if sampling_weights is not None and sampling_weights.shape != (predictions.shape[0],):
        raise ValueError("sampling_weights has an incompatible shape")

    transformed_predictions = (
        _masked_softmax(predictions, action_masks) if strategy_targets else predictions
    )
    _require_finite_tensor("predictions", predictions)
    _require_finite_tensor("targets", targets)
    _require_finite_tensor("transformed predictions", transformed_predictions)
    _require_finite_tensor("sample iterations", sample_iterations)
    if sampling_weights is not None:
        _require_finite_tensor("sampling weights", sampling_weights)
        if bool((sampling_weights < 0.0).any()):
            raise ValueError("sampling_weights must be non-negative")
    squared_errors = (transformed_predictions - targets).square()
    per_sample_errors = (squared_errors * action_masks).sum(dim=1)
    weights = 2.0 * sample_iterations / current_iteration
    if sampling_weights is not None:
        weights = weights * sampling_weights
    loss = (weights * per_sample_errors).mean()
    _require_finite_tensor("loss", loss)
    return loss


def _masked_softmax(logits: Tensor, action_masks: Tensor) -> Tensor:
    masked_logits = logits.masked_fill(~action_masks, -torch.inf)
    return torch.softmax(masked_logits, dim=1)


def _train_network(
    *,
    network: DeepCFRNetwork,
    samples: tuple[AdvantageSample, ...] | tuple[StrategySample, ...],
    current_iteration: int,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    data_seed: int,
    training_seed: int,
    strategy_targets: bool,
    validation_fraction: float,
    max_gradient_norm: float | None,
    validation_split_id: str = "sample",
) -> _LossMetrics:
    """Train one network with deterministic uniform reservoir minibatches."""
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
    sampling_weights = (
        torch.tensor([sample.sampling_weight for sample in samples], dtype=torch.float32)
        if any(sample.sampling_weight != 1.0 for sample in samples)
        else None
    )
    return _train_network_tensors(
        network=network,
        states=states,
        action_masks=action_masks,
        targets=targets,
        sample_iterations=sample_iterations,
        sampling_weights=sampling_weights,
        current_iteration=current_iteration,
        training_steps=training_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        data_seed=data_seed,
        training_seed=training_seed,
        strategy_targets=strategy_targets,
        validation_fraction=validation_fraction,
        max_gradient_norm=max_gradient_norm,
        validation_split_id=validation_split_id,
    )


def _train_network_tensors(
    *,
    network: DeepCFRNetwork,
    states: Tensor,
    action_masks: Tensor,
    targets: Tensor,
    sample_iterations: Tensor,
    sampling_weights: Tensor | None = None,
    current_iteration: int,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    data_seed: int,
    training_seed: int,
    strategy_targets: bool,
    validation_fraction: float,
    max_gradient_norm: float | None,
    validation_split_id: str = "sample",
) -> _LossMetrics:
    """Train directly from packed sample tensors with the reference loss and schedule."""
    point = train_network_tensor_milestones(
        network=network,
        states=states,
        action_masks=action_masks,
        targets=targets,
        sample_iterations=sample_iterations,
        sampling_weights=sampling_weights,
        current_iteration=current_iteration,
        update_milestones=(training_steps,),
        batch_size=batch_size,
        learning_rate=learning_rate,
        data_seed=data_seed,
        training_seed=training_seed,
        strategy_targets=strategy_targets,
        validation_fraction=validation_fraction,
        max_gradient_norm=max_gradient_norm,
        validation_split_id=validation_split_id,
    )[0]
    return _LossMetrics(
        training_samples=point.training_samples,
        validation_samples=point.validation_samples,
        training_loss=point.training_loss,
        validation_loss=point.validation_loss,
    )


def train_network_tensor_milestones(
    *,
    network: DeepCFRNetwork,
    states: Tensor,
    action_masks: Tensor,
    targets: Tensor,
    sample_iterations: Tensor,
    sampling_weights: Tensor | None = None,
    current_iteration: int,
    update_milestones: tuple[int, ...],
    batch_size: int,
    learning_rate: float,
    data_seed: int,
    training_seed: int,
    strategy_targets: bool,
    validation_fraction: float,
    max_gradient_norm: float | None,
    validation_split_id: str = "sample",
    milestone_callback: Callable[[NetworkFitPoint], None] | None = None,
) -> tuple[NetworkFitPoint, ...]:
    """Train continuously and evaluate one fixed split at cumulative update milestones."""
    if states.ndim != 2 or len(states) == 0:
        raise ValueError("cannot train a network from empty or malformed states")
    if len(action_masks) != len(states) or len(targets) != len(states):
        raise ValueError("packed action data must match the state count")
    if sample_iterations.shape != (len(states),):
        raise ValueError("packed sample iterations must match the state count")
    if sampling_weights is not None and sampling_weights.shape != (len(states),):
        raise ValueError("packed sampling weights must match the state count")
    if (
        not isinstance(update_milestones, tuple)
        or not update_milestones
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step < 1
            for step in update_milestones
        )
        or tuple(sorted(set(update_milestones))) != update_milestones
    ):
        raise ValueError("update milestones must be unique, positive, and increasing")
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(data_seed)
    training_indices, validation_indices = _split_training_indices(
        states,
        action_masks,
        validation_fraction,
        generator,
        validation_split_id=validation_split_id,
    )
    network_device = next(network.parameters()).device
    cuda_devices = (
        [network_device.index if network_device.index is not None else torch.cuda.current_device()]
        if network_device.type == "cuda"
        else []
    )
    points: list[NetworkFitPoint] = []
    completed_steps = 0
    cumulative_training_seconds = 0.0
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(training_seed)
        if cuda_devices:
            torch.cuda.manual_seed(training_seed)
        for milestone in update_milestones:
            _synchronise_network_device(network)
            started = perf_counter()
            _fit_network(
                network=network,
                optimiser=optimiser,
                states=states,
                targets=targets,
                action_masks=action_masks,
                sample_iterations=sample_iterations,
                sampling_weights=sampling_weights,
                training_indices=training_indices,
                current_iteration=current_iteration,
                training_steps=milestone - completed_steps,
                batch_size=batch_size,
                generator=generator,
                strategy_targets=strategy_targets,
                max_gradient_norm=max_gradient_norm,
            )
            _synchronise_network_device(network)
            cumulative_training_seconds += perf_counter() - started
            completed_steps = milestone
            training_evaluation_indices = _bounded_evaluation_indices(training_indices, batch_size)
            validation_evaluation_indices = _bounded_evaluation_indices(
                validation_indices, batch_size
            )
            training_loss = _evaluate_network_loss(
                network,
                states,
                targets,
                action_masks,
                sample_iterations,
                sampling_weights,
                training_evaluation_indices,
                current_iteration,
                strategy_targets=strategy_targets,
            )
            validation_loss = (
                _evaluate_network_loss(
                    network,
                    states,
                    targets,
                    action_masks,
                    sample_iterations,
                    sampling_weights,
                    validation_evaluation_indices,
                    current_iteration,
                    strategy_targets=strategy_targets,
                )
                if len(validation_indices) > 0
                else None
            )
            point = NetworkFitPoint(
                update_steps=milestone,
                training_samples=len(training_indices),
                validation_samples=len(validation_indices),
                training_loss=training_loss,
                validation_loss=validation_loss,
                training_seconds=cumulative_training_seconds,
            )
            points.append(point)
            if milestone_callback is not None:
                milestone_callback(point)
    return tuple(points)


def _synchronise_network_device(network: DeepCFRNetwork) -> None:
    """Wait for queued CUDA work so measured optimiser time is complete."""
    device = next(network.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit_network(
    *,
    network: DeepCFRNetwork,
    optimiser: torch.optim.Optimizer,
    states: Tensor,
    targets: Tensor,
    action_masks: Tensor,
    sample_iterations: Tensor,
    sampling_weights: Tensor | None,
    training_indices: Tensor,
    current_iteration: int,
    training_steps: int,
    batch_size: int,
    generator: torch.Generator,
    strategy_targets: bool,
    max_gradient_norm: float | None,
) -> None:
    """Apply a fixed number of uniformly sampled minibatch updates."""
    _require_finite_parameters(network)
    device = next(network.parameters()).device
    network.train()
    for _ in range(training_steps):
        positions = torch.randint(
            len(training_indices),
            (batch_size,),
            generator=generator,
        )
        indices = training_indices[positions]
        optimiser.zero_grad(set_to_none=True)
        batch_states = states[indices].to(device=device, dtype=torch.float32)
        loss = linear_cfr_loss(
            network(batch_states),
            targets[indices].to(device),
            action_masks[indices].to(device),
            sample_iterations[indices].to(device),
            current_iteration,
            strategy_targets=strategy_targets,
            sampling_weights=(
                None if sampling_weights is None else sampling_weights[indices].to(device)
            ),
        )
        loss.backward()
        _require_finite_gradients(network)
        if max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                network.parameters(),
                max_gradient_norm,
                error_if_nonfinite=True,
            )
        optimiser.step()
        _require_finite_parameters(network)
    network.eval()


def _split_training_indices(
    states: Tensor,
    action_masks: Tensor,
    validation_fraction: float,
    generator: torch.Generator,
    *,
    validation_split_id: str,
) -> tuple[Tensor, Tensor]:
    """Return one deterministic sample- or information-state-grouped split."""
    sample_count = len(states)
    indices = torch.randperm(sample_count, generator=generator)
    if sample_count < 2 or validation_fraction == 0.0:
        return indices, torch.empty(0, dtype=torch.int64)
    if validation_split_id == "sample":
        validation_count = min(sample_count - 1, max(1, int(sample_count * validation_fraction)))
        return indices[validation_count:], indices[:validation_count]
    if validation_split_id != "holdem_information_state":
        raise ValueError("validation_split_id is unsupported")

    state_values = states.detach().cpu().numpy()
    mask_values = action_masks.detach().cpu().numpy()
    threshold = int(validation_fraction * (1 << 64))
    seed = generator.initial_seed().to_bytes(8, byteorder="big", signed=False)
    validation = np.empty(sample_count, dtype=np.bool)
    for index, (state, mask) in enumerate(zip(state_values, mask_values, strict=True)):
        digest = blake2b(digest_size=8, person=b"ac_cfr_split_v1")
        digest.update(seed)
        digest.update(memoryview(np.ascontiguousarray(state)).cast("B"))
        digest.update(memoryview(np.ascontiguousarray(mask)).cast("B"))
        validation[index] = int.from_bytes(digest.digest(), "big") < threshold
    if not validation.any() or validation.all():
        raise RuntimeError("grouped validation split produced an empty partition")
    return (
        torch.from_numpy(np.flatnonzero(~validation)),
        torch.from_numpy(np.flatnonzero(validation)),
    )


def _bounded_evaluation_indices(indices: Tensor, batch_size: int) -> Tensor:
    """Keep loss diagnostics representative without scanning a growing reservoir."""
    limit = _LOSS_EVALUATION_BATCHES * batch_size
    return indices[:limit]


def _evaluate_network_loss(
    network: DeepCFRNetwork,
    states: Tensor,
    targets: Tensor,
    action_masks: Tensor,
    sample_iterations: Tensor,
    sampling_weights: Tensor | None,
    indices: Tensor,
    current_iteration: int,
    *,
    strategy_targets: bool,
) -> float:
    """Evaluate one fixed subset with dropout disabled and gradients omitted."""
    device = next(network.parameters()).device
    with torch.inference_mode():
        loss = linear_cfr_loss(
            network(states[indices].to(device=device, dtype=torch.float32)),
            targets[indices].to(device),
            action_masks[indices].to(device),
            sample_iterations[indices].to(device),
            current_iteration,
            strategy_targets=strategy_targets,
            sampling_weights=(
                None if sampling_weights is None else sampling_weights[indices].to(device)
            ),
        )
    return float(loss)


def _require_finite_tensor(name: str, values: Tensor) -> None:
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"{name} must contain only finite values")


def _require_finite_gradients(network: DeepCFRNetwork) -> None:
    for parameter in network.parameters():
        if parameter.grad is not None:
            _require_finite_tensor("gradients", parameter.grad)


def _require_finite_parameters(network: DeepCFRNetwork) -> None:
    for parameter in network.parameters():
        _require_finite_tensor("network parameters", parameter)


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


def exploratory_opponent_probabilities(
    strategy: tuple[float, ...] | NDArray[np.float64],
    epsilon: float,
) -> tuple[float, ...] | NDArray[np.float64]:
    """Mix an opponent policy with uniform exploration over its legal actions."""
    if epsilon == 0.0:
        return strategy
    action_count = len(strategy)
    if action_count < 1:
        raise ValueError("strategy must contain at least one action")
    return np.asarray(strategy, dtype=np.float64) * (1.0 - epsilon) + epsilon / action_count


def opponent_importance_ratio(
    strategy: tuple[float, ...] | NDArray[np.float64],
    action_position: int,
    epsilon: float,
) -> float:
    """Return sigma/q for one action sampled from the exploratory behaviour policy."""
    if epsilon == 0.0:
        return 1.0
    behaviour = exploratory_opponent_probabilities(strategy, epsilon)
    return float(strategy[action_position]) / float(behaviour[action_position])


def _quantile_or_default(
    values: NDArray[np.float64],
    quantile: float,
    default: float,
) -> float:
    if len(values) == 0:
        return default
    return float(np.quantile(values, quantile))


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


def _metric_schedule(
    metrics: tuple[NetworkTrainingMetrics, ...],
) -> tuple[tuple[int, str, int | None], ...]:
    return tuple((metric.iteration, metric.network_role, metric.player) for metric in metrics)


def _expected_metric_schedule(
    config: DeepCFRTrainingConfig,
    iteration: int,
) -> tuple[tuple[int, str, int | None], ...]:
    schedule: list[tuple[int, str, int | None]] = []
    for completed_iteration in range(1, iteration + 1):
        schedule.extend(
            (
                (completed_iteration, "advantage", 0),
                (completed_iteration, "advantage", 1),
            )
        )
        if (
            completed_iteration in config.snapshot_iterations
            and completed_iteration < config.iterations
        ):
            schedule.append((completed_iteration, "strategy", None))
    if iteration == config.iterations:
        schedule.append((iteration, "strategy", None))
    return tuple(schedule)


def _restored_random(state: object, name: str) -> Random:
    if not isinstance(state, tuple):
        raise ValueError(f"checkpoint {name} RNG state is invalid")
    rng = Random()
    try:
        rng.setstate(state)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint {name} RNG state is invalid") from error
    return rng


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
