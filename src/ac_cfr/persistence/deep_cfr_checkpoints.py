"""Atomic, validated training checkpoints for Deep CFR implementations."""

import pickle
from dataclasses import dataclass
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ac_cfr.common.config import DeepCFRImplementationId, GameConfigurationId
from ac_cfr.games.base import GameId
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, LEDUC_NEURAL_STATE_SIZE
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.models import DeepCFRNetwork, build_deep_cfr_network, deep_cfr_network_config
from ac_cfr.persistence.compatibility import (
    ACTION_SPACE_ID,
    holdem_compatibility_digest,
    tree_compatibility_digest,
)
from ac_cfr.persistence.files import staged_atomic_binary_writer
from ac_cfr.solvers.deep_cfr_selection import deep_cfr_implementation, deep_cfr_solver_type
from ac_cfr.solvers.naive_deep_cfr import NaiveDeepCFR, NetworkTrainingMetrics
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import (
    DEEP_CFR_RESERVOIR_SCHEMA_VERSION,
    DEEP_CFR_WEIGHTED_RESERVOIR_SCHEMA_VERSION,
    AdvantageSample,
    PackedAdvantageReservoir,
    PackedStrategyReservoir,
    StrategySample,
    UniformReservoir,
)

DEEP_CFR_CHECKPOINT_SCHEMA_VERSION = 4
PROJECT_VERSION = version("ac-cfr")
_RNG_CONTRACT = "python_random_and_derived_torch_v1"


@dataclass(frozen=True, slots=True)
class LoadedDeepCFRCheckpoint:
    """Validated metadata and a fully reconstructed Deep CFR solver."""

    metadata: dict[str, Any]
    solver: NaiveDeepCFR
    run_state: dict[str, Any]


def save_deep_cfr_checkpoint(
    path: Path,
    *,
    solver: NaiveDeepCFR,
    run_id: str,
    checkpoint_id: str,
    code_revision: str,
    elapsed_training_seconds: float = 0.0,
    metric_records: tuple[dict[str, str], ...] = (),
) -> None:
    """Atomically save every value needed at a completed outer iteration."""
    implementation = deep_cfr_implementation(solver)
    for name, value in (
        ("run_id", run_id),
        ("checkpoint_id", checkpoint_id),
        ("code_revision", code_revision),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if (
        isinstance(elapsed_training_seconds, bool)
        or not isinstance(elapsed_training_seconds, (int, float))
        or not isfinite(elapsed_training_seconds)
        or elapsed_training_seconds < 0.0
    ):
        raise ValueError("elapsed_training_seconds must be finite and non-negative")
    if not isinstance(metric_records, tuple) or any(
        not isinstance(record, dict) for record in metric_records
    ):
        raise TypeError("metric_records must be a tuple of dictionaries")

    config = solver.config
    game, game_version, compatibility_digest = _solver_game_metadata(solver)
    architecture = deep_cfr_network_config(
        config.model_config_id,
        dropout_probability=config.dropout_probability,
    )
    completed_snapshots = sorted(solver.snapshot_networks)
    metrics = [metric.to_dict() for metric in solver.training_metrics]
    weighted_sampling = config.opponent_exploration_epsilon > 0.0
    metadata = {
        "checkpoint_schema_version": DEEP_CFR_CHECKPOINT_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "code_revision": code_revision,
        "game": game,
        "game_version": game_version,
        "state_encoding": config.state_encoding_id.value,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": compatibility_digest,
        "solver": implementation.value,
        "model_config_id": config.model_config_id.value,
        "optimizer_id": config.optimizer_id.value,
        "reservoir_schema_version": _reservoir_schema_version(config),
        "rng_contract": _RNG_CONTRACT,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "iteration": solver.iteration,
        "training_config": config.to_dict(),
        "runtime_config": solver.runtime.to_dict(),
        "architecture_config": architecture.to_dict(),
        "schedule_state": {
            "completed_snapshot_iterations": completed_snapshots,
            "final_strategy_network_trained": solver.final_strategy_network is not None,
        },
        "metric_logger_state": {
            "record_count": len(metrics),
            "last_iteration": metrics[-1]["iteration"] if metrics else 0,
        },
        # Optimisers are deliberately short-lived and absent at safe iteration boundaries.
        "optimizer_state_required": False,
    }
    payload = {
        "metadata": metadata,
        "advantage_networks": [_network_state(network) for network in solver.advantage_networks],
        "snapshot_networks": {
            str(iteration): _network_state(network)
            for iteration, network in solver.snapshot_networks.items()
        },
        "final_strategy_network": _network_state(solver.final_strategy_network),
        "optimizer_states": {},
        "advantage_reservoirs": [
            _pack_advantage_storage(
                reservoir,
                architecture.input_size,
                architecture.output_size,
                weighted_sampling=weighted_sampling,
            )
            for reservoir in solver.advantage_reservoirs
        ],
        "strategy_reservoir": _pack_strategy_storage(
            solver.strategy_reservoir,
            architecture.input_size,
            architecture.output_size,
            weighted_sampling=weighted_sampling,
        ),
        "rng_state": solver.training_rng_state(),
        "training_metrics": metrics,
        "run_state": {
            "elapsed_training_seconds": float(elapsed_training_seconds),
            "metric_records": [record.copy() for record in metric_records],
        },
    }
    with staged_atomic_binary_writer(path) as checkpoint_file:
        torch.save(payload, checkpoint_file)


def load_deep_cfr_checkpoint(
    path: Path,
    game: IndexedGameTree | HoldemConfig,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedDeepCFRCheckpoint:
    """Safely load, validate, and reconstruct the recorded Deep CFR implementation."""
    _game_metadata(game)
    try:
        payload = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("Deep CFR checkpoint is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != {
        "metadata",
        "advantage_networks",
        "snapshot_networks",
        "final_strategy_network",
        "optimizer_states",
        "advantage_reservoirs",
        "strategy_reservoir",
        "rng_state",
        "training_metrics",
        "run_state",
    }:
        raise ValueError("Deep CFR checkpoint fields are incomplete or unexpected")

    metadata = _validated_metadata(payload["metadata"], game)
    implementation = DeepCFRImplementationId(metadata["solver"])
    config = DeepCFRTrainingConfig.from_dict(metadata["training_config"])
    runtime = DeepCFRRuntimeConfig.from_dict(metadata["runtime_config"])
    architecture = deep_cfr_network_config(
        config.model_config_id,
        dropout_probability=config.dropout_probability,
    )
    expected_architecture = architecture.to_dict()
    if metadata["architecture_config"] != expected_architecture:
        raise ValueError("Deep CFR checkpoint architecture is incompatible")
    if payload["optimizer_states"] != {} or metadata["optimizer_state_required"] is not False:
        raise ValueError("Deep CFR checkpoint contains unexpected optimiser state")

    iteration = metadata["iteration"]
    metrics = _load_metrics(payload["training_metrics"])
    _validate_logger_state(metadata["metric_logger_state"], metrics)
    _validate_run_state(payload["run_state"], metadata)
    advantage_networks = _load_advantage_networks(payload["advantage_networks"], config)
    snapshot_networks = _load_snapshot_networks(payload["snapshot_networks"], config)
    final_strategy_network = _load_network(
        payload["final_strategy_network"],
        config,
        frozen=True,
        name="final strategy network",
    )
    for network in (
        *advantage_networks,
        *snapshot_networks.values(),
        final_strategy_network,
    ):
        if network is not None:
            network.to(runtime.device)
    rng_state = _validated_rng_state(payload["rng_state"])

    if isinstance(game, HoldemConfig) and implementation is not DeepCFRImplementationId.OPTIMISED:
        raise ValueError("modified HULHE checkpoints require the optimised implementation")
    if isinstance(game, HoldemConfig):
        from ac_cfr.solvers.deep_cfr import DeepCFR

        solver = DeepCFR(game, config, runtime)
    else:
        solver = deep_cfr_solver_type(implementation)(game, config, runtime)
    if metadata["checkpoint_schema_version"] >= 4 and isinstance(
        solver.advantage_reservoirs[0], PackedAdvantageReservoir
    ):
        _restore_packed_reservoirs(
            solver,
            payload["advantage_reservoirs"],
            payload["strategy_reservoir"],
            iteration,
            rng_state,
            weighted_sampling=config.opponent_exploration_epsilon > 0.0,
        )
    else:
        advantage_reservoirs = _load_advantage_reservoirs(
            payload["advantage_reservoirs"],
            config,
            iteration,
            architecture.input_size,
            architecture.output_size,
            weighted_sampling=config.opponent_exploration_epsilon > 0.0,
        )
        strategy_reservoir = _unpack_strategy_reservoir(
            payload["strategy_reservoir"],
            config.strategy_reservoir_capacity,
            iteration,
            architecture.input_size,
            architecture.output_size,
            weighted_sampling=config.opponent_exploration_epsilon > 0.0,
        )
        for player, samples in enumerate(advantage_reservoirs):
            reservoir_state = payload["advantage_reservoirs"][player]
            solver.advantage_reservoirs[player].restore_training_state(
                samples=samples,
                samples_seen=reservoir_state["samples_seen"],
                rng_state=rng_state[f"advantage_reservoir_{player}"],
            )
        solver.strategy_reservoir.restore_training_state(
            samples=strategy_reservoir,
            samples_seen=payload["strategy_reservoir"]["samples_seen"],
            rng_state=rng_state["strategy_reservoir"],
        )
    solver.restore_training_state(
        iteration=iteration,
        advantage_networks=advantage_networks,
        snapshot_networks=snapshot_networks,
        final_strategy_network=final_strategy_network,
        training_metrics=metrics,
        chance_rng_state=rng_state["chance"],
        policy_rng_state=rng_state["policy"],
    )
    return LoadedDeepCFRCheckpoint(
        metadata=metadata,
        solver=solver,
        run_state=payload["run_state"],
    )


def _validated_metadata(
    value: object,
    game_context: IndexedGameTree | HoldemConfig,
) -> dict[str, Any]:
    """Validate identifiers and metadata before constructing mutable solver state."""
    required_fields = {
        "checkpoint_schema_version",
        "project_version",
        "code_revision",
        "game",
        "game_version",
        "state_encoding",
        "action_space",
        "tree_digest",
        "solver",
        "model_config_id",
        "optimizer_id",
        "reservoir_schema_version",
        "rng_contract",
        "run_id",
        "checkpoint_id",
        "iteration",
        "training_config",
        "runtime_config",
        "architecture_config",
        "schedule_state",
        "metric_logger_state",
        "optimizer_state_required",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise ValueError("Deep CFR checkpoint metadata is incomplete or unexpected")
    metadata = value.copy()
    game, game_version, compatibility_digest = _game_metadata(game_context)
    expected = {
        "project_version": PROJECT_VERSION,
        "game": game,
        "game_version": game_version,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": compatibility_digest,
        "rng_contract": _RNG_CONTRACT,
    }
    if metadata["checkpoint_schema_version"] not in (3, DEEP_CFR_CHECKPOINT_SCHEMA_VERSION):
        raise ValueError("Deep CFR checkpoint has incompatible checkpoint_schema_version")
    for field_name, expected_value in expected.items():
        if metadata[field_name] != expected_value:
            raise ValueError(f"Deep CFR checkpoint has incompatible {field_name}")
    try:
        DeepCFRImplementationId(metadata["solver"])
    except (TypeError, ValueError) as error:
        raise ValueError("Deep CFR checkpoint has incompatible solver") from error
    for field_name in ("code_revision", "run_id", "checkpoint_id"):
        if not isinstance(metadata[field_name], str) or not metadata[field_name]:
            raise ValueError(f"Deep CFR checkpoint {field_name} is invalid")
    iteration = metadata["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Deep CFR checkpoint iteration is invalid")
    config = DeepCFRTrainingConfig.from_dict(metadata["training_config"])
    if metadata["reservoir_schema_version"] != _reservoir_schema_version(config):
        raise ValueError("Deep CFR checkpoint has incompatible reservoir_schema_version")
    DeepCFRRuntimeConfig.from_dict(metadata["runtime_config"])
    if iteration > config.iterations:
        raise ValueError("Deep CFR checkpoint iteration exceeds its training budget")
    config_identifiers = {
        "game_version": config.game_configuration_id.value,
        "state_encoding": config.state_encoding_id.value,
        "model_config_id": config.model_config_id.value,
        "optimizer_id": config.optimizer_id.value,
    }
    for field_name, expected_value in config_identifiers.items():
        if metadata[field_name] != expected_value:
            raise ValueError(f"Deep CFR checkpoint has incompatible {field_name}")
    expected_schedule = {
        "completed_snapshot_iterations": [
            item
            for item in config.snapshot_iterations
            if item <= iteration and item < config.iterations
        ],
        "final_strategy_network_trained": iteration == config.iterations,
    }
    if metadata["schedule_state"] != expected_schedule:
        raise ValueError("Deep CFR checkpoint schedule state is incompatible")
    return metadata


def _solver_game_metadata(solver: NaiveDeepCFR) -> tuple[str, str, str]:
    holdem_configuration = getattr(solver, "holdem_configuration", None)
    if isinstance(holdem_configuration, HoldemConfig):
        game, game_version, digest = _game_metadata(holdem_configuration)
        return game, game_version, digest
    return _game_metadata(solver.tree)


def _reservoir_schema_version(config: DeepCFRTrainingConfig) -> int:
    """Keep default and Hold'em checkpoints on the original reservoir schema."""
    if config.opponent_exploration_epsilon > 0.0:
        return DEEP_CFR_WEIGHTED_RESERVOIR_SCHEMA_VERSION
    return DEEP_CFR_RESERVOIR_SCHEMA_VERSION


def _game_metadata(
    game: IndexedGameTree | HoldemConfig,
) -> tuple[str, str, str]:
    if isinstance(game, IndexedGameTree):
        if game.game_id is not GameId.LEDUC:
            raise ValueError("indexed Deep CFR checkpoints require Leduc")
        return (
            GameId.LEDUC.value,
            GameConfigurationId.LEDUC.value,
            tree_compatibility_digest(game),
        )
    if isinstance(game, HoldemConfig):
        if game.configuration_id is not GameConfigurationId.MODIFIED_HULHE:
            raise ValueError("Hold'em Deep CFR checkpoints require modified HULHE")
        return (
            GameId.HOLD_EM.value,
            GameConfigurationId.MODIFIED_HULHE.value,
            holdem_compatibility_digest(game),
        )
    raise TypeError("game must be an IndexedGameTree or HoldemConfig")


def _network_state(network: DeepCFRNetwork | None) -> dict[str, Tensor] | None:
    if network is None:
        return None
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


def _load_advantage_networks(
    value: object,
    config: DeepCFRTrainingConfig,
) -> tuple[DeepCFRNetwork | None, DeepCFRNetwork | None]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Deep CFR checkpoint advantage networks are invalid")
    return (
        _load_network(value[0], config, frozen=False, name="Player 0 advantage network"),
        _load_network(value[1], config, frozen=False, name="Player 1 advantage network"),
    )


def _load_snapshot_networks(
    value: object,
    config: DeepCFRTrainingConfig,
) -> dict[int, DeepCFRNetwork]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Deep CFR checkpoint milestone networks are invalid")
    networks: dict[int, DeepCFRNetwork] = {}
    for key, state in value.items():
        try:
            iteration = int(key)
        except ValueError as error:
            raise ValueError("Deep CFR checkpoint milestone iteration is invalid") from error
        if str(iteration) != key or iteration in networks:
            raise ValueError("Deep CFR checkpoint milestone iteration is invalid")
        network = _load_network(state, config, frozen=True, name="milestone strategy network")
        if network is None:
            raise ValueError("Deep CFR checkpoint milestone network is missing")
        networks[iteration] = network
    return networks


def _load_network(
    value: object,
    config: DeepCFRTrainingConfig,
    *,
    frozen: bool,
    name: str,
) -> DeepCFRNetwork | None:
    if value is None:
        return None
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(tensor, Tensor) for key, tensor in value.items()
    ):
        raise ValueError(f"Deep CFR checkpoint {name} state is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        network = build_deep_cfr_network(
            config.model_config_id,
            dropout_probability=config.dropout_probability,
        )
    expected_state = network.state_dict()
    if set(value) != set(expected_state):
        raise ValueError(f"Deep CFR checkpoint {name} fields are incompatible")
    for key, tensor in value.items():
        if tensor.shape != expected_state[key].shape or tensor.dtype != expected_state[key].dtype:
            raise ValueError(f"Deep CFR checkpoint {name} tensor {key} is incompatible")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Deep CFR checkpoint {name} tensors must be finite")
    try:
        network.load_state_dict(value, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Deep CFR checkpoint {name} state is incompatible") from error
    network.eval()
    if frozen:
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    return network


def _pack_advantage_reservoir(
    capacity: int,
    samples_seen: int,
    samples: tuple[AdvantageSample, ...],
    state_size: int,
    action_count: int,
    *,
    weighted_sampling: bool,
) -> dict[str, object]:
    values = {
        "capacity": capacity,
        "samples_seen": samples_seen,
        "states": _states_tensor(samples, state_size),
        "action_masks": _masks_tensor(samples, action_count),
        "iterations": torch.tensor([sample.iteration for sample in samples], dtype=torch.int64),
        "advantages": _action_values_tensor(
            [sample.advantages for sample in samples], len(samples), action_count
        ),
    }
    if weighted_sampling:
        values["sampling_weights"] = torch.tensor(
            [sample.sampling_weight for sample in samples], dtype=torch.float64
        )
    return values


def _pack_advantage_storage(
    reservoir: UniformReservoir[AdvantageSample] | PackedAdvantageReservoir,
    state_size: int,
    action_count: int,
    *,
    weighted_sampling: bool,
) -> dict[str, object]:
    """Pack a reservoir directly when contiguous arrays are available."""
    if not isinstance(reservoir, PackedAdvantageReservoir):
        return _pack_advantage_reservoir(
            reservoir.capacity,
            reservoir.samples_seen,
            reservoir.samples,
            state_size,
            action_count,
            weighted_sampling=weighted_sampling,
        )
    states, action_masks, iterations, advantages = reservoir.arrays
    values = {
        "capacity": reservoir.capacity,
        "samples_seen": reservoir.samples_seen,
        "states": torch.from_numpy(states),
        "action_masks": torch.from_numpy(action_masks),
        "iterations": torch.from_numpy(iterations.astype("int64")),
        "advantages": torch.from_numpy(advantages),
    }
    if weighted_sampling:
        sampling_weights = reservoir.sampling_weights
        if sampling_weights is None:
            raise ValueError("weighted checkpoint requires reservoir sampling weights")
        values["sampling_weights"] = torch.from_numpy(sampling_weights)
    return values


def _pack_strategy_reservoir(
    capacity: int,
    samples_seen: int,
    samples: tuple[StrategySample, ...],
    state_size: int,
    action_count: int,
    *,
    weighted_sampling: bool,
) -> dict[str, object]:
    values = {
        "capacity": capacity,
        "samples_seen": samples_seen,
        "players": torch.tensor([sample.player for sample in samples], dtype=torch.int8),
        "states": _states_tensor(samples, state_size),
        "action_masks": _masks_tensor(samples, action_count),
        "iterations": torch.tensor([sample.iteration for sample in samples], dtype=torch.int64),
        "strategies": _action_values_tensor(
            [sample.strategy for sample in samples], len(samples), action_count
        ),
    }
    if weighted_sampling:
        values["sampling_weights"] = torch.tensor(
            [sample.sampling_weight for sample in samples], dtype=torch.float64
        )
    return values


def _pack_strategy_storage(
    reservoir: UniformReservoir[StrategySample] | PackedStrategyReservoir,
    state_size: int,
    action_count: int,
    *,
    weighted_sampling: bool,
) -> dict[str, object]:
    """Pack a strategy reservoir without constructing millions of objects."""
    if not isinstance(reservoir, PackedStrategyReservoir):
        return _pack_strategy_reservoir(
            reservoir.capacity,
            reservoir.samples_seen,
            reservoir.samples,
            state_size,
            action_count,
            weighted_sampling=weighted_sampling,
        )
    players, states, action_masks, iterations, strategies = reservoir.arrays
    values = {
        "capacity": reservoir.capacity,
        "samples_seen": reservoir.samples_seen,
        "players": torch.from_numpy(players),
        "states": torch.from_numpy(states),
        "action_masks": torch.from_numpy(action_masks),
        "iterations": torch.from_numpy(iterations.astype("int64")),
        "strategies": torch.from_numpy(strategies),
    }
    if weighted_sampling:
        sampling_weights = reservoir.sampling_weights
        if sampling_weights is None:
            raise ValueError("weighted checkpoint requires reservoir sampling weights")
        values["sampling_weights"] = torch.from_numpy(sampling_weights)
    return values


def _states_tensor(
    samples: tuple[AdvantageSample, ...] | tuple[StrategySample, ...],
    state_size: int,
) -> Tensor:
    if not samples:
        return torch.empty((0, state_size), dtype=torch.float32)
    return torch.tensor([sample.state for sample in samples], dtype=torch.float32)


def _masks_tensor(
    samples: tuple[AdvantageSample, ...] | tuple[StrategySample, ...],
    action_count: int,
) -> Tensor:
    if not samples:
        return torch.empty((0, action_count), dtype=torch.bool)
    return torch.tensor([sample.action_mask for sample in samples], dtype=torch.bool)


def _action_values_tensor(
    values: list[tuple[float, ...]],
    sample_count: int,
    action_count: int,
) -> Tensor:
    if not values:
        return torch.empty((0, action_count), dtype=torch.float64)
    return torch.tensor(values, dtype=torch.float64).reshape(sample_count, action_count)


def _load_advantage_reservoirs(
    value: object,
    config: DeepCFRTrainingConfig,
    checkpoint_iteration: int,
    state_size: int,
    action_count: int,
    *,
    weighted_sampling: bool,
) -> tuple[tuple[AdvantageSample, ...], tuple[AdvantageSample, ...]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Deep CFR checkpoint advantage reservoirs are invalid")
    return (
        _unpack_advantage_reservoir(
            value[0],
            config.advantage_reservoir_capacity,
            checkpoint_iteration,
            state_size,
            action_count,
            weighted_sampling=weighted_sampling,
        ),
        _unpack_advantage_reservoir(
            value[1],
            config.advantage_reservoir_capacity,
            checkpoint_iteration,
            state_size,
            action_count,
            weighted_sampling=weighted_sampling,
        ),
    )


def _restore_packed_reservoirs(
    solver: NaiveDeepCFR,
    advantage_values: object,
    strategy_value: object,
    checkpoint_iteration: int,
    rng_state: dict[str, object],
    *,
    weighted_sampling: bool,
) -> None:
    """Restore contiguous optimiser memories without constructing sample objects."""
    advantage_reservoirs = solver.advantage_reservoirs
    strategy_reservoir = solver.strategy_reservoir
    if not all(
        isinstance(reservoir, PackedAdvantageReservoir) for reservoir in advantage_reservoirs
    ) or not isinstance(strategy_reservoir, PackedStrategyReservoir):
        raise TypeError("solver reservoirs are not packed")
    if not isinstance(advantage_values, list) or len(advantage_values) != 2:
        raise ValueError("Deep CFR checkpoint advantage reservoirs are invalid")

    for player, reservoir in enumerate(advantage_reservoirs):
        assert isinstance(reservoir, PackedAdvantageReservoir)
        raw_reservoir = advantage_values[player]
        if not isinstance(raw_reservoir, dict):
            raise ValueError("Deep CFR checkpoint advantage reservoir is invalid")
        state_dtype = _torch_dtype(reservoir.arrays[0].dtype)
        fields = ("states", "action_masks", "iterations", "advantages")
        if weighted_sampling:
            fields += ("sampling_weights",)
        tensors, sample_count = _validated_reservoir(
            raw_reservoir,
            reservoir.capacity,
            fields,
            state_size=reservoir.state_size,
            action_count=reservoir.action_count,
            state_dtype=state_dtype,
        )
        _validate_tensor(
            tensors["advantages"],
            (sample_count, reservoir.action_count),
            torch.float32,
            "advantages",
        )
        iterations = _packed_iterations(tensors["iterations"], checkpoint_iteration)
        sampling_weights = _packed_sampling_weights(tensors, sample_count, weighted_sampling)
        reservoir.restore_arrays(
            states=tensors["states"].cpu().numpy(),
            action_masks=tensors["action_masks"].cpu().numpy(),
            iterations=iterations,
            advantages=tensors["advantages"].cpu().numpy(),
            sampling_weights=sampling_weights,
            samples_seen=raw_reservoir["samples_seen"],
            rng_state=rng_state[f"advantage_reservoir_{player}"],
        )

    if not isinstance(strategy_value, dict):
        raise ValueError("Deep CFR checkpoint strategy reservoir is invalid")
    state_dtype = _torch_dtype(strategy_reservoir.arrays[1].dtype)
    fields = ("players", "states", "action_masks", "iterations", "strategies")
    if weighted_sampling:
        fields += ("sampling_weights",)
    tensors, sample_count = _validated_reservoir(
        strategy_value,
        strategy_reservoir.capacity,
        fields,
        state_size=strategy_reservoir.state_size,
        action_count=strategy_reservoir.action_count,
        state_dtype=state_dtype,
    )
    _validate_tensor(tensors["players"], (sample_count,), torch.int8, "players")
    _validate_tensor(
        tensors["strategies"],
        (sample_count, strategy_reservoir.action_count),
        torch.float32,
        "strategies",
    )
    strategy_reservoir.restore_arrays(
        players=tensors["players"].cpu().numpy(),
        states=tensors["states"].cpu().numpy(),
        action_masks=tensors["action_masks"].cpu().numpy(),
        iterations=_packed_iterations(tensors["iterations"], checkpoint_iteration),
        strategies=tensors["strategies"].cpu().numpy(),
        sampling_weights=_packed_sampling_weights(tensors, sample_count, weighted_sampling),
        samples_seen=strategy_value["samples_seen"],
        rng_state=rng_state["strategy_reservoir"],
    )


def _packed_iterations(
    values: Tensor,
    checkpoint_iteration: int,
) -> NDArray[np.uint32]:
    _validate_tensor(values, (len(values),), torch.int64, "iterations")
    if len(values) and (int(values.min()) < 1 or int(values.max()) > checkpoint_iteration):
        raise ValueError("Deep CFR checkpoint reservoir contains invalid sample iterations")
    return values.cpu().numpy().astype("uint32")


def _packed_sampling_weights(
    tensors: dict[str, Tensor],
    sample_count: int,
    weighted_sampling: bool,
) -> NDArray[np.float32] | None:
    if not weighted_sampling:
        return None
    values = tensors["sampling_weights"]
    _validate_tensor(values, (sample_count,), torch.float32, "sampling_weights")
    if bool((values < 0.0).any()):
        raise ValueError("Deep CFR checkpoint sampling weights must be non-negative")
    return values.cpu().numpy()


def _torch_dtype(dtype: np.dtype[Any]) -> torch.dtype:
    if dtype == np.dtype(np.float16):
        return torch.float16
    if dtype == np.dtype(np.float32):
        return torch.float32
    raise ValueError("packed state dtype is unsupported")


def _unpack_advantage_reservoir(
    value: object,
    expected_capacity: int,
    checkpoint_iteration: int,
    state_size: int,
    action_count: int,
    *,
    weighted_sampling: bool,
) -> tuple[AdvantageSample, ...]:
    fields = ("states", "action_masks", "iterations", "advantages")
    if weighted_sampling:
        fields += ("sampling_weights",)
    tensors, sample_count = _validated_reservoir(
        value,
        expected_capacity,
        fields,
        state_size=state_size,
        action_count=action_count,
    )
    _validate_tensor(
        tensors["advantages"],
        (sample_count, action_count),
        torch.float64,
        "advantages",
    )
    if weighted_sampling:
        _validate_sampling_weight_tensor(tensors["sampling_weights"], sample_count)
    samples = tuple(
        AdvantageSample(
            state=tuple(float(item) for item in tensors["states"][row].tolist()),
            action_mask=tuple(bool(item) for item in tensors["action_masks"][row].tolist()),
            iteration=int(tensors["iterations"][row]),
            advantages=tuple(float(item) for item in tensors["advantages"][row].tolist()),
            sampling_weight=(float(tensors["sampling_weights"][row]) if weighted_sampling else 1.0),
        )
        for row in range(sample_count)
    )
    _validate_sample_iterations(samples, checkpoint_iteration)
    return samples


def _unpack_strategy_reservoir(
    value: object,
    expected_capacity: int,
    checkpoint_iteration: int,
    state_size: int,
    action_count: int,
    *,
    weighted_sampling: bool,
) -> tuple[StrategySample, ...]:
    fields = ("players", "states", "action_masks", "iterations", "strategies")
    if weighted_sampling:
        fields += ("sampling_weights",)
    tensors, sample_count = _validated_reservoir(
        value,
        expected_capacity,
        fields,
        state_size=state_size,
        action_count=action_count,
    )
    _validate_tensor(tensors["players"], (sample_count,), torch.int8, "players")
    _validate_tensor(
        tensors["strategies"],
        (sample_count, action_count),
        torch.float64,
        "strategies",
    )
    if weighted_sampling:
        _validate_sampling_weight_tensor(tensors["sampling_weights"], sample_count)
    samples = tuple(
        StrategySample(
            player=int(tensors["players"][row]),
            state=tuple(float(item) for item in tensors["states"][row].tolist()),
            action_mask=tuple(bool(item) for item in tensors["action_masks"][row].tolist()),
            iteration=int(tensors["iterations"][row]),
            strategy=tuple(float(item) for item in tensors["strategies"][row].tolist()),
            sampling_weight=(float(tensors["sampling_weights"][row]) if weighted_sampling else 1.0),
        )
        for row in range(sample_count)
    )
    _validate_sample_iterations(samples, checkpoint_iteration)
    return samples


def _validated_reservoir(
    value: object,
    expected_capacity: int,
    tensor_fields: tuple[str, ...],
    *,
    state_size: int = LEDUC_NEURAL_STATE_SIZE,
    action_count: int = LEDUC_ACTION_COUNT,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, Tensor], int]:
    expected_fields = {"capacity", "samples_seen", *tensor_fields}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("Deep CFR checkpoint reservoir fields are incompatible")
    if value["capacity"] != expected_capacity:
        raise ValueError("Deep CFR checkpoint reservoir capacity is incompatible")
    samples_seen = value["samples_seen"]
    if isinstance(samples_seen, bool) or not isinstance(samples_seen, int) or samples_seen < 0:
        raise ValueError("Deep CFR checkpoint reservoir sample count is invalid")
    tensors = {name: value[name] for name in tensor_fields}
    if any(not isinstance(tensor, Tensor) for tensor in tensors.values()):
        raise ValueError("Deep CFR checkpoint reservoir arrays are invalid")
    first_tensor = tensors[tensor_fields[0]]
    if first_tensor.ndim == 0:
        raise ValueError("Deep CFR checkpoint reservoir arrays are invalid")
    sample_count = int(first_tensor.shape[0])
    if sample_count != min(samples_seen, expected_capacity):
        raise ValueError("Deep CFR checkpoint reservoir occupancy is inconsistent")
    _validate_tensor(
        tensors["states"],
        (sample_count, state_size),
        state_dtype,
        "states",
    )
    _validate_tensor(
        tensors["action_masks"],
        (sample_count, action_count),
        torch.bool,
        "action_masks",
    )
    _validate_tensor(tensors["iterations"], (sample_count,), torch.int64, "iterations")
    return tensors, sample_count


def _validate_tensor(tensor: Tensor, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> None:
    if tensor.shape != shape or tensor.dtype != dtype:
        raise ValueError(f"Deep CFR checkpoint reservoir {name} is incompatible")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Deep CFR checkpoint reservoir {name} must be finite")


def _validate_sampling_weight_tensor(values: Tensor, sample_count: int) -> None:
    _validate_tensor(values, (sample_count,), torch.float64, "sampling_weights")
    if bool((values < 0.0).any()):
        raise ValueError("Deep CFR checkpoint sampling weights must be non-negative")


def _validate_sample_iterations(
    samples: tuple[AdvantageSample, ...] | tuple[StrategySample, ...],
    checkpoint_iteration: int,
) -> None:
    if any(sample.iteration > checkpoint_iteration for sample in samples):
        raise ValueError("Deep CFR checkpoint reservoir contains future samples")


def _validated_rng_state(value: object) -> dict[str, object]:
    expected_fields = {
        "chance",
        "policy",
        "advantage_reservoir_0",
        "advantage_reservoir_1",
        "strategy_reservoir",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("Deep CFR checkpoint RNG state is incomplete or unexpected")
    return value.copy()


def _load_metrics(value: object) -> tuple[NetworkTrainingMetrics, ...]:
    if not isinstance(value, list):
        raise ValueError("Deep CFR checkpoint training metrics are invalid")
    return tuple(NetworkTrainingMetrics.from_dict(record) for record in value)


def _validate_logger_state(
    value: object,
    metrics: tuple[NetworkTrainingMetrics, ...],
) -> None:
    expected = {
        "record_count": len(metrics),
        "last_iteration": metrics[-1].iteration if metrics else 0,
    }
    if value != expected:
        raise ValueError("Deep CFR checkpoint metric logger state is inconsistent")


def _validate_run_state(value: object, metadata: dict[str, Any]) -> None:
    """Validate resume-safe elapsed time and compact periodic metric records."""
    if not isinstance(value, dict) or set(value) != {
        "elapsed_training_seconds",
        "metric_records",
    }:
        raise ValueError("Deep CFR checkpoint run state is incomplete or unexpected")
    elapsed = value["elapsed_training_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not isfinite(elapsed)
        or elapsed < 0.0
    ):
        raise ValueError("Deep CFR checkpoint elapsed training time is invalid")
    records = value["metric_records"]
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("Deep CFR checkpoint metric records are invalid")
    for record in records:
        try:
            run_id = record["run_id"]
            iteration = int(record["iteration"])
            seed = int(record["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Deep CFR checkpoint metric record is invalid") from error
        if (
            run_id != metadata["run_id"]
            or iteration < 1
            or iteration > metadata["iteration"]
            or seed != metadata["training_config"]["seed"]
        ):
            raise ValueError("Deep CFR checkpoint metric record is inconsistent")
