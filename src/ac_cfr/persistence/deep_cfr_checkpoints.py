"""Atomic, validated training checkpoints for reference Deep CFR."""

import pickle
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ac_cfr.common.config import GameConfigurationId
from ac_cfr.games.base import GameId
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, LEDUC_NEURAL_STATE_SIZE
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.models import DeepCFRNetwork, build_deep_cfr_network, deep_cfr_network_config
from ac_cfr.persistence.compatibility import ACTION_SPACE_ID, tree_compatibility_digest
from ac_cfr.persistence.files import atomic_binary_writer
from ac_cfr.solvers.naive_deep_cfr import NaiveDeepCFR, NetworkTrainingMetrics
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import (
    DEEP_CFR_RESERVOIR_SCHEMA_VERSION,
    AdvantageSample,
    StrategySample,
)

DEEP_CFR_CHECKPOINT_SCHEMA_VERSION = 1
PROJECT_VERSION = version("ac-cfr")
_SOLVER_ID = "naive_deep_cfr"
_RNG_CONTRACT = "python_random_and_derived_torch_v1"


@dataclass(frozen=True, slots=True)
class LoadedDeepCFRCheckpoint:
    """Validated metadata and a fully reconstructed reference solver."""

    metadata: dict[str, Any]
    solver: NaiveDeepCFR


def save_deep_cfr_checkpoint(
    path: Path,
    *,
    solver: NaiveDeepCFR,
    run_id: str,
    checkpoint_id: str,
    code_revision: str,
) -> None:
    """Atomically save every value needed at a completed outer iteration."""
    if not isinstance(solver, NaiveDeepCFR):
        raise TypeError("solver must be a NaiveDeepCFR")
    for name, value in (
        ("run_id", run_id),
        ("checkpoint_id", checkpoint_id),
        ("code_revision", code_revision),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    config = solver.config
    architecture = deep_cfr_network_config(
        config.model_config_id,
        dropout_probability=config.dropout_probability,
    )
    completed_snapshots = sorted(solver.snapshot_networks)
    metrics = [metric.to_dict() for metric in solver.training_metrics]
    metadata = {
        "checkpoint_schema_version": DEEP_CFR_CHECKPOINT_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "code_revision": code_revision,
        "game": GameId.LEDUC.value,
        "game_version": GameConfigurationId.LEDUC.value,
        "state_encoding": config.state_encoding_id.value,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": tree_compatibility_digest(solver.tree),
        "solver": _SOLVER_ID,
        "model_config_id": config.model_config_id.value,
        "optimizer_id": config.optimizer_id.value,
        "reservoir_schema_version": DEEP_CFR_RESERVOIR_SCHEMA_VERSION,
        "rng_contract": _RNG_CONTRACT,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "iteration": solver.iteration,
        "training_config": config.to_dict(),
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
            _pack_advantage_reservoir(reservoir.capacity, reservoir.samples_seen, reservoir.samples)
            for reservoir in solver.advantage_reservoirs
        ],
        "strategy_reservoir": _pack_strategy_reservoir(
            solver.strategy_reservoir.capacity,
            solver.strategy_reservoir.samples_seen,
            solver.strategy_reservoir.samples,
        ),
        "rng_state": solver.training_rng_state(),
        "training_metrics": metrics,
    }
    with atomic_binary_writer(path) as checkpoint_file:
        torch.save(payload, checkpoint_file)


def load_deep_cfr_checkpoint(
    path: Path,
    tree: IndexedGameTree,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedDeepCFRCheckpoint:
    """Safely load, validate, and reconstruct one reference Deep CFR solver."""
    if not isinstance(tree, IndexedGameTree):
        raise TypeError("tree must be an IndexedGameTree")
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
    }:
        raise ValueError("Deep CFR checkpoint fields are incomplete or unexpected")

    metadata = _validated_metadata(payload["metadata"], tree)
    config = DeepCFRTrainingConfig.from_dict(metadata["training_config"])
    expected_architecture = deep_cfr_network_config(
        config.model_config_id,
        dropout_probability=config.dropout_probability,
    ).to_dict()
    if metadata["architecture_config"] != expected_architecture:
        raise ValueError("Deep CFR checkpoint architecture is incompatible")
    if payload["optimizer_states"] != {} or metadata["optimizer_state_required"] is not False:
        raise ValueError("Deep CFR checkpoint contains unexpected optimiser state")

    iteration = metadata["iteration"]
    metrics = _load_metrics(payload["training_metrics"])
    _validate_logger_state(metadata["metric_logger_state"], metrics)
    advantage_networks = _load_advantage_networks(payload["advantage_networks"], config)
    snapshot_networks = _load_snapshot_networks(payload["snapshot_networks"], config)
    final_strategy_network = _load_network(
        payload["final_strategy_network"],
        config,
        frozen=True,
        name="final strategy network",
    )
    advantage_reservoirs = _load_advantage_reservoirs(
        payload["advantage_reservoirs"], config, iteration
    )
    strategy_reservoir = _unpack_strategy_reservoir(
        payload["strategy_reservoir"], config.strategy_reservoir_capacity, iteration
    )
    rng_state = _validated_rng_state(payload["rng_state"])

    solver = NaiveDeepCFR(tree, config)
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
    return LoadedDeepCFRCheckpoint(metadata=metadata, solver=solver)


def _validated_metadata(value: object, tree: IndexedGameTree) -> dict[str, Any]:
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
        "architecture_config",
        "schedule_state",
        "metric_logger_state",
        "optimizer_state_required",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise ValueError("Deep CFR checkpoint metadata is incomplete or unexpected")
    metadata = value.copy()
    expected = {
        "checkpoint_schema_version": DEEP_CFR_CHECKPOINT_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "game": GameId.LEDUC.value,
        "game_version": GameConfigurationId.LEDUC.value,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": tree_compatibility_digest(tree),
        "solver": _SOLVER_ID,
        "reservoir_schema_version": DEEP_CFR_RESERVOIR_SCHEMA_VERSION,
        "rng_contract": _RNG_CONTRACT,
    }
    for field_name, expected_value in expected.items():
        if metadata[field_name] != expected_value:
            raise ValueError(f"Deep CFR checkpoint has incompatible {field_name}")
    for field_name in ("code_revision", "run_id", "checkpoint_id"):
        if not isinstance(metadata[field_name], str) or not metadata[field_name]:
            raise ValueError(f"Deep CFR checkpoint {field_name} is invalid")
    iteration = metadata["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Deep CFR checkpoint iteration is invalid")
    config = DeepCFRTrainingConfig.from_dict(metadata["training_config"])
    if iteration > config.iterations:
        raise ValueError("Deep CFR checkpoint iteration exceeds its training budget")
    config_identifiers = {
        "state_encoding": config.state_encoding_id.value,
        "model_config_id": config.model_config_id.value,
        "optimizer_id": config.optimizer_id.value,
    }
    for field_name, expected_value in config_identifiers.items():
        if metadata[field_name] != expected_value:
            raise ValueError(f"Deep CFR checkpoint has incompatible {field_name}")
    expected_schedule = {
        "completed_snapshot_iterations": [
            item for item in config.snapshot_iterations if item <= iteration
        ],
        "final_strategy_network_trained": iteration == config.iterations,
    }
    if metadata["schedule_state"] != expected_schedule:
        raise ValueError("Deep CFR checkpoint schedule state is incompatible")
    return metadata


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
) -> dict[str, object]:
    return {
        "capacity": capacity,
        "samples_seen": samples_seen,
        "states": _states_tensor(samples),
        "action_masks": _masks_tensor(samples),
        "iterations": torch.tensor([sample.iteration for sample in samples], dtype=torch.int64),
        "advantages": _action_values_tensor(
            [sample.advantages for sample in samples], len(samples)
        ),
    }


def _pack_strategy_reservoir(
    capacity: int,
    samples_seen: int,
    samples: tuple[StrategySample, ...],
) -> dict[str, object]:
    return {
        "capacity": capacity,
        "samples_seen": samples_seen,
        "players": torch.tensor([sample.player for sample in samples], dtype=torch.int8),
        "states": _states_tensor(samples),
        "action_masks": _masks_tensor(samples),
        "iterations": torch.tensor([sample.iteration for sample in samples], dtype=torch.int64),
        "strategies": _action_values_tensor([sample.strategy for sample in samples], len(samples)),
    }


def _states_tensor(samples: tuple[AdvantageSample, ...] | tuple[StrategySample, ...]) -> Tensor:
    if not samples:
        return torch.empty((0, LEDUC_NEURAL_STATE_SIZE), dtype=torch.float32)
    return torch.tensor([sample.state for sample in samples], dtype=torch.float32)


def _masks_tensor(samples: tuple[AdvantageSample, ...] | tuple[StrategySample, ...]) -> Tensor:
    if not samples:
        return torch.empty((0, LEDUC_ACTION_COUNT), dtype=torch.bool)
    return torch.tensor([sample.action_mask for sample in samples], dtype=torch.bool)


def _action_values_tensor(values: list[tuple[float, ...]], sample_count: int) -> Tensor:
    if not values:
        return torch.empty((0, LEDUC_ACTION_COUNT), dtype=torch.float64)
    return torch.tensor(values, dtype=torch.float64).reshape(sample_count, LEDUC_ACTION_COUNT)


def _load_advantage_reservoirs(
    value: object,
    config: DeepCFRTrainingConfig,
    checkpoint_iteration: int,
) -> tuple[tuple[AdvantageSample, ...], tuple[AdvantageSample, ...]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Deep CFR checkpoint advantage reservoirs are invalid")
    return (
        _unpack_advantage_reservoir(
            value[0], config.advantage_reservoir_capacity, checkpoint_iteration
        ),
        _unpack_advantage_reservoir(
            value[1], config.advantage_reservoir_capacity, checkpoint_iteration
        ),
    )


def _unpack_advantage_reservoir(
    value: object,
    expected_capacity: int,
    checkpoint_iteration: int,
) -> tuple[AdvantageSample, ...]:
    tensors, sample_count = _validated_reservoir(
        value,
        expected_capacity,
        ("states", "action_masks", "iterations", "advantages"),
    )
    _validate_tensor(
        tensors["advantages"],
        (sample_count, LEDUC_ACTION_COUNT),
        torch.float64,
        "advantages",
    )
    samples = tuple(
        AdvantageSample(
            state=tuple(float(item) for item in tensors["states"][row].tolist()),
            action_mask=tuple(bool(item) for item in tensors["action_masks"][row].tolist()),
            iteration=int(tensors["iterations"][row]),
            advantages=tuple(float(item) for item in tensors["advantages"][row].tolist()),
        )
        for row in range(sample_count)
    )
    _validate_sample_iterations(samples, checkpoint_iteration)
    return samples


def _unpack_strategy_reservoir(
    value: object,
    expected_capacity: int,
    checkpoint_iteration: int,
) -> tuple[StrategySample, ...]:
    tensors, sample_count = _validated_reservoir(
        value,
        expected_capacity,
        ("players", "states", "action_masks", "iterations", "strategies"),
    )
    _validate_tensor(tensors["players"], (sample_count,), torch.int8, "players")
    _validate_tensor(
        tensors["strategies"],
        (sample_count, LEDUC_ACTION_COUNT),
        torch.float64,
        "strategies",
    )
    samples = tuple(
        StrategySample(
            player=int(tensors["players"][row]),
            state=tuple(float(item) for item in tensors["states"][row].tolist()),
            action_mask=tuple(bool(item) for item in tensors["action_masks"][row].tolist()),
            iteration=int(tensors["iterations"][row]),
            strategy=tuple(float(item) for item in tensors["strategies"][row].tolist()),
        )
        for row in range(sample_count)
    )
    _validate_sample_iterations(samples, checkpoint_iteration)
    return samples


def _validated_reservoir(
    value: object,
    expected_capacity: int,
    tensor_fields: tuple[str, ...],
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
        (sample_count, LEDUC_NEURAL_STATE_SIZE),
        torch.float32,
        "states",
    )
    _validate_tensor(
        tensors["action_masks"],
        (sample_count, LEDUC_ACTION_COUNT),
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
