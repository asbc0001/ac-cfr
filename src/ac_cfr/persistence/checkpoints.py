"""Atomic, validated checkpoints for tabular CFR training."""

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.tabular import TabularGame
from ac_cfr.persistence.compatibility import ACTION_SPACE_ID, tree_compatibility_digest
from ac_cfr.persistence.files import atomic_binary_writer
from ac_cfr.persistence.results import RESULT_FIELDS, RESULT_KEY_FIELDS, ResultRecord
from ac_cfr.solvers.cfr import CFR
from ac_cfr.solvers.naive_cfr import NaiveCFR

CHECKPOINT_SCHEMA_VERSION = 1
PROJECT_VERSION = version("ac-cfr")


@dataclass(frozen=True, slots=True)
class LoadedTabularCheckpoint:
    """Fully parsed checkpoint data ready for solver reconstruction."""

    metadata: dict[str, Any]
    regret_sum: NDArray[np.float64]
    strategy_sum: NDArray[np.float64]


def save_tabular_checkpoint(
    path: Path,
    *,
    solver: NaiveCFR | CFR,
    tabular_game: TabularGame,
    solver_id: str,
    run_id: str,
    seed: int,
    training_config: dict[str, object],
    elapsed_training_seconds: float,
    checkpoint_id: str,
    schedule_state: dict[str, object],
    metric_records: tuple[ResultRecord, ...],
    code_revision: str,
) -> None:
    """Atomically save every value needed to continue tabular training."""
    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "code_revision": code_revision,
        "game": tabular_game.game_id.value,
        "game_version": tabular_game.configuration_id.value,
        "state_encoding": tabular_game.state_encoding_id.value,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": tree_compatibility_digest(tabular_game.tree),
        "solver": solver_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "iteration": solver.iteration,
        "seed": seed,
        "training_config": training_config,
        "elapsed_training_seconds": elapsed_training_seconds,
        "schedule_state": schedule_state,
        "metric_records": list(metric_records),
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    regret_sum = _flatten_table(solver.regret_sum)
    strategy_sum = _flatten_table(solver.strategy_sum)
    with atomic_binary_writer(path) as checkpoint_file:
        np.savez_compressed(
            checkpoint_file,
            metadata=np.asarray(metadata_json),
            regret_sum=regret_sum,
            strategy_sum=strategy_sum,
        )


def load_tabular_checkpoint(path: Path) -> LoadedTabularCheckpoint:
    """Read checkpoint data without executing or unpickling stored objects."""
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            if set(checkpoint.files) != {"metadata", "regret_sum", "strategy_sum"}:
                raise ValueError("checkpoint contains unexpected fields")
            metadata_value = checkpoint["metadata"]
            if metadata_value.shape != () or metadata_value.dtype.kind not in {"U", "S"}:
                raise ValueError("checkpoint metadata has an invalid format")
            metadata = json.loads(str(metadata_value.item()))
            regret_sum = _validated_array(checkpoint["regret_sum"], "regret_sum")
            strategy_sum = _validated_array(checkpoint["strategy_sum"], "strategy_sum")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint is unreadable") from error
    _validate_metadata(metadata)
    regret_sum.setflags(write=False)
    strategy_sum.setflags(write=False)
    return LoadedTabularCheckpoint(metadata, regret_sum, strategy_sum)


def validate_checkpoint_compatibility(
    checkpoint: LoadedTabularCheckpoint,
    tabular_game: TabularGame,
    solver_id: str,
) -> None:
    """Reject incompatible checkpoint metadata before restoring a solver."""
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "game": tabular_game.game_id.value,
        "game_version": tabular_game.configuration_id.value,
        "state_encoding": tabular_game.state_encoding_id.value,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": tree_compatibility_digest(tabular_game.tree),
        "solver": solver_id,
    }
    for field_name, expected_value in expected.items():
        if checkpoint.metadata[field_name] != expected_value:
            raise ValueError(f"checkpoint has incompatible {field_name}")
    expected_size = len(tabular_game.tree.information_set_actions)
    if checkpoint.regret_sum.shape != (expected_size,):
        raise ValueError("checkpoint regret_sum has an incompatible shape")
    if checkpoint.strategy_sum.shape != (expected_size,):
        raise ValueError("checkpoint strategy_sum has an incompatible shape")


def _flatten_table(table: tuple[tuple[float, ...], ...]) -> NDArray[np.float64]:
    return np.fromiter(
        (value for row in table for value in row),
        dtype=np.float64,
        count=sum(len(row) for row in table),
    )


def _validated_array(value: object, name: str) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise ValueError(f"checkpoint {name} must be a one-dimensional array")
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"checkpoint {name} must use float64")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"checkpoint {name} must contain only finite values")
    return value.copy()


def _validate_metadata(metadata: object) -> None:
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
        "run_id",
        "checkpoint_id",
        "iteration",
        "seed",
        "training_config",
        "elapsed_training_seconds",
        "schedule_state",
        "metric_records",
    }
    if not isinstance(metadata, dict) or set(metadata) != required_fields:
        raise ValueError("checkpoint metadata fields are incomplete or unexpected")
    string_fields = (
        "project_version",
        "code_revision",
        "game",
        "game_version",
        "state_encoding",
        "action_space",
        "tree_digest",
        "solver",
        "run_id",
        "checkpoint_id",
    )
    if any(not isinstance(metadata[field], str) or not metadata[field] for field in string_fields):
        raise ValueError("checkpoint identifiers must be non-empty strings")
    schema_version = metadata["checkpoint_schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("checkpoint schema version is invalid")
    if (
        isinstance(metadata["iteration"], bool)
        or not isinstance(metadata["iteration"], int)
        or metadata["iteration"] < 0
    ):
        raise ValueError("checkpoint iteration is invalid")
    if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int):
        raise ValueError("checkpoint seed is invalid")
    if not isinstance(metadata["training_config"], dict):
        raise ValueError("checkpoint training_config is invalid")
    if not isinstance(metadata["schedule_state"], dict):
        raise ValueError("checkpoint schedule_state is invalid")
    records = metadata["metric_records"]
    if not isinstance(records, list):
        raise ValueError("checkpoint metric_records is invalid")
    record_keys: set[tuple[str, ...]] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != set(RESULT_FIELDS):
            raise ValueError("checkpoint contains an invalid metric record")
        if any(not isinstance(value, str) for value in record.values()):
            raise ValueError("checkpoint metric values must be strings")
        required_result_fields = ("game", "game_version", "solver", "run_id", "iteration", "seed")
        if any(not record[field] for field in required_result_fields):
            raise ValueError("checkpoint metric record is missing a required field")
        record_key = tuple(record[field] for field in RESULT_KEY_FIELDS)
        if record_key in record_keys:
            raise ValueError("checkpoint contains duplicate metric records")
        record_keys.add(record_key)
