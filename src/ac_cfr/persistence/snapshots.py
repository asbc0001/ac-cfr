"""Safe tabular average-strategy snapshot export and loading."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.tabular import TabularGame
from ac_cfr.persistence.compatibility import (
    ACTION_SPACE_ID,
    TABULAR_MODEL_CONFIG_ID,
    tree_compatibility_digest,
)
from ac_cfr.persistence.files import atomic_binary_writer

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TabularSnapshotMetadata:
    """Compatibility and provenance data stored beside one average policy."""

    artifact_schema_version: int
    snapshot_id: str
    game: str
    game_version: str
    solver: str
    agent_type: str
    training_iteration: int
    run_id: str
    seed: int
    source_checkpoint_id: str
    model_config_id: str
    state_encoding: str
    action_space: str
    tree_digest: str


@dataclass(frozen=True, slots=True)
class LoadedTabularSnapshot:
    """Validated metadata and policy from a non-executable NPZ snapshot."""

    metadata: TabularSnapshotMetadata
    average_policy: NDArray[np.float64]


def export_tabular_snapshot(
    path: Path,
    *,
    tabular_game: TabularGame,
    average_policy: NDArray[np.float64],
    snapshot_id: str,
    solver: str,
    iteration: int,
    run_id: str,
    seed: int,
    source_checkpoint_id: str,
) -> TabularSnapshotMetadata:
    """Atomically export a validated playable average policy."""
    policy = _validated_policy(average_policy, tabular_game)
    metadata = TabularSnapshotMetadata(
        artifact_schema_version=SNAPSHOT_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        game=tabular_game.game_id.value,
        game_version=tabular_game.configuration_id.value,
        solver=solver,
        agent_type="tabular",
        training_iteration=iteration,
        run_id=run_id,
        seed=seed,
        source_checkpoint_id=source_checkpoint_id,
        model_config_id=TABULAR_MODEL_CONFIG_ID,
        state_encoding=tabular_game.state_encoding_id.value,
        action_space=ACTION_SPACE_ID,
        tree_digest=tree_compatibility_digest(tabular_game.tree),
    )
    metadata = _parse_metadata(asdict(metadata))
    _validate_compatibility(metadata, tabular_game)
    metadata_json = json.dumps(asdict(metadata), sort_keys=True, separators=(",", ":"))
    with atomic_binary_writer(path) as snapshot_file:
        np.savez_compressed(
            snapshot_file,
            metadata=np.asarray(metadata_json),
            average_policy=policy,
        )
    return metadata


def load_tabular_snapshot(path: Path, tabular_game: TabularGame) -> LoadedTabularSnapshot:
    """Load a tabular snapshot and reject incompatible or unsafe contents."""
    try:
        with np.load(path, allow_pickle=False) as snapshot:
            if set(snapshot.files) != {"metadata", "average_policy"}:
                raise ValueError("strategy snapshot contains unexpected fields")
            metadata_value = snapshot["metadata"]
            if metadata_value.shape != () or metadata_value.dtype.kind not in {"U", "S"}:
                raise ValueError("strategy snapshot metadata has an invalid format")
            raw_metadata = json.loads(str(metadata_value.item()))
            metadata = _parse_metadata(raw_metadata)
            policy = _validated_policy(snapshot["average_policy"], tabular_game)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("strategy snapshot is unreadable") from error

    _validate_compatibility(metadata, tabular_game)
    policy.setflags(write=False)
    return LoadedTabularSnapshot(metadata, policy)


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_metadata(raw_metadata: object) -> TabularSnapshotMetadata:
    if not isinstance(raw_metadata, dict):
        raise ValueError("strategy snapshot metadata must be an object")
    expected_fields = set(TabularSnapshotMetadata.__dataclass_fields__)
    if set(raw_metadata) != expected_fields:
        raise ValueError("strategy snapshot metadata fields are incomplete or unexpected")
    try:
        metadata = TabularSnapshotMetadata(**raw_metadata)
    except TypeError as error:
        raise ValueError("strategy snapshot metadata has invalid values") from error
    string_fields = (
        metadata.snapshot_id,
        metadata.game,
        metadata.game_version,
        metadata.solver,
        metadata.agent_type,
        metadata.run_id,
        metadata.source_checkpoint_id,
        metadata.model_config_id,
        metadata.state_encoding,
        metadata.action_space,
        metadata.tree_digest,
    )
    if any(not isinstance(value, str) or not value for value in string_fields):
        raise ValueError("strategy snapshot identifiers must be non-empty strings")
    integer_fields = (
        metadata.artifact_schema_version,
        metadata.training_iteration,
        metadata.seed,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
        raise ValueError("strategy snapshot schema and iteration must be integers")
    return metadata


def _validate_compatibility(
    metadata: TabularSnapshotMetadata,
    tabular_game: TabularGame,
) -> None:
    expected = {
        "artifact_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "game": tabular_game.game_id.value,
        "game_version": tabular_game.configuration_id.value,
        "agent_type": "tabular",
        "model_config_id": TABULAR_MODEL_CONFIG_ID,
        "state_encoding": tabular_game.state_encoding_id.value,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": tree_compatibility_digest(tabular_game.tree),
    }
    for field_name, expected_value in expected.items():
        if getattr(metadata, field_name) != expected_value:
            raise ValueError(f"strategy snapshot has incompatible {field_name}")
    if metadata.training_iteration < 0:
        raise ValueError("strategy snapshot iteration must not be negative")
    if not metadata.snapshot_id or not metadata.run_id or not metadata.source_checkpoint_id:
        raise ValueError("strategy snapshot provenance identifiers must not be empty")


def _validated_policy(
    average_policy: NDArray[np.float64],
    tabular_game: TabularGame,
) -> NDArray[np.float64]:
    if not isinstance(average_policy, np.ndarray):
        raise TypeError("average_policy must be a NumPy array")
    expected_size = len(tabular_game.tree.information_set_actions)
    if average_policy.shape != (expected_size,):
        raise ValueError("average_policy has an incompatible shape")
    policy = np.asarray(average_policy, dtype=np.float64)
    if not np.all(np.isfinite(policy)) or np.any(policy < 0.0):
        raise ValueError("average_policy must contain finite non-negative probabilities")
    for offset, count in zip(
        tabular_game.tree.information_set_action_offsets,
        tabular_game.tree.information_set_action_counts,
        strict=True,
    ):
        start = int(offset)
        if not np.isclose(policy[start : start + int(count)].sum(), 1.0, atol=1e-12):
            raise ValueError("each information-set policy must sum to one")
    return policy.copy()
