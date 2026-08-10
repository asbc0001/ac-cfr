"""Validated strategy registry and trusted playable-agent resolution."""

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.typing import NDArray

from ac_cfr.agents.base import PlayableAgent
from ac_cfr.agents.baselines import BaselineAgent
from ac_cfr.agents.tabular import TabularAgent
from ac_cfr.games.base import GameId
from ac_cfr.games.tabular import TabularGame, create_tabular_game
from ac_cfr.persistence.compatibility import ACTION_SPACE_ID, TABULAR_MODEL_CONFIG_ID
from ac_cfr.persistence.snapshots import (
    SNAPSHOT_SCHEMA_VERSION,
    TabularSnapshotMetadata,
    file_sha256,
    load_tabular_snapshot,
)

REGISTRY_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class StrategyRegistryEntry:
    """One validated playable strategy description."""

    strategy_id: str
    label: str
    game: str
    game_version: str
    algorithm: str
    agent_type: str
    snapshot_id: str | None
    training_iteration: int
    local_path: str | None
    evaluation: dict[str, object]
    model_config_id: str
    state_encoding: str
    action_space: str
    tree_digest: str | None
    artifact_schema_version: int | None
    release_id: str | None
    file_size: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ResolvedStrategy:
    """A registry entry, its canonical game, and its loaded agent."""

    entry: StrategyRegistryEntry
    tabular_game: TabularGame
    agent: PlayableAgent
    policy: NDArray[np.float64]
    snapshot_metadata: TabularSnapshotMetadata | None


class StrategyRegistry:
    """Resolve only schema-validated, registry-owned strategy identifiers."""

    __slots__ = ("_entries", "_project_root")

    def __init__(
        self,
        entries: tuple[StrategyRegistryEntry, ...],
        project_root: Path,
    ) -> None:
        self._entries = {entry.strategy_id: entry for entry in entries}
        self._project_root = project_root.resolve()

    @property
    def entries(self) -> tuple[StrategyRegistryEntry, ...]:
        """Return registry entries in their declared order."""
        return tuple(self._entries.values())

    def resolve(self, strategy_id: str) -> ResolvedStrategy:
        """Validate and load one named baseline or tabular strategy."""
        try:
            entry = self._entries[strategy_id]
        except KeyError as error:
            raise ValueError(f"unknown strategy_id: {strategy_id}") from error
        tabular_game = create_tabular_game(GameId(entry.game))
        if entry.game_version != tabular_game.configuration_id.value:
            raise ValueError("registry entry has an incompatible game_version")
        if entry.state_encoding != tabular_game.state_encoding_id.value:
            raise ValueError("registry entry has an incompatible state_encoding")
        if entry.action_space != ACTION_SPACE_ID:
            raise ValueError("registry entry has an incompatible action_space")

        if entry.agent_type == "baseline":
            policy = _uniform_policy(tabular_game)
            policy.setflags(write=False)
            return ResolvedStrategy(entry, tabular_game, BaselineAgent(), policy, None)
        if entry.agent_type != "tabular":
            raise ValueError(f"unsupported agent_type: {entry.agent_type}")

        path = self._resolve_file_path(entry)
        snapshot = load_tabular_snapshot(path, tabular_game)
        expected_metadata = {
            "snapshot_id": entry.snapshot_id,
            "solver": entry.algorithm,
            "training_iteration": entry.training_iteration,
            "model_config_id": entry.model_config_id,
            "tree_digest": entry.tree_digest,
            "artifact_schema_version": entry.artifact_schema_version,
        }
        for field_name, expected_value in expected_metadata.items():
            if getattr(snapshot.metadata, field_name) != expected_value:
                raise ValueError(f"snapshot does not match registry {field_name}")
        return ResolvedStrategy(
            entry,
            tabular_game,
            TabularAgent(tabular_game, snapshot),
            snapshot.average_policy,
            snapshot.metadata,
        )

    def _resolve_file_path(self, entry: StrategyRegistryEntry) -> Path:
        """Resolve and integrity-check a registry-owned artefact path."""
        if entry.local_path is None or entry.file_size is None or entry.sha256 is None:
            raise ValueError("file-backed registry entry is incomplete")
        relative_path = PurePosixPath(entry.local_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("registry local_path must remain within the project root")
        path = (self._project_root / Path(*relative_path.parts)).resolve()
        if not path.is_relative_to(self._project_root):
            raise ValueError("registry local_path escapes the project root")
        if not path.is_file():
            raise ValueError("registered strategy file does not exist")
        if path.stat().st_size != entry.file_size:
            raise ValueError("registered strategy file size does not match")
        if file_sha256(path) != entry.sha256:
            raise ValueError("registered strategy checksum does not match")
        return path


def load_strategy_registry(path: Path, *, project_root: Path) -> StrategyRegistry:
    """Load the explicit JSON registry schema without accepting extra fields."""
    try:
        raw_registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("strategy registry is unreadable") from error
    if not isinstance(raw_registry, dict) or set(raw_registry) != {
        "schema_version",
        "strategies",
    }:
        raise ValueError("strategy registry fields are incomplete or unexpected")
    schema_version = raw_registry["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != REGISTRY_SCHEMA_VERSION
    ):
        raise ValueError("strategy registry schema_version is incompatible")
    raw_entries = raw_registry["strategies"]
    if not isinstance(raw_entries, list):
        raise ValueError("strategy registry strategies must be a list")

    entries = tuple(_parse_entry(raw_entry) for raw_entry in raw_entries)
    strategy_ids = tuple(entry.strategy_id for entry in entries)
    if len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("strategy registry contains duplicate strategy_id values")
    return StrategyRegistry(entries, project_root)


def _parse_entry(raw_entry: object) -> StrategyRegistryEntry:
    """Parse one exact-schema registry entry and validate its values."""
    expected_fields = set(StrategyRegistryEntry.__dataclass_fields__)
    if not isinstance(raw_entry, dict) or set(raw_entry) != expected_fields:
        raise ValueError("strategy entry fields are incomplete or unexpected")
    try:
        entry = StrategyRegistryEntry(**raw_entry)
    except TypeError as error:
        raise ValueError("strategy entry values are invalid") from error
    _validate_entry(entry)
    return entry


def _validate_entry(entry: StrategyRegistryEntry) -> None:
    """Validate common, baseline-specific, and tabular entry invariants."""
    string_fields = (
        entry.strategy_id,
        entry.label,
        entry.game,
        entry.game_version,
        entry.algorithm,
        entry.agent_type,
        entry.model_config_id,
        entry.state_encoding,
        entry.action_space,
    )
    if any(not isinstance(value, str) or not value for value in string_fields):
        raise ValueError("strategy entry identifiers and labels must be non-empty strings")
    try:
        game_id = GameId(entry.game)
    except ValueError as error:
        raise ValueError("strategy entry game is unknown") from error
    if game_id not in (GameId.KUHN, GameId.LEDUC):
        raise ValueError("strategy entry game does not support tabular policies")
    if isinstance(entry.training_iteration, bool) or not isinstance(entry.training_iteration, int):
        raise ValueError("strategy entry training_iteration must be an integer")
    if entry.training_iteration < 0 or not isinstance(entry.evaluation, dict):
        raise ValueError("strategy entry training metadata is invalid")

    if entry.agent_type == "baseline":
        optional_values = (
            entry.snapshot_id,
            entry.local_path,
            entry.tree_digest,
            entry.artifact_schema_version,
            entry.release_id,
            entry.file_size,
            entry.sha256,
        )
        if any(value is not None for value in optional_values):
            raise ValueError("baseline entries must not reference a strategy file")
        return
    if entry.agent_type != "tabular":
        raise ValueError("strategy entry agent_type is unsupported")
    if entry.model_config_id != TABULAR_MODEL_CONFIG_ID:
        raise ValueError("tabular strategy entry model_config_id is incompatible")
    required_strings = (
        entry.snapshot_id,
        entry.local_path,
        entry.tree_digest,
        entry.release_id,
        entry.sha256,
    )
    if any(not isinstance(value, str) or not value for value in required_strings):
        raise ValueError("tabular strategy entry file metadata is incomplete")
    if (
        isinstance(entry.artifact_schema_version, bool)
        or entry.artifact_schema_version != SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError("tabular strategy entry artifact schema is incompatible")
    if isinstance(entry.file_size, bool) or not isinstance(entry.file_size, int):
        raise ValueError("tabular strategy entry file_size must be an integer")
    if entry.file_size <= 0 or _SHA256_PATTERN.fullmatch(entry.sha256 or "") is None:
        raise ValueError("tabular strategy entry file integrity metadata is invalid")
    assert entry.local_path is not None
    if PurePosixPath(entry.local_path).parts[:1] != ("artifacts",):
        raise ValueError("tabular strategy files must be located under artifacts")


def _uniform_policy(tabular_game: TabularGame) -> NDArray[np.float64]:
    """Build a flat uniform policy for every information set in a game."""
    tree = tabular_game.tree
    policy = np.empty(len(tree.information_set_actions), dtype=np.float64)
    for offset, count in zip(
        tree.information_set_action_offsets,
        tree.information_set_action_counts,
        strict=True,
    ):
        start = int(offset)
        action_count = int(count)
        policy[start : start + action_count] = 1.0 / action_count
    return policy
