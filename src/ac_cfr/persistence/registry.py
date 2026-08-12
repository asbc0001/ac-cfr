"""Validated strategy registry and trusted playable-agent resolution."""

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.typing import NDArray

from ac_cfr.agents.base import PlayableAgent
from ac_cfr.agents.baselines import RULE_BASED_AGENT_ID, BaselineAgent, RuleBasedAgent
from ac_cfr.agents.neural import NeuralAgent
from ac_cfr.agents.tabular import TabularAgent
from ac_cfr.common.config import GameConfigurationId, ModelConfigId, StateEncodingId
from ac_cfr.games.base import GameId
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.games.tabular import TabularGame, create_tabular_game
from ac_cfr.persistence.compatibility import ACTION_SPACE_ID, TABULAR_MODEL_CONFIG_ID
from ac_cfr.persistence.deep_cfr_snapshots import (
    DEEP_CFR_SNAPSHOT_SCHEMA_VERSION,
    DeepCFRSnapshotMetadata,
    deep_cfr_policy,
    load_deep_cfr_snapshot,
)
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
    """A registry entry, compatible game configuration, and loaded agent."""

    entry: StrategyRegistryEntry
    game: TabularGame | HoldemConfig
    agent: PlayableAgent
    _policy: NDArray[np.float64] | None
    snapshot_metadata: TabularSnapshotMetadata | DeepCFRSnapshotMetadata | None

    @property
    def tabular_game(self) -> TabularGame:
        """Return the exact-evaluation game or reject a Hold'em strategy."""
        if not isinstance(self.game, TabularGame):
            raise ValueError("strategy does not have a tabular game tree")
        return self.game

    @property
    def policy(self) -> NDArray[np.float64]:
        """Return the enumerated policy or reject an on-demand Hold'em strategy."""
        if self._policy is None:
            raise ValueError("strategy does not have an enumerated policy")
        return self._policy


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
        """Validate and load one named baseline, tabular, or neural strategy."""
        try:
            entry = self._entries[strategy_id]
        except KeyError as error:
            raise ValueError(f"unknown strategy_id: {strategy_id}") from error
        game = _create_registry_game(entry)
        configuration_id = game.configuration_id
        assert configuration_id is not None
        if entry.game_version != configuration_id.value:
            raise ValueError("registry entry has an incompatible game_version")
        if entry.action_space != ACTION_SPACE_ID:
            raise ValueError("registry entry has an incompatible action_space")

        if entry.agent_type == "baseline":
            _require_state_encoding(entry, game.state_encoding_id.value)
            agent = BaselineAgent() if entry.algorithm == "uniform_random" else RuleBasedAgent()
            policy = _uniform_policy(game) if isinstance(game, TabularGame) else None
            if policy is not None:
                policy.setflags(write=False)
            return ResolvedStrategy(entry, game, agent, policy, None)
        path = self._resolve_file_path(entry)
        if entry.agent_type == "tabular":
            if not isinstance(game, TabularGame):
                raise ValueError("tabular strategies require Kuhn or Leduc")
            _require_state_encoding(entry, game.state_encoding_id.value)
            snapshot = load_tabular_snapshot(path, game)
            _require_snapshot_metadata(entry, snapshot.metadata)
            return ResolvedStrategy(
                entry,
                game,
                TabularAgent(game, snapshot),
                snapshot.average_policy,
                snapshot.metadata,
            )
        if entry.agent_type == "neural":
            snapshot_game = game.tree if isinstance(game, TabularGame) else game
            snapshot = load_deep_cfr_snapshot(path, snapshot_game)
            _require_snapshot_metadata(entry, snapshot.metadata)
            if snapshot.metadata.solver != "optimised":
                raise ValueError("registered neural strategy must use the optimised solver")
            policy = (
                deep_cfr_policy(game.tree, snapshot.network)
                if isinstance(game, TabularGame)
                else None
            )
            return ResolvedStrategy(
                entry,
                game,
                NeuralAgent(snapshot),
                policy,
                snapshot.metadata,
            )
        raise ValueError(f"unsupported agent_type: {entry.agent_type}")

    def _resolve_file_path(self, entry: StrategyRegistryEntry) -> Path:
        """Resolve and integrity-check a registry-owned artefact path."""
        if entry.local_path is None or entry.file_size is None or entry.sha256 is None:
            raise ValueError("file-backed registry entry is incomplete")
        path = strategy_artifact_path(entry, project_root=self._project_root)
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


def strategy_artifact_path(entry: StrategyRegistryEntry, *, project_root: Path) -> Path:
    """Resolve one registry-owned artefact destination without accessing it."""
    if entry.local_path is None:
        raise ValueError("strategy entry does not reference an artefact")
    relative_path = PurePosixPath(entry.local_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("registry local_path must remain within the project root")
    root = project_root.resolve()
    path = (root / Path(*relative_path.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("registry local_path escapes the project root")
    return path


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
    """Validate common and agent-specific registry invariants."""
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
    if game_id not in (GameId.KUHN, GameId.LEDUC, GameId.HOLD_EM):
        raise ValueError("strategy entry game is unsupported")
    expected_game_version = {
        GameId.KUHN: GameConfigurationId.KUHN.value,
        GameId.LEDUC: GameConfigurationId.LEDUC.value,
        GameId.HOLD_EM: GameConfigurationId.MODIFIED_HULHE.value,
    }[game_id]
    if entry.game_version != expected_game_version:
        raise ValueError("strategy entry game_version is incompatible")
    if entry.action_space != ACTION_SPACE_ID:
        raise ValueError("strategy entry action_space is incompatible")
    if isinstance(entry.training_iteration, bool) or not isinstance(entry.training_iteration, int):
        raise ValueError("strategy entry training_iteration must be an integer")
    if entry.training_iteration < 0 or not isinstance(entry.evaluation, dict):
        raise ValueError("strategy entry training metadata is invalid")

    if entry.agent_type == "baseline":
        if entry.algorithm not in {"uniform_random", RULE_BASED_AGENT_ID}:
            raise ValueError("baseline strategy entry algorithm is unsupported")
        if entry.model_config_id != entry.algorithm:
            raise ValueError("baseline strategy entry model_config_id is incompatible")
        if entry.training_iteration != 0:
            raise ValueError("baseline strategy entry must have iteration zero")
        if entry.algorithm == RULE_BASED_AGENT_ID and game_id is not GameId.HOLD_EM:
            raise ValueError("rule-based strategy entry requires modified HULHE")
        expected_encoding = {
            GameId.KUHN: StateEncodingId.KUHN.value,
            GameId.LEDUC: StateEncodingId.LEDUC.value,
            GameId.HOLD_EM: StateEncodingId.HOLD_EM.value,
        }[game_id]
        if entry.state_encoding != expected_encoding:
            raise ValueError("baseline strategy entry state_encoding is incompatible")
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
    if entry.agent_type not in {"tabular", "neural"}:
        raise ValueError("strategy entry agent_type is unsupported")
    if entry.agent_type == "tabular" and game_id not in (GameId.KUHN, GameId.LEDUC):
        raise ValueError("tabular strategy entry requires Kuhn or Leduc")
    if entry.agent_type == "tabular" and entry.model_config_id != TABULAR_MODEL_CONFIG_ID:
        raise ValueError("tabular strategy entry model_config_id is incompatible")
    if entry.agent_type == "tabular":
        expected_encoding = (
            StateEncodingId.KUHN.value if game_id is GameId.KUHN else StateEncodingId.LEDUC.value
        )
        if entry.state_encoding != expected_encoding:
            raise ValueError("tabular strategy entry state_encoding is incompatible")
    if entry.agent_type == "neural":
        if game_id not in (GameId.LEDUC, GameId.HOLD_EM) or entry.algorithm != "deep_cfr":
            raise ValueError("neural strategy entry must describe Deep CFR")
        try:
            ModelConfigId(entry.model_config_id)
        except ValueError as error:
            raise ValueError("neural strategy model_config_id is incompatible") from error
        expected_model = (
            ModelConfigId.LEDUC_DEEP_CFR
            if game_id is GameId.LEDUC
            else ModelConfigId.MODIFIED_HULHE_DEEP_CFR
        )
        expected_encoding = (
            StateEncodingId.LEDUC_NEURAL if game_id is GameId.LEDUC else StateEncodingId.HOLD_EM
        )
        if entry.model_config_id != expected_model.value:
            raise ValueError("neural strategy model_config_id is incompatible")
        if entry.state_encoding != expected_encoding.value:
            raise ValueError("neural strategy state_encoding is incompatible")
    required_strings = (
        entry.snapshot_id,
        entry.local_path,
        entry.tree_digest,
        entry.release_id,
        entry.sha256,
    )
    if any(not isinstance(value, str) or not value for value in required_strings):
        raise ValueError("strategy entry file metadata is incomplete")
    if isinstance(entry.artifact_schema_version, bool) or entry.artifact_schema_version != (
        SNAPSHOT_SCHEMA_VERSION
        if entry.agent_type == "tabular"
        else DEEP_CFR_SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError("strategy entry artifact schema is incompatible")
    if isinstance(entry.file_size, bool) or not isinstance(entry.file_size, int):
        raise ValueError("tabular strategy entry file_size must be an integer")
    if entry.file_size <= 0 or _SHA256_PATTERN.fullmatch(entry.sha256 or "") is None:
        raise ValueError("strategy entry file integrity metadata is invalid")
    assert entry.local_path is not None
    if PurePosixPath(entry.local_path).parts[:1] != ("artifacts",):
        raise ValueError("strategy files must be located under artifacts")


def _require_state_encoding(entry: StrategyRegistryEntry, expected: str) -> None:
    """Require one registry entry to use the expected game-state encoding."""
    if entry.state_encoding != expected:
        raise ValueError("registry entry has an incompatible state_encoding")


def _create_registry_game(entry: StrategyRegistryEntry) -> TabularGame | HoldemConfig:
    """Construct the exact game configuration declared by a registry entry."""
    game_id = GameId(entry.game)
    if game_id in (GameId.KUHN, GameId.LEDUC):
        return create_tabular_game(game_id)
    if entry.game_version != GameConfigurationId.MODIFIED_HULHE.value:
        raise ValueError("registry supports only the modified Hold'em configuration")
    return HoldemConfig.modified()


def _require_snapshot_metadata(
    entry: StrategyRegistryEntry,
    metadata: TabularSnapshotMetadata | DeepCFRSnapshotMetadata,
) -> None:
    """Require loaded snapshot metadata to match its registry entry."""
    expected_metadata = {
        "snapshot_id": entry.snapshot_id,
        "game": entry.game,
        "game_version": entry.game_version,
        "training_iteration": entry.training_iteration,
        "model_config_id": entry.model_config_id,
        "state_encoding": entry.state_encoding,
        "action_space": entry.action_space,
        "tree_digest": entry.tree_digest,
        "artifact_schema_version": entry.artifact_schema_version,
    }
    if entry.agent_type == "tabular":
        expected_metadata["solver"] = entry.algorithm
    for field_name, expected_value in expected_metadata.items():
        if getattr(metadata, field_name) != expected_value:
            raise ValueError(f"snapshot does not match registry {field_name}")


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
