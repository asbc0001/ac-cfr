"""Validated inference snapshots for Leduc Deep CFR strategies."""

import pickle
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ac_cfr.common.config import GameConfigurationId, ModelConfigId, StateEncodingId
from ac_cfr.games.base import GameId
from ac_cfr.games.leduc_neural import build_leduc_neural_data
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.models import (
    DeepCFRNetwork,
    DeepCFRNetworkConfig,
    build_deep_cfr_network,
    deep_cfr_network_config,
)
from ac_cfr.persistence.compatibility import ACTION_SPACE_ID, tree_compatibility_digest
from ac_cfr.persistence.files import atomic_binary_writer
from ac_cfr.training.config import DeepCFRTrainingConfig

DEEP_CFR_SNAPSHOT_SCHEMA_VERSION = 1
PROJECT_VERSION = version("ac-cfr")
_SOLVER_ID = "naive_deep_cfr"


@dataclass(frozen=True, slots=True)
class DeepCFRSnapshotMetadata:
    """Compatibility and provenance data for one frozen neural strategy."""

    artifact_schema_version: int
    project_version: str
    snapshot_id: str
    game: str
    game_version: str
    solver: str
    training_iteration: int
    run_id: str
    seed: int
    source_checkpoint_id: str
    model_config_id: str
    state_encoding: str
    action_space: str
    tree_digest: str
    architecture_config: dict[str, object]


@dataclass(frozen=True, slots=True)
class LoadedDeepCFRSnapshot:
    """Validated snapshot metadata and reconstructed frozen strategy network."""

    metadata: DeepCFRSnapshotMetadata
    network: DeepCFRNetwork


def export_deep_cfr_snapshot(
    path: Path,
    *,
    network: DeepCFRNetwork,
    tree: IndexedGameTree,
    config: DeepCFRTrainingConfig,
    snapshot_id: str,
    iteration: int,
    run_id: str,
    source_checkpoint_id: str,
) -> None:
    """Atomically export only the data needed to query one average strategy."""
    if not isinstance(network, DeepCFRNetwork):
        raise TypeError("network must be a DeepCFRNetwork")
    _validate_tree(tree)
    if network.config != deep_cfr_network_config(
        config.model_config_id,
        dropout_probability=config.dropout_probability,
    ):
        raise ValueError("network architecture does not match the training configuration")
    _validate_positive_iteration(iteration, config.iterations)
    for name, value in (
        ("snapshot_id", snapshot_id),
        ("run_id", run_id),
        ("source_checkpoint_id", source_checkpoint_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    metadata = DeepCFRSnapshotMetadata(
        artifact_schema_version=DEEP_CFR_SNAPSHOT_SCHEMA_VERSION,
        project_version=PROJECT_VERSION,
        snapshot_id=snapshot_id,
        game=GameId.LEDUC.value,
        game_version=GameConfigurationId.LEDUC.value,
        solver=_SOLVER_ID,
        training_iteration=iteration,
        run_id=run_id,
        seed=config.seed,
        source_checkpoint_id=source_checkpoint_id,
        model_config_id=config.model_config_id.value,
        state_encoding=config.state_encoding_id.value,
        action_space=ACTION_SPACE_ID,
        tree_digest=tree_compatibility_digest(tree),
        architecture_config=network.config.to_dict(),
    )
    payload = {
        "metadata": {
            field: getattr(metadata, field)
            for field in DeepCFRSnapshotMetadata.__dataclass_fields__
        },
        "strategy_network": {
            name: value.detach().cpu().clone() for name, value in network.state_dict().items()
        },
    }
    with atomic_binary_writer(path) as snapshot_file:
        torch.save(payload, snapshot_file)


def load_deep_cfr_snapshot(
    path: Path,
    tree: IndexedGameTree,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedDeepCFRSnapshot:
    """Safely load and validate one frozen Leduc average-strategy network."""
    _validate_tree(tree)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("Deep CFR snapshot is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != {"metadata", "strategy_network"}:
        raise ValueError("Deep CFR snapshot fields are incomplete or unexpected")

    metadata = _validated_metadata(payload["metadata"], tree)
    try:
        model_config_id = ModelConfigId(metadata.model_config_id)
        raw_dropout_probability = metadata.architecture_config["dropout_probability"]
    except (KeyError, ValueError) as error:
        raise ValueError("Deep CFR snapshot architecture is invalid") from error
    if isinstance(raw_dropout_probability, bool) or not isinstance(
        raw_dropout_probability,
        (int, float),
    ):
        raise ValueError("Deep CFR snapshot dropout probability is invalid")
    dropout_probability = float(raw_dropout_probability)
    expected_config = deep_cfr_network_config(
        model_config_id,
        dropout_probability=dropout_probability,
    )
    if metadata.architecture_config != expected_config.to_dict():
        raise ValueError("Deep CFR snapshot architecture is incompatible")

    network = _load_network(payload["strategy_network"], expected_config, map_location)
    return LoadedDeepCFRSnapshot(metadata=metadata, network=network)


def deep_cfr_policy(
    tree: IndexedGameTree,
    network: DeepCFRNetwork,
) -> NDArray[np.float64]:
    """Evaluate a frozen network into the exact evaluator's flat policy layout."""
    _validate_tree(tree)
    if not isinstance(network, DeepCFRNetwork):
        raise TypeError("network must be a DeepCFRNetwork")
    neural_data = build_leduc_neural_data(tree)
    device = next(network.parameters()).device
    states = torch.from_numpy(neural_data.states.copy()).to(device)
    masks = torch.from_numpy(neural_data.action_masks.copy()).to(device)
    network.eval()
    with torch.inference_mode():
        logits = network(states)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("strategy network logits must be finite")
        probabilities = torch.softmax(logits.masked_fill(~masks, float("-inf")), dim=1)
        if not bool(torch.isfinite(probabilities).all()):
            raise FloatingPointError("strategy network probabilities must be finite")
        probability_array = probabilities.cpu().numpy()

    policy = np.empty(len(tree.information_set_actions), dtype=np.float64)
    for information_set_id in range(tree.information_set_count):
        offset = int(tree.information_set_action_offsets[information_set_id])
        count = int(tree.information_set_action_counts[information_set_id])
        actions = tuple(
            int(action) for action in tree.information_set_actions[offset : offset + count]
        )
        values = np.asarray(probability_array[information_set_id, actions], dtype=np.float64)
        policy[offset : offset + count] = values / values.sum()
    policy.setflags(write=False)
    return policy


def _validated_metadata(value: object, tree: IndexedGameTree) -> DeepCFRSnapshotMetadata:
    """Parse exact snapshot metadata and reject incompatible identifiers."""
    expected_fields = set(DeepCFRSnapshotMetadata.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("Deep CFR snapshot metadata is incomplete or unexpected")
    try:
        metadata = DeepCFRSnapshotMetadata(**value)
    except TypeError as error:
        raise ValueError("Deep CFR snapshot metadata is invalid") from error
    expected = {
        "artifact_schema_version": DEEP_CFR_SNAPSHOT_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "game": GameId.LEDUC.value,
        "game_version": GameConfigurationId.LEDUC.value,
        "solver": _SOLVER_ID,
        "state_encoding": StateEncodingId.LEDUC_NEURAL.value,
        "action_space": ACTION_SPACE_ID,
        "tree_digest": tree_compatibility_digest(tree),
    }
    for field, expected_value in expected.items():
        if getattr(metadata, field) != expected_value:
            raise ValueError(f"Deep CFR snapshot has incompatible {field}")
    _validate_positive_iteration(metadata.training_iteration)
    if isinstance(metadata.seed, bool) or not isinstance(metadata.seed, int):
        raise ValueError("Deep CFR snapshot seed is invalid")
    for field in ("snapshot_id", "run_id", "source_checkpoint_id", "model_config_id"):
        if not isinstance(getattr(metadata, field), str) or not getattr(metadata, field):
            raise ValueError(f"Deep CFR snapshot {field} is invalid")
    if not isinstance(metadata.architecture_config, dict):
        raise ValueError("Deep CFR snapshot architecture is invalid")
    return metadata


def _load_network(
    value: object,
    expected_config: DeepCFRNetworkConfig,
    map_location: str | torch.device,
) -> DeepCFRNetwork:
    """Reconstruct a frozen network after strict tensor validation."""
    if not isinstance(value, dict) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in value.items()
    ):
        raise ValueError("Deep CFR snapshot network state is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        network = build_deep_cfr_network(
            expected_config.model_config_id,
            dropout_probability=expected_config.dropout_probability,
        ).to(map_location)
    expected_state = network.state_dict()
    if set(value) != set(expected_state):
        raise ValueError("Deep CFR snapshot network fields are incompatible")
    for name, tensor in value.items():
        expected_tensor = expected_state[name]
        if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"Deep CFR snapshot tensor {name} is incompatible")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Deep CFR snapshot tensors must be finite")
    network.load_state_dict(value, strict=True)
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def _validate_tree(tree: IndexedGameTree) -> None:
    if not isinstance(tree, IndexedGameTree):
        raise TypeError("tree must be an IndexedGameTree")
    if tree.game_id is not GameId.LEDUC:
        raise ValueError("Deep CFR snapshots currently support only Leduc")


def _validate_positive_iteration(iteration: int, maximum: int | None = None) -> None:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("training iteration must be a positive integer")
    if maximum is not None and iteration > maximum:
        raise ValueError("training iteration exceeds the configured budget")
