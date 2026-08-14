import errno
from pathlib import Path

import pytest
import torch

from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
from ac_cfr.persistence import files as persistence_files
from ac_cfr.persistence.deep_cfr_checkpoints import (
    load_deep_cfr_checkpoint,
    save_deep_cfr_checkpoint,
)
from ac_cfr.solvers.naive_deep_cfr import NaiveDeepCFR
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig


def _config() -> DeepCFRTrainingConfig:
    return DeepCFRTrainingConfig(
        iterations=2,
        traversals_per_player=1,
        advantage_reservoir_capacity=100,
        strategy_reservoir_capacity=100,
        advantage_training_steps=1,
        strategy_training_steps=1,
        advantage_batch_size=128,
        strategy_batch_size=128,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=2026,
        snapshot_iterations=(1,),
        opponent_exploration_epsilon=0.1,
    )


def _runtime() -> DeepCFRRuntimeConfig:
    return DeepCFRRuntimeConfig(inference_batch_size=64, cpu_threads=1, device="cpu")


def test_staged_checkpoint_publication_retries_transient_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "remote" / "checkpoint.pt"
    staging_directory = tmp_path / "local"
    publish_once = persistence_files._publish_staged_file_once
    attempts = 0

    def flaky_publish(staged_path: Path, staged_digest: bytes, path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EIO, "simulated transient storage failure")
        publish_once(staged_path, staged_digest, path)

    monkeypatch.setattr(persistence_files, "_CHECKPOINT_RETRY_DELAYS_SECONDS", (0.0,))
    monkeypatch.setattr(persistence_files, "_publish_staged_file_once", flaky_publish)

    with persistence_files.staged_atomic_binary_writer(
        target,
        staging_directory=staging_directory,
    ) as output_file:
        output_file.write(b"complete checkpoint")

    assert attempts == 2
    assert target.read_bytes() == b"complete checkpoint"
    assert not list(staging_directory.glob("*.staged"))


def test_staged_checkpoint_exhaustion_preserves_old_target_and_local_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "remote" / "checkpoint.pt"
    target.parent.mkdir()
    target.write_bytes(b"previous checkpoint")
    staging_directory = tmp_path / "local"

    def failed_publish(staged_path: Path, staged_digest: bytes, path: Path) -> None:
        raise OSError(errno.EIO, "simulated persistent storage failure")

    monkeypatch.setattr(persistence_files, "_CHECKPOINT_RETRY_DELAYS_SECONDS", (0.0,))
    monkeypatch.setattr(persistence_files, "_publish_staged_file_once", failed_publish)

    with (
        pytest.raises(OSError, match="persistent storage failure"),
        persistence_files.staged_atomic_binary_writer(
            target,
            staging_directory=staging_directory,
        ) as output_file,
    ):
        output_file.write(b"new checkpoint")

    staged_paths = list(staging_directory.glob("*.staged"))
    assert target.read_bytes() == b"previous checkpoint"
    assert len(staged_paths) == 1
    assert staged_paths[0].read_bytes() == b"new checkpoint"


def test_interrupted_deep_cfr_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    uninterrupted = NaiveDeepCFR(tree, _config(), _runtime())
    uninterrupted.train(2)

    interrupted = NaiveDeepCFR(tree, _config(), _runtime())
    interrupted.train(1)
    checkpoint_path = tmp_path / "latest.pt"
    save_deep_cfr_checkpoint(
        checkpoint_path,
        solver=interrupted,
        run_id="deep_cfr_resume_test",
        checkpoint_id="deep_cfr_resume_test_iter_1",
        code_revision="test-revision",
    )
    loaded = load_deep_cfr_checkpoint(checkpoint_path, tree, map_location="cpu")
    resumed = loaded.solver
    resumed.train(1)

    assert loaded.metadata["optimizer_state_required"] is False
    assert loaded.metadata["reservoir_schema_version"] == 2
    assert resumed.runtime == _runtime()
    assert resumed.iteration == uninterrupted.iteration == 2
    assert resumed.training_metrics == uninterrupted.training_metrics
    assert resumed.training_rng_state() == uninterrupted.training_rng_state()
    assert tuple(reservoir.samples for reservoir in resumed.advantage_reservoirs) == tuple(
        reservoir.samples for reservoir in uninterrupted.advantage_reservoirs
    )
    assert resumed.strategy_reservoir.samples == uninterrupted.strategy_reservoir.samples
    _assert_networks_equal(resumed, uninterrupted)


def test_deep_cfr_checkpoint_rejects_incompatible_metadata(tmp_path: Path) -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    solver = NaiveDeepCFR(tree, _config(), _runtime())
    solver.train(1)
    checkpoint_path = tmp_path / "latest.pt"
    save_deep_cfr_checkpoint(
        checkpoint_path,
        solver=solver,
        run_id="deep_cfr_validation_test",
        checkpoint_id="deep_cfr_validation_test_iter_1",
        code_revision="test-revision",
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["metadata"]["architecture_config"]["input_size"] += 1
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="architecture"):
        load_deep_cfr_checkpoint(checkpoint_path, tree)

    del payload["rng_state"]
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="incomplete"):
        load_deep_cfr_checkpoint(checkpoint_path, tree)


def _assert_networks_equal(left: NaiveDeepCFR, right: NaiveDeepCFR) -> None:
    left_networks = (
        *left.advantage_networks,
        *left.snapshot_networks.values(),
        left.final_strategy_network,
    )
    right_networks = (
        *right.advantage_networks,
        *right.snapshot_networks.values(),
        right.final_strategy_network,
    )
    assert len(left_networks) == len(right_networks)
    for left_network, right_network in zip(left_networks, right_networks, strict=True):
        assert left_network is not None and right_network is not None
        assert all(
            torch.equal(left_value, right_value)
            for left_value, right_value in zip(
                left_network.state_dict().values(),
                right_network.state_dict().values(),
                strict=True,
            )
        )
