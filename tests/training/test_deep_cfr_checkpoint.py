from pathlib import Path

import pytest
import torch

from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
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
        training_batch_size=128,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=2026,
        snapshot_iterations=(1,),
    )


def _runtime() -> DeepCFRRuntimeConfig:
    return DeepCFRRuntimeConfig(inference_batch_size=64, cpu_threads=1, device="cpu")


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
