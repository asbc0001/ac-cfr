import csv
import json
from pathlib import Path

import pytest
import torch

from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
from ac_cfr.persistence.deep_cfr_snapshots import (
    deep_cfr_policy,
    load_deep_cfr_snapshot,
)
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_runner import (
    DeepCFRRunConfig,
    resume_deep_cfr_training,
    start_deep_cfr_training,
)


def _run_config() -> DeepCFRRunConfig:
    return DeepCFRRunConfig(
        run_id="deep_cfr_workflow_test",
        checkpoint_interval=1,
        training=DeepCFRTrainingConfig(
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
        ),
        runtime=DeepCFRRuntimeConfig(
            inference_batch_size=64,
            cpu_threads=1,
            device="cpu",
        ),
    )


def test_deep_cfr_run_resumes_metrics_and_exports_exactly_evaluable_snapshots(
    tmp_path: Path,
) -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    outcome = start_deep_cfr_training(_run_config(), runs_root=tmp_path)
    saved_config = json.loads(
        (outcome.run_directory / "run_config.json").read_text(encoding="utf-8")
    )
    assert saved_config["run_config"] == _run_config().to_dict()
    resumed = resume_deep_cfr_training(outcome.run_directory / "checkpoints" / "iter_1.pt")
    with (resumed.run_directory / "metrics.csv").open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert [record["iteration"] for record in records] == ["1", "2"]

    final_snapshot = load_deep_cfr_snapshot(
        resumed.run_directory / "strategy_snapshots" / "deep_cfr_workflow_test_iter_2.pt",
        tree,
    )
    metrics = evaluate_strategy(tree, deep_cfr_policy(tree, final_snapshot.network))
    assert metrics.exploitability == pytest.approx(float(records[-1]["exploitability"]))
    assert all(not parameter.requires_grad for parameter in final_snapshot.network.parameters())


def test_deep_cfr_snapshot_rejects_incompatible_architecture(tmp_path: Path) -> None:
    outcome = start_deep_cfr_training(_run_config(), runs_root=tmp_path)
    path = outcome.run_directory / "strategy_snapshots" / "deep_cfr_workflow_test_iter_2.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["architecture_config"]["input_size"] += 1
    torch.save(payload, path)

    tree = compile_game_tree(LeducGame(), LeducConfig())
    with pytest.raises(ValueError, match="architecture"):
        load_deep_cfr_snapshot(path, tree)
