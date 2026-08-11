import csv
import json
from pathlib import Path

import pytest
import torch

from ac_cfr.agents import NeuralAgent
from ac_cfr.common.config import DeepCFRImplementationId
from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.base import Action, InformationState
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.persistence.deep_cfr_checkpoints import load_deep_cfr_checkpoint
from ac_cfr.persistence.deep_cfr_snapshots import (
    deep_cfr_policy,
    load_deep_cfr_snapshot,
)
from ac_cfr.persistence.registry import load_strategy_registry
from ac_cfr.persistence.snapshots import file_sha256
from ac_cfr.solvers import DeepCFR
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_runner import (
    DeepCFRRunConfig,
    resume_deep_cfr_training,
    start_deep_cfr_training,
)


def _run_config() -> DeepCFRRunConfig:
    return DeepCFRRunConfig(
        run_id="deep_cfr_workflow_test",
        implementation=DeepCFRImplementationId.OPTIMISED,
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
    resume_path = outcome.run_directory / "checkpoints" / "iter_1.pt"
    assert type(load_deep_cfr_checkpoint(resume_path, tree).solver) is DeepCFR
    resumed = resume_deep_cfr_training(resume_path)
    with (resumed.run_directory / "metrics.csv").open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert [record["iteration"] for record in records] == ["1", "2"]

    final_snapshot_path = (
        resumed.run_directory / "strategy_snapshots" / "deep_cfr_workflow_test_iter_2.pt"
    )
    final_snapshot = load_deep_cfr_snapshot(final_snapshot_path, tree)
    metrics = evaluate_strategy(tree, deep_cfr_policy(tree, final_snapshot.network))
    assert metrics.exploitability == pytest.approx(float(records[-1]["exploitability"]))
    assert all(not parameter.requires_grad for parameter in final_snapshot.network.parameters())

    information_state = _information_state(tree, 0)
    agent = NeuralAgent(final_snapshot)
    agent_strategy = agent.get_strategy(information_state, information_state.legal_actions)
    policy = deep_cfr_policy(tree, final_snapshot.network)
    action_count = len(information_state.legal_actions)
    assert agent_strategy == pytest.approx(policy[:action_count])

    artifact_path = tmp_path / "artifacts" / "deep_cfr" / "test.pt"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(final_snapshot_path.read_bytes())
    metadata = final_snapshot.metadata
    registry_path = tmp_path / "strategy_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategies": [
                    {
                        "strategy_id": "leduc_deep_cfr_test",
                        "label": "Deep CFR test",
                        "game": metadata.game,
                        "game_version": metadata.game_version,
                        "algorithm": "deep_cfr",
                        "agent_type": "neural",
                        "snapshot_id": metadata.snapshot_id,
                        "training_iteration": metadata.training_iteration,
                        "local_path": "artifacts/deep_cfr/test.pt",
                        "evaluation": {"exploitability": metrics.exploitability},
                        "model_config_id": metadata.model_config_id,
                        "state_encoding": metadata.state_encoding,
                        "action_space": metadata.action_space,
                        "tree_digest": metadata.tree_digest,
                        "artifact_schema_version": metadata.artifact_schema_version,
                        "release_id": "test",
                        "file_size": artifact_path.stat().st_size,
                        "sha256": file_sha256(artifact_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resolved = load_strategy_registry(registry_path, project_root=tmp_path).resolve(
        "leduc_deep_cfr_test"
    )
    assert isinstance(resolved.agent, NeuralAgent)
    assert resolved.policy == pytest.approx(policy)

    saved_config["run_config"]["implementation"] = "reference"
    (outcome.run_directory / "run_config.json").write_text(
        json.dumps(saved_config),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="implementation"):
        resume_deep_cfr_training(resume_path)


def test_deep_cfr_snapshot_rejects_incompatible_architecture(tmp_path: Path) -> None:
    outcome = start_deep_cfr_training(_run_config(), runs_root=tmp_path)
    path = outcome.run_directory / "strategy_snapshots" / "deep_cfr_workflow_test_iter_2.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["architecture_config"]["input_size"] += 1
    torch.save(payload, path)

    tree = compile_game_tree(LeducGame(), LeducConfig())
    with pytest.raises(ValueError, match="architecture"):
        load_deep_cfr_snapshot(path, tree)


def _information_state(
    tree: IndexedGameTree,
    information_set_id: int,
) -> InformationState:
    """Reconstruct one information state from the indexed Leduc tree."""
    encoding_offset = int(tree.information_set_encoding_offsets[information_set_id])
    encoding_count = int(tree.information_set_encoding_counts[information_set_id])
    action_offset = int(tree.information_set_action_offsets[information_set_id])
    action_count = int(tree.information_set_action_counts[information_set_id])
    return InformationState(
        game_id=tree.game_id,
        player=int(tree.information_set_players[information_set_id]),
        encoding=tuple(
            int(value)
            for value in tree.information_set_encodings[
                encoding_offset : encoding_offset + encoding_count
            ]
        ),
        legal_actions=tuple(
            Action(int(value))
            for value in tree.information_set_actions[action_offset : action_offset + action_count]
        ),
    )
