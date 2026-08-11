import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ac_cfr.agents import TabularAgent
from ac_cfr.benchmarking import run_tabular_benchmark
from ac_cfr.cli.evaluate import main as evaluate_main
from ac_cfr.cli.plot_results import main as plot_main
from ac_cfr.cli.train import main as train_main
from ac_cfr.games.base import GameId
from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.tabular import create_tabular_game
from ac_cfr.persistence.checkpoints import load_tabular_checkpoint
from ac_cfr.persistence.files import atomic_binary_writer
from ac_cfr.persistence.registry import load_strategy_registry
from ac_cfr.persistence.results import EVALUATION_RESULT_FIELDS, TRAINING_METRIC_FIELDS
from ac_cfr.persistence.snapshots import (
    export_tabular_snapshot,
    file_sha256,
)
from ac_cfr.solvers import NaiveCFR
from ac_cfr.training import (
    TabularTrainingConfig,
    resume_tabular_training,
    start_tabular_training,
)


@pytest.mark.parametrize(
    ("solver_id", "game"),
    (
        ("naive_cfr", "kuhn"),
        ("cfr", "kuhn"),
        ("naive_mccfr", "leduc"),
        ("mccfr", "leduc"),
    ),
)
def test_checkpoint_resume_matches_uninterrupted_training_and_reconciles_metrics(
    tmp_path: Path,
    solver_id: str,
    game: str,
) -> None:
    config = TabularTrainingConfig(
        game=game,
        solver=solver_id,
        iterations=4,
        seed=7,
        run_id=f"{solver_id}_resume_test",
        evaluation_interval=2,
        checkpoint_interval=2,
        snapshot_iterations=(2,),
    )
    first_outcome = start_tabular_training(config, runs_root=tmp_path)
    uninterrupted = load_tabular_checkpoint(first_outcome.latest_checkpoint)

    resume_path = first_outcome.run_directory / "checkpoints" / "iter_2.npz"
    if solver_id == "cfr":
        resume_path = _schema_one_checkpoint_copy(resume_path)
    resumed_outcome = resume_tabular_training(resume_path)
    resumed = load_tabular_checkpoint(resumed_outcome.latest_checkpoint)
    assert resumed_outcome.final_iteration == 4
    assert np.array_equal(resumed.regret_sum, uninterrupted.regret_sum)
    assert np.array_equal(resumed.strategy_sum, uninterrupted.strategy_sum)
    assert resumed.metadata["rng_state"] == uninterrupted.metadata["rng_state"]

    with (resumed_outcome.run_directory / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as metrics_file:
        records = list(csv.DictReader(metrics_file))
    assert [record["iteration"] for record in records] == ["2", "4"]


def test_atomic_writer_preserves_previous_file_after_interrupted_write(tmp_path: Path) -> None:
    target = tmp_path / "latest.npz"
    target.write_bytes(b"previous checkpoint")

    with (
        pytest.raises(RuntimeError, match="interrupted"),
        atomic_binary_writer(target) as checkpoint_file,
    ):
        checkpoint_file.write(b"partial replacement")
        raise RuntimeError("interrupted")

    assert target.read_bytes() == b"previous checkpoint"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_predeclared_early_stopping_ends_a_run_at_an_evaluation_boundary(
    tmp_path: Path,
) -> None:
    config = TabularTrainingConfig(
        game="kuhn",
        solver="naive_cfr",
        iterations=10,
        seed=0,
        run_id="early_stop_test",
        evaluation_interval=1,
        checkpoint_interval=5,
        snapshot_iterations=(),
        early_stopping_minimum_improvement=1.0,
        early_stopping_patience=1,
    )

    outcome = start_tabular_training(config, runs_root=tmp_path)

    assert outcome.stopped_early
    assert outcome.final_iteration == 2
    assert outcome.latest_checkpoint.is_file()


def test_snapshot_registry_loads_tabular_agent_and_rejects_tampering(tmp_path: Path) -> None:
    tabular_game = create_tabular_game(GameId.KUHN)
    solver = NaiveCFR(tabular_game.tree)
    solver.train(2)
    snapshot_path = tmp_path / "artifacts" / "kuhn_test.npz"
    export_tabular_snapshot(
        snapshot_path,
        tabular_game=tabular_game,
        average_policy=solver.average_policy(),
        snapshot_id="kuhn_test",
        solver="naive_cfr",
        iteration=2,
        run_id="snapshot_test",
        seed=7,
        source_checkpoint_id="snapshot_test_iter_2",
    )
    registry_path = tmp_path / "configs" / "strategy_registry.json"
    registry_path.parent.mkdir()
    registry_path.write_text(
        json.dumps(_registry_for_snapshot(snapshot_path, tabular_game.tree.game_id)),
        encoding="utf-8",
    )

    resolved = load_strategy_registry(registry_path, project_root=tmp_path).resolve("kuhn_test")
    assert isinstance(resolved.agent, TabularAgent)
    root = KuhnGame().initial_state(KuhnConfig())
    state = root.apply_action(root.chance_outcomes()[0].outcome)
    information_state = state.information_state()
    strategy = resolved.agent.get_strategy(
        information_state,
        information_state.legal_actions,
    )
    assert strategy == pytest.approx(resolved.policy[: len(strategy)])

    snapshot_bytes = snapshot_path.read_bytes()
    snapshot_path.write_bytes(snapshot_bytes[:-1] + bytes((snapshot_bytes[-1] ^ 1,)))
    with pytest.raises(ValueError, match="checksum"):
        load_strategy_registry(registry_path, project_root=tmp_path).resolve("kuhn_test")


def test_mccfr_training_command_exports_a_registry_loadable_agent(tmp_path: Path) -> None:
    run_id = "mccfr_agent_test"
    assert (
        train_main(
            (
                "--game",
                "leduc",
                "--solver",
                "mccfr",
                "--iterations",
                "4",
                "--run-id",
                run_id,
                "--runs-root",
                str(tmp_path),
                "--evaluation-interval",
                "2",
                "--checkpoint-interval",
                "2",
            )
        )
        == 0
    )
    source_snapshot = tmp_path / run_id / "strategy_snapshots" / f"{run_id}_iter_4.npz"
    snapshot_path = tmp_path / "artifacts" / "leduc_mccfr_test.npz"
    snapshot_path.parent.mkdir()
    snapshot_path.write_bytes(source_snapshot.read_bytes())
    registry_path = tmp_path / "configs" / "strategy_registry.json"
    registry_path.parent.mkdir()
    registry_path.write_text(
        json.dumps(_registry_for_snapshot(snapshot_path, GameId.LEDUC)),
        encoding="utf-8",
    )

    resolved = load_strategy_registry(registry_path, project_root=tmp_path).resolve("leduc_test")
    assert isinstance(resolved.agent, TabularAgent)


def test_evaluation_results_plot_and_benchmark_foundations(tmp_path: Path) -> None:
    results_path = tmp_path / "evaluation" / "metrics.csv"
    arguments = (
        "kuhn_random",
        "--strategy-registry",
        "configs/strategy_registry.json",
        "--results",
        str(results_path),
        "--project-root",
        ".",
    )
    assert evaluate_main(arguments) == 0
    assert evaluate_main(arguments) == 0
    with results_path.open(encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        assert tuple(reader.fieldnames or ()) == EVALUATION_RESULT_FIELDS
        assert len(list(reader)) == 1

    plot_path = results_path.parent / "plots" / "exploitability_by_iteration.png"
    assert plot_main((str(results_path.parent), "--metric", "exploitability")) == 0
    assert plot_path.stat().st_size > 0

    benchmark = run_tabular_benchmark(
        game="kuhn",
        solver_id="naive_cfr",
        iterations=2,
        repeats=2,
    )
    assert benchmark.traversals == 4
    assert benchmark.median_seconds > 0.0
    assert benchmark.memory_metric in {"pss", "uss", "rss"}
    assert benchmark.median_peak_memory_mb > 0.0
    assert len(benchmark.repeat_results) == 2


def test_training_command_uses_useful_output_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "cli_defaults"
    assert (
        train_main(
            (
                "--game",
                "kuhn",
                "--solver",
                "cfr",
                "--iterations",
                "200",
                "--run-id",
                run_id,
                "--runs-root",
                str(tmp_path),
                "--plot",
            )
        )
        == 0
    )
    terminal_output = capsys.readouterr().out
    assert "progress: 5% (10/200 iterations)" in terminal_output
    assert "progress: 100% (200/200 iterations)" in terminal_output

    run_directory = tmp_path / run_id
    run_config = json.loads((run_directory / "run_config.json").read_text(encoding="utf-8"))
    training_config = run_config["training_config"]
    assert training_config["evaluation_interval"] == 2
    assert training_config["checkpoint_interval"] == 20

    with (run_directory / "metrics.csv").open(encoding="utf-8", newline="") as metrics_file:
        reader = csv.DictReader(metrics_file)
        assert tuple(reader.fieldnames or ()) == TRAINING_METRIC_FIELDS
        assert len(list(reader)) == 100
    assert len(tuple((run_directory / "strategy_snapshots").glob("*.npz"))) == 1
    assert len(tuple((run_directory / "checkpoints").glob("*.npz"))) == 11
    assert (run_directory / "plots" / "training_diagnostics.png").stat().st_size > 0
    summary = (run_directory / "summary.txt").read_text(encoding="utf-8")
    assert "Status: completed" in summary
    assert "Iterations: 200" in summary
    assert "Exact exploitability:" in summary

    comparison_config = TabularTrainingConfig(
        game="kuhn",
        solver="naive_cfr",
        iterations=1,
        seed=0,
        run_id="comparison_reference",
        evaluation_interval=1,
        checkpoint_interval=1,
        snapshot_iterations=(),
    )
    comparison_outcome = start_tabular_training(comparison_config, runs_root=tmp_path)
    assert plot_main((str(run_directory), str(comparison_outcome.run_directory))) == 0
    comparison_plot = (
        tmp_path / "plots" / "exploitability_comparison__cli_defaults__comparison_reference.png"
    )
    assert comparison_plot.stat().st_size > 0


def _registry_for_snapshot(snapshot_path: Path, game_id: GameId) -> dict[str, object]:
    metadata = _snapshot_metadata(snapshot_path)
    strategy_id = f"{game_id.value}_test"
    return {
        "schema_version": 1,
        "strategies": [
            {
                "strategy_id": strategy_id,
                "label": f"{game_id.value.capitalize()} test strategy",
                "game": game_id.value,
                "game_version": metadata["game_version"],
                "algorithm": metadata["solver"],
                "agent_type": "tabular",
                "snapshot_id": metadata["snapshot_id"],
                "training_iteration": metadata["training_iteration"],
                "local_path": f"artifacts/{snapshot_path.name}",
                "evaluation": {},
                "model_config_id": metadata["model_config_id"],
                "state_encoding": metadata["state_encoding"],
                "action_space": metadata["action_space"],
                "tree_digest": metadata["tree_digest"],
                "artifact_schema_version": metadata["artifact_schema_version"],
                "release_id": "local-test",
                "file_size": snapshot_path.stat().st_size,
                "sha256": file_sha256(snapshot_path),
            }
        ],
    }


def _snapshot_metadata(snapshot_path: Path) -> dict[str, object]:
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        metadata = json.loads(str(snapshot["metadata"].item()))
    assert isinstance(metadata, dict)
    return metadata


def _schema_one_checkpoint_copy(checkpoint_path: Path) -> Path:
    """Create the deterministic checkpoint shape written before RNG persistence."""
    legacy_path = checkpoint_path.with_name("schema_one_iter_2.npz")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"].item()))
        metadata["checkpoint_schema_version"] = 1
        metadata.pop("rng_state")
        with atomic_binary_writer(legacy_path) as checkpoint_file:
            np.savez_compressed(
                checkpoint_file,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
                regret_sum=checkpoint["regret_sum"],
                strategy_sum=checkpoint["strategy_sum"],
            )
    return legacy_path
