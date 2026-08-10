"""Command-line entry point for checkpointed tabular poker training."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from ac_cfr.evaluation.plotting import plot_training_diagnostics
from ac_cfr.persistence.results import TrainingMetricStore
from ac_cfr.training.runner import (
    SOLVER_IDS,
    TabularTrainingConfig,
    new_run_id,
    resume_tabular_training,
    start_tabular_training,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume one configured CFR training job."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    report_progress = _progress_reporter()
    if arguments.resume is not None:
        if any(
            value is not None
            for value in (
                arguments.game,
                arguments.solver,
                arguments.iterations,
                arguments.seed,
                arguments.run_id,
                arguments.runs_root,
                arguments.evaluation_interval,
                arguments.checkpoint_interval,
                arguments.snapshot_iterations,
                arguments.averaging_delay,
                arguments.early_stopping_minimum_improvement,
                arguments.early_stopping_patience,
            )
        ):
            parser.error("--resume cannot be combined with new-run options")
        outcome = resume_tabular_training(
            arguments.resume,
            progress_callback=report_progress,
        )
    else:
        if arguments.game is None or arguments.solver is None or arguments.iterations is None:
            parser.error("new training requires --game, --solver, and --iterations")
        evaluation_interval = (
            _interval_for_target_count(arguments.iterations, target_count=100)
            if arguments.evaluation_interval is None
            else arguments.evaluation_interval
        )
        checkpoint_interval = (
            _interval_for_target_count(arguments.iterations, target_count=10)
            if arguments.checkpoint_interval is None
            else arguments.checkpoint_interval
        )
        try:
            snapshots = _parse_iterations(arguments.snapshot_iterations)
        except ValueError as error:
            parser.error(str(error))
        config = TabularTrainingConfig(
            game=arguments.game,
            solver=arguments.solver,
            iterations=arguments.iterations,
            seed=0 if arguments.seed is None else arguments.seed,
            run_id=arguments.run_id or new_run_id(),
            evaluation_interval=evaluation_interval,
            checkpoint_interval=checkpoint_interval,
            snapshot_iterations=snapshots,
            averaging_delay=0 if arguments.averaging_delay is None else arguments.averaging_delay,
            early_stopping_minimum_improvement=arguments.early_stopping_minimum_improvement,
            early_stopping_patience=arguments.early_stopping_patience,
        )
        outcome = start_tabular_training(
            config,
            runs_root=Path("runs") if arguments.runs_root is None else arguments.runs_root,
            progress_callback=report_progress,
        )

    print(f"run: {outcome.run_directory}")
    print(f"iteration: {outcome.final_iteration}")
    print(f"checkpoint: {outcome.latest_checkpoint}")
    for snapshot_path in outcome.snapshot_paths:
        print(f"snapshot: {snapshot_path}")
    _print_final_metrics(outcome.run_directory / "metrics.csv")
    if arguments.plot:
        plot_path = outcome.run_directory / "plots" / "training_diagnostics.png"
        plot_training_diagnostics(outcome.run_directory / "metrics.csv", plot_path)
        print(f"plot: {plot_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CFR or CFR+ on Kuhn or Leduc.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--game", choices=("kuhn", "leduc"))
    parser.add_argument("--solver", choices=SOLVER_IDS)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument(
        "--snapshot-iterations",
        help="Comma-separated outer iterations; the final policy is always exported.",
    )
    parser.add_argument("--averaging-delay", type=int)
    parser.add_argument("--early-stopping-minimum-improvement", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create training diagnostics after training.",
    )
    return parser


def _parse_iterations(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    try:
        return tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("snapshot iterations must be comma-separated integers") from error


def _interval_for_target_count(iterations: int, *, target_count: int) -> int:
    return max(1, (iterations + target_count - 1) // target_count)


def _progress_reporter() -> Callable[[int, int], None]:
    reported_percentage: int | None = None

    def report(completed: int, total: int) -> None:
        nonlocal reported_percentage
        percentage = min(100, completed * 100 // total)
        rounded_percentage = percentage // 5 * 5
        if reported_percentage is None:
            reported_percentage = rounded_percentage
        elif rounded_percentage > reported_percentage:
            reported_percentage = rounded_percentage
            print(f"progress: {rounded_percentage}% ({completed}/{total} iterations)")

    return report


def _print_final_metrics(metrics_path: Path) -> None:
    records = tuple(
        record for record in TrainingMetricStore(metrics_path).records if record["exploitability"]
    )
    if not records:
        return
    final_record = max(records, key=lambda record: int(record["iteration"]))
    print(
        f"player-zero average-policy value: {float(final_record['expected_value_player_zero']):.12g}"
    )
    print(f"exploitability: {float(final_record['exploitability']):.12g}")
    print(f"solver training time: {float(final_record['elapsed_training_seconds']):.6g} seconds")
    print(f"average traversal throughput: {float(final_record['traversals_per_second']):.6g}/s")
