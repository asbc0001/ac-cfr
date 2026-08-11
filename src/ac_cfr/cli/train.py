"""Command-line entry point for checkpointed poker-solver training."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from ac_cfr.common.config import DeepCFRImplementationId, ModelConfigId
from ac_cfr.evaluation.plotting import (
    plot_deep_cfr_training_diagnostics,
    plot_training_diagnostics,
)
from ac_cfr.persistence.results import DeepCFRMetricStore, TrainingMetricStore
from ac_cfr.training.deep_cfr_config import load_deep_cfr_run_config
from ac_cfr.training.deep_cfr_runner import (
    DEEP_CFR_SOLVER_ID,
    DeepCFRRunConfig,
    resume_deep_cfr_training,
    start_deep_cfr_training,
)
from ac_cfr.training.runner import (
    SOLVER_IDS,
    TabularTrainingConfig,
    new_run_id,
    resume_tabular_training,
    start_tabular_training,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume one configured poker-solver training job."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    report_progress = _progress_reporter()
    is_deep_cfr = (
        arguments.config is not None
        or arguments.solver == DEEP_CFR_SOLVER_ID
        or (arguments.resume is not None and arguments.resume.suffix == ".pt")
    )
    if arguments.resume is not None:
        if any(
            value is not None
            for value in (
                arguments.game,
                arguments.solver,
                arguments.config,
                arguments.iterations,
                arguments.seed,
                arguments.run_id,
                arguments.runs_root,
                arguments.evaluation_interval,
                arguments.checkpoint_interval,
                arguments.implementation,
                arguments.snapshot_iterations,
                arguments.averaging_delay,
                arguments.early_stopping_minimum_improvement,
                arguments.early_stopping_patience,
                arguments.traversals_per_player,
                arguments.advantage_reservoir_capacity,
                arguments.strategy_reservoir_capacity,
                arguments.advantage_training_steps,
                arguments.strategy_training_steps,
                arguments.advantage_batch_size,
                arguments.strategy_batch_size,
                arguments.inference_batch_size,
                arguments.cpu_threads,
                arguments.device,
                arguments.model_config_id,
                arguments.learning_rate,
                arguments.validation_fraction,
                arguments.max_gradient_norm,
                arguments.dropout_probability,
            )
        ):
            parser.error("--resume cannot be combined with new-run options")
        outcome = (
            resume_deep_cfr_training(arguments.resume, progress_callback=report_progress)
            if is_deep_cfr
            else resume_tabular_training(arguments.resume, progress_callback=report_progress)
        )
    else:
        if arguments.config is None and (
            arguments.game is None or arguments.solver is None or arguments.iterations is None
        ):
            parser.error("new training requires --game, --solver, and --iterations")
        if arguments.config is not None and arguments.game not in (None, "leduc", "holdem"):
            parser.error("Deep CFR configuration has an incompatible game")
        if arguments.config is not None and arguments.solver not in (None, DEEP_CFR_SOLVER_ID):
            parser.error("Deep CFR configuration is incompatible with the selected solver")
        try:
            snapshots = (
                None
                if arguments.snapshot_iterations is None
                else _parse_iterations(arguments.snapshot_iterations)
            )
        except ValueError as error:
            parser.error(str(error))
        if is_deep_cfr:
            if arguments.config is None:
                parser.error("new Deep CFR training requires --config")
            if arguments.game not in (None, "leduc", "holdem"):
                parser.error("Deep CFR has an incompatible game")
            if any(
                value is not None
                for value in (
                    arguments.evaluation_interval,
                    arguments.averaging_delay,
                    arguments.early_stopping_minimum_improvement,
                    arguments.early_stopping_patience,
                )
            ):
                parser.error(
                    "Deep CFR evaluates snapshots and does not use tabular evaluation, "
                    "averaging, or early-stopping options"
                )
            deep_config = _deep_cfr_config(arguments, snapshots)
            outcome = start_deep_cfr_training(
                deep_config,
                runs_root=Path("runs") if arguments.runs_root is None else arguments.runs_root,
                progress_callback=report_progress,
            )
        else:
            _reject_deep_cfr_options(parser, arguments)
            assert arguments.game is not None
            assert arguments.solver is not None
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
            config = TabularTrainingConfig(
                game=arguments.game,
                solver=arguments.solver,
                iterations=arguments.iterations,
                seed=0 if arguments.seed is None else arguments.seed,
                run_id=arguments.run_id or new_run_id(),
                evaluation_interval=evaluation_interval,
                checkpoint_interval=checkpoint_interval,
                snapshot_iterations=() if snapshots is None else snapshots,
                averaging_delay=0
                if arguments.averaging_delay is None
                else arguments.averaging_delay,
                early_stopping_minimum_improvement=arguments.early_stopping_minimum_improvement,
                early_stopping_patience=arguments.early_stopping_patience,
            )
            outcome = start_tabular_training(
                config,
                runs_root=Path("runs") if arguments.runs_root is None else arguments.runs_root,
                progress_callback=report_progress,
            )

    print(f"run: {outcome.run_directory}")
    print(f"summary: {outcome.run_directory / 'summary.txt'}")
    print(f"iteration: {outcome.final_iteration}")
    print(f"checkpoint: {outcome.latest_checkpoint}")
    for snapshot_path in outcome.snapshot_paths:
        print(f"snapshot: {snapshot_path}")
    _print_final_metrics(outcome.run_directory / "metrics.csv", deep_cfr=is_deep_cfr)
    if arguments.plot:
        plot_path = outcome.run_directory / "plots" / "training_diagnostics.png"
        if is_deep_cfr:
            plot_deep_cfr_training_diagnostics(
                outcome.run_directory / "metrics.csv",
                plot_path,
            )
        else:
            plot_training_diagnostics(outcome.run_directory / "metrics.csv", plot_path)
        print(f"plot: {plot_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the shared training command-line parser."""
    parser = argparse.ArgumentParser(description="Train a poker solver.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--config", type=Path, help="Deep CFR TOML configuration preset.")
    parser.add_argument("--game", choices=("kuhn", "leduc", "holdem"))
    parser.add_argument("--solver", choices=(*SOLVER_IDS, DEEP_CFR_SOLVER_ID))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument(
        "--implementation",
        choices=tuple(implementation.value for implementation in DeepCFRImplementationId),
    )
    parser.add_argument(
        "--snapshot-iterations",
        help="Comma-separated outer iterations; the final policy is always exported.",
    )
    parser.add_argument("--averaging-delay", type=int)
    parser.add_argument("--early-stopping-minimum-improvement", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--traversals-per-player", type=int)
    parser.add_argument("--advantage-reservoir-capacity", type=int)
    parser.add_argument("--strategy-reservoir-capacity", type=int)
    parser.add_argument(
        "--advantage-training-steps",
        type=int,
        help="Minibatch updates for each freshly initialised advantage network.",
    )
    parser.add_argument(
        "--strategy-training-steps",
        type=int,
        help="Minibatch updates for each exported average-strategy network.",
    )
    parser.add_argument(
        "--advantage-batch-size",
        type=int,
    )
    parser.add_argument("--strategy-batch-size", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--model-config-id",
        choices=tuple(config_id.value for config_id in ModelConfigId),
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--max-gradient-norm", type=float)
    parser.add_argument("--dropout-probability", type=float)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create training diagnostics after training.",
    )
    return parser


def _parse_iterations(value: str | None) -> tuple[int, ...]:
    """Parse unique snapshot iterations from a comma-separated option."""
    if value is None or not value.strip():
        return ()
    try:
        return tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("snapshot iterations must be comma-separated integers") from error


def _interval_for_target_count(iterations: int, *, target_count: int) -> int:
    """Choose an interval producing at most roughly the requested event count."""
    return max(1, (iterations + target_count - 1) // target_count)


def _progress_reporter() -> Callable[[int, int], None]:
    """Create a callback that prints progress at five-percentage-point steps."""
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


def _print_final_metrics(metrics_path: Path, *, deep_cfr: bool) -> None:
    """Print the final exact metrics recorded for a completed run."""
    store = DeepCFRMetricStore(metrics_path) if deep_cfr else TrainingMetricStore(metrics_path)
    records = tuple(record for record in store.records if record["exploitability"])
    if not records:
        return
    final_record = max(records, key=lambda record: int(record["iteration"]))
    print(
        f"player-zero average-policy value: {float(final_record['expected_value_player_zero']):.12g}"
    )
    print(f"exploitability: {float(final_record['exploitability']):.12g}")
    print(f"solver training time: {float(final_record['elapsed_training_seconds']):.6g} seconds")
    print(f"average traversal throughput: {float(final_record['traversals_per_second']):.6g}/s")


def _deep_cfr_config(
    arguments: argparse.Namespace,
    snapshots: tuple[int, ...] | None,
) -> DeepCFRRunConfig:
    """Load a Deep CFR preset and apply only explicitly supplied overrides."""
    if arguments.config is None:
        raise ValueError("new Deep CFR training requires --config")
    run_id = arguments.run_id or new_run_id()
    overrides = _deep_cfr_overrides(arguments, snapshots)
    config = load_deep_cfr_run_config(
        arguments.config,
        run_id=run_id,
        overrides=overrides,
    )
    expected_game = (
        "holdem"
        if config.training.game_configuration_id.value == "modified_hulhe"
        else config.training.game_configuration_id.value
    )
    if arguments.game is not None and arguments.game != expected_game:
        raise ValueError("--game does not match the Deep CFR preset")
    return config


def _deep_cfr_overrides(
    arguments: argparse.Namespace,
    snapshots: tuple[int, ...] | None,
) -> dict[str, object]:
    """Return only explicitly supplied values that override a TOML preset."""
    names = (
        "iterations",
        "seed",
        "checkpoint_interval",
        "implementation",
        "traversals_per_player",
        "advantage_reservoir_capacity",
        "strategy_reservoir_capacity",
        "advantage_training_steps",
        "strategy_training_steps",
        "advantage_batch_size",
        "strategy_batch_size",
        "inference_batch_size",
        "cpu_threads",
        "device",
        "learning_rate",
        "validation_fraction",
        "max_gradient_norm",
        "dropout_probability",
        "model_config_id",
    )
    overrides = {
        name: getattr(arguments, name) for name in names if getattr(arguments, name) is not None
    }
    if snapshots is not None:
        overrides["snapshot_iterations"] = snapshots
    return overrides


def _reject_deep_cfr_options(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> None:
    """Reject neural-only options when a tabular solver was selected."""
    names = (
        "traversals_per_player",
        "advantage_reservoir_capacity",
        "strategy_reservoir_capacity",
        "advantage_training_steps",
        "strategy_training_steps",
        "advantage_batch_size",
        "strategy_batch_size",
        "inference_batch_size",
        "cpu_threads",
        "device",
        "model_config_id",
        "learning_rate",
        "validation_fraction",
        "max_gradient_norm",
        "dropout_probability",
        "implementation",
    )
    if any(getattr(arguments, name) is not None for name in names):
        parser.error("Deep CFR options require --solver deep_cfr")
