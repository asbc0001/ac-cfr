"""Command-line entry point for poker-solver benchmarks and validation suites."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ac_cfr.benchmarking.cfr_gate import BENCHMARK_ID, run_cfr_gate
from ac_cfr.benchmarking.deep_cfr_benchmark import (
    run_deep_cfr_benchmark,
    run_deep_cfr_convergence_comparison,
)
from ac_cfr.benchmarking.deep_cfr_profiling import run_deep_cfr_profiling
from ac_cfr.benchmarking.deep_cfr_reference_validation import (
    run_deep_cfr_reference_validation,
)
from ac_cfr.benchmarking.deep_cfr_selected_validation import (
    run_deep_cfr_selected_validation,
)
from ac_cfr.benchmarking.deep_cfr_sensitivity import run_deep_cfr_sensitivity_study
from ac_cfr.benchmarking.harness import run_tabular_benchmark
from ac_cfr.benchmarking.mccfr_gate import run_mccfr_gate
from ac_cfr.benchmarking.mccfr_validation import run_mccfr_validation
from ac_cfr.benchmarking.modified_hulhe_calibration import (
    DEFAULT_OUTPUT_DIRECTORY as MODIFIED_HULHE_CALIBRATION_OUTPUT,
)
from ac_cfr.benchmarking.modified_hulhe_calibration import run_modified_hulhe_calibration
from ac_cfr.training.runner import SOLVER_IDS


def main(argv: Sequence[str] | None = None) -> int:
    """Run a declared benchmark workload or validation suite."""
    parser = argparse.ArgumentParser(description="Benchmark and validate poker solvers.")
    parser.add_argument(
        "--suite",
        choices=(
            "cfr-cfr-plus",
            "mccfr-validation",
            "mccfr-gate",
            "deep-cfr-reference",
            "deep-cfr-profile",
            "deep-cfr-benchmark",
            "deep-cfr-comparison",
            "deep-cfr-sensitivity",
            "deep-cfr-validation",
            "modified-hulhe-calibration",
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--game", choices=("kuhn", "leduc"))
    parser.add_argument("--solver", choices=SOLVER_IDS)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--averaging-delay", type=int)
    arguments = parser.parse_args(argv)
    if arguments.suite is not None:
        if any(
            value is not None
            for value in (
                arguments.game,
                arguments.solver,
                arguments.iterations,
                arguments.repeats,
                arguments.averaging_delay,
            )
        ):
            parser.error("--suite cannot be combined with individual-workload options")
        if arguments.suite == "cfr-cfr-plus":
            output_directory = arguments.output or Path("results") / BENCHMARK_ID
            result_path = run_cfr_gate(output_directory, progress_callback=print)
            result_label = "gate"
        elif arguments.suite == "mccfr-validation":
            output_directory = arguments.output or Path("results/mccfr")
            result_path = run_mccfr_validation(
                output_directory,
                progress_callback=print,
            )
            result_label = "validation"
        elif arguments.suite == "mccfr-gate":
            output_directory = arguments.output or Path("results/mccfr")
            result_path = run_mccfr_gate(output_directory, progress_callback=print)
            result_label = "gate"
        elif arguments.suite == "deep-cfr-reference":
            output_directory = arguments.output or Path("results/deep_cfr")
            result_path = run_deep_cfr_reference_validation(
                output_directory,
                progress_callback=print,
            )
            result_label = "validation"
        elif arguments.suite == "deep-cfr-profile":
            output_directory = arguments.output or Path("results/deep_cfr")
            result_path = run_deep_cfr_profiling(
                output_directory,
                progress_callback=print,
            )
            result_label = "profiling"
        elif arguments.suite == "deep-cfr-benchmark":
            output_directory = arguments.output or Path("results/deep_cfr")
            result_path = run_deep_cfr_benchmark(
                output_directory,
                progress_callback=print,
            )
            result_label = "benchmark"
        elif arguments.suite == "deep-cfr-comparison":
            output_directory = arguments.output or Path("results/deep_cfr")
            result_path = run_deep_cfr_convergence_comparison(
                output_directory,
                progress_callback=print,
            )
            result_label = "comparison"
        elif arguments.suite == "deep-cfr-sensitivity":
            output_directory = arguments.output or Path("results/deep_cfr")
            result_path = run_deep_cfr_sensitivity_study(
                output_directory,
                progress_callback=print,
            )
            result_label = "configuration study"
        elif arguments.suite == "deep-cfr-validation":
            output_directory = arguments.output or Path("results/deep_cfr")
            result_path = run_deep_cfr_selected_validation(
                output_directory,
                progress_callback=print,
            )
            result_label = "selected validation"
        else:
            output_directory = arguments.output or MODIFIED_HULHE_CALIBRATION_OUTPUT
            result_path = run_modified_hulhe_calibration(
                output_directory,
                progress_callback=print,
            )
            result_label = "modified-HULHE calibration"
        print(f"{result_label}: {result_path}")
        return 0

    if arguments.output is not None:
        parser.error("--output applies only to --suite")
    if arguments.game is None or arguments.solver is None or arguments.iterations is None:
        parser.error("an individual benchmark requires --game, --solver, and --iterations")
    result = run_tabular_benchmark(
        game=arguments.game,
        solver_id=arguments.solver,
        iterations=arguments.iterations,
        repeats=5 if arguments.repeats is None else arguments.repeats,
        averaging_delay=0 if arguments.averaging_delay is None else arguments.averaging_delay,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0
