"""Command-line entry point for repeated tabular solver timings."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ac_cfr.benchmarking.cfr_gate import BENCHMARK_ID, run_cfr_gate
from ac_cfr.benchmarking.harness import run_tabular_benchmark
from ac_cfr.training.runner import SOLVER_IDS


def main(argv: Sequence[str] | None = None) -> int:
    """Run a user-declared fixed benchmark workload and print JSON."""
    parser = argparse.ArgumentParser(description="Benchmark tabular poker solvers.")
    parser.add_argument("--suite", choices=("cfr-cfr-plus",))
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
        output_directory = arguments.output or Path("results") / BENCHMARK_ID
        gate_path = run_cfr_gate(output_directory, progress_callback=print)
        print(f"gate: {gate_path}")
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
