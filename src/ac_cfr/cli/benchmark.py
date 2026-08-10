"""Command-line entry point for repeated tabular solver timings."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from ac_cfr.benchmarking.harness import run_tabular_benchmark
from ac_cfr.training.runner import SOLVER_IDS


def main(argv: Sequence[str] | None = None) -> int:
    """Run a user-declared fixed benchmark workload and print JSON."""
    parser = argparse.ArgumentParser(description="Benchmark one fixed tabular solver workload.")
    parser.add_argument("--game", choices=("kuhn", "leduc"), required=True)
    parser.add_argument("--solver", choices=SOLVER_IDS, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--averaging-delay", type=int, default=0)
    arguments = parser.parse_args(argv)
    result = run_tabular_benchmark(
        game=arguments.game,
        solver_id=arguments.solver,
        iterations=arguments.iterations,
        repeats=arguments.repeats,
        averaging_delay=arguments.averaging_delay,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0
