"""Command-line entry point for exact-metric result plots."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ac_cfr.evaluation.plotting import plot_exact_metric


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one plot directly from compact result files."""
    parser = argparse.ArgumentParser(description="Plot exact strategy metrics from result CSVs.")
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("exploitability", "nash_conv"), required=True)
    parser.add_argument(
        "--x-axis",
        choices=("iteration", "elapsed_training_seconds"),
        required=True,
    )
    arguments = parser.parse_args(argv)
    plot_exact_metric(
        tuple(arguments.results),
        arguments.output,
        metric=arguments.metric,
        x_axis=arguments.x_axis,
    )
    print(f"plot: {arguments.output}")
    return 0
