"""Command-line entry point for exact-metric result plots."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ac_cfr.evaluation.plotting import (
    plot_exact_metric,
    plot_exploitability_comparison,
    plot_training_diagnostics,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one plot from run directories or result files."""
    parser = argparse.ArgumentParser(description="Plot exact strategy metrics from result CSVs.")
    parser.add_argument("inputs", type=Path, nargs="+", metavar="RUN_OR_RESULTS")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metric",
        choices=("exploitability", "nash_conv"),
    )
    parser.add_argument(
        "--x-axis",
        choices=("iteration", "elapsed_training_seconds"),
        default="iteration",
    )
    arguments = parser.parse_args(argv)
    result_paths = tuple(
        input_path / "metrics.csv" if input_path.is_dir() else input_path
        for input_path in arguments.inputs
    )
    if len(result_paths) == 1 and arguments.metric is None:
        output_path = arguments.output or _default_output_path(
            arguments.inputs,
            filename="training_diagnostics.png",
        )
        plot_training_diagnostics(result_paths[0], output_path)
    elif arguments.metric is None:
        output_path = arguments.output or _default_output_path(
            arguments.inputs,
            filename="exploitability_comparison.png",
        )
        plot_exploitability_comparison(result_paths, output_path)
    else:
        metric = arguments.metric
        output_path = arguments.output or _default_output_path(
            arguments.inputs,
            filename=f"{metric}_by_{arguments.x_axis}.png",
        )
        plot_exact_metric(
            result_paths,
            output_path,
            metric=metric,
            x_axis=arguments.x_axis,
        )
    print(f"plot: {output_path}")
    return 0


def _default_output_path(inputs: list[Path], *, filename: str) -> Path:
    if len(inputs) == 1:
        input_path = inputs[0]
        parent = input_path if input_path.is_dir() else input_path.parent
        return parent / "plots" / filename
    input_parents = {input_path.parent for input_path in inputs}
    parent = input_parents.pop() if len(input_parents) == 1 else Path.cwd()
    return parent / "plots" / filename
