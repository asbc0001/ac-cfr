"""Command-line entry point for exact-metric result plots."""

import argparse
import csv
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
            filename=_comparison_filename(result_paths, plot_name="exploitability_comparison"),
        )
        plot_exploitability_comparison(result_paths, output_path)
    else:
        metric = arguments.metric
        plot_name = f"{metric}_by_{arguments.x_axis}"
        filename = (
            _comparison_filename(result_paths, plot_name=plot_name)
            if len(result_paths) > 1
            else f"{plot_name}.png"
        )
        output_path = arguments.output or _default_output_path(
            arguments.inputs,
            filename=filename,
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


def _comparison_filename(result_paths: tuple[Path, ...], *, plot_name: str) -> str:
    run_ids: list[str] = []
    for result_path in result_paths:
        with result_path.open(encoding="utf-8", newline="") as results_file:
            reader = csv.DictReader(results_file)
            if "run_id" not in (reader.fieldnames or ()):
                raise ValueError(f"results file is missing run_id: {result_path}")
            file_run_ids = {record["run_id"] for record in reader if record["run_id"]}
        if not file_run_ids:
            raise ValueError(f"results file contains no run IDs: {result_path}")
        run_ids.extend(sorted(file_run_ids))
    return f"{plot_name}__{'__'.join(dict.fromkeys(run_ids))}.png"
