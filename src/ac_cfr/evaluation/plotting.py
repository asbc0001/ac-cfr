"""Training diagnostics and exact-exploitability comparison plots."""

import csv
from pathlib import Path
from typing import Any

KUHN_PLAYER_ZERO_EQUILIBRIUM_VALUE = -1 / 18


def plot_training_diagnostics(result_path: Path, output_path: Path) -> None:
    """Plot convergence, Kuhn's known value, and throughput for one run."""
    records = _training_records(result_path)
    iterations = [int(record["iteration"]) for record in records]
    elapsed_seconds = [float(record["elapsed_training_seconds"]) for record in records]
    exploitability = [float(record["exploitability"]) for record in records]
    player_zero_values = [float(record["expected_value_player_zero"]) for record in records]
    interval_throughput = _interval_throughput(records)

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8))
    axes = figure.subplots(2, 2)
    iteration_axis, time_axis = axes[0]
    value_axis, throughput_axis = axes[1]

    iteration_axis.plot(iterations, exploitability)
    iteration_axis.set_xlabel("Iterations")
    iteration_axis.set_ylabel("Exact exploitability (chips)")
    _use_log_scale_for_positive_values(iteration_axis, exploitability)

    time_axis.plot(elapsed_seconds, exploitability)
    time_axis.set_xlabel("Solver training time (seconds)")
    time_axis.set_ylabel("Exact exploitability (chips)")
    _use_log_scale_for_positive_values(time_axis, exploitability)

    value_axis.plot(iterations, player_zero_values, label="Average-policy value")
    if records[0]["game"] == "kuhn":
        value_axis.axhline(
            KUHN_PLAYER_ZERO_EQUILIBRIUM_VALUE,
            color="tab:red",
            linestyle="--",
            label="Kuhn equilibrium value (-1/18)",
        )
    value_axis.set_xlabel("Iterations")
    value_axis.set_ylabel("Player 0 expected value (chips)")
    value_axis.legend()

    throughput_axis.plot(iterations, interval_throughput)
    throughput_axis.set_xlabel("Iterations")
    throughput_axis.set_ylabel("Full-tree traversals / second")
    throughput_axis.set_title("Diagnostic throughput")

    for axis in (iteration_axis, time_axis, value_axis, throughput_axis):
        axis.grid(alpha=0.25)

    first_record = records[0]
    figure.suptitle(
        f"{first_record['game'].capitalize()} {_solver_label(first_record['solver'])} training"
    )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_exploitability_comparison(
    result_paths: tuple[Path, ...],
    output_path: Path,
) -> None:
    """Compare exact exploitability by iterations and training time."""
    series = _exact_metric_series(result_paths, metric="exploitability")

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4.8))
    iteration_axis, time_axis = figure.subplots(1, 2)
    _plot_series(iteration_axis, series, metric="exploitability", x_axis="iteration")
    _plot_series(
        time_axis,
        series,
        metric="exploitability",
        x_axis="elapsed_training_seconds",
    )
    iteration_axis.set_xlabel("Iterations")
    time_axis.set_xlabel("Solver training time (seconds)")
    for axis in (iteration_axis, time_axis):
        axis.set_ylabel("Exact exploitability (chips)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_exact_metric(
    result_paths: tuple[Path, ...],
    output_path: Path,
    *,
    metric: str,
    x_axis: str,
) -> None:
    """Plot one explicitly selected exact poker metric and x-axis."""
    if metric not in {"exploitability", "nash_conv"}:
        raise ValueError("metric must be exploitability or nash_conv")
    if x_axis not in {"iteration", "elapsed_training_seconds"}:
        raise ValueError("x_axis must be iteration or elapsed_training_seconds")
    series = _exact_metric_series(result_paths, metric=metric)

    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.subplots()
    _plot_series(axis, series, metric=metric, x_axis=x_axis)
    axis.set_xlabel("Iterations" if x_axis == "iteration" else "Solver training time (seconds)")
    axis.set_ylabel("Exact exploitability (chips)" if metric == "exploitability" else "NashConv")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_cfr_gate_results(
    convergence_path: Path,
    benchmark_summary_path: Path,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Plot CFR/CFR+ convergence and implementation-performance evidence."""
    convergence = _read_records(
        convergence_path,
        {
            "game",
            "algorithm",
            "implementation",
            "iteration",
            "elapsed_training_seconds",
            "exploitability",
        },
    )
    benchmarks = _read_records(
        benchmark_summary_path,
        {
            "game",
            "algorithm",
            "implementation",
            "median_seconds",
            "median_absolute_deviation_seconds",
            "traversals_per_second",
            "median_absolute_deviation_traversals_per_second",
            "median_peak_memory_mb",
            "median_absolute_deviation_memory_mb",
        },
    )
    plot_paths = tuple(
        _plot_gate_convergence(convergence, game, output_directory) for game in ("kuhn", "leduc")
    )
    performance_path = output_directory / "implementation_performance.png"
    _plot_gate_performance(benchmarks, performance_path)
    return (*plot_paths, performance_path)


def _plot_gate_convergence(
    records: list[dict[str, str]],
    game: str,
    output_directory: Path,
) -> Path:
    """Plot one game's reference and optimised convergence by work and time."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4.8))
    iteration_axis, time_axis = figure.subplots(1, 2)
    game_records = [record for record in records if record["game"] == game]
    algorithm_colours = {"cfr": "tab:blue", "cfr_plus": "tab:orange"}
    for (algorithm, implementation), label in _gate_series_labels().items():
        series = sorted(
            (
                record
                for record in game_records
                if record["algorithm"] == algorithm and record["implementation"] == implementation
            ),
            key=lambda record: int(record["iteration"]),
        )
        if not series:
            raise ValueError(f"convergence results are missing {game} {label}")
        exploitability = [float(record["exploitability"]) for record in series]
        line_style = "--" if implementation == "reference" else "-"
        marker = "s" if implementation == "reference" else "o"
        iteration_axis.plot(
            [int(record["iteration"]) for record in series],
            exploitability,
            linestyle=line_style,
            marker=marker,
            color=algorithm_colours[algorithm],
            markerfacecolor="none" if implementation == "reference" else None,
            label=label,
        )
        time_axis.plot(
            [float(record["elapsed_training_seconds"]) for record in series],
            exploitability,
            linestyle=line_style,
            marker=marker,
            color=algorithm_colours[algorithm],
            markerfacecolor="none" if implementation == "reference" else None,
            label=label,
        )

    iteration_axis.set_xlabel("Iterations")
    time_axis.set_xlabel("Solver training time (seconds)")
    time_axis.set_xscale("log")
    for axis in (iteration_axis, time_axis):
        axis.set_ylabel("Exact exploitability (chips)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    handles, labels = iteration_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4)
    figure.suptitle(f"{game.capitalize()} exact exploitability")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    output_path = output_directory / f"{game}_convergence.png"
    _save_figure(figure, output_path)
    return output_path


def _plot_gate_performance(records: list[dict[str, str]], output_path: Path) -> None:
    """Plot fixed-workload runtime, throughput, and memory comparisons."""
    import numpy as np
    from matplotlib.figure import Figure

    ordered_keys = tuple(
        (game, algorithm) for game in ("kuhn", "leduc") for algorithm in ("cfr", "cfr_plus")
    )
    by_key = {
        (record["game"], record["algorithm"], record["implementation"]): record
        for record in records
    }
    if len(by_key) != len(ordered_keys) * 2:
        raise ValueError(
            "benchmark summary must contain one reference and optimised row per workload"
        )

    positions = np.arange(len(ordered_keys), dtype=np.float64)
    width = 0.36
    figure = Figure(figsize=(14, 4.8))
    runtime_axis, throughput_axis, memory_axis = figure.subplots(1, 3)
    for offset, implementation, label in (
        (-width / 2, "reference", "Reference"),
        (width / 2, "optimised", "Optimised"),
    ):
        selected = [by_key[(*key, implementation)] for key in ordered_keys]
        runtime_bars = runtime_axis.bar(
            positions + offset,
            [float(record["median_seconds"]) for record in selected],
            width,
            yerr=[float(record["median_absolute_deviation_seconds"]) for record in selected],
            label=label,
            capsize=3,
        )
        if implementation == "optimised":
            speedup_labels = [
                f"{float(by_key[(*key, 'reference')]['median_seconds']) / float(record['median_seconds']):.1f}×"
                for key, record in zip(ordered_keys, selected, strict=True)
            ]
            runtime_axis.bar_label(runtime_bars, labels=speedup_labels, padding=3, fontsize=8)
        throughput_axis.bar(
            positions + offset,
            [float(record["traversals_per_second"]) for record in selected],
            width,
            yerr=[
                float(record["median_absolute_deviation_traversals_per_second"])
                for record in selected
            ],
            label=label,
            capsize=3,
        )
        memory_axis.bar(
            positions + offset,
            [float(record["median_peak_memory_mb"]) for record in selected],
            width,
            yerr=[float(record["median_absolute_deviation_memory_mb"]) for record in selected],
            label=label,
            capsize=3,
        )

    labels = [
        f"{game.capitalize()}\n{'CFR+' if algorithm == 'cfr_plus' else 'CFR'}"
        for game, algorithm in ordered_keys
    ]
    for axis in (runtime_axis, throughput_axis, memory_axis):
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.25)
    runtime_axis.set_yscale("log")
    throughput_axis.set_yscale("log")
    runtime_axis.set_ylabel("Median training time (seconds)")
    throughput_axis.set_ylabel("Median traversals / second")
    memory_axis.set_ylabel("Median peak process-tree memory (MB)")
    handles, legend_labels = runtime_axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncols=2)
    figure.suptitle(
        "Reference versus optimised fixed-workload performance\n"
        "Kuhn: 10,000 iterations; Leduc: 5,000 iterations"
    )
    figure.tight_layout(rect=(0.0, 0.11, 1.0, 0.9))
    _save_figure(figure, output_path)


def _gate_series_labels() -> dict[tuple[str, str], str]:
    """Return display labels for gate algorithm and implementation pairs."""
    return {
        ("cfr", "reference"): "Reference CFR",
        ("cfr", "optimised"): "Optimised CFR",
        ("cfr_plus", "reference"): "Reference CFR+",
        ("cfr_plus", "optimised"): "Optimised CFR+",
    }


def _training_records(result_path: Path) -> list[dict[str, str]]:
    """Load and validate ordered metric records belonging to one training run."""
    required_fields = {
        "game",
        "solver",
        "run_id",
        "iteration",
        "elapsed_training_seconds",
        "expected_value_player_zero",
        "exploitability",
        "traversals",
    }
    records = _read_records(result_path, required_fields)
    identities = {
        (record["game"], record["solver"], record["run_id"], record["seed"]) for record in records
    }
    if not records or len(identities) != 1:
        raise ValueError("training diagnostics require metrics from exactly one run")
    ordered_records = sorted(records, key=lambda record: int(record["iteration"]))
    if len({record["iteration"] for record in ordered_records}) != len(ordered_records):
        raise ValueError("training metrics contain duplicate iterations")
    return ordered_records


def _exact_metric_series(
    result_paths: tuple[Path, ...],
    *,
    metric: str,
) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    """Group one exact metric by game, solver, run, and seed."""
    if not result_paths:
        raise ValueError("at least one results file is required")
    required_fields = {
        "game",
        "solver",
        "run_id",
        "seed",
        "iteration",
        metric,
    }
    series: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for result_path in result_paths:
        for record in _read_records(result_path, required_fields):
            if not record[metric]:
                continue
            key = (record["game"], record["solver"], record["run_id"], record["seed"])
            series.setdefault(key, []).append(record)
    if not series:
        raise ValueError("results files contain no plottable exact metrics")
    for records in series.values():
        iterations = [record["iteration"] for record in records]
        if len(set(iterations)) != len(iterations):
            raise ValueError("the same run contains duplicate metric iterations")
    return series


def _plot_series(
    axis: Any,
    series: dict[tuple[str, str, str, str], list[dict[str, str]]],
    *,
    metric: str,
    x_axis: str,
) -> None:
    """Plot grouped metric records on one supplied Matplotlib axis."""
    labels = _series_labels(tuple(series))
    all_values: list[float] = []
    plotted_series = 0
    for key, records in sorted(series.items()):
        points = sorted(
            (
                (float(record[x_axis]), float(record[metric]))
                for record in records
                if record.get(x_axis)
            ),
            key=lambda point: point[0],
        )
        if not points:
            continue
        plotted_series += 1
        values = [point[1] for point in points]
        all_values.extend(values)
        marker = "o" if len(points) <= 20 else None
        axis.plot(
            [point[0] for point in points],
            values,
            marker=marker,
            label=labels[key],
        )
    if plotted_series == 0:
        raise ValueError(f"results files contain no values for {x_axis}")
    _use_log_scale_for_positive_values(axis, all_values)


def _series_labels(
    keys: tuple[tuple[str, str, str, str], ...],
) -> dict[tuple[str, str, str, str], str]:
    """Build labels that identify the source run and seed of each series."""
    labels: dict[tuple[str, str, str, str], str] = {}
    for key in keys:
        game, solver, run_id, seed = key
        labels[key] = f"{game.capitalize()} {_solver_label(solver)} ({run_id}, seed {seed})"
    return labels


def _read_records(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    """Read CSV records after checking all required columns are present."""
    with path.open(encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        fields = set(reader.fieldnames or ())
        missing_fields = required_fields - fields
        if missing_fields:
            raise ValueError(f"results file is missing fields: {sorted(missing_fields)}")
        return [dict(record) for record in reader]


def _interval_throughput(records: list[dict[str, str]]) -> list[float]:
    """Calculate throughput between successive cumulative training records."""
    throughput: list[float] = []
    previous_traversals = 0
    previous_seconds = 0.0
    for record in records:
        traversals = int(record["traversals"])
        elapsed_seconds = float(record["elapsed_training_seconds"])
        traversal_delta = traversals - previous_traversals
        time_delta = elapsed_seconds - previous_seconds
        if traversal_delta <= 0 or time_delta <= 0.0:
            raise ValueError("training metrics must increase in traversal count and elapsed time")
        throughput.append(traversal_delta / time_delta)
        previous_traversals = traversals
        previous_seconds = elapsed_seconds
    return throughput


def _use_log_scale_for_positive_values(axis: Any, values: list[float]) -> None:
    """Use logarithmic scaling only when every plotted value is positive."""
    if values and all(value > 0.0 for value in values):
        axis.set_yscale("log")


def _solver_label(solver: str) -> str:
    """Return a concise human-readable solver name."""
    return {
        "cfr": "CFR",
        "cfr_plus": "CFR+",
        "naive_cfr": "Naive CFR",
        "naive_cfr_plus": "Naive CFR+",
    }.get(solver, solver)


def _save_figure(figure: Any, output_path: Path) -> None:
    """Create the destination directory and save a consistently sized image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
