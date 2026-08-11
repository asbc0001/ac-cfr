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


def plot_deep_cfr_training_diagnostics(result_path: Path, output_path: Path) -> None:
    """Plot exact Leduc quality, neural losses, and traversal throughput."""
    records = _read_records(
        result_path,
        {
            "solver",
            "iteration",
            "elapsed_training_seconds",
            "exploitability",
            "traversals_per_second",
            "player_zero_advantage_training_loss",
            "player_zero_advantage_validation_loss",
            "player_one_advantage_training_loss",
            "player_one_advantage_validation_loss",
            "strategy_training_loss",
            "strategy_validation_loss",
        },
    )
    records.sort(key=lambda record: int(record["iteration"]))
    iterations = [int(record["iteration"]) for record in records]
    exploitability = [float(record["exploitability"]) for record in records]

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8))
    iteration_axis, time_axis, loss_axis, throughput_axis = figure.subplots(2, 2).flat
    iteration_axis.plot(iterations, exploitability, marker="o")
    time_axis.plot(
        [float(record["elapsed_training_seconds"]) for record in records],
        exploitability,
        marker="o",
    )
    for axis in (iteration_axis, time_axis):
        axis.set_ylabel("Exact exploitability (chips)")
        _use_log_scale_for_positive_values(axis, exploitability)
    iteration_axis.set_xlabel("Deep CFR outer iterations")
    time_axis.set_xlabel("Training time (seconds)")

    for field, label in (
        ("player_zero_advantage_validation_loss", "Player 0 advantage validation"),
        ("player_one_advantage_validation_loss", "Player 1 advantage validation"),
    ):
        values = [float(record[field]) for record in records if record[field]]
        x_values = [int(record["iteration"]) for record in records if record[field]]
        if values:
            loss_axis.plot(x_values, values, marker="o", label=label)
    loss_axis.set_xlabel("Deep CFR outer iterations")
    loss_axis.set_ylabel("Advantage held-out loss")

    strategy_loss_axis = loss_axis.twinx()
    strategy_values = [
        float(record["strategy_validation_loss"])
        for record in records
        if record["strategy_validation_loss"]
    ]
    strategy_iterations = [
        int(record["iteration"]) for record in records if record["strategy_validation_loss"]
    ]
    if strategy_values:
        strategy_loss_axis.plot(
            strategy_iterations,
            strategy_values,
            color="tab:green",
            marker="o",
            label="Average-strategy validation",
        )
    strategy_loss_axis.set_ylabel("Average-strategy held-out loss")
    lines = (*loss_axis.lines, *strategy_loss_axis.lines)
    loss_axis.legend(lines, [line.get_label() for line in lines])

    throughput_axis.plot(
        iterations,
        [float(record["traversals_per_second"]) for record in records],
        marker="o",
    )
    throughput_axis.set_xlabel("Deep CFR outer iterations")
    throughput_axis.set_ylabel("Average sampled traversals / second")
    for axis in (iteration_axis, time_axis, loss_axis, throughput_axis):
        axis.grid(alpha=0.25)
    implementation = records[0]["solver"].capitalize()
    figure.suptitle(f"{implementation} Deep CFR training on Leduc")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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
            "memory_metric",
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


def plot_deep_cfr_implementation_convergence(
    convergence_path: Path,
    output_path: Path,
) -> None:
    """Compare matched reference and optimised Deep CFR learning trajectories."""
    records = _read_records(
        convergence_path,
        {
            "implementation",
            "iteration",
            "elapsed_training_seconds",
            "exploitability",
        },
    )
    implementations = {record["implementation"] for record in records}
    if implementations != {"reference", "optimised"}:
        raise ValueError("Deep CFR convergence requires reference and optimised records")

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4.8))
    iteration_axis, time_axis = figure.subplots(1, 2)
    for implementation, label, colour in (
        ("reference", "Reference", "tab:orange"),
        ("optimised", "Optimised", "tab:blue"),
    ):
        series = sorted(
            (record for record in records if record["implementation"] == implementation),
            key=lambda record: int(record["iteration"]),
        )
        values = [float(record["exploitability"]) for record in series]
        iteration_axis.plot(
            [int(record["iteration"]) for record in series],
            values,
            marker="o",
            color=colour,
            label=label,
        )
        time_axis.plot(
            [float(record["elapsed_training_seconds"]) for record in series],
            values,
            marker="o",
            color=colour,
            label=label,
        )
    iteration_axis.set_xlabel("Deep CFR outer iterations")
    time_axis.set_xlabel("Training time (seconds)")
    for axis in (iteration_axis, time_axis):
        axis.set_ylabel("Exact exploitability (chips)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Matched Deep CFR convergence on Leduc")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_figure(figure, output_path)


def plot_deep_cfr_sensitivity(result_path: Path, output_path: Path) -> None:
    """Compare quality, phase timing, validation loss, and throughput by configuration."""
    records = _read_records(
        result_path,
        {
            "case",
            "exploitability",
            "traversal_seconds",
            "advantage_training_seconds",
            "strategy_training_seconds",
            "collection_traversals_per_second",
            "player_zero_advantage_training_loss",
            "player_zero_advantage_validation_loss",
            "player_one_advantage_training_loss",
            "player_one_advantage_validation_loss",
            "strategy_training_loss",
            "strategy_validation_loss",
        },
    )
    expected_cases = (
        "baseline",
        "lower_k",
        "lower_advantage_steps",
        "advantage_steps_150",
        "advantage_steps_200",
        "smaller_network",
    )
    by_case = {record["case"]: record for record in records}
    if set(by_case) != set(expected_cases):
        raise ValueError("Deep CFR sensitivity results contain unexpected cases")
    ordered = [by_case[name] for name in expected_cases]
    labels = (
        "Baseline",
        "Lower K",
        "50 advantage steps",
        "150 advantage steps",
        "200 advantage steps",
        "Smaller network",
    )
    positions = list(range(len(ordered)))

    from matplotlib.figure import Figure

    figure = Figure(figsize=(13, 8))
    quality_axis, time_axis, loss_axis, throughput_axis = figure.subplots(2, 2).flat
    quality_axis.bar(
        positions,
        [float(record["exploitability"]) for record in ordered],
        color="tab:blue",
    )
    quality_axis.set_ylabel("Exact exploitability (chips)")

    bottoms = [0.0] * len(ordered)
    for field, label, colour in (
        ("traversal_seconds", "Traversal", "tab:blue"),
        ("advantage_training_seconds", "Advantage training", "tab:orange"),
        ("strategy_training_seconds", "Strategy training", "tab:green"),
    ):
        values = [float(record[field]) for record in ordered]
        time_axis.bar(positions, values, bottom=bottoms, label=label, color=colour)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    time_axis.set_ylabel("Training time (seconds)")
    time_axis.legend()

    width = 0.25
    for offset, training_field, validation_field, label in (
        (
            -width,
            "player_zero_advantage_training_loss",
            "player_zero_advantage_validation_loss",
            "Player 0 advantage",
        ),
        (
            0.0,
            "player_one_advantage_training_loss",
            "player_one_advantage_validation_loss",
            "Player 1 advantage",
        ),
        (
            width,
            "strategy_training_loss",
            "strategy_validation_loss",
            "Average strategy",
        ),
    ):
        loss_axis.bar(
            [position + offset for position in positions],
            [float(record[validation_field]) / float(record[training_field]) for record in ordered],
            width=width,
            label=label,
        )
    loss_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    loss_axis.set_ylabel("Held-out / training loss")
    loss_axis.legend()

    throughput_axis.bar(
        positions,
        [float(record["collection_traversals_per_second"]) for record in ordered],
        color="tab:purple",
    )
    throughput_axis.set_ylabel("Traversal collection / second")
    for axis in (quality_axis, time_axis, loss_axis, throughput_axis):
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Optimised Leduc Deep CFR configuration sensitivity")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(figure, output_path)


def plot_selected_deep_cfr_validation(
    result_path: Path,
    output_path: Path,
    *,
    main_seed: int,
) -> None:
    """Plot multi-seed quality and moderate-run neural-training diagnostics."""
    records = _read_records(
        result_path,
        {
            "seed",
            "iteration",
            "elapsed_training_seconds",
            "exploitability",
            "player_zero_advantage_training_loss",
            "player_zero_advantage_validation_loss",
            "player_one_advantage_training_loss",
            "player_one_advantage_validation_loss",
            "strategy_training_loss",
            "strategy_validation_loss",
        },
    )
    seeds = sorted({int(record["seed"]) for record in records})
    if main_seed not in seeds or len(seeds) != 3:
        raise ValueError(
            "selected Deep CFR validation requires three seeds including the main seed"
        )
    main_records = sorted(
        (record for record in records if int(record["seed"]) == main_seed),
        key=lambda record: int(record["iteration"]),
    )

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8))
    quality_axis, time_axis, advantage_axis, strategy_axis = figure.subplots(2, 2).flat
    for seed in seeds:
        seed_records = sorted(
            (record for record in records if int(record["seed"]) == seed),
            key=lambda record: int(record["iteration"]),
        )
        quality_axis.plot(
            [int(record["iteration"]) for record in seed_records],
            [float(record["exploitability"]) for record in seed_records],
            marker="o",
            label=f"Seed {seed}" + (" (moderate)" if seed == main_seed else ""),
        )
    quality_axis.set_xlabel("Deep CFR outer iterations")
    quality_axis.set_ylabel("Exact exploitability (chips)")
    quality_axis.legend()

    time_axis.plot(
        [float(record["elapsed_training_seconds"]) for record in main_records],
        [float(record["exploitability"]) for record in main_records],
        marker="o",
    )
    time_axis.set_xlabel("Moderate-run training time (seconds)")
    time_axis.set_ylabel("Exact exploitability (chips)")

    iterations = [int(record["iteration"]) for record in main_records]
    advantage_training = [
        (
            float(record["player_zero_advantage_training_loss"])
            + float(record["player_one_advantage_training_loss"])
        )
        / 2.0
        for record in main_records
    ]
    advantage_validation = [
        (
            float(record["player_zero_advantage_validation_loss"])
            + float(record["player_one_advantage_validation_loss"])
        )
        / 2.0
        for record in main_records
    ]
    advantage_axis.plot(iterations, advantage_training, marker="o", label="Training")
    advantage_axis.plot(iterations, advantage_validation, marker="o", label="Held-out")
    advantage_axis.set_xlabel("Deep CFR outer iterations")
    advantage_axis.set_ylabel("Mean advantage-network loss")
    advantage_axis.legend()

    strategy_axis.plot(
        iterations,
        [float(record["strategy_training_loss"]) for record in main_records],
        marker="o",
        label="Training",
    )
    strategy_axis.plot(
        iterations,
        [float(record["strategy_validation_loss"]) for record in main_records],
        marker="o",
        label="Held-out",
    )
    strategy_axis.set_xlabel("Deep CFR outer iterations")
    strategy_axis.set_ylabel("Average-strategy network loss")
    strategy_axis.legend()

    for axis in (quality_axis, time_axis, advantage_axis, strategy_axis):
        axis.grid(alpha=0.25)
    figure.suptitle("Selected optimised Deep CFR validation on Leduc")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(figure, output_path)


def plot_mccfr_validation(
    convergence_path: Path,
    summary_path: Path,
    output_path: Path,
    *,
    reference_exploitability_limit: float,
) -> None:
    """Plot reference and optimised MCCFR seed distributions and medians."""
    convergence = _read_records(
        convergence_path,
        {"implementation", "seed", "iteration", "elapsed_training_seconds", "exploitability"},
    )
    summary = _read_records(
        summary_path,
        {
            "implementation",
            "iteration",
            "median_elapsed_training_seconds",
            "median_exploitability",
        },
    )

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4.8))
    iteration_axis, time_axis = figure.subplots(1, 2)
    colours = {"reference": "tab:orange", "optimised": "tab:blue"}
    labels = {"reference": "Reference", "optimised": "Optimised"}
    for implementation in ("reference", "optimised"):
        implementation_records = [
            record for record in convergence if record["implementation"] == implementation
        ]
        for seed in sorted({record["seed"] for record in implementation_records}, key=int):
            seed_records = sorted(
                (record for record in implementation_records if record["seed"] == seed),
                key=lambda record: int(record["iteration"]),
            )
            values = [float(record["exploitability"]) for record in seed_records]
            iteration_axis.plot(
                [int(record["iteration"]) for record in seed_records],
                values,
                color=colours[implementation],
                alpha=0.14,
                linewidth=0.8,
            )
            time_axis.plot(
                [float(record["elapsed_training_seconds"]) for record in seed_records],
                values,
                color=colours[implementation],
                alpha=0.14,
                linewidth=0.8,
            )

        aggregate = sorted(
            (record for record in summary if record["implementation"] == implementation),
            key=lambda record: int(record["iteration"]),
        )
        median_values = [float(record["median_exploitability"]) for record in aggregate]
        iteration_axis.plot(
            [int(record["iteration"]) for record in aggregate],
            median_values,
            color=colours[implementation],
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"{labels[implementation]} median",
        )
        time_axis.plot(
            [float(record["median_elapsed_training_seconds"]) for record in aggregate],
            median_values,
            color=colours[implementation],
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"{labels[implementation]} median",
        )

    for axis in (iteration_axis, time_axis):
        axis.axhline(
            reference_exploitability_limit,
            color="tab:green",
            linestyle="--",
            label="CFR/CFR+ validation ceiling",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel("Exact exploitability (chips)")
        axis.grid(alpha=0.25)
    iteration_axis.set_xlabel("MCCFR iterations")
    time_axis.set_xlabel("Solver training time per seed (seconds)")
    handles, legend_labels = iteration_axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncols=3)
    figure.suptitle("MCCFR strategy quality on Leduc across five seeds")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    _save_figure(figure, output_path)


def plot_mccfr_performance(summary_path: Path, output_path: Path) -> None:
    """Plot fixed-workload MCCFR runtime, throughput, and peak memory."""
    records = _read_records(
        summary_path,
        {
            "implementation",
            "iterations",
            "traversals",
            "median_seconds",
            "median_absolute_deviation_seconds",
            "traversals_per_second",
            "median_absolute_deviation_traversals_per_second",
            "memory_metric",
            "median_peak_memory_mb",
            "median_absolute_deviation_memory_mb",
        },
    )
    by_implementation = {record["implementation"]: record for record in records}
    if set(by_implementation) != {"reference", "optimised"}:
        raise ValueError("MCCFR performance results require reference and optimised records")

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4.4))
    runtime_axis, throughput_axis, memory_axis = figure.subplots(1, 3)
    implementations = ("reference", "optimised")
    labels = ("Reference", "Optimised")
    colours = ("tab:orange", "tab:blue")
    iterations = {int(record["iterations"]) for record in records}
    traversals = {int(record["traversals"]) for record in records}
    if len(iterations) != 1 or len(traversals) != 1:
        raise ValueError("MCCFR performance results use inconsistent workloads")
    iteration_count = iterations.pop()
    traversal_count = traversals.pop()
    speedup = float(by_implementation["reference"]["median_seconds"]) / float(
        by_implementation["optimised"]["median_seconds"]
    )
    memory_metric = by_implementation["reference"]["memory_metric"]
    if by_implementation["optimised"]["memory_metric"] != memory_metric:
        raise ValueError("MCCFR performance results use inconsistent memory metrics")
    values_and_errors = (
        (
            runtime_axis,
            "median_seconds",
            "median_absolute_deviation_seconds",
            "Training time (seconds)",
        ),
        (
            throughput_axis,
            "traversals_per_second",
            "median_absolute_deviation_traversals_per_second",
            "Sampled traversals / second",
        ),
        (
            memory_axis,
            "median_peak_memory_mb",
            "median_absolute_deviation_memory_mb",
            f"Peak process-tree {memory_metric.upper()} (MB)",
        ),
    )
    for axis, value_field, error_field, ylabel in values_and_errors:
        values = [float(by_implementation[name][value_field]) for name in implementations]
        errors = [float(by_implementation[name][error_field]) for name in implementations]
        bars = axis.bar(labels, values, yerr=errors, color=colours, capsize=4)
        if value_field == "traversals_per_second":
            value_labels = [f"{value:,.0f}" for value in values]
        elif value_field == "median_seconds":
            value_labels = [f"{value:.2f}" for value in values]
        else:
            value_labels = [f"{value:.1f}" for value in values]
        if value_field == "median_seconds":
            value_labels[1] = f"{value_labels[1]}\n{speedup:.1f}× faster"
        axis.bar_label(bars, labels=value_labels, padding=3)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    runtime_axis.set_yscale("log")
    throughput_axis.set_yscale("log")
    figure.suptitle(
        "Reference versus optimised MCCFR performance on Leduc\n"
        f"{iteration_count:,} iterations; {traversal_count:,} sampled traversals"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    _save_figure(figure, output_path)


def plot_deep_cfr_performance(summary_path: Path, output_path: Path) -> None:
    """Plot Deep CFR total time, phase costs, traversal throughput, and memory."""
    records = _read_records(
        summary_path,
        {
            "implementation",
            "traversals",
            "optimizer_steps",
            "median_seconds",
            "median_absolute_deviation_seconds",
            "median_traversal_seconds",
            "median_advantage_training_seconds",
            "median_strategy_training_seconds",
            "median_other_seconds",
            "collection_traversals_per_second",
            "median_absolute_deviation_collection_traversals_per_second",
            "memory_metric",
            "median_peak_memory_mb",
            "median_absolute_deviation_memory_mb",
        },
    )
    by_implementation = {record["implementation"]: record for record in records}
    if set(by_implementation) != {"reference", "optimised"}:
        raise ValueError("Deep CFR performance requires reference and optimised records")
    traversal_counts = {int(record["traversals"]) for record in records}
    optimizer_steps = {int(record["optimizer_steps"]) for record in records}
    memory_metrics = {record["memory_metric"] for record in records}
    if len(traversal_counts) != 1 or len(optimizer_steps) != 1 or len(memory_metrics) != 1:
        raise ValueError("Deep CFR performance records use inconsistent workloads")

    from matplotlib.figure import Figure

    implementations = ("reference", "optimised")
    labels = ("Reference", "Optimised")
    colours = ("tab:orange", "tab:blue")
    figure = Figure(figsize=(13, 8))
    axes = figure.subplots(2, 2)
    runtime_axis, phase_axis, throughput_axis, memory_axis = axes.flat

    runtimes = [float(by_implementation[name]["median_seconds"]) for name in implementations]
    runtime_errors = [
        float(by_implementation[name]["median_absolute_deviation_seconds"])
        for name in implementations
    ]
    runtime_bars = runtime_axis.bar(labels, runtimes, yerr=runtime_errors, color=colours, capsize=4)
    speedup = runtimes[0] / runtimes[1]
    runtime_axis.bar_label(
        runtime_bars,
        labels=(f"{runtimes[0]:.2f}", f"{runtimes[1]:.2f}\n{speedup:.2f}× faster"),
        padding=3,
    )
    runtime_axis.set_ylabel("Complete training time (seconds)")

    phase_fields = (
        ("median_traversal_seconds", "Traversal collection", "tab:blue"),
        ("median_advantage_training_seconds", "Advantage training", "tab:orange"),
        ("median_strategy_training_seconds", "Strategy training", "tab:green"),
        ("median_other_seconds", "Other", "tab:gray"),
    )
    bottoms = [0.0, 0.0]
    for field, label, colour in phase_fields:
        values = [float(by_implementation[name][field]) for name in implementations]
        phase_axis.bar(labels, values, bottom=bottoms, label=label, color=colour)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    phase_axis.set_ylabel("Median time (seconds)")
    phase_axis.legend(fontsize=8)

    throughput = [
        float(by_implementation[name]["collection_traversals_per_second"])
        for name in implementations
    ]
    throughput_errors = [
        float(by_implementation[name]["median_absolute_deviation_collection_traversals_per_second"])
        for name in implementations
    ]
    throughput_bars = throughput_axis.bar(
        labels,
        throughput,
        yerr=throughput_errors,
        color=colours,
        capsize=4,
    )
    throughput_axis.bar_label(
        throughput_bars,
        labels=tuple(f"{value:,.0f}" for value in throughput),
        padding=3,
    )
    throughput_axis.set_ylabel("Traversal collection / second")

    memory = [float(by_implementation[name]["median_peak_memory_mb"]) for name in implementations]
    memory_errors = [
        float(by_implementation[name]["median_absolute_deviation_memory_mb"])
        for name in implementations
    ]
    memory_bars = memory_axis.bar(labels, memory, yerr=memory_errors, color=colours, capsize=4)
    memory_axis.bar_label(
        memory_bars,
        labels=tuple(f"{value:.1f}" for value in memory),
        padding=3,
    )
    memory_axis.set_ylabel(f"Peak process-tree {memory_metrics.pop().upper()} (MB)")

    for axis in (runtime_axis, phase_axis, throughput_axis, memory_axis):
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Reference versus optimised Deep CFR on Leduc\n"
        f"{traversal_counts.pop():,} traversals; {optimizer_steps.pop():,} optimizer steps"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_figure(figure, output_path)


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

    validation_ceiling = 1e-3 if game == "kuhn" else 5e-3
    for axis in (iteration_axis, time_axis):
        axis.axhline(
            validation_ceiling,
            color="tab:green",
            linestyle=":",
            label="Validation ceiling" if axis is iteration_axis else None,
        )

    iteration_axis.set_xlabel("Iterations")
    iteration_axis.set_xscale("log")
    time_axis.set_xlabel("Solver training time (seconds)")
    time_axis.set_xscale("log")
    for axis in (iteration_axis, time_axis):
        axis.set_ylabel("Exact exploitability (chips)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    handles, labels = iteration_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=5)
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
    colours = {"reference": "tab:orange", "optimised": "tab:blue"}
    memory_metrics = {record["memory_metric"] for record in records}
    if len(memory_metrics) != 1:
        raise ValueError("benchmark summary uses inconsistent memory metrics")
    memory_metric = memory_metrics.pop().upper()
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
            color=colours[implementation],
            capsize=3,
        )
        runtime_labels = [f"{float(record['median_seconds']):.3g}" for record in selected]
        if implementation == "optimised":
            speedup_labels = [
                f"{float(by_key[(*key, 'reference')]['median_seconds']) / float(record['median_seconds']):.1f}×"
                for key, record in zip(ordered_keys, selected, strict=True)
            ]
            runtime_labels = [
                f"{value}\n{speedup} faster"
                for value, speedup in zip(runtime_labels, speedup_labels, strict=True)
            ]
        runtime_axis.bar_label(runtime_bars, labels=runtime_labels, padding=3, fontsize=8)
        throughput_bars = throughput_axis.bar(
            positions + offset,
            [float(record["traversals_per_second"]) for record in selected],
            width,
            yerr=[
                float(record["median_absolute_deviation_traversals_per_second"])
                for record in selected
            ],
            label=label,
            color=colours[implementation],
            capsize=3,
        )
        throughput_axis.bar_label(
            throughput_bars,
            labels=[f"{float(record['traversals_per_second']):,.0f}" for record in selected],
            padding=3,
            fontsize=8,
        )
        memory_bars = memory_axis.bar(
            positions + offset,
            [float(record["median_peak_memory_mb"]) for record in selected],
            width,
            yerr=[float(record["median_absolute_deviation_memory_mb"]) for record in selected],
            label=label,
            color=colours[implementation],
            capsize=3,
        )
        memory_axis.bar_label(
            memory_bars,
            labels=[f"{float(record['median_peak_memory_mb']):.1f}" for record in selected],
            padding=3,
            fontsize=8,
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
    memory_axis.set_ylabel(f"Median peak process-tree {memory_metric} (MB)")
    handles, legend_labels = runtime_axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncols=2)
    figure.suptitle(
        "Reference versus optimised fixed-workload performance\n"
        "Kuhn: 10,000 iterations / 20,000 traversals; "
        "Leduc: 5,000 iterations / 10,000 traversals"
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
