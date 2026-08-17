"""Plots for modified-HULHE calibration and policy progression."""

import csv
from pathlib import Path
from typing import Any


def plot_modified_hulhe_generalisation(result_path: Path, output_path: Path) -> None:
    """Plot training and held-out advantage loss for each seed and player."""
    records = _read_records(
        result_path,
        {"seed", "player", "update_steps", "training_loss", "validation_loss"},
    )
    series: dict[tuple[int, int], list[dict[str, str]]] = {}
    for record in records:
        key = (int(record["seed"]), int(record["player"]))
        series.setdefault(key, []).append(record)
    if not series:
        raise ValueError("generalisation results contain no records")

    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4.8))
    training_axis, validation_axis = figure.subplots(1, 2)
    for (seed, player), values in sorted(series.items()):
        ordered = sorted(values, key=lambda record: int(record["update_steps"]))
        updates = [int(record["update_steps"]) for record in ordered]
        label = f"Seed {seed}, Player {player}"
        training_axis.plot(
            updates,
            [float(record["training_loss"]) for record in ordered],
            marker="o",
            label=label,
        )
        validation_axis.plot(
            updates,
            [float(record["validation_loss"]) for record in ordered],
            marker="o",
            label=label,
        )
    training_axis.set_yscale("log")
    training_axis.set_ylabel("Advantage training loss")
    validation_axis.set_ylabel("Advantage held-out loss")
    for axis in (training_axis, validation_axis):
        axis.set_xlabel("Continuous SGD updates")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Modified-HULHE advantage-network generalisation")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_figure(figure, output_path)


def plot_modified_hulhe_network_fit(result_path: Path, output_path: Path) -> None:
    """Plot the continuous advantage-network calibration fit."""
    records = _read_records(
        result_path,
        {"update_steps", "training_loss", "validation_loss"},
    )
    ordered = sorted(records, key=lambda record: int(record["update_steps"]))
    if not ordered:
        raise ValueError("network-fit results contain no records")

    updates = [int(record["update_steps"]) for record in ordered]
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 6.4))
    validation_axis, training_axis = figure.subplots(2, 1, sharex=True)
    validation_axis.plot(
        updates,
        [float(record["validation_loss"]) for record in ordered],
        marker="o",
        color="tab:orange",
    )
    training_axis.plot(
        updates,
        [float(record["training_loss"]) for record in ordered],
        marker="o",
    )
    validation_axis.set_ylabel("Held-out loss")
    training_axis.set_ylabel("Training loss (log scale)")
    training_axis.set_yscale("log")
    training_axis.set_xlabel("SGD updates")
    for axis in (validation_axis, training_axis):
        axis.grid(alpha=0.25)
    figure.suptitle("Modified-HULHE advantage-network fit")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(figure, output_path)


def plot_modified_hulhe_batch_throughput(result_path: Path, output_path: Path) -> None:
    """Plot warmed GPU update throughput by batch size."""
    records = _read_records(result_path, {"batch_size", "updates_per_second"})
    ordered = sorted(records, key=lambda record: int(record["batch_size"]))
    if not ordered:
        raise ValueError("batch-throughput results contain no records")

    labels = [f"{int(record['batch_size']):,}" for record in ordered]
    throughput = [float(record["updates_per_second"]) for record in ordered]
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 4.4))
    axis = figure.subplots()
    bars = axis.bar(labels, throughput, color=("tab:blue", "tab:orange"))
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in throughput], padding=3)
    axis.set_ylim(0.0, max(throughput) * 1.15)
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Updates per second")
    axis.set_title("Modified-HULHE GPU batch throughput")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_modified_hulhe_worker_scaling(result_path: Path, output_path: Path) -> None:
    """Plot traversal throughput and complete-iteration time by worker count."""
    records = _read_records(
        result_path,
        {"workers", "collection_traversals_per_second", "iteration_seconds"},
    )
    ordered = sorted(records, key=lambda record: int(record["workers"]))
    if not ordered:
        raise ValueError("worker-scaling results contain no records")

    workers = [int(record["workers"]) for record in ordered]
    throughput = [float(record["collection_traversals_per_second"]) for record in ordered]
    iteration_seconds = [float(record["iteration_seconds"]) for record in ordered]
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 4.6))
    throughput_axis = figure.subplots()
    timing_axis = throughput_axis.twinx()
    bars = throughput_axis.bar(
        workers,
        throughput,
        width=1.6,
        color="tab:blue",
        alpha=0.9,
        label="Collection throughput",
    )
    timing_axis.plot(
        workers,
        iteration_seconds,
        marker="o",
        color="tab:orange",
        linewidth=2.0,
        label="Complete iteration",
    )
    throughput_axis.bar_label(
        bars,
        labels=[f"{value:.1f}" for value in throughput],
        padding=3,
    )
    throughput_axis.set_xticks(workers)
    throughput_axis.set_xlabel("Traversal workers")
    throughput_axis.set_ylabel("Collection traversals per second")
    timing_axis.set_ylabel("Complete iteration seconds")
    throughput_axis.grid(axis="y", alpha=0.25)
    handles = (*throughput_axis.containers, *timing_axis.lines)
    throughput_axis.legend(handles, ["Collection throughput", "Complete iteration"])
    figure.suptitle("Modified-HULHE traversal-worker scaling")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(figure, output_path)


def plot_modified_hulhe_h2h(result_path: Path, output_path: Path) -> None:
    """Plot snapshot progression with paired-bootstrap intervals by opponent."""
    records = _read_records(
        result_path,
        {
            "iteration",
            "seed",
            "opponent_id",
            "mbb_per_game",
            "confidence_interval_low",
            "confidence_interval_high",
        },
    )
    series: dict[tuple[str, int], list[dict[str, str]]] = {}
    for record in records:
        series.setdefault((record["opponent_id"], int(record["seed"])), []).append(record)
    if not series:
        raise ValueError("modified-HULHE H2H results contain no records")

    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 5.5))
    axis = figure.subplots()
    for (opponent_id, seed), values in sorted(series.items()):
        ordered = sorted(
            values,
            key=lambda record: (int(record["iteration"]), int(record["seed"])),
        )
        iterations = [int(record["iteration"]) for record in ordered]
        estimates = [float(record["mbb_per_game"]) for record in ordered]
        lows = [float(record["confidence_interval_low"]) for record in ordered]
        highs = [float(record["confidence_interval_high"]) for record in ordered]
        axis.plot(iterations, estimates, marker="o", label=f"vs {opponent_id}, seed {seed}")
        axis.vlines(iterations, lows, highs, alpha=0.7)
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Focal snapshot training iteration")
    axis.set_ylabel("Focal policy result (mbb/g)")
    axis.set_title("Modified-HULHE duplicate-deal progression")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_modified_hulhe_final_timing(result_path: Path, output_path: Path) -> None:
    """Plot the recorded training phases and remaining iteration overhead."""
    records = _read_records(
        result_path,
        {
            "iteration",
            "iteration_seconds",
            "traversal_seconds",
            "advantage_training_seconds",
            "strategy_training_seconds",
        },
    )
    ordered = sorted(records, key=lambda record: int(record["iteration"]))
    if not ordered:
        raise ValueError("modified-HULHE iteration results contain no records")

    iterations = [int(record["iteration"]) for record in ordered]
    totals = [float(record["iteration_seconds"]) / 60.0 for record in ordered]
    traversal = [float(record["traversal_seconds"]) / 60.0 for record in ordered]
    advantage = [float(record["advantage_training_seconds"]) / 60.0 for record in ordered]
    strategy = [float(record["strategy_training_seconds"] or 0.0) / 60.0 for record in ordered]
    overhead = [
        max(0.0, total - traversal_time - advantage_time - strategy_time)
        for total, traversal_time, advantage_time, strategy_time in zip(
            totals,
            traversal,
            advantage,
            strategy,
            strict=True,
        )
    ]

    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.25, 6.2))
    axis = figure.subplots()
    axis.stackplot(
        iterations,
        traversal,
        advantage,
        strategy,
        overhead,
        labels=(
            "Traversal",
            "Advantage training",
            "Strategy training",
            "Checkpoint, snapshot and other overhead",
        ),
        alpha=0.8,
    )
    axis.plot(iterations, totals, color="black", linewidth=1.0, label="Total iteration")
    axis.set_xlabel("Outer iteration")
    axis.set_ylabel("Minutes")
    axis.set_title("Per-iteration training and publication time")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_modified_hulhe_advantage_losses(result_path: Path, output_path: Path) -> None:
    """Plot per-player training and grouped-validation advantage losses."""
    fields = {
        "iteration",
        "player_zero_advantage_training_loss",
        "player_zero_advantage_validation_loss",
        "player_one_advantage_training_loss",
        "player_one_advantage_validation_loss",
    }
    records = _read_records(result_path, fields)
    ordered = sorted(records, key=lambda record: int(record["iteration"]))
    if not ordered:
        raise ValueError("modified-HULHE iteration results contain no records")

    iterations = [int(record["iteration"]) for record in ordered]
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.25, 9.0))
    axes = figure.subplots(2, 1, sharex=True)
    for player, axis, colour in ((0, axes[0], "#829dcc"), (1, axes[1], "#e5a07a")):
        prefix = "player_zero" if player == 0 else "player_one"
        axis.plot(
            iterations,
            [float(record[f"{prefix}_advantage_training_loss"]) for record in ordered],
            color=colour,
            label=f"P{player} training",
        )
        axis.plot(
            iterations,
            [float(record[f"{prefix}_advantage_validation_loss"]) for record in ordered],
            color="black",
            label=f"P{player} grouped validation",
        )
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_xlabel("Outer iteration")
    figure.suptitle("Advantage-network losses (targets evolve each iteration)")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(figure, output_path)


def plot_modified_hulhe_reservoir_growth(result_path: Path, output_path: Path) -> None:
    """Plot retained samples in each bounded training reservoir."""
    fields = {
        "iteration",
        "player_zero_advantage_samples_retained",
        "player_one_advantage_samples_retained",
        "strategy_samples_retained",
    }
    records = _read_records(result_path, fields)
    ordered = sorted(records, key=lambda record: int(record["iteration"]))
    if not ordered:
        raise ValueError("modified-HULHE iteration results contain no records")

    iterations = [int(record["iteration"]) for record in ordered]
    series = {
        "P0 advantage": [
            int(record["player_zero_advantage_samples_retained"]) for record in ordered
        ],
        "P1 advantage": [
            int(record["player_one_advantage_samples_retained"]) for record in ordered
        ],
        "Strategy": [int(record["strategy_samples_retained"]) for record in ordered],
    }
    capacity = max(max(values) for values in series.values())
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.25, 6.2))
    axis = figure.subplots()
    for label, values in series.items():
        axis.plot(iterations, [value / 1_000_000 for value in values], label=label)
    axis.axhline(
        capacity / 1_000_000,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Capacity",
    )
    axis.set_xlabel("Outer iteration")
    axis.set_ylabel("Retained samples (millions)")
    axis.set_title("Reservoir occupancy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_modified_hulhe_policy_progression(result_path: Path, output_path: Path) -> None:
    """Plot selected snapshots against the fixed rule-based and random agents."""
    fields = {
        "iteration",
        "rule_mbb_per_game",
        "rule_ci_low",
        "rule_ci_high",
        "random_mbb_per_game",
        "random_ci_low",
        "random_ci_high",
    }
    records = _read_records(result_path, fields)
    ordered = sorted(records, key=lambda record: int(record["iteration"]))
    if not ordered:
        raise ValueError("modified-HULHE milestone results contain no records")

    from matplotlib.figure import Figure

    figure = Figure(figsize=(10.125, 6.2))
    axis = figure.subplots()
    for prefix, label, colour in (
        ("rule", "Rule-based v1", "#c44e52"),
        ("random", "Uniform random", "#4c72b0"),
    ):
        selected = [record for record in ordered if record[f"{prefix}_mbb_per_game"]]
        iterations = [int(record["iteration"]) for record in selected]
        estimates = [float(record[f"{prefix}_mbb_per_game"]) for record in selected]
        lows = [float(record[f"{prefix}_ci_low"]) for record in selected]
        highs = [float(record[f"{prefix}_ci_high"]) for record in selected]
        axis.errorbar(
            iterations,
            estimates,
            yerr=(
                [estimate - low for estimate, low in zip(estimates, lows, strict=True)],
                [high - estimate for estimate, high in zip(estimates, highs, strict=True)],
            ),
            marker="o",
            capsize=3,
            color=colour,
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("Outer iteration")
    axis.set_ylabel("mbb/g (95% paired-bootstrap CI)")
    axis.set_title("Modified-HULHE policy progression")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_modified_hulhe_snapshot_progression(
    result_path: Path,
    output_path: Path,
) -> None:
    """Plot same-run head-to-head results against genuinely earlier snapshots."""
    records = _read_records(
        result_path,
        {
            "run_id",
            "strategy_snapshot_id",
            "iteration",
            "opponent_snapshot_id",
            "opponent_iteration",
            "mbb_per_game",
            "confidence_interval_low",
            "confidence_interval_high",
        },
    )
    series: dict[int, list[dict[str, str]]] = {}
    for record in records:
        focal_iteration = int(record["iteration"])
        opponent_iteration = int(record["opponent_iteration"])
        opponent_snapshot_id = record["opponent_snapshot_id"]
        same_run_prefix = f"{record['run_id']}_iter_"
        if (
            not opponent_snapshot_id.startswith(same_run_prefix)
            or opponent_snapshot_id == record["strategy_snapshot_id"]
            or opponent_iteration >= focal_iteration
        ):
            continue
        series.setdefault(opponent_iteration, []).append(record)
    if not series:
        raise ValueError("modified-HULHE policy results contain no earlier-snapshot matches")

    from matplotlib.figure import Figure

    figure = Figure(figsize=(10.125, 6.2))
    axis = figure.subplots()
    focal_iterations: set[int] = set()
    for opponent_iteration, values in sorted(series.items()):
        ordered = sorted(values, key=lambda record: int(record["iteration"]))
        iterations = [int(record["iteration"]) for record in ordered]
        estimates = [float(record["mbb_per_game"]) for record in ordered]
        lows = [float(record["confidence_interval_low"]) for record in ordered]
        highs = [float(record["confidence_interval_high"]) for record in ordered]
        focal_iterations.update(iterations)
        axis.errorbar(
            iterations,
            estimates,
            yerr=(
                [estimate - low for estimate, low in zip(estimates, lows, strict=True)],
                [high - estimate for estimate, high in zip(estimates, highs, strict=True)],
            ),
            marker="o",
            capsize=3,
            label=f"Against iteration {opponent_iteration}",
        )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xticks(sorted(focal_iterations))
    axis.set_xlabel("Focal iteration")
    axis.set_ylabel("Focal policy result (mbb/g, 95% CI)")
    axis.set_title("Head-to-head results against earlier snapshots")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    _save_figure(figure, output_path)


def _read_records(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    """Read CSV records after checking all required columns are present."""
    with path.open(encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(f"results file is missing fields: {sorted(missing_fields)}")
        return [dict(record) for record in reader]


def _save_figure(figure: Any, output_path: Path) -> None:
    """Create the destination directory and save a compact raster figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
