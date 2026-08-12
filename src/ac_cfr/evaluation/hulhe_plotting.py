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
