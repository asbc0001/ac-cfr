"""Reproducible exact-metric plots generated from compact CSV results."""

from pathlib import Path

from ac_cfr.persistence.results import CsvResultStore


def plot_exact_metric(
    result_paths: tuple[Path, ...],
    output_path: Path,
    *,
    metric: str,
    x_axis: str,
) -> None:
    """Plot one exact poker metric grouped by game and solver."""
    if metric not in {"exploitability", "nash_conv"}:
        raise ValueError("metric must be exploitability or nash_conv")
    if x_axis not in {"iteration", "elapsed_training_seconds"}:
        raise ValueError("x_axis must be iteration or elapsed_training_seconds")
    if not result_paths:
        raise ValueError("at least one results file is required")

    series: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for result_path in result_paths:
        for row in CsvResultStore(result_path).records:
            if not row[metric] or not row[x_axis]:
                continue
            key = (row["game"], row["solver"])
            series.setdefault(key, []).append((float(row[x_axis]), float(row[metric])))
    if not series:
        raise ValueError("results files contain no plottable exact metrics")

    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.subplots()
    for (game, solver), points in sorted(series.items()):
        ordered_points = sorted(points)
        axis.plot(
            [point[0] for point in ordered_points],
            [point[1] for point in ordered_points],
            marker="o",
            label=f"{game} {solver}",
        )
    axis.set_xlabel("Iteration" if x_axis == "iteration" else "Training time (seconds)")
    axis.set_ylabel("Exploitability (chips)" if metric == "exploitability" else "NashConv")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
