"""Multi-seed convergence validation for reference MCCFR on Leduc poker."""

import csv
import json
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Final

from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.evaluation.plotting import plot_mccfr_reference_convergence
from ac_cfr.games.base import GameId
from ac_cfr.games.tabular import create_tabular_game
from ac_cfr.persistence.files import atomic_text_writer
from ac_cfr.solvers import NaiveMCCFR

VALIDATION_ID = "mccfr_reference_convergence"
SEEDS: Final = (20260810, 20260811, 20260812, 20260813, 20260814)
MILESTONES: Final = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000)
REFERENCE_EXPLOITABILITY_LIMIT = 0.005
FINAL_MEDIAN_EXPLOITABILITY_LIMIT = 10 * REFERENCE_EXPLOITABILITY_LIMIT

_CONVERGENCE_FIELDS: Final = (
    "validation_id",
    "game",
    "solver",
    "seed",
    "iteration",
    "traversals",
    "elapsed_training_seconds",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
)
_SUMMARY_FIELDS: Final = (
    "validation_id",
    "game",
    "solver",
    "iteration",
    "traversals_per_seed",
    "seed_count",
    "median_elapsed_training_seconds",
    "median_expected_value_player_zero",
    "median_exploitability",
    "minimum_exploitability",
    "maximum_exploitability",
    "median_nash_conv",
)


def run_mccfr_reference_convergence_validation(
    output_directory: Path = Path("results") / VALIDATION_ID,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run the declared Leduc seed set and write exact convergence evidence."""
    tree = create_tabular_game(GameId.LEDUC).tree
    convergence_records: list[dict[str, object]] = []

    for seed_number, seed in enumerate(SEEDS, start=1):
        _report_progress(progress_callback, f"seed {seed_number}/{len(SEEDS)}: {seed}")
        solver = NaiveMCCFR(tree, seed=seed)
        elapsed_training_seconds = 0.0
        previous_iteration = 0
        for milestone in MILESTONES:
            start_time = perf_counter()
            solver.train(milestone - previous_iteration)
            elapsed_training_seconds += perf_counter() - start_time
            metrics = evaluate_strategy(tree, solver.average_policy())
            convergence_records.append(
                {
                    "validation_id": VALIDATION_ID,
                    "game": GameId.LEDUC.value,
                    "solver": "naive_mccfr",
                    "seed": seed,
                    "iteration": milestone,
                    "traversals": 2 * milestone,
                    "elapsed_training_seconds": elapsed_training_seconds,
                    "expected_value_player_zero": metrics.expected_values[0],
                    "exploitability": metrics.exploitability,
                    "nash_conv": metrics.nash_conv,
                }
            )
            previous_iteration = milestone

    summary_records = _summarise(convergence_records)
    checks = _convergence_checks(convergence_records, summary_records)
    output_directory.mkdir(parents=True, exist_ok=True)
    convergence_path = output_directory / "convergence.csv"
    summary_path = output_directory / "summary.csv"
    plot_path = output_directory / "convergence.png"
    _write_csv(convergence_path, _CONVERGENCE_FIELDS, convergence_records)
    _write_csv(summary_path, _SUMMARY_FIELDS, summary_records)
    plot_mccfr_reference_convergence(
        convergence_path,
        summary_path,
        plot_path,
        reference_exploitability_limit=REFERENCE_EXPLOITABILITY_LIMIT,
    )

    passed = all(bool(check["passed"]) for check in checks)
    validation_path = output_directory / "validation.json"
    _write_json(
        validation_path,
        {
            "about": (
                "Machine-readable configuration, checks, and file index for the reference "
                "MCCFR multi-seed Leduc convergence validation."
            ),
            "validation_id": VALIDATION_ID,
            "passed": passed,
            "configuration": {
                "game": GameId.LEDUC.value,
                "solver": "naive_mccfr",
                "seeds": list(SEEDS),
                "milestones": list(MILESTONES),
                "traversals_per_iteration": 2,
                "timed_region": "solver.train only",
                "early_stopping": False,
                "reference_exploitability_limit": REFERENCE_EXPLOITABILITY_LIMIT,
                "final_median_exploitability_limit": FINAL_MEDIAN_EXPLOITABILITY_LIMIT,
            },
            "checks": checks,
            "files": {
                "convergence": convergence_path.name,
                "summary": summary_path.name,
                "plot": plot_path.name,
            },
            "file_descriptions": {
                "convergence": "Exact measurements for every seed and milestone.",
                "summary": "Median and full seed range at each milestone.",
                "plot": "Per-seed and median exploitability by iterations and training time.",
            },
        },
    )
    if not passed:
        raise RuntimeError(f"MCCFR convergence validation failed; see {validation_path}")
    return validation_path


def _summarise(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate stochastic measurements at each declared milestone."""
    summary: list[dict[str, object]] = []
    for milestone in MILESTONES:
        milestone_records = [record for record in records if record["iteration"] == milestone]
        exploitabilities = [
            _numeric_field(record, "exploitability") for record in milestone_records
        ]
        summary.append(
            {
                "validation_id": VALIDATION_ID,
                "game": GameId.LEDUC.value,
                "solver": "naive_mccfr",
                "iteration": milestone,
                "traversals_per_seed": 2 * milestone,
                "seed_count": len(milestone_records),
                "median_elapsed_training_seconds": median(
                    _numeric_field(record, "elapsed_training_seconds")
                    for record in milestone_records
                ),
                "median_expected_value_player_zero": median(
                    _numeric_field(record, "expected_value_player_zero")
                    for record in milestone_records
                ),
                "median_exploitability": median(exploitabilities),
                "minimum_exploitability": min(exploitabilities),
                "maximum_exploitability": max(exploitabilities),
                "median_nash_conv": median(
                    _numeric_field(record, "nash_conv") for record in milestone_records
                ),
            }
        )
    return summary


def _convergence_checks(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Check consistent improvement and a coarse final quality boundary."""
    first_median = _numeric_field(summary[0], "median_exploitability")
    final_median = _numeric_field(summary[-1], "median_exploitability")
    medians = [_numeric_field(record, "median_exploitability") for record in summary]
    per_seed = {
        seed: (
            _exploitability_at(records, seed, MILESTONES[0]),
            _exploitability_at(records, seed, MILESTONES[-1]),
        )
        for seed in SEEDS
    }
    return [
        {
            "name": "every_seed_improves_over_the_declared_workload",
            "passed": all(final < first for first, final in per_seed.values()),
            "first_iteration": MILESTONES[0],
            "final_iteration": MILESTONES[-1],
            "seed_exploitabilities": {
                str(seed): {"first": first, "final": final}
                for seed, (first, final) in per_seed.items()
            },
        },
        {
            "name": "median_exploitability_decreases_at_every_milestone",
            "passed": all(
                later < earlier for earlier, later in zip(medians, medians[1:], strict=False)
            ),
            "median_exploitabilities": medians,
        },
        {
            "name": "final_median_is_within_ten_times_the_cfr_reference_ceiling",
            "passed": final_median <= FINAL_MEDIAN_EXPLOITABILITY_LIMIT,
            "limit": FINAL_MEDIAN_EXPLOITABILITY_LIMIT,
            "final_median_exploitability": final_median,
            "initial_median_exploitability": first_median,
        },
    ]


def _exploitability_at(records: list[dict[str, object]], seed: int, iteration: int) -> float:
    """Return the unique exploitability for one seed and milestone."""
    matches = [
        _numeric_field(record, "exploitability")
        for record in records
        if record["seed"] == seed and record["iteration"] == iteration
    ]
    if len(matches) != 1:
        raise RuntimeError("MCCFR convergence records are incomplete or duplicated")
    return matches[0]


def _numeric_field(record: dict[str, object], field: str) -> float:
    """Read one numeric field from an internal validation record."""
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"MCCFR validation field is not numeric: {field}")
    return float(value)


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    records: list[dict[str, object]],
) -> None:
    """Atomically write records using a fixed column order."""
    with atomic_text_writer(path) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, values: dict[str, object]) -> None:
    """Atomically write deterministic, readable JSON."""
    with atomic_text_writer(path) as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    """Send a progress message when the caller supplied one."""
    if callback is not None:
        callback(message)
