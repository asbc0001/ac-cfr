"""Reference-versus-optimised MCCFR validation on Leduc poker."""

import csv
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Final

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ac_cfr.benchmarking.harness import report_progress
from ac_cfr.common.provenance import code_revision
from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.evaluation.metrics import StrategyMetrics, evaluate_strategy
from ac_cfr.evaluation.plotting import plot_mccfr_validation
from ac_cfr.evaluation.self_play import evaluate_duplicate_self_play
from ac_cfr.games.base import GameId
from ac_cfr.games.tabular import create_tabular_game
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.persistence.checkpoints import save_tabular_checkpoint
from ac_cfr.persistence.files import write_csv, write_json
from ac_cfr.persistence.snapshots import export_tabular_snapshot, load_tabular_snapshot
from ac_cfr.solvers import MCCFR, NaiveMCCFR
from ac_cfr.training.runner import TabularTrainingConfig

VALIDATION_ID = "mccfr"
SEEDS: Final = (20260810, 20260811, 20260812, 20260813, 20260814)
SHARED_MILESTONES: Final = (
    1_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
)
OPTIMISED_MILESTONES: Final = (
    *SHARED_MILESTONES,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
)
FINAL_EXPLOITABILITY_LIMIT = 0.005
REFERENCE_FINAL_MEDIAN_EXPLOITABILITY_LIMIT = 10 * FINAL_EXPLOITABILITY_LIMIT
SELF_PLAY_DUPLICATE_PAIRS = 20_000
SELF_PLAY_SEED = 20260815
EQUIVALENCE_ITERATIONS = 10
EQUIVALENCE_ABSOLUTE_TOLERANCE = 1e-12

_TABULAR_REFERENCE_PATH = Path("results/tabular_policies/evaluations.csv")
_SNAPSHOT_PATH = Path("artifacts/tabular/leduc-mccfr-final.npz")
_CHECKPOINT_PATH = Path("runs/leduc-mccfr-final/checkpoints/latest.npz")
_CONVERGENCE_FIELDS: Final = (
    "validation_id",
    "game",
    "implementation",
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
    "implementation",
    "solver",
    "iteration",
    "seed_count",
    "median_elapsed_training_seconds",
    "median_expected_value_player_zero",
    "median_exploitability",
    "minimum_exploitability",
    "maximum_exploitability",
    "median_nash_conv",
)


def run_mccfr_validation(
    output_directory: Path = Path("results") / VALIDATION_ID,
    *,
    tabular_reference_path: Path = _TABULAR_REFERENCE_PATH,
    snapshot_path: Path = _SNAPSHOT_PATH,
    checkpoint_path: Path = _CHECKPOINT_PATH,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Validate optimised MCCFR against its reference and exact Leduc evaluation."""
    tabular_references = _load_tabular_references(tabular_reference_path)
    tabular_game = create_tabular_game(GameId.LEDUC)
    tree = tabular_game.tree
    reference_records = _load_or_run_reference_records(
        output_directory / "convergence.csv",
        tree,
        progress_callback,
    )
    optimised_records: list[dict[str, object]] = []
    final_solvers: dict[int, MCCFR] = {}
    final_metrics: dict[int, StrategyMetrics] = {}

    report_progress(progress_callback, "semantic equivalence: identical sampled draws")
    equivalence = _same_draw_equivalence(tree)
    for seed_number, seed in enumerate(SEEDS, start=1):
        report_progress(progress_callback, f"optimised seed {seed_number}/{len(SEEDS)}: {seed}")
        solver = MCCFR(tree, seed=seed)
        solver.train(1)  # Compile before measuring solver-training time.
        solver = MCCFR(tree, seed=seed)
        elapsed_training_seconds = 0.0
        previous_iteration = 0
        for milestone in OPTIMISED_MILESTONES:
            start_time = perf_counter()
            solver.train(milestone - previous_iteration)
            elapsed_training_seconds += perf_counter() - start_time
            metrics = evaluate_strategy(tree, solver.average_policy())
            optimised_records.append(
                _strategy_record(
                    implementation="optimised",
                    solver="mccfr",
                    seed=seed,
                    iteration=milestone,
                    elapsed_training_seconds=elapsed_training_seconds,
                    expected_value_player_zero=metrics.expected_values[0],
                    exploitability=metrics.exploitability,
                    nash_conv=metrics.nash_conv,
                )
            )
            previous_iteration = milestone
            report_progress(progress_callback, f"seed {seed}: {milestone:,} iterations")
        final_solvers[seed] = solver
        final_metrics[seed] = evaluate_strategy(tree, solver.average_policy())

    records = [*reference_records, *optimised_records]
    summary = _summarise(records)
    selected_seed = _median_seed(final_metrics)
    selected_solver = final_solvers[selected_seed]
    selected_metrics = final_metrics[selected_seed]
    snapshot_id = f"leduc-mccfr-final-seed-{selected_seed}-iter-{OPTIMISED_MILESTONES[-1]}"
    checkpoint_id = f"leduc-mccfr-final_iter_{OPTIMISED_MILESTONES[-1]}"
    revision = code_revision()
    training_config = TabularTrainingConfig(
        game=GameId.LEDUC.value,
        solver="mccfr",
        iterations=OPTIMISED_MILESTONES[-1],
        seed=selected_seed,
        run_id="leduc-mccfr-final",
        evaluation_interval=OPTIMISED_MILESTONES[-1],
        checkpoint_interval=OPTIMISED_MILESTONES[-1],
        snapshot_iterations=(OPTIMISED_MILESTONES[-1],),
    )
    save_tabular_checkpoint(
        checkpoint_path,
        solver=selected_solver,
        tabular_game=tabular_game,
        solver_id="mccfr",
        run_id=training_config.run_id,
        seed=selected_seed,
        training_config=training_config.to_dict(),
        elapsed_training_seconds=_final_elapsed_seconds(optimised_records, selected_seed),
        checkpoint_id=checkpoint_id,
        schedule_state={
            "best_exploitability": selected_metrics.exploitability,
            "evaluations_without_improvement": 0,
        },
        metric_records=(),
        code_revision=revision,
    )
    export_tabular_snapshot(
        snapshot_path,
        tabular_game=tabular_game,
        average_policy=selected_solver.average_policy(),
        snapshot_id=snapshot_id,
        solver="mccfr",
        iteration=OPTIMISED_MILESTONES[-1],
        run_id="leduc-mccfr-final",
        seed=selected_seed,
        source_checkpoint_id=checkpoint_id,
    )
    exported_policy = load_tabular_snapshot(snapshot_path, tabular_game).average_policy
    self_play = evaluate_duplicate_self_play(
        tree,
        exported_policy,
        duplicate_pairs=SELF_PLAY_DUPLICATE_PAIRS,
        seed=SELF_PLAY_SEED,
    )
    checks = _validation_checks(records, summary, equivalence, self_play.includes_zero)

    output_directory.mkdir(parents=True, exist_ok=True)
    convergence_path = output_directory / "convergence.csv"
    summary_path = output_directory / "summary.csv"
    plot_path = output_directory / "plots" / "convergence.png"
    write_csv(convergence_path, _CONVERGENCE_FIELDS, records)
    write_csv(summary_path, _SUMMARY_FIELDS, summary)
    plot_mccfr_validation(
        convergence_path,
        summary_path,
        plot_path,
        reference_exploitability_limit=FINAL_EXPLOITABILITY_LIMIT,
    )

    passed = all(bool(check["passed"]) for check in checks)
    validation_path = output_directory / "validation.json"
    write_json(
        validation_path,
        {
            "about": (
                "Machine-readable configuration, checks, and file index for the Leduc "
                "reference-versus-optimised MCCFR validation."
            ),
            "validation_id": VALIDATION_ID,
            "passed": passed,
            "code_revision": revision,
            "configuration": {
                "game": GameId.LEDUC.value,
                "seeds": list(SEEDS),
                "shared_milestones": list(SHARED_MILESTONES),
                "optimised_milestones": list(OPTIMISED_MILESTONES),
                "traversals_per_iteration": 2,
                "timed_region": "solver.train only",
                "early_stopping": False,
                "final_exploitability_limit": FINAL_EXPLOITABILITY_LIMIT,
                "reference_final_median_exploitability_limit": (
                    REFERENCE_FINAL_MEDIAN_EXPLOITABILITY_LIMIT
                ),
                "equivalence_iterations": EQUIVALENCE_ITERATIONS,
                "equivalence_absolute_tolerance": EQUIVALENCE_ABSOLUTE_TOLERANCE,
                "self_play_duplicate_pairs": SELF_PLAY_DUPLICATE_PAIRS,
                "self_play_hands": 2 * SELF_PLAY_DUPLICATE_PAIRS,
                "self_play_seed": SELF_PLAY_SEED,
                "self_play_confidence_level": 0.99,
                "self_play_confidence_interval_method": "normal",
            },
            "tabular_references": tabular_references,
            "selected_snapshot": {
                "snapshot_id": snapshot_id,
                "path": str(snapshot_path),
                "source_checkpoint": str(checkpoint_path),
                "selection": "seed with the median final exploitability",
                "seed": selected_seed,
                "iteration": OPTIMISED_MILESTONES[-1],
                "expected_value_player_zero": selected_metrics.expected_values[0],
                "exploitability": selected_metrics.exploitability,
                "nash_conv": selected_metrics.nash_conv,
            },
            "self_play": {
                "mean_chips": self_play.mean_chips,
                "standard_error_chips": self_play.standard_error_chips,
                "confidence_interval_low": self_play.confidence_interval_low,
                "confidence_interval_high": self_play.confidence_interval_high,
            },
            "checks": checks,
            "files": {
                "convergence": convergence_path.name,
                "summary": summary_path.name,
                "plot": str(plot_path.relative_to(output_directory)),
            },
            "file_descriptions": {
                "convergence": "Exact measurements for every implementation, seed, and milestone.",
                "summary": "Median and full seed range for each implementation and milestone.",
                "plot": "Multi-seed exploitability by MCCFR iterations and training time.",
            },
        },
    )
    if not passed:
        raise RuntimeError(f"MCCFR validation failed; see {validation_path}")
    return validation_path


def _load_or_run_reference_records(
    convergence_path: Path,
    tree: IndexedGameTree,
    progress_callback: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    """Reuse final evidence when available, otherwise run the reference workload."""
    if not convergence_path.is_file():
        return _run_reference_records(tree, progress_callback)
    records: list[dict[str, object]] = []
    try:
        with convergence_path.open(encoding="utf-8", newline="") as input_file:
            for record in csv.DictReader(input_file):
                if record.get("implementation") != "reference":
                    continue
                if (
                    record.get("game") != GameId.LEDUC.value
                    or record.get("solver") != "naive_mccfr"
                ):
                    raise ValueError("reference MCCFR record has incompatible identifiers")
                records.append(
                    _strategy_record(
                        implementation="reference",
                        solver="naive_mccfr",
                        seed=int(record["seed"]),
                        iteration=int(record["iteration"]),
                        elapsed_training_seconds=float(record["elapsed_training_seconds"]),
                        expected_value_player_zero=float(record["expected_value_player_zero"]),
                        exploitability=float(record["exploitability"]),
                        nash_conv=float(record["nash_conv"]),
                    )
                )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("reference MCCFR convergence records are unreadable") from error
    expected_keys = {(seed, milestone) for seed in SEEDS for milestone in SHARED_MILESTONES}
    actual_keys = {
        (int(_number(record, "seed")), int(_number(record, "iteration"))) for record in records
    }
    if len(records) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("reference MCCFR convergence records are incomplete")
    return records


def _run_reference_records(
    tree: IndexedGameTree,
    progress_callback: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    """Generate reference convergence rows when no committed rows are available."""
    records: list[dict[str, object]] = []
    for seed_number, seed in enumerate(SEEDS, start=1):
        report_progress(
            progress_callback,
            f"reference seed {seed_number}/{len(SEEDS)}: {seed}",
        )
        solver = NaiveMCCFR(tree, seed=seed)
        elapsed_training_seconds = 0.0
        previous_iteration = 0
        for milestone in SHARED_MILESTONES:
            start_time = perf_counter()
            solver.train(milestone - previous_iteration)
            elapsed_training_seconds += perf_counter() - start_time
            metrics = evaluate_strategy(tree, solver.average_policy())
            records.append(
                _strategy_record(
                    implementation="reference",
                    solver="naive_mccfr",
                    seed=seed,
                    iteration=milestone,
                    elapsed_training_seconds=elapsed_training_seconds,
                    expected_value_player_zero=metrics.expected_values[0],
                    exploitability=metrics.exploitability,
                    nash_conv=metrics.nash_conv,
                )
            )
            previous_iteration = milestone
    return records


def _load_tabular_references(path: Path) -> dict[str, dict[str, object]]:
    """Load the selected exact Leduc CFR and CFR+ evaluation records."""
    references: dict[str, dict[str, object]] = {}
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            for record in csv.DictReader(input_file):
                solver = record["solver"]
                if record["game"] == GameId.LEDUC.value and solver in ("cfr", "cfr_plus"):
                    if solver in references:
                        raise ValueError("tabular reference contains duplicate solver results")
                    references[solver] = {
                        "source": str(path),
                        "iteration": int(record["iteration"]),
                        "expected_value_player_zero": float(record["expected_value_player_zero"]),
                        "exploitability": float(record["exploitability"]),
                        "nash_conv": float(record["nash_conv"]),
                    }
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("tabular Leduc reference evaluations are unreadable") from error
    if set(references) != {"cfr", "cfr_plus"}:
        raise ValueError("tabular Leduc reference evaluations are incomplete")
    return references


def _strategy_record(
    *,
    implementation: str,
    solver: str,
    seed: int,
    iteration: int,
    elapsed_training_seconds: float,
    expected_value_player_zero: float,
    exploitability: float,
    nash_conv: float,
) -> dict[str, object]:
    """Build one exact strategy-quality result row."""
    return {
        "validation_id": VALIDATION_ID,
        "game": GameId.LEDUC.value,
        "implementation": implementation,
        "solver": solver,
        "seed": seed,
        "iteration": iteration,
        "traversals": 2 * iteration,
        "elapsed_training_seconds": elapsed_training_seconds,
        "expected_value_player_zero": expected_value_player_zero,
        "exploitability": exploitability,
        "nash_conv": nash_conv,
    }


def _summarise(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate the seed distribution for each implementation and milestone."""
    summary: list[dict[str, object]] = []
    for implementation, solver, milestones in (
        ("reference", "naive_mccfr", SHARED_MILESTONES),
        ("optimised", "mccfr", OPTIMISED_MILESTONES),
    ):
        for milestone in milestones:
            selected = [
                record
                for record in records
                if record["implementation"] == implementation and record["iteration"] == milestone
            ]
            exploitabilities = [_number(record, "exploitability") for record in selected]
            summary.append(
                {
                    "validation_id": VALIDATION_ID,
                    "game": GameId.LEDUC.value,
                    "implementation": implementation,
                    "solver": solver,
                    "iteration": milestone,
                    "seed_count": len(selected),
                    "median_elapsed_training_seconds": median(
                        _number(record, "elapsed_training_seconds") for record in selected
                    ),
                    "median_expected_value_player_zero": median(
                        _number(record, "expected_value_player_zero") for record in selected
                    ),
                    "median_exploitability": median(exploitabilities),
                    "minimum_exploitability": min(exploitabilities),
                    "maximum_exploitability": max(exploitabilities),
                    "median_nash_conv": median(_number(record, "nash_conv") for record in selected),
                }
            )
    return summary


def _same_draw_equivalence(tree: IndexedGameTree) -> dict[str, object]:
    """Compare both solvers after feeding them identical random draws."""
    seed = SEEDS[0]
    seed_deriver = SeedDeriver(seed)
    reference = NaiveMCCFR(tree, seed=seed)
    reference._chance_rng = default_rng(  # pyright: ignore[reportAttributeAccessIssue]
        seed_deriver.derive(RngStream.CHANCE)
    )
    reference._policy_rng = default_rng(  # pyright: ignore[reportAttributeAccessIssue]
        seed_deriver.derive(RngStream.POLICY)
    )
    optimised = MCCFR(tree, seed=seed)
    reference.train(EQUIVALENCE_ITERATIONS)
    optimised.train(EQUIVALENCE_ITERATIONS)
    differences = {
        "maximum_regret_difference": _maximum_difference(
            _flatten(reference.regret_sum), _flatten(optimised.regret_sum)
        ),
        "maximum_strategy_sum_difference": _maximum_difference(
            _flatten(reference.strategy_sum), _flatten(optimised.strategy_sum)
        ),
        "maximum_current_policy_difference": _maximum_difference(
            reference.current_policy(), optimised.current_policy()
        ),
        "maximum_average_policy_difference": _maximum_difference(
            reference.average_policy(), optimised.average_policy()
        ),
    }
    return {
        "passed": max(differences.values()) <= EQUIVALENCE_ABSOLUTE_TOLERANCE,
        **differences,
    }


def _validation_checks(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
    equivalence: dict[str, object],
    self_play_includes_zero: bool,
) -> list[dict[str, object]]:
    """Apply predeclared semantic, stochastic-quality, and self-play checks."""
    reference_final = _summary_record(summary, "reference", SHARED_MILESTONES[-1])
    optimised_shared = _summary_record(summary, "optimised", SHARED_MILESTONES[-1])
    optimised_final = _summary_record(summary, "optimised", OPTIMISED_MILESTONES[-1])
    reference_minimum = _number(reference_final, "minimum_exploitability")
    reference_maximum = _number(reference_final, "maximum_exploitability")
    shared_median = _number(optimised_shared, "median_exploitability")
    final_median = _number(optimised_final, "median_exploitability")
    return [
        *_reference_convergence_checks(records, summary),
        {
            "name": "identical_draw_updates_match",
            **equivalence,
            "absolute_tolerance": EQUIVALENCE_ABSOLUTE_TOLERANCE,
        },
        {
            "name": "shared_workload_median_is_within_reference_seed_range",
            "passed": reference_minimum <= shared_median <= reference_maximum,
            "iteration": SHARED_MILESTONES[-1],
            "reference_seed_range": [reference_minimum, reference_maximum],
            "optimised_median_exploitability": shared_median,
        },
        {
            "name": "final_median_reaches_cfr_cfr_plus_validation_ceiling",
            "passed": final_median <= FINAL_EXPLOITABILITY_LIMIT,
            "iteration": OPTIMISED_MILESTONES[-1],
            "exploitability_limit": FINAL_EXPLOITABILITY_LIMIT,
            "median_exploitability": final_median,
        },
        {
            "name": "exported_snapshot_self_play_is_neutral",
            "passed": self_play_includes_zero,
            "confidence_level": 0.99,
        },
    ]


def _reference_convergence_checks(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Check sustained learning by the reference implementation and every seed."""
    seed_exploitabilities: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        seed_records = [
            record
            for record in records
            if record["implementation"] == "reference" and record["seed"] == seed
        ]
        first = _record_number(seed_records, SHARED_MILESTONES[0], "exploitability")
        final = _record_number(seed_records, SHARED_MILESTONES[-1], "exploitability")
        seed_exploitabilities[str(seed)] = {"first": first, "final": final}

    median_exploitabilities = [
        _number(_summary_record(summary, "reference", milestone), "median_exploitability")
        for milestone in SHARED_MILESTONES
    ]
    return [
        {
            "name": "reference_every_seed_improves_over_the_declared_workload",
            "passed": all(
                values["final"] < values["first"] for values in seed_exploitabilities.values()
            ),
            "first_iteration": SHARED_MILESTONES[0],
            "final_iteration": SHARED_MILESTONES[-1],
            "seed_exploitabilities": seed_exploitabilities,
        },
        {
            "name": "reference_median_exploitability_decreases_at_every_milestone",
            "passed": all(later < earlier for earlier, later in pairwise(median_exploitabilities)),
            "median_exploitabilities": median_exploitabilities,
        },
        {
            "name": "reference_final_median_reaches_declared_correctness_scale",
            "passed": (median_exploitabilities[-1] <= REFERENCE_FINAL_MEDIAN_EXPLOITABILITY_LIMIT),
            "limit": REFERENCE_FINAL_MEDIAN_EXPLOITABILITY_LIMIT,
            "final_median_exploitability": median_exploitabilities[-1],
        },
    ]


def _median_seed(metrics: dict[int, StrategyMetrics]) -> int:
    """Select the middle final seed by exploitability without favouring the best run."""
    return sorted(metrics, key=lambda seed: (metrics[seed].exploitability, seed))[len(metrics) // 2]


def _final_elapsed_seconds(records: list[dict[str, object]], seed: int) -> float:
    """Return the selected seed's final measured solver-training time."""
    matches = [
        _number(record, "elapsed_training_seconds")
        for record in records
        if record["seed"] == seed and record["iteration"] == OPTIMISED_MILESTONES[-1]
    ]
    if len(matches) != 1:
        raise RuntimeError("selected MCCFR final timing is missing or duplicated")
    return matches[0]


def _summary_record(
    summary: list[dict[str, object]], implementation: str, iteration: int
) -> dict[str, object]:
    """Return one unique aggregate record."""
    matches = [
        record
        for record in summary
        if record["implementation"] == implementation and record["iteration"] == iteration
    ]
    if len(matches) != 1:
        raise RuntimeError("MCCFR summary records are incomplete or duplicated")
    return matches[0]


def _record_number(
    records: list[dict[str, object]],
    iteration: int,
    field: str,
) -> float:
    """Return one numeric field from a unique iteration record."""
    matches = [_number(record, field) for record in records if record["iteration"] == iteration]
    if len(matches) != 1:
        raise RuntimeError("MCCFR convergence records are incomplete or duplicated")
    return matches[0]


def _number(record: dict[str, object], field: str) -> float:
    """Return one numeric internal result field as a float."""
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"MCCFR result field is not numeric: {field}")
    return float(value)


def _flatten(table: tuple[tuple[float, ...], ...]) -> NDArray[np.float64]:
    """Flatten a solver table in stable information-set action order."""
    return np.fromiter((value for row in table for value in row), dtype=np.float64)


def _maximum_difference(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Return the largest absolute element difference between two arrays."""
    return float(np.max(np.abs(first - second)))
