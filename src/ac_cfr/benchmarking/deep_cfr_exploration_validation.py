"""Resumable matched Leduc validation for exploratory Deep CFR sampling."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Final, TypedDict

from ac_cfr.benchmarking.harness import report_progress
from ac_cfr.evaluation.deep_cfr_sampling import validate_exploratory_sampling_estimators
from ac_cfr.persistence.files import write_csv, write_json
from ac_cfr.training.deep_cfr_config import load_deep_cfr_run_config
from ac_cfr.training.deep_cfr_runner import (
    DeepCFRRunConfig,
    _load_run_config,
    resume_deep_cfr_training,
    start_deep_cfr_training,
)

VALIDATION_ID = "deep_cfr_exploratory_sampling"
SEEDS: Final = (20260811, 20260812, 20260813)
MILESTONES: Final = (5, 10, 15, 20)
ESTIMATOR_SAMPLE_COUNT = 100_000
ADVANTAGE_RMSE_LIMIT = 0.15
STRATEGY_RMSE_LIMIT = 0.005
MINIMUM_EFFECTIVE_SAMPLE_FRACTION = 0.8

_RESULT_FIELDS: Final = (
    "sampling",
    "epsilon",
    "seed",
    "iteration",
    "exploitability",
    "nash_conv",
    "elapsed_training_seconds",
    "run_id",
)
_SUMMARY_FIELDS: Final = (
    "sampling",
    "epsilon",
    "iteration",
    "seed_count",
    "median_exploitability",
    "minimum_exploitability",
    "maximum_exploitability",
    "median_elapsed_training_seconds",
)


class _ExperimentRecord(TypedDict):
    sampling: str
    epsilon: float
    seed: int
    iteration: int
    exploitability: float
    nash_conv: float
    elapsed_training_seconds: float
    run_id: str


class _SummaryRecord(TypedDict):
    sampling: str
    epsilon: float
    iteration: int
    seed_count: int
    median_exploitability: float
    minimum_exploitability: float
    maximum_exploitability: float
    median_elapsed_training_seconds: float


def run_deep_cfr_exploration_validation(
    output_directory: Path = Path("results") / "deep_cfr" / "exploratory_sampling",
    *,
    runs_root: Path = Path("runs"),
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run estimator checks and a resumable paired three-seed Leduc comparison."""
    estimator = validate_exploratory_sampling_estimators(
        sample_count=ESTIMATOR_SAMPLE_COUNT,
        epsilon=0.1,
    )
    estimator_checks = _estimator_checks(estimator.to_dict())
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(
        output_directory / "estimator_validation.json",
        {
            "about": (
                "Full-tree versus importance-corrected sampled advantage and "
                "strategy-memory estimates on Leduc."
            ),
            "measurements": estimator.to_dict(),
            "checks": estimator_checks,
            "passed": all(check["passed"] for check in estimator_checks),
        },
    )
    if not all(check["passed"] for check in estimator_checks):
        raise RuntimeError("exploratory-sampling estimator validation failed")

    records: list[_ExperimentRecord] = []
    run_files: dict[str, dict[str, str]] = {}
    for sampling, preset_name, epsilon in (
        ("baseline", "leduc_exploration_baseline.toml", 0.0),
        ("exploratory", "leduc_exploratory_sampling.toml", 0.1),
    ):
        for seed in SEEDS:
            run_id = f"leduc-{sampling}-sampling-seed-{seed}"
            config = _experiment_config(preset_name, run_id, seed)
            report_progress(progress_callback, f"{sampling} seed {seed}: checking run state")
            _complete_run(config, runs_root, progress_callback)
            run_directory = runs_root / run_id
            records.extend(_read_metrics(run_directory / "metrics.csv", sampling, epsilon))
            run_files[f"{sampling}:{seed}"] = {
                "run_directory": str(run_directory),
                "latest_checkpoint": str(run_directory / "checkpoints" / "latest.pt"),
                "iteration_metrics": str(run_directory / "iteration_metrics.csv"),
                "exploration_metrics": (
                    str(run_directory / "exploration_metrics.csv")
                    if epsilon > 0.0
                    else "not generated for epsilon=0"
                ),
            }

    _validate_record_grid(records)
    summary = _summarise(records)
    training_checks = _training_checks(records, summary)
    write_csv(output_directory / "training_results.csv", _RESULT_FIELDS, records)
    write_csv(output_directory / "training_summary.csv", _SUMMARY_FIELDS, summary)
    passed = all(check["passed"] for check in (*estimator_checks, *training_checks))
    validation_path = output_directory / "validation.json"
    write_json(
        validation_path,
        {
            "about": (
                "Leduc-only importance-corrected exploratory opponent-sampling gate. "
                "No Kuhn or Hold'em runs are part of this validation."
            ),
            "validation_id": VALIDATION_ID,
            "passed": passed,
            "configuration": {
                "baseline_preset": "configs/deep_cfr/leduc_exploration_baseline.toml",
                "exploratory_preset": "configs/deep_cfr/leduc_exploratory_sampling.toml",
                "checkpoint_interval": 1,
                "comparison_rule": (
                    "Exploratory sampling must improve at least two paired final seeds "
                    "and lower median final exact exploitability."
                ),
            },
            "seeds": list(SEEDS),
            "milestones": list(MILESTONES),
            "estimator_checks": estimator_checks,
            "training_checks": training_checks,
            "run_files": run_files,
            "files": {
                "estimator_validation": "estimator_validation.json",
                "training_results": "training_results.csv",
                "training_summary": "training_summary.csv",
            },
        },
    )
    return validation_path


def _experiment_config(preset_name: str, run_id: str, seed: int) -> DeepCFRRunConfig:
    preset = Path("configs") / "deep_cfr" / preset_name
    return load_deep_cfr_run_config(
        preset,
        run_id=run_id,
        overrides={"seed": seed},
    )


def _complete_run(
    config: DeepCFRRunConfig,
    runs_root: Path,
    progress_callback: Callable[[str], None] | None,
) -> None:
    run_directory = runs_root / config.run_id
    if not run_directory.exists():
        start_deep_cfr_training(
            config,
            runs_root=runs_root,
            progress_callback=_iteration_reporter(config, progress_callback),
        )
        return
    stored_config = _load_run_config(run_directory / "run_config.json")
    if stored_config != config:
        raise ValueError(f"existing run has incompatible configuration: {run_directory}")
    latest_checkpoint = run_directory / "checkpoints" / "latest.pt"
    if not latest_checkpoint.exists():
        raise ValueError(f"existing run has no recovery checkpoint: {run_directory}")
    outcome = resume_deep_cfr_training(
        latest_checkpoint,
        progress_callback=_iteration_reporter(config, progress_callback),
    )
    if outcome.final_iteration != config.training.iterations:
        raise RuntimeError(f"run stopped before its configured budget: {run_directory}")


def _iteration_reporter(
    config: DeepCFRRunConfig,
    progress_callback: Callable[[str], None] | None,
) -> Callable[[int, int], None] | None:
    if progress_callback is None:
        return None
    return lambda completed, total: report_progress(
        progress_callback,
        f"{config.run_id}: {completed}/{total} iterations",
    )


def _read_metrics(
    path: Path,
    sampling: str,
    epsilon: float,
) -> list[_ExperimentRecord]:
    records: list[_ExperimentRecord] = []
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            for raw in csv.DictReader(input_file):
                records.append(
                    {
                        "sampling": sampling,
                        "epsilon": epsilon,
                        "seed": int(raw["seed"]),
                        "iteration": int(raw["iteration"]),
                        "exploitability": float(raw["exploitability"]),
                        "nash_conv": float(raw["nash_conv"]),
                        "elapsed_training_seconds": float(raw["elapsed_training_seconds"]),
                        "run_id": raw["run_id"],
                    }
                )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Deep CFR experiment metrics are unreadable: {path}") from error
    return records


def _validate_record_grid(records: list[_ExperimentRecord]) -> None:
    expected = {
        (sampling, seed, iteration)
        for sampling in ("baseline", "exploratory")
        for seed in SEEDS
        for iteration in MILESTONES
    }
    actual = {(record["sampling"], record["seed"], record["iteration"]) for record in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("matched exploratory-sampling result grid is incomplete")


def _summarise(records: list[_ExperimentRecord]) -> list[_SummaryRecord]:
    summary: list[_SummaryRecord] = []
    for sampling, epsilon in (("baseline", 0.0), ("exploratory", 0.1)):
        for iteration in MILESTONES:
            matches = [
                record
                for record in records
                if record["sampling"] == sampling and record["iteration"] == iteration
            ]
            exploitabilities = [record["exploitability"] for record in matches]
            elapsed = [record["elapsed_training_seconds"] for record in matches]
            summary.append(
                {
                    "sampling": sampling,
                    "epsilon": epsilon,
                    "iteration": iteration,
                    "seed_count": len(matches),
                    "median_exploitability": median(exploitabilities),
                    "minimum_exploitability": min(exploitabilities),
                    "maximum_exploitability": max(exploitabilities),
                    "median_elapsed_training_seconds": median(elapsed),
                }
            )
    return summary


def _estimator_checks(measurements: dict[str, int | float]) -> list[dict[str, object]]:
    return [
        {
            "name": "corrected_advantage_estimates_match_full_tree",
            "passed": measurements["advantage_reach_weighted_rmse"] <= ADVANTAGE_RMSE_LIMIT,
            "value": measurements["advantage_reach_weighted_rmse"],
            "limit": ADVANTAGE_RMSE_LIMIT,
        },
        {
            "name": "corrected_strategy_memory_matches_full_tree",
            "passed": measurements["strategy_reach_weighted_rmse"] <= STRATEGY_RMSE_LIMIT,
            "value": measurements["strategy_reach_weighted_rmse"],
            "limit": STRATEGY_RMSE_LIMIT,
        },
        {
            "name": "all_reachable_information_sets_observed",
            "passed": measurements["observed_information_sets"]
            == measurements["expected_information_sets"],
            "observed": measurements["observed_information_sets"],
            "expected": measurements["expected_information_sets"],
        },
        {
            "name": "importance_ratios_bounded",
            "passed": measurements["maximum_importance_ratio"] <= 1.0 / 0.9 + 1e-12,
            "value": measurements["maximum_importance_ratio"],
            "limit": 1.0 / 0.9,
        },
        {
            "name": "effective_sample_fraction_healthy",
            "passed": measurements["effective_sample_fraction"]
            >= MINIMUM_EFFECTIVE_SAMPLE_FRACTION,
            "value": measurements["effective_sample_fraction"],
            "minimum": MINIMUM_EFFECTIVE_SAMPLE_FRACTION,
        },
    ]


def _training_checks(
    records: list[_ExperimentRecord],
    summary: list[_SummaryRecord],
) -> list[dict[str, object]]:
    final_records = [record for record in records if record["iteration"] == MILESTONES[-1]]
    by_key = {
        (record["sampling"], record["seed"]): record["exploitability"] for record in final_records
    }
    improved_seeds = [
        seed for seed in SEEDS if by_key[("exploratory", seed)] < by_key[("baseline", seed)]
    ]
    final_summary = {
        record["sampling"]: record["median_exploitability"]
        for record in summary
        if record["iteration"] == MILESTONES[-1]
    }
    return [
        {
            "name": "matched_seed_majority_improves",
            "passed": len(improved_seeds) >= 2,
            "improved_seeds": improved_seeds,
            "required": 2,
        },
        {
            "name": "median_final_exploitability_improves",
            "passed": final_summary["exploratory"] < final_summary["baseline"],
            "baseline": final_summary["baseline"],
            "exploratory": final_summary["exploratory"],
        },
    ]


def main() -> int:
    """Run the complete pre-Hold'em Leduc gate with console progress."""
    path = run_deep_cfr_exploration_validation(progress_callback=print)
    values = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path}")
    return 0 if values["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
