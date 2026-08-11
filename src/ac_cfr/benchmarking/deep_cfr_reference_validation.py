"""Multi-seed convergence validation for reference Deep CFR on Leduc."""

import csv
import json
from collections.abc import Callable
from math import isclose
from pathlib import Path
from statistics import median
from typing import Final

from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.persistence.deep_cfr_checkpoints import load_deep_cfr_checkpoint
from ac_cfr.persistence.deep_cfr_snapshots import deep_cfr_policy, load_deep_cfr_snapshot
from ac_cfr.persistence.files import atomic_text_writer
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig, start_deep_cfr_training

VALIDATION_ID = "deep_cfr"
SEEDS: Final = (20260810, 20260811, 20260812)
MILESTONES: Final = (1, 5, 10)
TABULAR_REFERENCE_EXPLOITABILITY = 0.005

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
    "player_zero_advantage_training_loss",
    "player_zero_advantage_validation_loss",
    "player_one_advantage_training_loss",
    "player_one_advantage_validation_loss",
    "strategy_training_loss",
    "strategy_validation_loss",
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
    "median_strategy_training_loss",
    "median_strategy_validation_loss",
)


def run_deep_cfr_reference_validation(
    output_directory: Path = Path("results") / VALIDATION_ID,
    *,
    runs_root: Path = Path("runs"),
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Train three declared seeds and write compact exact Leduc evidence."""
    records: list[dict[str, object]] = []
    run_files: dict[str, object] = {}
    tree = compile_game_tree(LeducGame(), LeducConfig())
    for seed_number, seed in enumerate(SEEDS, start=1):
        _report(progress_callback, f"seed {seed_number}/{len(SEEDS)}: {seed}")
        run_id = f"leduc-naive-deep-cfr-fixed-steps-seed-{seed}"
        run_directory = runs_root / run_id
        if run_directory.exists():
            _report(progress_callback, f"seed {seed}: reusing compatible completed run")
        else:
            start_deep_cfr_training(
                _run_config(run_id, seed),
                runs_root=runs_root,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda completed, total, seed=seed: _report(
                        progress_callback,
                        f"seed {seed}: {completed}/{total} outer iterations",
                    )
                ),
            )
        seed_records = _read_run_metrics(run_directory / "metrics.csv")
        revision = _validate_completed_run(run_directory, _run_config(run_id, seed), tree)
        _validate_exported_snapshots(run_directory, _run_config(run_id, seed), seed_records, tree)
        records.extend(seed_records)
        run_files[str(seed)] = {
            "code_revision": revision,
            "run_directory": str(run_directory),
            "latest_checkpoint": str(run_directory / "checkpoints" / "latest.pt"),
            "snapshots": [
                str(run_directory / "strategy_snapshots" / f"{run_id}_iter_{iteration}.pt")
                for iteration in MILESTONES
            ],
        }

    summary = _summarise(records)
    checks = _checks(records, summary)
    output_directory.mkdir(parents=True, exist_ok=True)
    convergence_path = output_directory / "convergence.csv"
    summary_path = output_directory / "summary.csv"
    _write_csv(convergence_path, _CONVERGENCE_FIELDS, records)
    _write_csv(summary_path, _SUMMARY_FIELDS, summary)
    passed = all(bool(check["passed"]) for check in checks)
    validation_path = output_directory / "validation.json"
    _write_json(
        validation_path,
        {
            "about": (
                "Machine-readable configuration and checks for multi-seed reference "
                "Deep CFR convergence on Leduc."
            ),
            "validation_id": VALIDATION_ID,
            "passed": passed,
            "configuration": {
                "game": "leduc",
                "solver": "naive_deep_cfr",
                "seeds": list(SEEDS),
                "milestones": list(MILESTONES),
                "outer_iterations": MILESTONES[-1],
                "traversals_per_player_per_iteration": 200,
                "traversals_per_outer_iteration": 400,
                "advantage_reservoir_capacity": 100_000,
                "strategy_reservoir_capacity": 100_000,
                "advantage_training_steps": 20,
                "strategy_training_steps": 40,
                "batch_size": 512,
                "learning_rate": 0.001,
                "validation_fraction": 0.1,
                "max_gradient_norm": 10.0,
                "dropout_probability": 0.0,
                "early_stopping": False,
                "timed_region": (
                    "solver.train, including configured milestone strategy-network training"
                ),
                "tabular_reference_exploitability": TABULAR_REFERENCE_EXPLOITABILITY,
            },
            "checks": checks,
            "run_files": run_files,
            "files": {
                "convergence": convergence_path.name,
                "summary": summary_path.name,
            },
            "file_descriptions": {
                "convergence": "Exact measurements for every seed and milestone snapshot.",
                "summary": "Median and complete seed range at each milestone.",
            },
        },
    )
    if not passed:
        raise RuntimeError(f"reference Deep CFR validation failed; see {validation_path}")
    return validation_path


def _run_config(run_id: str, seed: int) -> DeepCFRRunConfig:
    """Return the explicit version-controlled reference-validation workload."""
    return DeepCFRRunConfig(
        run_id=run_id,
        checkpoint_interval=5,
        training=DeepCFRTrainingConfig(
            iterations=MILESTONES[-1],
            traversals_per_player=200,
            advantage_reservoir_capacity=100_000,
            strategy_reservoir_capacity=100_000,
            advantage_training_steps=20,
            strategy_training_steps=40,
            batch_size=512,
            learning_rate=1e-3,
            validation_fraction=0.1,
            max_gradient_norm=10.0,
            dropout_probability=0.0,
            seed=seed,
            snapshot_iterations=MILESTONES[:-1],
        ),
    )


def _read_run_metrics(path: Path) -> list[dict[str, object]]:
    """Project one run's CSV onto the committed convergence schema."""
    records: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8", newline="") as input_file:
            for raw in csv.DictReader(input_file):
                records.append(
                    {
                        "validation_id": VALIDATION_ID,
                        "game": raw["game"],
                        "solver": raw["solver"],
                        "seed": int(raw["seed"]),
                        "iteration": int(raw["iteration"]),
                        "traversals": int(raw["traversals"]),
                        "elapsed_training_seconds": float(raw["elapsed_training_seconds"]),
                        "expected_value_player_zero": float(raw["expected_value_player_zero"]),
                        "exploitability": float(raw["exploitability"]),
                        "nash_conv": float(raw["nash_conv"]),
                        "player_zero_advantage_training_loss": float(
                            raw["player_zero_advantage_training_loss"]
                        ),
                        "player_zero_advantage_validation_loss": float(
                            raw["player_zero_advantage_validation_loss"]
                        ),
                        "player_one_advantage_training_loss": float(
                            raw["player_one_advantage_training_loss"]
                        ),
                        "player_one_advantage_validation_loss": float(
                            raw["player_one_advantage_validation_loss"]
                        ),
                        "strategy_training_loss": float(raw["strategy_training_loss"]),
                        "strategy_validation_loss": float(raw["strategy_validation_loss"]),
                    }
                )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Deep CFR run metrics are unreadable") from error
    if tuple(int(_number(record, "iteration")) for record in records) != MILESTONES:
        raise ValueError("Deep CFR run metrics do not match the declared milestones")
    return records


def _validate_completed_run(
    run_directory: Path,
    config: DeepCFRRunConfig,
    tree: IndexedGameTree,
) -> str:
    """Reuse only a complete run with exactly the declared configuration."""
    try:
        values = json.loads((run_directory / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing Deep CFR run configuration is unreadable") from error
    if not isinstance(values, dict) or values.get("run_config") != config.to_dict():
        raise ValueError("existing Deep CFR run uses a different configuration")
    revision = values.get("code_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("existing Deep CFR run code revision is invalid")
    required_paths = [
        run_directory / "checkpoints" / "latest.pt",
        run_directory / "metrics.csv",
        *(
            run_directory / "strategy_snapshots" / f"{config.run_id}_iter_{iteration}.pt"
            for iteration in MILESTONES
        ),
    ]
    if any(not path.is_file() for path in required_paths):
        raise ValueError("existing Deep CFR run is incomplete")
    loaded = load_deep_cfr_checkpoint(required_paths[0], tree)
    if loaded.solver.iteration != config.training.iterations:
        raise ValueError("existing Deep CFR final checkpoint is incomplete")
    return revision


def _validate_exported_snapshots(
    run_directory: Path,
    config: DeepCFRRunConfig,
    records: list[dict[str, object]],
    tree: IndexedGameTree,
) -> None:
    """Reload every frozen policy and reproduce its exact exploitability."""
    if not isinstance(tree, IndexedGameTree):
        raise TypeError("tree must be an IndexedGameTree")
    for record in records:
        iteration = int(_number(record, "iteration"))
        path = run_directory / "strategy_snapshots" / f"{config.run_id}_iter_{iteration}.pt"
        snapshot = load_deep_cfr_snapshot(path, tree)
        metrics = evaluate_strategy(tree, deep_cfr_policy(tree, snapshot.network))
        if not isclose(
            metrics.exploitability,
            _number(record, "exploitability"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("exported Deep CFR snapshot does not reproduce its exact metric")


def _summarise(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate the seed distribution at each strategy milestone."""
    summary: list[dict[str, object]] = []
    for milestone in MILESTONES:
        selected = [record for record in records if record["iteration"] == milestone]
        exploitabilities = [_number(record, "exploitability") for record in selected]
        summary.append(
            {
                "validation_id": VALIDATION_ID,
                "game": "leduc",
                "solver": "naive_deep_cfr",
                "iteration": milestone,
                "traversals_per_seed": 400 * milestone,
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
                "median_strategy_training_loss": median(
                    _number(record, "strategy_training_loss") for record in selected
                ),
                "median_strategy_validation_loss": median(
                    _number(record, "strategy_validation_loss") for record in selected
                ),
            }
        )
    return summary


def _checks(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Check robust improvement without fitting a near-equilibrium threshold."""
    seed_changes = {
        seed: {
            "initial": _exploitability_at(records, seed, MILESTONES[0]),
            "final": _exploitability_at(records, seed, MILESTONES[-1]),
        }
        for seed in SEEDS
    }
    medians = [_number(record, "median_exploitability") for record in summary]
    return [
        {
            "name": "every_seed_improves_over_the_declared_workload",
            "passed": all(values["final"] < values["initial"] for values in seed_changes.values()),
            "seed_exploitabilities": {str(seed): values for seed, values in seed_changes.items()},
        },
        {
            "name": "median_exploitability_decreases_at_every_milestone",
            "passed": all(
                later < earlier for earlier, later in zip(medians, medians[1:], strict=False)
            ),
            "median_exploitabilities": medians,
        },
    ]


def _exploitability_at(records: list[dict[str, object]], seed: int, iteration: int) -> float:
    """Return the unique exploitability for one seed and snapshot."""
    matches = [
        _number(record, "exploitability")
        for record in records
        if record["seed"] == seed and record["iteration"] == iteration
    ]
    if len(matches) != 1:
        raise RuntimeError("Deep CFR convergence records are incomplete or duplicated")
    return matches[0]


def _number(record: dict[str, object], field: str) -> float:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Deep CFR validation field is not numeric: {field}")
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
    """Atomically write deterministic readable JSON."""
    with atomic_text_writer(path) as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _report(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
