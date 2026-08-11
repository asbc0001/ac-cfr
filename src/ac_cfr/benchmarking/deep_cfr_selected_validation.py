"""Selected-configuration lifecycle validation for optimised Leduc Deep CFR."""

import json
from collections.abc import Callable
from dataclasses import replace
from math import isclose
from pathlib import Path
from statistics import median
from typing import Final

from ac_cfr.agents import NeuralAgent
from ac_cfr.benchmarking.harness import report_progress
from ac_cfr.common.config import DeepCFRImplementationId
from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.evaluation.plotting import plot_selected_deep_cfr_validation
from ac_cfr.games.base import Action, InformationState
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.persistence.deep_cfr_checkpoints import load_deep_cfr_checkpoint
from ac_cfr.persistence.deep_cfr_snapshots import deep_cfr_policy, load_deep_cfr_snapshot
from ac_cfr.persistence.files import write_csv, write_json
from ac_cfr.persistence.results import DeepCFRMetricStore
from ac_cfr.training.deep_cfr_config import load_deep_cfr_run_config
from ac_cfr.training.deep_cfr_runner import (
    DeepCFRRunConfig,
    resume_deep_cfr_training,
    start_deep_cfr_training,
)

VALIDATION_ID = "deep_cfr_selected"
SELECTED_PRESET = Path("configs/deep_cfr/leduc_selected.toml")
MAIN_SEED = 20260811
SHORT_SEEDS: Final = (20260810, 20260812)
SHORT_MILESTONES: Final = (1, 20)
MAIN_MILESTONES: Final = (1, 20, 50, 100)
INTERRUPTION_ITERATION = 20
SNAPSHOT_REPRODUCTION_TOLERANCE = 1e-8

_CONVERGENCE_FIELDS: Final = (
    "validation_id",
    "run_role",
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


class _PlannedInterruption(RuntimeError):
    """Stop the main run immediately after its declared recovery checkpoint."""


def run_deep_cfr_selected_validation(
    output_directory: Path = Path("results/deep_cfr"),
    *,
    runs_root: Path = Path("runs"),
    preset_path: Path = SELECTED_PRESET,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Validate the selected optimised setup across three seeds and one resumed run."""
    selected = load_deep_cfr_run_config(preset_path, run_id="validation-metadata")
    _validate_selected_config(selected)
    tree = compile_game_tree(LeducGame(), LeducConfig())
    records: list[dict[str, object]] = []
    run_files: dict[str, object] = {}

    for index, seed in enumerate(SHORT_SEEDS, start=1):
        report_progress(progress_callback, f"short seed {index}/{len(SHORT_SEEDS)}: {seed}")
        config = _short_config(selected, seed)
        _complete_run(config, runs_root, progress_callback)
        seed_records, files = _validated_run(config, SHORT_MILESTONES, runs_root, tree)
        records.extend(_project_records(seed_records, "short"))
        run_files[str(seed)] = files

    main_config = replace(selected, run_id=_run_id(selected, MAIN_SEED))
    report_progress(progress_callback, f"moderate seed: {MAIN_SEED}")
    resumed = _complete_run(
        main_config,
        runs_root,
        progress_callback,
        interrupt_at=INTERRUPTION_ITERATION,
    )
    main_records, main_files = _validated_run(
        main_config,
        MAIN_MILESTONES,
        runs_root,
        tree,
    )
    records.extend(_project_records(main_records, "moderate"))
    run_files[str(MAIN_SEED)] = main_files

    agent_maximum_difference = _validate_agents(run_files, tree)
    summary = _summarise(records)
    checks = _checks(records, summary, resumed, agent_maximum_difference)

    output_directory.mkdir(parents=True, exist_ok=True)
    convergence_path = output_directory / "selected_convergence.csv"
    plot_path = output_directory / "plots" / "selected_validation.png"
    write_csv(convergence_path, _CONVERGENCE_FIELDS, records)
    plot_selected_deep_cfr_validation(convergence_path, plot_path, main_seed=MAIN_SEED)

    passed = all(bool(check["passed"]) for check in checks)
    validation_path = output_directory / "selected_validation.json"
    write_json(
        validation_path,
        {
            "about": (
                "Machine-readable checks and file index for the selected optimised Leduc "
                "Deep CFR validation preceding final-policy training."
            ),
            "validation_id": VALIDATION_ID,
            "passed": passed,
            "selected_preset": str(preset_path),
            "configuration": {
                "implementation": selected.implementation.value,
                "short_seeds": [*SHORT_SEEDS, MAIN_SEED],
                "short_milestones": list(SHORT_MILESTONES),
                "main_seed": MAIN_SEED,
                "main_milestones": list(MAIN_MILESTONES),
                "interruption_iteration": INTERRUPTION_ITERATION,
                "snapshot_reproduction_absolute_tolerance": (SNAPSHOT_REPRODUCTION_TOLERANCE),
                "training": selected.training.to_dict(),
                "runtime": selected.runtime.to_dict(),
                "timed_region": (
                    "solver.train, including configured milestone strategy-network training"
                ),
            },
            "short_seed_summary": summary,
            "checks": checks,
            "run_files": run_files,
            "files": {
                "convergence": convergence_path.name,
                "plot": str(plot_path.relative_to(output_directory)),
            },
            "file_descriptions": {
                "convergence": "Exact measurements and neural losses for every snapshot.",
                "plot": "Multi-seed learning and moderate-run loss diagnostics.",
            },
        },
    )
    if not passed:
        raise RuntimeError(f"selected Deep CFR validation failed; see {validation_path}")
    return validation_path


def _validate_selected_config(config: DeepCFRRunConfig) -> None:
    """Require the declared optimised 100-iteration validation schedule."""
    if config.implementation is not DeepCFRImplementationId.OPTIMISED:
        raise ValueError("selected validation requires the optimised implementation")
    if config.training.seed != MAIN_SEED:
        raise ValueError("selected validation preset has an unexpected seed")
    if config.training.iterations != MAIN_MILESTONES[-1]:
        raise ValueError("selected validation preset has an unexpected iteration budget")
    if config.training.snapshot_iterations != MAIN_MILESTONES[:-1]:
        raise ValueError("selected validation preset has unexpected snapshot milestones")
    if config.checkpoint_interval > INTERRUPTION_ITERATION:
        raise ValueError("selected validation cannot checkpoint before interruption")


def _short_config(selected: DeepCFRRunConfig, seed: int) -> DeepCFRRunConfig:
    """Derive one short seed while preserving all selected learning settings."""
    training = replace(
        selected.training,
        iterations=SHORT_MILESTONES[-1],
        seed=seed,
        snapshot_iterations=SHORT_MILESTONES[:-1],
    )
    return replace(
        selected,
        run_id=_run_id(selected, seed, short=True),
        training=training,
    )


def _run_id(config: DeepCFRRunConfig, seed: int, *, short: bool = False) -> str:
    """Identify validation runs by seed and the corrected advantage-update budget."""
    role = "short-" if short else ""
    steps = config.training.advantage_training_steps
    return f"leduc-deep-cfr-selected-{role}a{steps}-seed-{seed}"


def _complete_run(
    config: DeepCFRRunConfig,
    runs_root: Path,
    progress_callback: Callable[[str], None] | None,
    *,
    interrupt_at: int | None = None,
) -> bool:
    """Start or resume one exact configuration and report whether resume was exercised."""
    run_directory = runs_root / config.run_id
    if run_directory.exists():
        _require_run_config(run_directory, config)
        checkpoint = run_directory / "checkpoints" / "latest.pt"
        loaded = load_deep_cfr_checkpoint(checkpoint, compile_game_tree(LeducGame(), LeducConfig()))
        if loaded.solver.iteration < config.training.iterations:
            outcome = resume_deep_cfr_training(
                checkpoint, progress_callback=_progress(config, progress_callback)
            )
            if interrupt_at is not None:
                _write_resume_marker(
                    run_directory,
                    config,
                    loaded.solver.iteration,
                    outcome.final_iteration,
                )
            return True
        return _resume_marker_verified(run_directory, config)

    if interrupt_at is None:
        start_deep_cfr_training(
            config,
            runs_root=runs_root,
            progress_callback=_progress(config, progress_callback),
        )
        return False

    def interrupting_progress(completed: int, total: int) -> None:
        _report_iteration(config, progress_callback, completed, total)
        if completed == interrupt_at:
            raise _PlannedInterruption

    try:
        start_deep_cfr_training(
            config,
            runs_root=runs_root,
            progress_callback=interrupting_progress,
        )
    except _PlannedInterruption as interruption:
        checkpoint = run_directory / "checkpoints" / "latest.pt"
        loaded = load_deep_cfr_checkpoint(checkpoint, compile_game_tree(LeducGame(), LeducConfig()))
        if loaded.solver.iteration != interrupt_at:
            raise RuntimeError(
                "planned interruption did not leave the expected checkpoint"
            ) from interruption
        outcome = resume_deep_cfr_training(
            checkpoint, progress_callback=_progress(config, progress_callback)
        )
        _write_resume_marker(run_directory, config, interrupt_at, outcome.final_iteration)
        return True
    raise RuntimeError("selected Deep CFR run completed without its planned interruption")


def _progress(
    config: DeepCFRRunConfig,
    callback: Callable[[str], None] | None,
) -> Callable[[int, int], None] | None:
    """Adapt the training runner's numeric progress to the suite callback."""
    if callback is None:
        return None
    return lambda completed, total: _report_iteration(config, callback, completed, total)


def _report_iteration(
    config: DeepCFRRunConfig,
    callback: Callable[[str], None] | None,
    completed: int,
    total: int,
) -> None:
    """Report one concise run-specific iteration update."""
    report_progress(callback, f"{config.run_id}: {completed}/{total} outer iterations")


def _require_run_config(run_directory: Path, config: DeepCFRRunConfig) -> None:
    """Reject reuse of an existing directory with different resolved settings."""
    try:
        values = json.loads((run_directory / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing Deep CFR run configuration is unreadable") from error
    if not isinstance(values, dict) or values.get("run_config") != config.to_dict():
        raise ValueError("existing Deep CFR run uses a different configuration")


def _validated_run(
    config: DeepCFRRunConfig,
    milestones: tuple[int, ...],
    runs_root: Path,
    tree: IndexedGameTree,
) -> tuple[tuple[dict[str, str], ...], dict[str, object]]:
    """Validate one completed run and return its compact metrics and file index."""
    run_directory = runs_root / config.run_id
    latest_checkpoint = run_directory / "checkpoints" / "latest.pt"
    loaded = load_deep_cfr_checkpoint(latest_checkpoint, tree)
    if loaded.solver.iteration != config.training.iterations:
        raise ValueError("selected Deep CFR run is incomplete")
    records = DeepCFRMetricStore(run_directory / "metrics.csv").records
    if tuple(int(record["iteration"]) for record in records) != milestones:
        raise ValueError("selected Deep CFR run metrics do not match its milestones")

    snapshot_paths: list[str] = []
    for record in records:
        snapshot_path = (
            run_directory / "strategy_snapshots" / f"{record['strategy_snapshot_id']}.pt"
        )
        snapshot = load_deep_cfr_snapshot(snapshot_path, tree)
        metrics = evaluate_strategy(tree, deep_cfr_policy(tree, snapshot.network))
        if not isclose(
            metrics.exploitability,
            float(record["exploitability"]),
            rel_tol=0.0,
            abs_tol=SNAPSHOT_REPRODUCTION_TOLERANCE,
        ):
            raise ValueError("selected Deep CFR snapshot does not reproduce its exact metric")
        snapshot_paths.append(str(snapshot_path))
    return records, {
        "code_revision": _run_revision(run_directory),
        "run_directory": str(run_directory),
        "latest_checkpoint": str(latest_checkpoint),
        "snapshots": snapshot_paths,
    }


def _run_revision(run_directory: Path) -> str:
    """Return the source revision stored with one immutable resolved run."""
    try:
        values = json.loads((run_directory / "run_config.json").read_text(encoding="utf-8"))
        revision = values["code_revision"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("selected Deep CFR run revision is unreadable") from error
    if not isinstance(revision, str) or not revision:
        raise ValueError("selected Deep CFR run revision is invalid")
    return revision


def _project_records(
    records: tuple[dict[str, str], ...],
    run_role: str,
) -> list[dict[str, object]]:
    """Project runner metrics onto the selected-validation result schema."""
    projected: list[dict[str, object]] = []
    for record in records:
        projected.append(
            {
                "validation_id": VALIDATION_ID,
                "run_role": run_role,
                **{
                    field: record[field]
                    for field in _CONVERGENCE_FIELDS
                    if field not in {"validation_id", "run_role"}
                },
            }
        )
    return projected


def _validate_agents(run_files: dict[str, object], tree: IndexedGameTree) -> float:
    """Load every snapshot through NeuralAgent and compare every information set."""
    maximum_difference = 0.0
    for raw_files in run_files.values():
        if not isinstance(raw_files, dict) or not isinstance(raw_files.get("snapshots"), list):
            raise ValueError("selected Deep CFR run file index is invalid")
        for raw_path in raw_files["snapshots"]:
            if not isinstance(raw_path, str):
                raise ValueError("selected Deep CFR snapshot path is invalid")
            snapshot = load_deep_cfr_snapshot(Path(raw_path), tree)
            agent = NeuralAgent(snapshot)
            policy = deep_cfr_policy(tree, snapshot.network)
            for information_set_id in range(tree.information_set_count):
                information_state = _information_state(tree, information_set_id)
                strategy = agent.get_strategy(
                    information_state,
                    information_state.legal_actions,
                )
                offset = int(tree.information_set_action_offsets[information_set_id])
                for index, probability in enumerate(strategy):
                    maximum_difference = max(
                        maximum_difference,
                        abs(probability - float(policy[offset + index])),
                    )
    return maximum_difference


def _information_state(tree: IndexedGameTree, information_set_id: int) -> InformationState:
    """Reconstruct one player-visible state from the indexed Leduc tree."""
    encoding_offset = int(tree.information_set_encoding_offsets[information_set_id])
    encoding_count = int(tree.information_set_encoding_counts[information_set_id])
    action_offset = int(tree.information_set_action_offsets[information_set_id])
    action_count = int(tree.information_set_action_counts[information_set_id])
    return InformationState(
        game_id=tree.game_id,
        player=int(tree.information_set_players[information_set_id]),
        encoding=tuple(
            int(value)
            for value in tree.information_set_encodings[
                encoding_offset : encoding_offset + encoding_count
            ]
        ),
        legal_actions=tuple(
            Action(int(value))
            for value in tree.information_set_actions[action_offset : action_offset + action_count]
        ),
    )


def _summarise(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarise the three-seed distribution at shared short milestones."""
    summary: list[dict[str, object]] = []
    for milestone in SHORT_MILESTONES:
        selected = [record for record in records if int(_number(record, "iteration")) == milestone]
        exploitabilities = [_number(record, "exploitability") for record in selected]
        summary.append(
            {
                "validation_id": VALIDATION_ID,
                "iteration": milestone,
                "seed_count": len(selected),
                "median_elapsed_training_seconds": median(
                    _number(record, "elapsed_training_seconds") for record in selected
                ),
                "median_exploitability": median(exploitabilities),
                "minimum_exploitability": min(exploitabilities),
                "maximum_exploitability": max(exploitabilities),
                "median_nash_conv": median(_number(record, "nash_conv") for record in selected),
            }
        )
    return summary


def _checks(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
    resume_verified: bool,
    agent_maximum_difference: float,
) -> list[dict[str, object]]:
    """Apply predeclared learning, recovery, and playable-policy checks."""
    seed_changes = {
        seed: {
            "initial": _exploitability_at(records, seed, SHORT_MILESTONES[0]),
            "final": _exploitability_at(records, seed, SHORT_MILESTONES[-1]),
        }
        for seed in (*SHORT_SEEDS, MAIN_SEED)
    }
    main_at_20 = _exploitability_at(records, MAIN_SEED, 20)
    main_at_100 = _exploitability_at(records, MAIN_SEED, 100)
    initial_median = _number(summary[0], "median_exploitability")
    final_median = _number(summary[-1], "median_exploitability")
    return [
        {
            "name": "every_short_seed_improves_from_iteration_1_to_20",
            "passed": all(values["final"] < values["initial"] for values in seed_changes.values()),
            "seed_exploitabilities": {str(seed): values for seed, values in seed_changes.items()},
        },
        {
            "name": "short_seed_median_exploitability_improves",
            "passed": final_median < initial_median,
            "initial_median_exploitability": initial_median,
            "final_median_exploitability": final_median,
        },
        {
            "name": "moderate_run_improves_from_iteration_20_to_100",
            "passed": main_at_100 < main_at_20,
            "iteration_20_exploitability": main_at_20,
            "iteration_100_exploitability": main_at_100,
        },
        {
            "name": "selected_run_checkpoint_resume_completed",
            "passed": resume_verified,
            "interruption_iteration": INTERRUPTION_ITERATION,
        },
        {
            "name": "all_snapshots_match_neural_agent_probabilities",
            "passed": agent_maximum_difference <= 1e-6,
            "absolute_tolerance": 1e-6,
            "maximum_probability_difference": agent_maximum_difference,
        },
    ]


def _exploitability_at(records: list[dict[str, object]], seed: int, iteration: int) -> float:
    """Return one seed's unique exact exploitability at a milestone."""
    matches = [
        _number(record, "exploitability")
        for record in records
        if int(_number(record, "seed")) == seed and int(_number(record, "iteration")) == iteration
    ]
    if len(matches) != 1:
        raise RuntimeError("selected Deep CFR convergence records are incomplete")
    return matches[0]


def _number(record: dict[str, object], field: str) -> float:
    """Return one finite-format numeric validation field."""
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RuntimeError(f"selected Deep CFR field is not numeric: {field}")
    try:
        return float(value)
    except ValueError as error:
        raise RuntimeError(f"selected Deep CFR field is not numeric: {field}") from error


def _write_resume_marker(
    run_directory: Path,
    config: DeepCFRRunConfig,
    interruption_iteration: int,
    final_iteration: int,
) -> None:
    """Record the completed selected-run recovery check beside the run."""
    write_json(
        run_directory / "resume_verification.json",
        {
            "run_id": config.run_id,
            "interruption_iteration": interruption_iteration,
            "final_iteration": final_iteration,
        },
    )


def _resume_marker_verified(run_directory: Path, config: DeepCFRRunConfig) -> bool:
    """Validate durable recovery evidence when a completed run is reused."""
    marker_path = run_directory / "resume_verification.json"
    if not marker_path.is_file():
        return False
    try:
        values = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("selected Deep CFR resume evidence is unreadable") from error
    return values == {
        "run_id": config.run_id,
        "interruption_iteration": INTERRUPTION_ITERATION,
        "final_iteration": config.training.iterations,
    }
