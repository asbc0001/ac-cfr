"""Checkpointed CFR and CFR+ training for Kuhn and Leduc poker."""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from secrets import token_hex
from time import perf_counter

from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.base import GameId, UtilityUnit
from ac_cfr.games.tabular import TabularGame, create_tabular_game
from ac_cfr.persistence.checkpoints import (
    load_tabular_checkpoint,
    save_tabular_checkpoint,
    validate_checkpoint_compatibility,
)
from ac_cfr.persistence.files import atomic_text_writer
from ac_cfr.persistence.results import CsvResultStore
from ac_cfr.persistence.snapshots import export_tabular_snapshot
from ac_cfr.solvers import CFR, CFRPlus, NaiveCFR, NaiveCFRPlus

SOLVER_IDS = ("naive_cfr", "naive_cfr_plus", "cfr", "cfr_plus")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True, slots=True)
class TabularTrainingConfig:
    """Complete predeclared schedule for one tabular training run."""

    game: str
    solver: str
    iterations: int
    seed: int
    run_id: str
    evaluation_interval: int
    checkpoint_interval: int
    snapshot_iterations: tuple[int, ...]
    averaging_delay: int = 0
    early_stopping_minimum_improvement: float | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        try:
            game_id = GameId(self.game)
        except ValueError as error:
            raise ValueError("game must be kuhn or leduc") from error
        if game_id not in (GameId.KUHN, GameId.LEDUC):
            raise ValueError("game must be kuhn or leduc")
        if self.solver not in SOLVER_IDS:
            raise ValueError(f"solver must be one of: {', '.join(SOLVER_IDS)}")
        _validate_positive_integer("iterations", self.iterations)
        _validate_integer("seed", self.seed)
        _validate_identifier("run_id", self.run_id)
        _validate_positive_integer("evaluation_interval", self.evaluation_interval)
        _validate_positive_integer("checkpoint_interval", self.checkpoint_interval)
        _validate_non_negative_integer("averaging_delay", self.averaging_delay)
        if self.solver in ("naive_cfr", "cfr") and self.averaging_delay != 0:
            raise ValueError("averaging_delay applies only to CFR+ solvers")
        if not isinstance(self.snapshot_iterations, tuple):
            raise TypeError("snapshot_iterations must be a tuple")
        if tuple(sorted(set(self.snapshot_iterations))) != self.snapshot_iterations:
            raise ValueError("snapshot_iterations must be sorted and unique")
        for iteration in self.snapshot_iterations:
            _validate_positive_integer("snapshot iteration", iteration)
            if iteration > self.iterations:
                raise ValueError("snapshot iterations must not exceed the training budget")
        _validate_early_stopping(self)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible configuration object."""
        values = asdict(self)
        values["snapshot_iterations"] = list(self.snapshot_iterations)
        return values

    @classmethod
    def from_dict(cls, values: object) -> "TabularTrainingConfig":
        """Reconstruct and validate an exact checkpointed configuration."""
        if not isinstance(values, dict) or set(values) != set(cls.__dataclass_fields__):
            raise ValueError("training configuration fields are incomplete or unexpected")
        parsed_values = values.copy()
        snapshots = parsed_values["snapshot_iterations"]
        if not isinstance(snapshots, list):
            raise ValueError("snapshot_iterations must be stored as a list")
        parsed_values["snapshot_iterations"] = tuple(snapshots)
        try:
            return cls(**parsed_values)
        except TypeError as error:
            raise ValueError("training configuration values are invalid") from error


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Paths and final state produced by a completed or early-stopped run."""

    run_directory: Path
    latest_checkpoint: Path
    final_iteration: int
    stopped_early: bool
    snapshot_paths: tuple[Path, ...]


@dataclass(slots=True)
class _ScheduleState:
    best_exploitability: float | None = None
    evaluations_without_improvement: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: object) -> "_ScheduleState":
        if not isinstance(values, dict) or set(values) != set(cls.__dataclass_fields__):
            raise ValueError("checkpoint schedule_state is incompatible")
        try:
            state = cls(**values)
        except TypeError as error:
            raise ValueError("checkpoint schedule_state values are invalid") from error
        if state.best_exploitability is not None and (
            isinstance(state.best_exploitability, bool)
            or not isinstance(state.best_exploitability, (int, float))
            or not isfinite(state.best_exploitability)
            or state.best_exploitability < 0.0
        ):
            raise ValueError("checkpoint best_exploitability is invalid")
        _validate_non_negative_integer(
            "evaluations_without_improvement",
            state.evaluations_without_improvement,
        )
        return state


def new_run_id() -> str:
    """Return a short collision-resistant local training-run identifier."""
    return f"run_{token_hex(8)}"


def start_tabular_training(
    config: TabularTrainingConfig,
    *,
    runs_root: Path = Path("runs"),
) -> TrainingOutcome:
    """Start a CFR or CFR+ training run."""
    run_directory = runs_root / config.run_id
    if run_directory.exists():
        raise FileExistsError(f"run directory already exists: {run_directory}")
    tabular_game = create_tabular_game(GameId(config.game))
    solver = _create_solver(config, tabular_game)
    code_revision = _code_revision()
    _write_run_config(run_directory / "run_config.json", config, code_revision)
    result_store = CsvResultStore(run_directory / "metrics.csv")
    return _execute_schedule(
        config=config,
        tabular_game=tabular_game,
        solver=solver,
        run_directory=run_directory,
        result_store=result_store,
        elapsed_training_seconds=0.0,
        schedule_state=_ScheduleState(),
        code_revision=code_revision,
    )


def resume_tabular_training(checkpoint_path: Path) -> TrainingOutcome:
    """Resume a CFR or CFR+ run from a checkpoint."""
    checkpoint = load_tabular_checkpoint(checkpoint_path)
    config = TabularTrainingConfig.from_dict(checkpoint.metadata["training_config"])
    tabular_game = create_tabular_game(GameId(config.game))
    validate_checkpoint_compatibility(checkpoint, tabular_game, config.solver)
    if checkpoint.metadata["run_id"] != config.run_id:
        raise ValueError("checkpoint run_id does not match its training configuration")
    run_directory = checkpoint_path.parent.parent
    _validate_run_config(run_directory / "run_config.json", config)
    result_store = CsvResultStore(run_directory / "metrics.csv")
    elapsed_training_seconds = checkpoint.metadata["elapsed_training_seconds"]
    if (
        isinstance(elapsed_training_seconds, bool)
        or not isinstance(elapsed_training_seconds, (int, float))
        or not isfinite(elapsed_training_seconds)
        or elapsed_training_seconds < 0.0
    ):
        raise ValueError("checkpoint elapsed_training_seconds is invalid")
    schedule_state = _ScheduleState.from_dict(checkpoint.metadata["schedule_state"])
    solver = _create_solver(config, tabular_game)
    solver.restore_training_state(
        iteration=checkpoint.metadata["iteration"],
        regret_sum=checkpoint.regret_sum,
        strategy_sum=checkpoint.strategy_sum,
    )
    result_store.replace(checkpoint.metadata["metric_records"])
    return _execute_schedule(
        config=config,
        tabular_game=tabular_game,
        solver=solver,
        run_directory=run_directory,
        result_store=result_store,
        elapsed_training_seconds=float(elapsed_training_seconds),
        schedule_state=schedule_state,
        code_revision=_code_revision(),
    )


def _execute_schedule(
    *,
    config: TabularTrainingConfig,
    tabular_game: TabularGame,
    solver: NaiveCFR | CFR,
    run_directory: Path,
    result_store: CsvResultStore,
    elapsed_training_seconds: float,
    schedule_state: _ScheduleState,
    code_revision: str,
) -> TrainingOutcome:
    if solver.iteration > config.iterations:
        raise ValueError("checkpoint iteration exceeds the configured training budget")
    snapshot_paths: list[Path] = []
    stopped_early = False
    for milestone in _remaining_milestones(config, solver.iteration):
        start_time = perf_counter()
        solver.train(milestone - solver.iteration)
        elapsed_training_seconds += perf_counter() - start_time

        should_evaluate = (
            milestone % config.evaluation_interval == 0
            or milestone in config.snapshot_iterations
            or milestone == config.iterations
        )
        metrics = (
            evaluate_strategy(tabular_game.tree, solver.average_policy())
            if should_evaluate
            else None
        )
        if metrics is not None:
            stopped_early = _update_early_stopping(config, schedule_state, metrics.exploitability)

        checkpoint_id = f"{config.run_id}_iter_{milestone}"
        # Every measured policy is frozen first so evaluation records identify an exact artefact.
        should_snapshot = metrics is not None or stopped_early
        snapshot_id = f"{config.run_id}_iter_{milestone}" if should_snapshot else ""
        if should_snapshot:
            snapshot_path = run_directory / "strategy_snapshots" / f"{snapshot_id}.npz"
            export_tabular_snapshot(
                snapshot_path,
                tabular_game=tabular_game,
                average_policy=solver.average_policy(),
                snapshot_id=snapshot_id,
                solver=config.solver,
                iteration=milestone,
                run_id=config.run_id,
                seed=config.seed,
                source_checkpoint_id=checkpoint_id,
            )
            snapshot_paths.append(snapshot_path)

        if metrics is not None:
            traversals = 2 * milestone
            result_store.upsert(
                {
                    "game": config.game,
                    "game_version": tabular_game.configuration_id.value,
                    "utility_unit": UtilityUnit.CHIP.value,
                    "solver": config.solver,
                    "run_id": config.run_id,
                    "strategy_snapshot_id": snapshot_id,
                    "source_checkpoint_id": checkpoint_id,
                    "iteration": milestone,
                    "seed": config.seed,
                    "elapsed_training_seconds": elapsed_training_seconds,
                    "expected_value_player_zero": metrics.expected_values[0],
                    "exploitability": metrics.exploitability,
                    "nash_conv": metrics.nash_conv,
                    "traversals": traversals,
                    "traversals_per_second": traversals / elapsed_training_seconds,
                }
            )

        should_checkpoint = (
            milestone % config.checkpoint_interval == 0 or metrics is not None or stopped_early
        )
        if should_checkpoint:
            _save_checkpoint_pair(
                run_directory=run_directory,
                solver=solver,
                tabular_game=tabular_game,
                config=config,
                elapsed_training_seconds=elapsed_training_seconds,
                checkpoint_id=checkpoint_id,
                schedule_state=schedule_state,
                result_store=result_store,
                code_revision=code_revision,
            )
        if stopped_early:
            break

    latest_checkpoint = run_directory / "checkpoints" / "latest.npz"
    if not latest_checkpoint.exists():
        checkpoint_id = f"{config.run_id}_iter_{solver.iteration}"
        _save_checkpoint_pair(
            run_directory=run_directory,
            solver=solver,
            tabular_game=tabular_game,
            config=config,
            elapsed_training_seconds=elapsed_training_seconds,
            checkpoint_id=checkpoint_id,
            schedule_state=schedule_state,
            result_store=result_store,
            code_revision=code_revision,
        )
    return TrainingOutcome(
        run_directory=run_directory,
        latest_checkpoint=latest_checkpoint,
        final_iteration=solver.iteration,
        stopped_early=stopped_early,
        snapshot_paths=tuple(snapshot_paths),
    )


def _remaining_milestones(config: TabularTrainingConfig, completed: int) -> tuple[int, ...]:
    milestones = set(
        range(config.evaluation_interval, config.iterations + 1, config.evaluation_interval)
    )
    milestones.update(
        range(config.checkpoint_interval, config.iterations + 1, config.checkpoint_interval)
    )
    milestones.update(config.snapshot_iterations)
    milestones.add(config.iterations)
    return tuple(sorted(iteration for iteration in milestones if iteration > completed))


def _update_early_stopping(
    config: TabularTrainingConfig,
    state: _ScheduleState,
    exploitability: float,
) -> bool:
    minimum_improvement = config.early_stopping_minimum_improvement
    patience = config.early_stopping_patience
    if minimum_improvement is None or patience is None:
        return False
    if state.best_exploitability is None or (
        state.best_exploitability - exploitability >= minimum_improvement
    ):
        state.best_exploitability = exploitability
        state.evaluations_without_improvement = 0
    else:
        state.evaluations_without_improvement += 1
    return state.evaluations_without_improvement >= patience


def _save_checkpoint_pair(
    *,
    run_directory: Path,
    solver: NaiveCFR | CFR,
    tabular_game: TabularGame,
    config: TabularTrainingConfig,
    elapsed_training_seconds: float,
    checkpoint_id: str,
    schedule_state: _ScheduleState,
    result_store: CsvResultStore,
    code_revision: str,
) -> None:
    checkpoint_directory = run_directory / "checkpoints"
    arguments = {
        "solver": solver,
        "tabular_game": tabular_game,
        "solver_id": config.solver,
        "run_id": config.run_id,
        "seed": config.seed,
        "training_config": config.to_dict(),
        "elapsed_training_seconds": elapsed_training_seconds,
        "checkpoint_id": checkpoint_id,
        "schedule_state": schedule_state.to_dict(),
        "metric_records": result_store.records,
        "code_revision": code_revision,
    }
    save_tabular_checkpoint(
        checkpoint_directory / f"iter_{solver.iteration}.npz",
        **arguments,
    )
    save_tabular_checkpoint(checkpoint_directory / "latest.npz", **arguments)


def _create_solver(config: TabularTrainingConfig, tabular_game: TabularGame) -> NaiveCFR | CFR:
    if config.solver == "naive_cfr":
        return NaiveCFR(tabular_game.tree)
    if config.solver == "naive_cfr_plus":
        return NaiveCFRPlus(tabular_game.tree, averaging_delay=config.averaging_delay)
    if config.solver == "cfr":
        return CFR(tabular_game.tree)
    return CFRPlus(tabular_game.tree, averaging_delay=config.averaging_delay)


def _write_run_config(path: Path, config: TabularTrainingConfig, code_revision: str) -> None:
    values = {"code_revision": code_revision, "training_config": config.to_dict()}
    with atomic_text_writer(path) as config_file:
        json.dump(values, config_file, indent=2, sort_keys=True)
        config_file.write("\n")


def _validate_run_config(path: Path, config: TabularTrainingConfig) -> None:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("run_config.json is unreadable") from error
    if not isinstance(values, dict) or values.get("training_config") != config.to_dict():
        raise ValueError("run_config.json does not match the checkpoint")


def _code_revision() -> str:
    try:
        revision_result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = revision_result.stdout.strip()
    if not revision:
        return "unknown"
    return f"{revision}-dirty" if status_result.stdout else revision


def _validate_early_stopping(config: TabularTrainingConfig) -> None:
    minimum_improvement = config.early_stopping_minimum_improvement
    patience = config.early_stopping_patience
    if (minimum_improvement is None) != (patience is None):
        raise ValueError("early stopping requires both minimum improvement and patience")
    if minimum_improvement is None:
        return
    if (
        isinstance(minimum_improvement, bool)
        or not isinstance(minimum_improvement, (int, float))
        or not isfinite(minimum_improvement)
        or minimum_improvement <= 0.0
    ):
        raise ValueError("early stopping minimum improvement must be finite and positive")
    assert patience is not None
    _validate_positive_integer("early stopping patience", patience)


def _validate_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsupported characters")


def _validate_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _validate_non_negative_integer(name: str, value: int) -> None:
    _validate_integer(name, value)
    if value < 0:
        raise ValueError(f"{name} must not be negative")


def _validate_positive_integer(name: str, value: int) -> None:
    _validate_integer(name, value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
