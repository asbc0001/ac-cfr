"""Checkpointed Deep CFR training and exact Leduc evaluation."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from ac_cfr.common.config import DeepCFRImplementationId, GameConfigurationId
from ac_cfr.common.provenance import code_revision
from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.base import GameId, UtilityUnit
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
from ac_cfr.models import DeepCFRNetwork
from ac_cfr.persistence.deep_cfr_checkpoints import (
    load_deep_cfr_checkpoint,
    save_deep_cfr_checkpoint,
)
from ac_cfr.persistence.deep_cfr_snapshots import (
    deep_cfr_policy,
    export_deep_cfr_snapshot,
)
from ac_cfr.persistence.files import atomic_text_writer
from ac_cfr.persistence.results import DeepCFRMetricStore
from ac_cfr.solvers.deep_cfr_selection import deep_cfr_implementation, deep_cfr_solver_type
from ac_cfr.solvers.naive_deep_cfr import NaiveDeepCFR, NetworkTrainingMetrics
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig

DEEP_CFR_SOLVER_ID = "deep_cfr"
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True, slots=True)
class DeepCFRRunConfig:
    """Complete outer schedule for one configured Leduc Deep CFR run."""

    run_id: str
    implementation: DeepCFRImplementationId
    checkpoint_interval: int
    training: DeepCFRTrainingConfig
    runtime: DeepCFRRuntimeConfig

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _IDENTIFIER_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id contains unsupported characters")
        if not isinstance(self.implementation, DeepCFRImplementationId):
            raise TypeError("implementation must be a DeepCFRImplementationId")
        if isinstance(self.checkpoint_interval, bool) or not isinstance(
            self.checkpoint_interval, int
        ):
            raise TypeError("checkpoint_interval must be an integer")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if not isinstance(self.training, DeepCFRTrainingConfig):
            raise TypeError("training must be a DeepCFRTrainingConfig")
        if not isinstance(self.runtime, DeepCFRRuntimeConfig):
            raise TypeError("runtime must be a DeepCFRRuntimeConfig")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible run configuration values."""
        return {
            "run_id": self.run_id,
            "implementation": self.implementation.value,
            "checkpoint_interval": self.checkpoint_interval,
            "training": self.training.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: object) -> "DeepCFRRunConfig":
        """Reconstruct and validate one stored run configuration."""
        if not isinstance(values, dict) or set(values) != {
            "run_id",
            "implementation",
            "checkpoint_interval",
            "training",
            "runtime",
        }:
            raise ValueError("Deep CFR run configuration fields are incompatible")
        try:
            return cls(
                run_id=values["run_id"],
                implementation=DeepCFRImplementationId(values["implementation"]),
                checkpoint_interval=values["checkpoint_interval"],
                training=DeepCFRTrainingConfig.from_dict(values["training"]),
                runtime=DeepCFRRuntimeConfig.from_dict(values["runtime"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Deep CFR run configuration is invalid") from error


@dataclass(frozen=True, slots=True)
class DeepCFRTrainingOutcome:
    """Paths and completed state produced by a Deep CFR run."""

    run_directory: Path
    latest_checkpoint: Path
    final_iteration: int
    snapshot_paths: tuple[Path, ...]


def start_deep_cfr_training(
    config: DeepCFRRunConfig,
    *,
    runs_root: Path = Path("runs"),
    progress_callback: Callable[[int, int], None] | None = None,
) -> DeepCFRTrainingOutcome:
    """Start one configured Deep CFR implementation in a new directory."""
    if not isinstance(config, DeepCFRRunConfig):
        raise TypeError("config must be a DeepCFRRunConfig")
    run_directory = runs_root / config.run_id
    if run_directory.exists():
        raise FileExistsError(f"run directory already exists: {run_directory}")
    _apply_runtime(config.runtime)
    tree = compile_game_tree(LeducGame(), LeducConfig())
    solver = deep_cfr_solver_type(config.implementation)(tree, config.training, config.runtime)
    revision = code_revision()
    _write_run_config(run_directory / "run_config.json", config, revision)
    return _execute_schedule(
        config=config,
        solver=solver,
        run_directory=run_directory,
        result_store=DeepCFRMetricStore(run_directory / "metrics.csv"),
        elapsed_training_seconds=0.0,
        revision=revision,
        progress_callback=progress_callback,
    )


def resume_deep_cfr_training(
    checkpoint_path: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DeepCFRTrainingOutcome:
    """Resume the configured Deep CFR implementation from a complete iteration."""
    run_directory = checkpoint_path.parent.parent
    config = _load_run_config(run_directory / "run_config.json")
    _apply_runtime(config.runtime)
    tree = compile_game_tree(LeducGame(), LeducConfig())
    loaded = load_deep_cfr_checkpoint(checkpoint_path, tree, map_location=config.runtime.device)
    if config.run_id != loaded.metadata["run_id"]:
        raise ValueError("checkpoint run_id does not match run_config.json")
    if config.training != loaded.solver.config:
        raise ValueError("checkpoint training configuration does not match run_config.json")
    if config.runtime != loaded.solver.runtime:
        raise ValueError("checkpoint runtime configuration does not match run_config.json")
    if config.implementation is not deep_cfr_implementation(loaded.solver):
        raise ValueError("checkpoint implementation does not match run_config.json")
    result_store = DeepCFRMetricStore(run_directory / "metrics.csv")
    raw_records = loaded.run_state["metric_records"]
    if not isinstance(raw_records, list):
        raise ValueError("checkpoint metric records are invalid")
    result_store.replace(raw_records)
    return _execute_schedule(
        config=config,
        solver=loaded.solver,
        run_directory=run_directory,
        result_store=result_store,
        elapsed_training_seconds=float(loaded.run_state["elapsed_training_seconds"]),
        revision=code_revision(),
        progress_callback=progress_callback,
    )


def _execute_schedule(
    *,
    config: DeepCFRRunConfig,
    solver: NaiveDeepCFR,
    run_directory: Path,
    result_store: DeepCFRMetricStore,
    elapsed_training_seconds: float,
    revision: str,
    progress_callback: Callable[[int, int], None] | None,
) -> DeepCFRTrainingOutcome:
    """Run pending checkpoint and average-strategy milestones in order."""
    total_iterations = config.training.iterations
    snapshot_paths: list[Path] = []
    if progress_callback is not None:
        progress_callback(solver.iteration, total_iterations)
    for milestone in _remaining_milestones(config, solver.iteration, progress_callback is not None):
        start_time = perf_counter()
        solver.train(milestone - solver.iteration)
        elapsed_training_seconds += perf_counter() - start_time

        should_snapshot = (
            milestone in config.training.snapshot_iterations or milestone == total_iterations
        )
        should_checkpoint = milestone % config.checkpoint_interval == 0 or should_snapshot
        checkpoint_id = f"{config.run_id}_iter_{milestone}" if should_checkpoint else ""
        if should_snapshot:
            network = (
                solver.final_strategy_network
                if milestone == total_iterations
                else solver.snapshot_networks[milestone]
            )
            if network is None:
                raise RuntimeError("scheduled Deep CFR strategy network was not trained")
            snapshot_id = f"{config.run_id}_iter_{milestone}"
            snapshot_path = run_directory / "strategy_snapshots" / f"{snapshot_id}.pt"
            export_deep_cfr_snapshot(
                snapshot_path,
                network=network,
                tree=solver.tree,
                config=config.training,
                implementation=config.implementation,
                snapshot_id=snapshot_id,
                iteration=milestone,
                run_id=config.run_id,
                source_checkpoint_id=checkpoint_id,
            )
            snapshot_paths.append(snapshot_path)
            _record_evaluation(
                result_store=result_store,
                solver=solver,
                network=network,
                snapshot_id=snapshot_id,
                checkpoint_id=checkpoint_id,
                run_id=config.run_id,
                elapsed_training_seconds=elapsed_training_seconds,
            )

        if should_checkpoint:
            _save_checkpoint_pair(
                run_directory=run_directory,
                solver=solver,
                config=config,
                elapsed_training_seconds=elapsed_training_seconds,
                checkpoint_id=checkpoint_id,
                metric_records=result_store.records,
                revision=revision,
            )
        if progress_callback is not None:
            progress_callback(solver.iteration, total_iterations)

    latest_checkpoint = run_directory / "checkpoints" / "latest.pt"
    if not latest_checkpoint.exists():
        checkpoint_id = f"{config.run_id}_iter_{solver.iteration}"
        _save_checkpoint_pair(
            run_directory=run_directory,
            solver=solver,
            config=config,
            elapsed_training_seconds=elapsed_training_seconds,
            checkpoint_id=checkpoint_id,
            metric_records=result_store.records,
            revision=revision,
        )
    outcome = DeepCFRTrainingOutcome(
        run_directory=run_directory,
        latest_checkpoint=latest_checkpoint,
        final_iteration=solver.iteration,
        snapshot_paths=tuple(snapshot_paths),
    )
    _write_summary(outcome, result_store)
    return outcome


def _record_evaluation(
    *,
    result_store: DeepCFRMetricStore,
    solver: NaiveDeepCFR,
    network: DeepCFRNetwork,
    snapshot_id: str,
    checkpoint_id: str,
    run_id: str,
    elapsed_training_seconds: float,
) -> None:
    """Record exact strategy quality and the matching network losses."""
    if not isinstance(network, DeepCFRNetwork):
        raise TypeError("network must be a DeepCFRNetwork")
    metrics = evaluate_strategy(solver.tree, deep_cfr_policy(solver.tree, network))
    player_metrics = tuple(
        _network_metric(solver.training_metrics, solver.iteration, "advantage", player)
        for player in (0, 1)
    )
    strategy_metric = _network_metric(
        solver.training_metrics,
        solver.iteration,
        "strategy",
        None,
    )
    traversals = 2 * solver.config.traversals_per_player * solver.iteration
    result_store.upsert(
        {
            "game": GameId.LEDUC.value,
            "game_version": GameConfigurationId.LEDUC.value,
            "utility_unit": UtilityUnit.CHIP.value,
            "solver": deep_cfr_implementation(solver).value,
            "run_id": run_id,
            "strategy_snapshot_id": snapshot_id,
            "source_checkpoint_id": checkpoint_id,
            "iteration": solver.iteration,
            "seed": solver.config.seed,
            "elapsed_training_seconds": elapsed_training_seconds,
            "expected_value_player_zero": metrics.expected_values[0],
            "exploitability": metrics.exploitability,
            "nash_conv": metrics.nash_conv,
            "traversals": traversals,
            "traversals_per_second": traversals / elapsed_training_seconds,
            "player_zero_advantage_training_loss": player_metrics[0].training_loss,
            "player_zero_advantage_validation_loss": player_metrics[0].validation_loss,
            "player_one_advantage_training_loss": player_metrics[1].training_loss,
            "player_one_advantage_validation_loss": player_metrics[1].validation_loss,
            "strategy_training_loss": strategy_metric.training_loss,
            "strategy_validation_loss": strategy_metric.validation_loss,
        }
    )


def _network_metric(
    metrics: tuple[NetworkTrainingMetrics, ...],
    iteration: int,
    role: str,
    player: int | None,
) -> NetworkTrainingMetrics:
    """Return the unique neural-training metric for one scheduled network."""
    matches = tuple(
        metric
        for metric in metrics
        if metric.iteration == iteration and metric.network_role == role and metric.player == player
    )
    if len(matches) != 1:
        raise RuntimeError("Deep CFR network training metrics are incomplete")
    return matches[0]


def _remaining_milestones(
    config: DeepCFRRunConfig,
    completed: int,
    include_progress: bool,
) -> tuple[int, ...]:
    """Merge checkpoint, snapshot, final and optional progress boundaries."""
    total = config.training.iterations
    milestones = set(range(config.checkpoint_interval, total + 1, config.checkpoint_interval))
    milestones.update(config.training.snapshot_iterations)
    if include_progress:
        milestones.update((total * percentage + 99) // 100 for percentage in range(5, 101, 5))
    milestones.add(total)
    return tuple(sorted(iteration for iteration in milestones if iteration > completed))


def _save_checkpoint_pair(
    *,
    run_directory: Path,
    solver: NaiveDeepCFR,
    config: DeepCFRRunConfig,
    elapsed_training_seconds: float,
    checkpoint_id: str,
    metric_records: tuple[dict[str, str], ...],
    revision: str,
) -> None:
    """Save an iteration checkpoint and atomically update the latest alias."""
    checkpoint_directory = run_directory / "checkpoints"
    arguments = {
        "solver": solver,
        "run_id": config.run_id,
        "checkpoint_id": checkpoint_id,
        "code_revision": revision,
        "elapsed_training_seconds": elapsed_training_seconds,
        "metric_records": metric_records,
    }
    save_deep_cfr_checkpoint(
        checkpoint_directory / f"iter_{solver.iteration}.pt",
        **arguments,
    )
    save_deep_cfr_checkpoint(checkpoint_directory / "latest.pt", **arguments)


def _write_run_config(path: Path, config: DeepCFRRunConfig, revision: str) -> None:
    """Write immutable run configuration and source provenance."""
    with atomic_text_writer(path) as config_file:
        json.dump(
            {"code_revision": revision, "run_config": config.to_dict()},
            config_file,
            indent=2,
            sort_keys=True,
        )
        config_file.write("\n")


def _apply_runtime(config: DeepCFRRuntimeConfig) -> None:
    """Apply the resolved CPU execution setting before constructing networks."""
    torch.set_num_threads(config.cpu_threads)


def _load_run_config(path: Path) -> DeepCFRRunConfig:
    """Read and validate the immutable configuration beside a checkpoint."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("run_config.json is unreadable") from error
    if not isinstance(values, dict) or set(values) != {"code_revision", "run_config"}:
        raise ValueError("run_config.json fields are incompatible")
    if not isinstance(values["code_revision"], str) or not values["code_revision"]:
        raise ValueError("run_config.json code revision is invalid")
    return DeepCFRRunConfig.from_dict(values["run_config"])


def _write_summary(outcome: DeepCFRTrainingOutcome, store: DeepCFRMetricStore) -> None:
    """Write a concise human-readable summary from the final exact measurement."""
    if not store.records:
        return
    record = max(store.records, key=lambda value: int(value["iteration"]))
    lines = (
        f"Run: {record['run_id']}",
        "Game: Leduc",
        f"Solver: Deep CFR ({record['solver']})",
        f"Iterations: {int(record['iteration']):,}",
        f"Traversals: {int(record['traversals']):,}",
        f"Player 0 average-policy value: {float(record['expected_value_player_zero']):.12g} chips",
        f"Exact exploitability: {float(record['exploitability']):.12g} chips",
        f"NashConv: {float(record['nash_conv']):.12g} chips",
        f"Solver training time: {float(record['elapsed_training_seconds']):.6g} seconds",
        f"Final checkpoint: {outcome.latest_checkpoint.relative_to(outcome.run_directory)}",
        f"Final strategy snapshot: strategy_snapshots/{record['strategy_snapshot_id']}.pt",
    )
    with atomic_text_writer(outcome.run_directory / "summary.txt") as summary_file:
        summary_file.write("\n".join(lines))
        summary_file.write("\n")
