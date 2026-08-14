"""Checkpointed Deep CFR training with exact evaluation where tractable."""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import psutil
import torch

from ac_cfr.common.config import DeepCFRImplementationId, GameConfigurationId
from ac_cfr.common.environment import effective_storage_remaining_bytes, environment_record
from ac_cfr.common.provenance import code_revision
from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.base import GameId, UtilityUnit
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.models import DeepCFRNetwork
from ac_cfr.persistence.deep_cfr_checkpoints import (
    load_deep_cfr_checkpoint,
    save_deep_cfr_checkpoint,
)
from ac_cfr.persistence.deep_cfr_snapshots import (
    deep_cfr_policy,
    export_deep_cfr_snapshot,
)
from ac_cfr.persistence.files import (
    atomic_text_writer,
    checkpoint_staging_directory,
    write_json,
)
from ac_cfr.persistence.results import DeepCFRIterationMetricStore, DeepCFRMetricStore
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
    checkpoint_retention: int = 2

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
        if isinstance(self.checkpoint_retention, bool) or not isinstance(
            self.checkpoint_retention, int
        ):
            raise TypeError("checkpoint_retention must be an integer")
        if self.checkpoint_retention < 1:
            raise ValueError("checkpoint_retention must be positive")
        if not isinstance(self.training, DeepCFRTrainingConfig):
            raise TypeError("training must be a DeepCFRTrainingConfig")
        if not isinstance(self.runtime, DeepCFRRuntimeConfig):
            raise TypeError("runtime must be a DeepCFRRuntimeConfig")
        if (
            self.training.game_configuration_id is GameConfigurationId.MODIFIED_HULHE
            and self.implementation is not DeepCFRImplementationId.OPTIMISED
        ):
            raise ValueError("modified HULHE requires the optimised Deep CFR implementation")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible run configuration values."""
        return {
            "run_id": self.run_id,
            "implementation": self.implementation.value,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_retention": self.checkpoint_retention,
            "training": self.training.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: object) -> "DeepCFRRunConfig":
        """Reconstruct and validate one stored run configuration."""
        if not isinstance(values, dict) or set(values) not in (
            {
                "run_id",
                "implementation",
                "checkpoint_interval",
                "training",
                "runtime",
            },
            {
                "run_id",
                "implementation",
                "checkpoint_interval",
                "checkpoint_retention",
                "training",
                "runtime",
            },
        ):
            raise ValueError("Deep CFR run configuration fields are incompatible")
        try:
            return cls(
                run_id=values["run_id"],
                implementation=DeepCFRImplementationId(values["implementation"]),
                checkpoint_interval=values["checkpoint_interval"],
                checkpoint_retention=values.get("checkpoint_retention", 2),
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
    stop_requested: Callable[[], bool] | None = None,
) -> DeepCFRTrainingOutcome:
    """Start one configured Deep CFR implementation in a new directory."""
    if not isinstance(config, DeepCFRRunConfig):
        raise TypeError("config must be a DeepCFRRunConfig")
    run_directory = runs_root / config.run_id
    if run_directory.exists():
        raise FileExistsError(f"run directory already exists: {run_directory}")
    _apply_runtime(config.runtime)
    game = _game_context(config.training)
    if isinstance(game, HoldemConfig):
        from ac_cfr.solvers.deep_cfr import DeepCFR

        solver = DeepCFR(game, config.training, config.runtime)
    else:
        solver = deep_cfr_solver_type(config.implementation)(game, config.training, config.runtime)
    revision = code_revision()
    _write_run_config(run_directory / "run_config.json", config, revision)
    write_json(
        run_directory / "environment.json",
        environment_record("torch", "numpy", "psutil", device=config.runtime.device),
    )
    _append_training_log(run_directory, "training started at iteration 0")
    return _execute_schedule(
        config=config,
        solver=solver,
        run_directory=run_directory,
        result_store=DeepCFRMetricStore(run_directory / "metrics.csv"),
        elapsed_training_seconds=0.0,
        revision=revision,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )


def resume_deep_cfr_training(
    checkpoint_path: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> DeepCFRTrainingOutcome:
    """Resume the configured Deep CFR implementation from a complete iteration."""
    run_directory = checkpoint_path.parent.parent
    config = _load_run_config(run_directory / "run_config.json")
    _apply_runtime(config.runtime)
    game = _game_context(config.training)
    # Reservoirs remain in host memory; reconstructed networks move to the configured device.
    loaded = load_deep_cfr_checkpoint(checkpoint_path, game, map_location="cpu")
    if config.run_id != loaded.metadata["run_id"]:
        raise ValueError("checkpoint run_id does not match run_config.json")
    if config.training != loaded.solver.config:
        raise ValueError("checkpoint training configuration does not match run_config.json")
    if config.runtime != loaded.solver.runtime:
        raise ValueError("checkpoint runtime configuration does not match run_config.json")
    if config.implementation is not deep_cfr_implementation(loaded.solver):
        raise ValueError("checkpoint implementation does not match run_config.json")
    resume_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(
        run_directory
        / f"resume_environment_iter_{loaded.solver.iteration}_{resume_timestamp}.json",
        environment_record("torch", "numpy", "psutil", device=config.runtime.device),
    )
    _append_training_log(
        run_directory,
        f"training resumed from iteration {loaded.solver.iteration}",
    )
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
        stop_requested=stop_requested,
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
    stop_requested: Callable[[], bool] | None,
) -> DeepCFRTrainingOutcome:
    """Run pending iterations and stop only at complete recovery boundaries."""
    total_iterations = config.training.iterations
    snapshot_paths: list[Path] = []
    iteration_store = DeepCFRIterationMetricStore(run_directory / "iteration_metrics.csv")
    iteration_store.retain_through(solver.iteration)
    if progress_callback is not None:
        progress_callback(solver.iteration, total_iterations)
    stopped = False
    while solver.iteration < total_iterations:
        iteration_started = perf_counter()
        if config.runtime.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        training_started = perf_counter()
        solver.train(1)
        elapsed_training_seconds += perf_counter() - training_started
        iteration = solver.iteration

        should_snapshot = (
            iteration in config.training.snapshot_iterations or iteration == total_iterations
        )
        should_checkpoint = iteration % config.checkpoint_interval == 0 or should_snapshot
        requested_stop = stop_requested is not None and stop_requested()
        should_checkpoint = should_checkpoint or requested_stop
        checkpoint_id = f"{config.run_id}_iter_{iteration}" if should_checkpoint else ""
        snapshot_id = ""
        if should_snapshot:
            network = (
                solver.final_strategy_network
                if iteration == total_iterations
                else solver.snapshot_networks[iteration]
            )
            if network is None:
                raise RuntimeError("scheduled Deep CFR strategy network was not trained")
            snapshot_id = f"{config.run_id}_iter_{iteration}"
            snapshot_path = run_directory / "strategy_snapshots" / f"{snapshot_id}.pt"
            export_deep_cfr_snapshot(
                snapshot_path,
                network=network,
                game=_solver_game_context(solver),
                config=config.training,
                implementation=config.implementation,
                snapshot_id=snapshot_id,
                iteration=iteration,
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
            _save_recovery_checkpoint(
                run_directory=run_directory,
                solver=solver,
                config=config,
                elapsed_training_seconds=elapsed_training_seconds,
                checkpoint_id=checkpoint_id,
                metric_records=result_store.records,
                revision=revision,
            )
        iteration_seconds = perf_counter() - iteration_started
        _record_iteration_diagnostics(
            store=iteration_store,
            solver=solver,
            config=config,
            run_directory=run_directory,
            snapshot_id=snapshot_id,
            elapsed_training_seconds=elapsed_training_seconds,
            iteration_seconds=iteration_seconds,
        )
        _append_training_log(
            run_directory,
            f"iteration {iteration}/{total_iterations} completed in {iteration_seconds:.6g}s",
        )
        if progress_callback is not None:
            progress_callback(solver.iteration, total_iterations)
        if requested_stop:
            stopped = True
            _append_training_log(
                run_directory,
                f"stop requested; recovery checkpoint saved at iteration {iteration}",
            )
            break

    latest_checkpoint = run_directory / "checkpoints" / "latest.pt"
    if not latest_checkpoint.exists():
        checkpoint_id = f"{config.run_id}_iter_{solver.iteration}"
        _save_recovery_checkpoint(
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
    if not stopped and solver.iteration == total_iterations:
        _append_training_log(run_directory, f"training completed at iteration {solver.iteration}")
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
    holdem_configuration = getattr(solver, "holdem_configuration", None)
    metrics = (
        None
        if isinstance(holdem_configuration, HoldemConfig)
        else evaluate_strategy(solver.tree, deep_cfr_policy(solver.tree, network))
    )
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
            "game": (GameId.HOLD_EM.value if metrics is None else GameId.LEDUC.value),
            "game_version": solver.config.game_configuration_id.value,
            "utility_unit": UtilityUnit.CHIP.value,
            "solver": deep_cfr_implementation(solver).value,
            "run_id": run_id,
            "strategy_snapshot_id": snapshot_id,
            "source_checkpoint_id": checkpoint_id,
            "iteration": solver.iteration,
            "seed": solver.config.seed,
            "elapsed_training_seconds": elapsed_training_seconds,
            "expected_value_player_zero": None if metrics is None else metrics.expected_values[0],
            "exploitability": None if metrics is None else metrics.exploitability,
            "nash_conv": None if metrics is None else metrics.nash_conv,
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


def _record_iteration_diagnostics(
    *,
    store: DeepCFRIterationMetricStore,
    solver: NaiveDeepCFR,
    config: DeepCFRRunConfig,
    run_directory: Path,
    snapshot_id: str,
    elapsed_training_seconds: float,
    iteration_seconds: float,
) -> None:
    """Persist loss, timing, reservoir, memory, GPU, and disk diagnostics."""
    player_metrics = tuple(
        _network_metric(solver.training_metrics, solver.iteration, "advantage", player)
        for player in (0, 1)
    )
    strategy_metrics = tuple(
        metric
        for metric in solver.training_metrics
        if metric.iteration == solver.iteration
        and metric.network_role == "strategy"
        and metric.player is None
    )
    if len(strategy_metrics) > 1:
        raise RuntimeError("Deep CFR strategy training metrics are ambiguous")
    strategy_metric = strategy_metrics[0] if strategy_metrics else None
    reservoirs = (*solver.advantage_reservoirs, solver.strategy_reservoir)
    samples_seen_values: list[int] = []
    for reservoir in reservoirs:
        value = reservoir.training_state()["samples_seen"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("Deep CFR reservoir sample count is invalid")
        samples_seen_values.append(value)
    samples_seen = tuple(samples_seen_values)
    phase_times = solver.recent_training_times
    if config.runtime.device == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_allocated = 0
        peak_reserved = 0
    game = (
        GameId.HOLD_EM.value
        if config.training.game_configuration_id is GameConfigurationId.MODIFIED_HULHE
        else GameId.LEDUC.value
    )
    store.upsert(
        {
            "game": game,
            "game_version": config.training.game_configuration_id.value,
            "solver": config.implementation.value,
            "run_id": config.run_id,
            "strategy_snapshot_id": snapshot_id,
            "iteration": solver.iteration,
            "seed": config.training.seed,
            "elapsed_training_seconds": elapsed_training_seconds,
            "iteration_seconds": iteration_seconds,
            "traversal_seconds": phase_times.traversal_seconds,
            "advantage_training_seconds": phase_times.advantage_training_seconds,
            "strategy_training_seconds": phase_times.strategy_training_seconds,
            "player_zero_advantage_training_loss": player_metrics[0].training_loss,
            "player_zero_advantage_validation_loss": player_metrics[0].validation_loss,
            "player_one_advantage_training_loss": player_metrics[1].training_loss,
            "player_one_advantage_validation_loss": player_metrics[1].validation_loss,
            "strategy_training_loss": (
                None if strategy_metric is None else strategy_metric.training_loss
            ),
            "strategy_validation_loss": (
                None if strategy_metric is None else strategy_metric.validation_loss
            ),
            "player_zero_advantage_samples_retained": len(reservoirs[0]),
            "player_zero_advantage_samples_seen": samples_seen[0],
            "player_one_advantage_samples_retained": len(reservoirs[1]),
            "player_one_advantage_samples_seen": samples_seen[1],
            "strategy_samples_retained": len(reservoirs[2]),
            "strategy_samples_seen": samples_seen[2],
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "free_disk_bytes": effective_storage_remaining_bytes(
                run_directory,
                config.runtime.storage_budget_bytes,
            ),
        }
    )


def _append_training_log(run_directory: Path, message: str) -> None:
    """Durably append one concise lifecycle event without buffering it in memory."""
    run_directory.mkdir(parents=True, exist_ok=True)
    with (run_directory / "train.log").open("a", encoding="utf-8") as log_file:
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        log_file.write(f"{timestamp} {message}\n")
        log_file.flush()
        os.fsync(log_file.fileno())


def _save_recovery_checkpoint(
    *,
    run_directory: Path,
    solver: NaiveDeepCFR,
    config: DeepCFRRunConfig,
    elapsed_training_seconds: float,
    checkpoint_id: str,
    metric_records: tuple[dict[str, str], ...],
    revision: str,
) -> None:
    """Save one recovery file, atomically repoint latest, and prune old files."""
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    _require_checkpoint_space(checkpoint_directory, solver)
    arguments = {
        "solver": solver,
        "run_id": config.run_id,
        "checkpoint_id": checkpoint_id,
        "code_revision": revision,
        "elapsed_training_seconds": elapsed_training_seconds,
        "metric_records": metric_records,
    }
    iteration_path = checkpoint_directory / f"iter_{solver.iteration}.pt"
    save_deep_cfr_checkpoint(iteration_path, **arguments)
    _replace_latest_symlink(iteration_path, checkpoint_directory / "latest.pt")
    _prune_recovery_checkpoints(checkpoint_directory, config.checkpoint_retention)


def _require_checkpoint_space(directory: Path, solver: NaiveDeepCFR) -> None:
    """Fail before serialisation when one atomic checkpoint cannot fit."""
    occupied_bytes = 0
    reservoirs = (*solver.advantage_reservoirs, solver.strategy_reservoir)
    for reservoir in reservoirs:
        arrays = getattr(reservoir, "arrays", None)
        if arrays is None:
            occupied_bytes += sum(len(repr(sample)) for sample in reservoir.samples)
            continue
        occupied_bytes += sum(array.nbytes for array in arrays)
        # Checkpoints widen packed uint32 iteration numbers to int64.
        occupied_bytes += len(reservoir) * 4
    network_bytes = sum(
        parameter.numel() * parameter.element_size()
        for network in (
            *solver.advantage_networks,
            *solver.snapshot_networks.values(),
            solver.final_strategy_network,
        )
        if network is not None
        for parameter in network.parameters()
    )
    estimated_bytes = occupied_bytes + network_bytes
    required_bytes = int(estimated_bytes * 1.1) + 64 * 1024 * 1024
    staging_directory = checkpoint_staging_directory()
    staging_directory.mkdir(parents=True, exist_ok=True)
    staging_free_bytes = psutil.disk_usage(str(staging_directory)).free
    if staging_free_bytes < required_bytes:
        raise RuntimeError(
            "insufficient local checkpoint staging space: "
            f"requires {required_bytes} bytes, found {staging_free_bytes}"
        )
    free_bytes = effective_storage_remaining_bytes(
        directory,
        solver.runtime.storage_budget_bytes,
    )
    if free_bytes < required_bytes:
        raise OSError(
            "insufficient free space for an atomic Deep CFR checkpoint: "
            f"need approximately {required_bytes:,} bytes, found {free_bytes:,}"
        )


def _replace_latest_symlink(source: Path, target: Path) -> None:
    """Atomically point latest.pt at one same-directory checkpoint."""
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.symlink_to(source.name)
        os.replace(temporary, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _prune_recovery_checkpoints(directory: Path, retention: int) -> None:
    """Retain only the newest bounded set of complete iteration checkpoints."""
    checkpoints: list[tuple[int, Path]] = []
    for path in directory.glob("iter_*.pt"):
        match = re.fullmatch(r"iter_(\d+)\.pt", path.name)
        if match is not None and path.is_file() and not path.is_symlink():
            checkpoints.append((int(match.group(1)), path))
    for _, path in sorted(checkpoints)[:-retention]:
        path.unlink()


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
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")


def _game_context(config: DeepCFRTrainingConfig) -> IndexedGameTree | HoldemConfig:
    """Construct the game representation selected by immutable training metadata."""
    if config.game_configuration_id is GameConfigurationId.LEDUC:
        return compile_game_tree(LeducGame(), LeducConfig())
    if config.game_configuration_id is GameConfigurationId.MODIFIED_HULHE:
        return HoldemConfig.modified()
    raise ValueError("Deep CFR game configuration is unsupported")


def _solver_game_context(solver: NaiveDeepCFR) -> IndexedGameTree | HoldemConfig:
    """Return the exact compatibility context used by a live solver."""
    holdem_configuration = getattr(solver, "holdem_configuration", None)
    return holdem_configuration if isinstance(holdem_configuration, HoldemConfig) else solver.tree


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
        with atomic_text_writer(outcome.run_directory / "summary.txt") as summary_file:
            summary_file.write(f"Run: {outcome.run_directory.name}\n")
            summary_file.write(f"Completed iterations: {outcome.final_iteration:,}\n")
            summary_file.write(
                "Latest recovery checkpoint: "
                f"{outcome.latest_checkpoint.relative_to(outcome.run_directory)}\n"
            )
            summary_file.write("Strategy snapshot: not yet scheduled\n")
        return
    record = max(store.records, key=lambda value: int(value["iteration"]))
    common_lines = (
        f"Run: {record['run_id']}",
        f"Game: {record['game_version']}",
        f"Solver: Deep CFR ({record['solver']})",
        f"Iterations: {int(record['iteration']):,}",
        f"Traversals: {int(record['traversals']):,}",
        f"Solver training time: {float(record['elapsed_training_seconds']):.6g} seconds",
        "Latest recovery checkpoint: "
        f"{outcome.latest_checkpoint.relative_to(outcome.run_directory)}",
        f"Latest strategy snapshot: strategy_snapshots/{record['strategy_snapshot_id']}.pt",
    )
    lines = common_lines
    if record["exploitability"]:
        exact_lines = (
            "Player 0 average-policy value: "
            f"{float(record['expected_value_player_zero']):.12g} chips",
            f"Exact exploitability: {float(record['exploitability']):.12g} chips",
            f"NashConv: {float(record['nash_conv']):.12g} chips",
        )
        lines = (*common_lines[:5], *exact_lines, *common_lines[5:])
    with atomic_text_writer(outcome.run_directory / "summary.txt") as summary_file:
        summary_file.write("\n".join(lines))
        summary_file.write("\n")
