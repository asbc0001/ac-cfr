"""Restart-safe monitoring for a modified-HULHE production run."""

import argparse
import json
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ac_cfr.agents import RULE_BASED_AGENT_ID
from ac_cfr.cli.evaluate_hulhe import main as evaluate_hulhe_main
from ac_cfr.evaluation.holdem_h2h import PAIRED_BOOTSTRAP_METHOD
from ac_cfr.evaluation.hulhe_plotting import plot_modified_hulhe_h2h
from ac_cfr.persistence.results import HoldemH2HResultStore, ResultRecord
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig

_MONITORING_FIELDS = {
    "run_id",
    "representative_iterations",
    "duplicate_pairs",
    "seed",
    "confidence_level",
    "bootstrap_resamples",
    "poll_seconds",
    "settle_seconds",
    "device",
    "cpu_threads",
}


@dataclass(frozen=True, slots=True)
class HulheMonitoringConfig:
    """Frozen policy-evaluation protocol for one production run."""

    run_id: str
    representative_iterations: tuple[int, ...]
    duplicate_pairs: int
    seed: int
    confidence_level: float
    bootstrap_resamples: int
    poll_seconds: int
    settle_seconds: int
    device: str
    cpu_threads: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("monitoring run_id must not be empty")
        milestones = self.representative_iterations
        if not milestones or tuple(sorted(set(milestones))) != milestones:
            raise ValueError("representative_iterations must be unique and increasing")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in milestones
        ):
            raise ValueError("representative_iterations must contain positive integers")
        for name in ("duplicate_pairs", "bootstrap_resamples", "poll_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.settle_seconds, bool)
            or not isinstance(self.settle_seconds, int)
            or self.settle_seconds < 0
        ):
            raise ValueError("settle_seconds must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, float)
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be a float between zero and one")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if (
            isinstance(self.cpu_threads, bool)
            or not isinstance(self.cpu_threads, int)
            or self.cpu_threads < 1
        ):
            raise ValueError("cpu_threads must be a positive integer")


def load_hulhe_monitoring_config(path: Path) -> HulheMonitoringConfig:
    """Load one strict monitoring TOML file."""
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"unable to load monitoring configuration: {path}") from error
    if (
        set(values) != {"format_version", "monitoring"}
        or type(values["format_version"]) is not int
        or values["format_version"] != 1
    ):
        raise ValueError("monitoring configuration format is incompatible")
    monitoring = values["monitoring"]
    if not isinstance(monitoring, dict) or set(monitoring) != _MONITORING_FIELDS:
        raise ValueError("monitoring configuration fields are incompatible")
    milestones = monitoring["representative_iterations"]
    if not isinstance(milestones, list):
        raise ValueError("representative_iterations must be stored as a list")
    return HulheMonitoringConfig(
        run_id=monitoring["run_id"],
        representative_iterations=tuple(milestones),
        duplicate_pairs=monitoring["duplicate_pairs"],
        seed=monitoring["seed"],
        confidence_level=monitoring["confidence_level"],
        bootstrap_resamples=monitoring["bootstrap_resamples"],
        poll_seconds=monitoring["poll_seconds"],
        settle_seconds=monitoring["settle_seconds"],
        device=monitoring["device"],
        cpu_threads=monitoring["cpu_threads"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Watch one run and evaluate only its declared representative snapshots."""
    parser = argparse.ArgumentParser(
        description="Watch and evaluate representative modified-HULHE snapshots."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args(argv)
    config = load_hulhe_monitoring_config(arguments.config)
    torch.set_num_threads(config.cpu_threads)
    results_path = (
        arguments.results or arguments.run_directory / "evaluation" / "production_h2h.csv"
    )
    plot_path = arguments.plot or arguments.run_directory / "evaluation" / "production_h2h.png"
    return watch_hulhe_run(
        config,
        arguments.run_directory,
        results_path=results_path,
        plot_path=plot_path,
        once=arguments.once,
    )


def watch_hulhe_run(
    config: HulheMonitoringConfig,
    run_directory: Path,
    *,
    results_path: Path,
    plot_path: Path,
    once: bool = False,
) -> int:
    """Evaluate available milestones and wait until the final one is complete."""
    run_config_path = run_directory / "run_config.json"
    while True:
        if not run_config_path.exists():
            if once:
                return 0
            print(f"waiting for run configuration: {run_config_path}", flush=True)
            time.sleep(config.poll_seconds)
            continue

        run_config = _load_and_validate_run_config(run_config_path, config)
        completed_final = _evaluate_available_snapshots(
            config,
            run_config,
            run_directory,
            results_path=results_path,
            plot_path=plot_path,
        )
        if completed_final or once:
            return 0
        print("waiting for the next representative snapshot", flush=True)
        time.sleep(config.poll_seconds)


def _load_and_validate_run_config(
    path: Path,
    monitoring: HulheMonitoringConfig,
) -> DeepCFRRunConfig:
    """Load the immutable run record and verify the monitoring schedule against it."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load run configuration: {path}") from error
    if not isinstance(values, dict) or set(values) != {"code_revision", "run_config"}:
        raise ValueError("run configuration record is incompatible")
    run_config = DeepCFRRunConfig.from_dict(values["run_config"])
    if run_config.run_id != monitoring.run_id:
        raise ValueError("monitoring run_id does not match the training run")
    final_iteration = run_config.training.iterations
    if monitoring.representative_iterations[-1] != final_iteration:
        raise ValueError("final representative iteration must equal the training budget")
    generated_snapshots = {*run_config.training.snapshot_iterations, final_iteration}
    if not set(monitoring.representative_iterations) <= generated_snapshots:
        raise ValueError("monitoring requests an unscheduled strategy snapshot")
    return run_config


def _evaluate_available_snapshots(
    config: HulheMonitoringConfig,
    run_config: DeepCFRRunConfig,
    run_directory: Path,
    *,
    results_path: Path,
    plot_path: Path,
) -> bool:
    """Evaluate available milestones without repeating completed protocols."""
    snapshot_directory = run_directory / "strategy_snapshots"
    previous_path: Path | None = None
    final_complete = False
    changed = False
    for iteration in config.representative_iterations:
        snapshot_id = f"{config.run_id}_iter_{iteration}"
        snapshot_path = snapshot_directory / f"{snapshot_id}.pt"
        if not snapshot_path.exists():
            break
        _settle_after_snapshot(snapshot_path, config.settle_seconds)
        records = HoldemH2HResultStore(results_path).records
        common_arguments = _evaluation_arguments(config, snapshot_path, results_path)
        baseline_arguments = list(common_arguments)
        if not _has_result(records, config, snapshot_id, iteration, "uniform_random", ""):
            baseline_arguments.append("--include-random")
        if not _has_result(records, config, snapshot_id, iteration, RULE_BASED_AGENT_ID, ""):
            baseline_arguments.append("--include-rule-based")
        if len(baseline_arguments) != len(common_arguments):
            evaluate_hulhe_main(baseline_arguments)
            changed = True

        if previous_path is not None:
            previous_id = previous_path.stem
            records = HoldemH2HResultStore(results_path).records
            if not _has_result(
                records,
                config,
                snapshot_id,
                iteration,
                previous_id,
                previous_id,
            ):
                evaluate_hulhe_main([*common_arguments, "--anchor-snapshot", str(previous_path)])
                changed = True
        previous_path = snapshot_path
        final_complete = iteration == run_config.training.iterations

    if results_path.exists() and (changed or not plot_path.exists()):
        plot_modified_hulhe_h2h(results_path, plot_path)
        print(f"plot: {plot_path}", flush=True)
    return final_complete and _final_protocol_complete(config, results_path)


def _evaluation_arguments(
    config: HulheMonitoringConfig,
    snapshot_path: Path,
    results_path: Path,
) -> list[str]:
    """Build the fixed common CLI arguments for one focal snapshot."""
    return [
        "--snapshot",
        str(snapshot_path),
        "--duplicate-pairs",
        str(config.duplicate_pairs),
        "--seed",
        str(config.seed),
        "--confidence-level",
        str(config.confidence_level),
        "--bootstrap-resamples",
        str(config.bootstrap_resamples),
        "--results",
        str(results_path),
        "--device",
        config.device,
    ]


def _has_result(
    records: tuple[ResultRecord, ...],
    config: HulheMonitoringConfig,
    snapshot_id: str,
    iteration: int,
    opponent_id: str,
    opponent_snapshot_id: str,
) -> bool:
    """Return whether the complete frozen protocol is already recorded."""
    return any(
        record["run_id"] == config.run_id
        and record["strategy_snapshot_id"] == snapshot_id
        and int(record["iteration"]) == iteration
        and record["seed"] == str(config.seed)
        and record["opponent_id"] == opponent_id
        and record["opponent_snapshot_id"] == opponent_snapshot_id
        and record["paired_deals"] == str(config.duplicate_pairs)
        and float(record["confidence_level"]) == config.confidence_level
        and record["confidence_interval_method"] == PAIRED_BOOTSTRAP_METHOD
        and record["bootstrap_resamples"] == str(config.bootstrap_resamples)
        for record in records
    )


def _final_protocol_complete(config: HulheMonitoringConfig, results_path: Path) -> bool:
    """Return whether all declared comparisons exist for the final snapshot."""
    records = HoldemH2HResultStore(results_path).records
    final_iteration = config.representative_iterations[-1]
    final_id = f"{config.run_id}_iter_{final_iteration}"
    previous_iteration = config.representative_iterations[-2]
    previous_id = f"{config.run_id}_iter_{previous_iteration}"
    return all(
        _has_result(records, config, final_id, final_iteration, opponent_id, opponent_snapshot_id)
        for opponent_id, opponent_snapshot_id in (
            ("uniform_random", ""),
            (RULE_BASED_AGENT_ID, ""),
            (previous_id, previous_id),
        )
    )


def _settle_after_snapshot(path: Path, settle_seconds: int) -> None:
    """Delay only long enough to avoid competing with the next traversal phase."""
    remaining = settle_seconds - max(0.0, time.time() - path.stat().st_mtime)
    if remaining > 0.0:
        time.sleep(remaining)
