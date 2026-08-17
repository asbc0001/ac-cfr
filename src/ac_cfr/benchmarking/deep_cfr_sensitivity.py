"""Bounded one-factor configuration study for optimised Leduc Deep CFR."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Final

import torch

from ac_cfr.benchmarking.deep_cfr_benchmark import warm_up_deep_cfr_solver
from ac_cfr.benchmarking.harness import environment_record, report_progress
from ac_cfr.common.config import ModelConfigId
from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.evaluation.plotting import plot_deep_cfr_sensitivity
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
from ac_cfr.persistence.deep_cfr_snapshots import deep_cfr_policy
from ac_cfr.persistence.files import write_csv, write_json
from ac_cfr.solvers import DeepCFR, NetworkTrainingMetrics
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_config import load_deep_cfr_run_config

STUDY_ID = "deep_cfr_configuration"
BASELINE_PRESET = Path("configs/deep_cfr/leduc_baseline.toml")
LOWER_K_DIVISOR = 2
LOWER_TRAINING_STEPS_DIVISOR = 2
HIGHER_TRAINING_STEP_COUNTS: Final = (150, 200)

_FIELDS: Final = (
    "study_id",
    "case",
    "changed_factor",
    "iterations",
    "traversals_per_player",
    "advantage_training_steps",
    "strategy_training_steps",
    "model_config_id",
    "total_traversals",
    "total_optimizer_steps",
    "training_seconds",
    "traversal_seconds",
    "advantage_training_seconds",
    "strategy_training_seconds",
    "end_to_end_traversals_per_second",
    "collection_traversals_per_second",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
    "player_zero_advantage_training_loss",
    "player_zero_advantage_validation_loss",
    "player_one_advantage_training_loss",
    "player_one_advantage_validation_loss",
    "strategy_training_loss",
    "strategy_validation_loss",
    "packed_reservoir_bytes",
    "network_parameter_bytes",
)


@dataclass(frozen=True, slots=True)
class DeepCFRSensitivityCase:
    """One named configuration differing from the baseline in at most one factor."""

    name: str
    changed_factor: str
    config: DeepCFRTrainingConfig


def run_deep_cfr_sensitivity_study(
    output_directory: Path = Path("results/leduc_deep_cfr"),
    *,
    preset_path: Path = BASELINE_PRESET,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run the bounded optimised configurations and save comparable evidence."""
    resolved = load_deep_cfr_run_config(preset_path, run_id="sensitivity-study")
    baseline = resolved.training
    runtime = resolved.runtime
    cases = deep_cfr_sensitivity_cases(baseline)
    torch.set_num_threads(runtime.cpu_threads)
    tree = compile_game_tree(LeducGame(), LeducConfig())
    warm_up_deep_cfr_solver(DeepCFR, tree, runtime, seed=baseline.seed)

    records: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        report_progress(
            progress_callback,
            f"configuration {index}/{len(cases)}: {case.name}",
        )
        solver = DeepCFR(tree, case.config, runtime)
        started = perf_counter()
        solver.train(case.config.iterations)
        training_seconds = perf_counter() - started
        records.append(_record(case, solver, training_seconds))

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "configuration_study.csv"
    plot_path = output_directory / "plots" / "configuration_sensitivity.png"
    write_csv(csv_path, _FIELDS, records)
    plot_deep_cfr_sensitivity(csv_path, plot_path)
    metadata_path = output_directory / "configuration_study.json"
    write_json(
        metadata_path,
        {
            "about": (
                "One-factor optimised Leduc Deep CFR sensitivity study for selecting a "
                "reasonable compute-quality configuration; this is not exhaustive tuning."
            ),
            "study_id": STUDY_ID,
            "baseline_preset": str(preset_path),
            "runtime": runtime.to_dict(),
            "execution_order": [case.name for case in cases],
            "cases": [
                {
                    "name": case.name,
                    "changed_factor": case.changed_factor,
                    "training": case.config.to_dict(),
                }
                for case in cases
            ],
            "method": {
                "implementation": "optimised",
                "seed": baseline.seed,
                "one_factor_at_a_time": True,
                "repetitions": 1,
                "timing_interpretation": (
                    "Coarse phase-cost comparison only; formal repeated implementation "
                    "timings remain in benchmark_summary.csv."
                ),
                "warm_up_before_measurement": True,
                "timed_region": "solver.train only",
                "exact_evaluation_outside_timing": True,
                "memory_measures": [
                    "allocated packed reservoir arrays",
                    "trained network parameters",
                ],
            },
            "environment": environment_record("torch", "numpy", "numba", device=runtime.device),
            "files": {
                "measurements": csv_path.name,
                "plot": str(plot_path.relative_to(output_directory)),
            },
        },
    )
    return metadata_path


def deep_cfr_sensitivity_cases(
    baseline: DeepCFRTrainingConfig,
) -> tuple[DeepCFRSensitivityCase, ...]:
    """Return the baseline and declared one-factor alternatives."""
    if not isinstance(baseline, DeepCFRTrainingConfig):
        raise TypeError("baseline must be a DeepCFRTrainingConfig")
    if baseline.traversals_per_player < LOWER_K_DIVISOR:
        raise ValueError("baseline traversals are too small for the lower-K case")
    if baseline.advantage_training_steps < LOWER_TRAINING_STEPS_DIVISOR:
        raise ValueError("baseline training steps are too small for the reduced-budget case")
    return (
        DeepCFRSensitivityCase("baseline", "none", baseline),
        DeepCFRSensitivityCase(
            "lower_k",
            "traversals_per_player",
            replace(
                baseline,
                traversals_per_player=baseline.traversals_per_player // LOWER_K_DIVISOR,
            ),
        ),
        DeepCFRSensitivityCase(
            "lower_advantage_steps",
            "advantage_training_steps",
            replace(
                baseline,
                advantage_training_steps=(
                    baseline.advantage_training_steps // LOWER_TRAINING_STEPS_DIVISOR
                ),
            ),
        ),
        *(
            DeepCFRSensitivityCase(
                f"advantage_steps_{step_count}",
                "advantage_training_steps",
                replace(baseline, advantage_training_steps=step_count),
            )
            for step_count in HIGHER_TRAINING_STEP_COUNTS
        ),
        DeepCFRSensitivityCase(
            "smaller_network",
            "model_config_id",
            replace(baseline, model_config_id=ModelConfigId.LEDUC_DEEP_CFR_SMALL),
        ),
    )


def _record(
    case: DeepCFRSensitivityCase,
    solver: DeepCFR,
    training_seconds: float,
) -> dict[str, object]:
    """Collect final exact quality, phase timing, losses, throughput, and memory."""
    config = case.config
    network = solver.final_strategy_network
    if network is None:
        raise RuntimeError("sensitivity run did not train its final strategy network")
    exact = evaluate_strategy(solver.tree, deep_cfr_policy(solver.tree, network))
    times = solver.recent_training_times
    player_metrics = tuple(_network_metric(solver, "advantage", player) for player in (0, 1))
    strategy_metric = _network_metric(solver, "strategy", None)
    traversals = 2 * config.iterations * config.traversals_per_player
    optimizer_steps = (
        2 * config.iterations * config.advantage_training_steps + config.strategy_training_steps
    )
    networks = (*solver.advantage_networks, network)
    return {
        "study_id": STUDY_ID,
        "case": case.name,
        "changed_factor": case.changed_factor,
        "iterations": config.iterations,
        "traversals_per_player": config.traversals_per_player,
        "advantage_training_steps": config.advantage_training_steps,
        "strategy_training_steps": config.strategy_training_steps,
        "model_config_id": config.model_config_id.value,
        "total_traversals": traversals,
        "total_optimizer_steps": optimizer_steps,
        "training_seconds": training_seconds,
        "traversal_seconds": times.traversal_seconds,
        "advantage_training_seconds": times.advantage_training_seconds,
        "strategy_training_seconds": times.strategy_training_seconds,
        "end_to_end_traversals_per_second": traversals / training_seconds,
        "collection_traversals_per_second": traversals / times.traversal_seconds,
        "expected_value_player_zero": exact.expected_values[0],
        "exploitability": exact.exploitability,
        "nash_conv": exact.nash_conv,
        "player_zero_advantage_training_loss": player_metrics[0].training_loss,
        "player_zero_advantage_validation_loss": player_metrics[0].validation_loss,
        "player_one_advantage_training_loss": player_metrics[1].training_loss,
        "player_one_advantage_validation_loss": player_metrics[1].validation_loss,
        "strategy_training_loss": strategy_metric.training_loss,
        "strategy_validation_loss": strategy_metric.validation_loss,
        "packed_reservoir_bytes": sum(
            reservoir.resident_bytes
            for reservoir in (*solver.advantage_reservoirs, solver.strategy_reservoir)
        ),
        "network_parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for trained_network in networks
            if trained_network is not None
            for parameter in trained_network.parameters()
        ),
    }


def _network_metric(
    solver: DeepCFR,
    role: str,
    player: int | None,
) -> NetworkTrainingMetrics:
    """Return the final matching training metric for one network role."""
    matches = tuple(
        metric
        for metric in solver.training_metrics
        if metric.iteration == solver.iteration
        and metric.network_role == role
        and metric.player == player
    )
    if len(matches) != 1:
        raise RuntimeError("sensitivity run has incomplete network metrics")
    return matches[0]
