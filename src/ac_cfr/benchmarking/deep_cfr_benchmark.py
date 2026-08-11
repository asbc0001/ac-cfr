"""Fixed-workload reference-versus-optimised Deep CFR benchmark."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise
from math import isfinite
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Final

import psutil
import torch

from ac_cfr.benchmarking.harness import (
    environment_record,
    median_absolute_deviation,
    preferred_memory_metric,
    process_tree_memory_bytes,
    report_progress,
)
from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.evaluation.plotting import (
    plot_deep_cfr_implementation_convergence,
    plot_deep_cfr_performance,
)
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.persistence.deep_cfr_snapshots import deep_cfr_policy
from ac_cfr.persistence.files import write_csv, write_json
from ac_cfr.solvers import DeepCFR, NaiveDeepCFR
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig

BENCHMARK_ID = "deep_cfr"
BENCHMARK_ITERATIONS = 4
TRAVERSALS_PER_PLAYER = 1_250
ADVANTAGE_TRAINING_STEPS = 10
STRATEGY_TRAINING_STEPS = 20
BENCHMARK_SEED = 20260811
REPEATS = 3
MEMORY_SAMPLING_INTERVAL_SECONDS = 0.01
PYTORCH_INTRAOP_THREADS = 1
PYTORCH_INTEROP_THREADS = 1
CONVERGENCE_MILESTONES: Final = (1, 2, 4)
FINAL_EXPLOITABILITY_TOLERANCE = 0.1

_IMPLEMENTATIONS: Final = (
    ("reference", NaiveDeepCFR),
    ("optimised", DeepCFR),
)
_RUN_FIELDS: Final = (
    "benchmark_id",
    "game",
    "implementation",
    "repeat",
    "seed",
    "outer_iterations",
    "traversals",
    "optimizer_steps",
    "seconds",
    "traversal_seconds",
    "advantage_training_seconds",
    "strategy_training_seconds",
    "other_seconds",
    "end_to_end_traversals_per_second",
    "collection_traversals_per_second",
    "memory_metric",
    "memory_sampling_interval_seconds",
    "peak_memory_mb",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
)
_SUMMARY_FIELDS: Final = (
    "benchmark_id",
    "game",
    "implementation",
    "seed",
    "outer_iterations",
    "traversals",
    "optimizer_steps",
    "repeats",
    "median_seconds",
    "median_absolute_deviation_seconds",
    "median_traversal_seconds",
    "median_advantage_training_seconds",
    "median_strategy_training_seconds",
    "median_other_seconds",
    "end_to_end_traversals_per_second",
    "median_absolute_deviation_end_to_end_traversals_per_second",
    "collection_traversals_per_second",
    "median_absolute_deviation_collection_traversals_per_second",
    "memory_metric",
    "memory_sampling_interval_seconds",
    "median_peak_memory_mb",
    "median_absolute_deviation_memory_mb",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
)
_CONVERGENCE_FIELDS: Final = (
    "benchmark_id",
    "game",
    "implementation",
    "seed",
    "iteration",
    "traversals",
    "optimizer_steps",
    "elapsed_training_seconds",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
)


@dataclass(frozen=True, slots=True)
class DeepCFRBenchmarkRepeat:
    """One isolated fixed-workload timing and evaluation."""

    repeat: int
    seconds: float
    traversal_seconds: float
    advantage_training_seconds: float
    strategy_training_seconds: float
    peak_memory_mb: float
    expected_value_player_zero: float
    exploitability: float
    nash_conv: float

    @property
    def other_seconds(self) -> float:
        """Return measured time outside the three principal phases."""
        measured = (
            self.traversal_seconds
            + self.advantage_training_seconds
            + self.strategy_training_seconds
        )
        return max(0.0, self.seconds - measured)


@dataclass(frozen=True, slots=True)
class DeepCFRBenchmarkResult:
    """Repeated measurements for one Deep CFR implementation."""

    implementation: str
    memory_metric: str
    repeats: tuple[DeepCFRBenchmarkRepeat, ...]


@dataclass(frozen=True, slots=True)
class DeepCFRConvergencePoint:
    """One exact milestone measurement from the matched comparison."""

    iteration: int
    elapsed_training_seconds: float
    expected_value_player_zero: float
    exploitability: float
    nash_conv: float


def run_deep_cfr_benchmark(
    output_directory: Path = Path("results") / BENCHMARK_ID,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run the declared comparison and write compact machine-readable evidence."""
    measurements: dict[str, list[DeepCFRBenchmarkRepeat]] = {
        implementation: [] for implementation, _ in _IMPLEMENTATIONS
    }
    memory_metric = preferred_memory_metric()
    for repeat in range(1, REPEATS + 1):
        ordered_implementations = (
            _IMPLEMENTATIONS if repeat % 2 == 1 else tuple(reversed(_IMPLEMENTATIONS))
        )
        for implementation, solver_type in ordered_implementations:
            report_progress(
                progress_callback,
                f"benchmark: repeat {repeat}/{REPEATS}, {implementation} Deep CFR",
            )
            measurements[implementation].append(
                _run_isolated_repeat(solver_type, repeat, memory_metric)
            )
    results = [
        DeepCFRBenchmarkResult(
            implementation,
            memory_metric,
            tuple(measurements[implementation]),
        )
        for implementation, _ in _IMPLEMENTATIONS
    ]

    output_directory.mkdir(parents=True, exist_ok=True)
    runs_path = output_directory / "benchmark_runs.csv"
    summary_path = output_directory / "benchmark_summary.csv"
    _write_results(runs_path, summary_path, results)
    plot_path = output_directory / "plots" / "implementation_performance.png"
    plot_deep_cfr_performance(summary_path, plot_path)
    checks = _benchmark_checks(results)
    benchmark_path = output_directory / "benchmark.json"
    write_json(
        benchmark_path,
        {
            "about": (
                "Machine-readable configuration, environment, checks, and file index for the "
                "fixed-workload Leduc Deep CFR implementation benchmark. This is not a "
                "convergence experiment."
            ),
            "benchmark_id": BENCHMARK_ID,
            "passed": all(bool(check["passed"]) for check in checks),
            "configuration": {
                **_benchmark_config().to_dict(),
                "game": "leduc",
                "implementations": [name for name, _ in _IMPLEMENTATIONS],
                "total_traversals": _total_traversals(),
                "total_optimizer_steps": _total_optimizer_steps(),
                "repeats": REPEATS,
                "execution_order": [
                    "reference, optimised",
                    "optimised, reference",
                    "reference, optimised",
                ],
                "timed_region": "solver.train only",
                "warm_up_before_timing": True,
                "evaluation_outside_timing": True,
                "early_stopping": False,
                "memory_sampling_interval_seconds": MEMORY_SAMPLING_INTERVAL_SECONDS,
                "pytorch_intraop_threads": PYTORCH_INTRAOP_THREADS,
                "pytorch_interop_threads": PYTORCH_INTEROP_THREADS,
            },
            "metric_definitions": {
                "traversal": "One sampled root traversal for one traversing player.",
                "optimizer_step": "One complete neural-network minibatch update.",
                "end_to_end_traversals_per_second": (
                    "Fixed-workload traversals divided by complete solver training time."
                ),
                "collection_traversals_per_second": (
                    "Fixed-workload traversals divided only by measured traversal collection time."
                ),
                "median_absolute_deviation": (
                    "Median distance from the median, used as a robust variation measure."
                ),
                "peak_memory": (
                    "Highest sampled memory total across the training process and its children."
                ),
            },
            "environment": environment_record("torch", "numpy", "numba", "psutil", device="cpu"),
            "checks": checks,
            "files": {
                "runs": runs_path.name,
                "summary": summary_path.name,
                "plot": str(plot_path.relative_to(output_directory)),
            },
            "file_descriptions": {
                "runs": "Every isolated timing, phase breakdown, memory peak, and exact metric.",
                "summary": "Median performance, variation, memory, and exact final quality.",
                "plot": "Total time, phase timing, traversal throughput, and peak memory.",
            },
        },
    )
    if not all(bool(check["passed"]) for check in checks):
        raise RuntimeError(f"Deep CFR benchmark failed; see {benchmark_path}")
    return benchmark_path


def run_deep_cfr_convergence_comparison(
    output_directory: Path = Path("results") / BENCHMARK_ID,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Compare exact reference and optimised learning at matched milestones."""
    results: dict[str, tuple[DeepCFRConvergencePoint, ...]] = {}
    for implementation, solver_type in _IMPLEMENTATIONS:
        report_progress(progress_callback, f"convergence: {implementation} Deep CFR")
        results[implementation] = _run_isolated_convergence(solver_type)

    records = _convergence_records(results)
    checks = _convergence_checks(results)
    output_directory.mkdir(parents=True, exist_ok=True)
    convergence_path = output_directory / "implementation_convergence.csv"
    plot_path = output_directory / "plots" / "implementation_convergence.png"
    write_csv(convergence_path, _CONVERGENCE_FIELDS, records)
    plot_deep_cfr_implementation_convergence(convergence_path, plot_path)
    comparison_path = output_directory / "comparison.json"
    write_json(
        comparison_path,
        {
            "about": (
                "Machine-readable configuration and checks for matched reference-versus-"
                "optimised Deep CFR convergence on Leduc."
            ),
            "benchmark_id": BENCHMARK_ID,
            "passed": all(bool(check["passed"]) for check in checks),
            "configuration": {
                **_convergence_config().to_dict(),
                "game": "leduc",
                "implementations": [name for name, _ in _IMPLEMENTATIONS],
                "milestones": list(CONVERGENCE_MILESTONES),
                "total_traversals": _total_traversals(),
                "total_optimizer_steps": _optimizer_steps_at_milestone(CONVERGENCE_MILESTONES[-1]),
                "final_exploitability_tolerance": FINAL_EXPLOITABILITY_TOLERANCE,
                "evaluation_outside_training_time": True,
                "early_stopping": False,
                "pytorch_intraop_threads": PYTORCH_INTRAOP_THREADS,
                "pytorch_interop_threads": PYTORCH_INTEROP_THREADS,
            },
            "environment": environment_record("torch", "numpy", "numba", "psutil", device="cpu"),
            "checks": checks,
            "files": {
                "convergence": convergence_path.name,
                "plot": str(plot_path.relative_to(output_directory)),
            },
            "file_descriptions": {
                "convergence": "Exact metrics at every matched implementation milestone.",
                "plot": "Reference and optimised exploitability by iterations and training time.",
            },
        },
    )
    if not all(bool(check["passed"]) for check in checks):
        raise RuntimeError(f"Deep CFR convergence comparison failed; see {comparison_path}")
    return comparison_path


def _run_isolated_repeat(
    solver_type: type[NaiveDeepCFR],
    repeat: int,
    memory_metric: str,
) -> DeepCFRBenchmarkRepeat:
    """Measure one warmed worker while sampling its process-tree memory."""
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(target=_benchmark_worker, args=(child_connection, solver_type))
    process.start()
    child_connection.close()
    try:
        _expect_message(parent_connection, "ready")
        measured_process = psutil.Process(process.pid)
        parent_connection.send("start")
        peak_memory_bytes = 0
        while not parent_connection.poll(MEMORY_SAMPLING_INTERVAL_SECONDS):
            peak_memory_bytes = max(
                peak_memory_bytes,
                process_tree_memory_bytes(measured_process, memory_metric),
            )
            if not process.is_alive():
                raise RuntimeError("Deep CFR benchmark worker stopped before returning a result")
        peak_memory_bytes = max(
            peak_memory_bytes,
            process_tree_memory_bytes(measured_process, memory_metric),
        )
        trained_message = parent_connection.recv()
        if trained_message[0] == "error":
            raise RuntimeError(f"Deep CFR benchmark worker failed: {trained_message[1]}")
        if trained_message[0] != "trained" or len(trained_message) != 5:
            raise RuntimeError("Deep CFR benchmark worker returned an invalid training result")
        training_values = tuple(float(value) for value in trained_message[1:])

        parent_connection.send("evaluate")
        result_message = parent_connection.recv()
        if result_message[0] == "error":
            raise RuntimeError(f"Deep CFR benchmark worker failed: {result_message[1]}")
        if result_message[0] != "result" or len(result_message) != 4:
            raise RuntimeError("Deep CFR benchmark worker returned an invalid evaluation result")
        strategy_values = tuple(float(value) for value in result_message[1:])
    finally:
        parent_connection.close()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Deep CFR benchmark worker exited with status {process.exitcode}")
    return DeepCFRBenchmarkRepeat(
        repeat=repeat,
        seconds=training_values[0],
        traversal_seconds=training_values[1],
        advantage_training_seconds=training_values[2],
        strategy_training_seconds=training_values[3],
        peak_memory_mb=peak_memory_bytes / (1024 * 1024),
        expected_value_player_zero=strategy_values[0],
        exploitability=strategy_values[1],
        nash_conv=strategy_values[2],
    )


def _run_isolated_convergence(
    solver_type: type[NaiveDeepCFR],
) -> tuple[DeepCFRConvergencePoint, ...]:
    """Run one warmed matched comparison in a fresh process."""
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(target=_convergence_worker, args=(child_connection, solver_type))
    process.start()
    child_connection.close()
    try:
        message = parent_connection.recv()
        if message[0] == "error":
            raise RuntimeError(f"Deep CFR convergence worker failed: {message[1]}")
        if message[0] != "result":
            raise RuntimeError("Deep CFR convergence worker returned an invalid result")
        points = tuple(DeepCFRConvergencePoint(*values) for values in message[1])
    finally:
        parent_connection.close()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Deep CFR convergence worker exited with status {process.exitcode}")
    return points


def _convergence_worker(connection: Connection, solver_type: type[NaiveDeepCFR]) -> None:
    """Train and exactly evaluate one implementation at declared milestones."""
    try:
        torch.set_num_threads(PYTORCH_INTRAOP_THREADS)
        torch.set_num_interop_threads(PYTORCH_INTEROP_THREADS)
        tree = compile_game_tree(LeducGame(), LeducConfig())
        _warm_up_solver(solver_type, tree)
        solver = solver_type(tree, _convergence_config(), _runtime_config())
        elapsed_training_seconds = 0.0
        points: list[tuple[int, float, float, float, float]] = []
        for iteration in range(1, CONVERGENCE_MILESTONES[-1] + 1):
            started = perf_counter()
            solver.train(1)
            elapsed_training_seconds += perf_counter() - started
            if iteration not in CONVERGENCE_MILESTONES:
                continue
            network = (
                solver.final_strategy_network
                if iteration == CONVERGENCE_MILESTONES[-1]
                else solver.snapshot_networks[iteration]
            )
            if network is None:
                raise RuntimeError("Deep CFR convergence milestone network is missing")
            metrics = evaluate_strategy(tree, deep_cfr_policy(tree, network))
            points.append(
                (
                    iteration,
                    elapsed_training_seconds,
                    metrics.expected_values[0],
                    metrics.exploitability,
                    metrics.nash_conv,
                )
            )
        connection.send(("result", points))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _benchmark_worker(connection: Connection, solver_type: type[NaiveDeepCFR]) -> None:
    """Warm, train, and evaluate one solver under parent-process timing control."""
    try:
        torch.set_num_threads(PYTORCH_INTRAOP_THREADS)
        torch.set_num_interop_threads(PYTORCH_INTEROP_THREADS)
        tree = compile_game_tree(LeducGame(), LeducConfig())
        _warm_up_solver(solver_type, tree)
        solver = solver_type(tree, _benchmark_config(), _runtime_config())
        connection.send(("ready",))
        if connection.recv() != "start":
            raise RuntimeError("Deep CFR benchmark worker did not receive the start command")
        started = perf_counter()
        solver.train(BENCHMARK_ITERATIONS)
        elapsed_seconds = perf_counter() - started
        phase_times = solver.recent_training_times
        connection.send(
            (
                "trained",
                elapsed_seconds,
                phase_times.traversal_seconds,
                phase_times.advantage_training_seconds,
                phase_times.strategy_training_seconds,
            )
        )
        if connection.recv() != "evaluate":
            raise RuntimeError("Deep CFR benchmark worker did not receive the evaluation command")

        network = solver.final_strategy_network
        if network is None:
            raise RuntimeError("Deep CFR benchmark did not train its final strategy network")
        metrics = evaluate_strategy(tree, deep_cfr_policy(tree, network))
        connection.send(
            (
                "result",
                metrics.expected_values[0],
                metrics.exploitability,
                metrics.nash_conv,
            )
        )
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _warm_up_solver(solver_type: type[NaiveDeepCFR], tree: IndexedGameTree) -> None:
    """Exercise PyTorch and optional compiled paths before formal timing."""
    config = DeepCFRTrainingConfig(
        iterations=1,
        traversals_per_player=16,
        advantage_reservoir_capacity=1_000,
        strategy_reservoir_capacity=1_000,
        advantage_training_steps=2,
        strategy_training_steps=2,
        training_batch_size=32,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=BENCHMARK_SEED,
    )
    solver_type(tree, config, _runtime_config(inference_batch_size=32)).train(1)


def _benchmark_config() -> DeepCFRTrainingConfig:
    """Return the immutable algorithmic workload shared by both implementations."""
    return DeepCFRTrainingConfig(
        iterations=BENCHMARK_ITERATIONS,
        traversals_per_player=TRAVERSALS_PER_PLAYER,
        advantage_reservoir_capacity=100_000,
        strategy_reservoir_capacity=100_000,
        advantage_training_steps=ADVANTAGE_TRAINING_STEPS,
        strategy_training_steps=STRATEGY_TRAINING_STEPS,
        training_batch_size=512,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=BENCHMARK_SEED,
    )


def _convergence_config() -> DeepCFRTrainingConfig:
    """Return the matched configuration used for trajectory comparison."""
    return replace(
        _benchmark_config(),
        snapshot_iterations=CONVERGENCE_MILESTONES[:-1],
    )


def _runtime_config(*, inference_batch_size: int = 512) -> DeepCFRRuntimeConfig:
    """Return the execution settings shared by matched benchmark runs."""
    return DeepCFRRuntimeConfig(
        inference_batch_size=inference_batch_size,
        cpu_threads=PYTORCH_INTRAOP_THREADS,
        device="cpu",
    )


def _total_traversals() -> int:
    """Return the sampled traversal count in one implementation repeat."""
    return 2 * BENCHMARK_ITERATIONS * TRAVERSALS_PER_PLAYER


def _total_optimizer_steps() -> int:
    """Return the neural minibatch-update count in one implementation repeat."""
    return 2 * BENCHMARK_ITERATIONS * ADVANTAGE_TRAINING_STEPS + STRATEGY_TRAINING_STEPS


def _optimizer_steps_at_milestone(iteration: int) -> int:
    """Return cumulative neural updates through one comparison milestone."""
    advantage_steps = 2 * iteration * ADVANTAGE_TRAINING_STEPS
    strategy_steps = sum(
        STRATEGY_TRAINING_STEPS for milestone in CONVERGENCE_MILESTONES if milestone <= iteration
    )
    return advantage_steps + strategy_steps


def _convergence_records(
    results: dict[str, tuple[DeepCFRConvergencePoint, ...]],
) -> list[dict[str, object]]:
    """Convert matched milestones into stable CSV records."""
    records: list[dict[str, object]] = []
    for implementation, _ in _IMPLEMENTATIONS:
        for point in results[implementation]:
            records.append(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "game": "leduc",
                    "implementation": implementation,
                    "seed": BENCHMARK_SEED,
                    "iteration": point.iteration,
                    "traversals": 2 * point.iteration * TRAVERSALS_PER_PLAYER,
                    "optimizer_steps": _optimizer_steps_at_milestone(point.iteration),
                    "elapsed_training_seconds": point.elapsed_training_seconds,
                    "expected_value_player_zero": point.expected_value_player_zero,
                    "exploitability": point.exploitability,
                    "nash_conv": point.nash_conv,
                }
            )
    return records


def _convergence_checks(
    results: dict[str, tuple[DeepCFRConvergencePoint, ...]],
) -> list[dict[str, object]]:
    """Check completeness, finite values, improvement, and final agreement."""
    complete = set(results) == {name for name, _ in _IMPLEMENTATIONS} and all(
        tuple(point.iteration for point in points) == CONVERGENCE_MILESTONES
        for points in results.values()
    )
    finite = complete and all(
        all(
            isfinite(value)
            for value in (
                point.elapsed_training_seconds,
                point.expected_value_player_zero,
                point.exploitability,
                point.nash_conv,
            )
        )
        and point.elapsed_training_seconds > 0.0
        and point.exploitability >= 0.0
        and point.nash_conv >= 0.0
        for points in results.values()
        for point in points
    )
    increasing_times = complete and all(
        all(
            later.elapsed_training_seconds > earlier.elapsed_training_seconds
            for earlier, later in pairwise(points)
        )
        for points in results.values()
    )
    improving = complete and all(
        points[-1].exploitability < points[0].exploitability for points in results.values()
    )
    final_difference = (
        abs(results["reference"][-1].exploitability - results["optimised"][-1].exploitability)
        if complete
        else float("inf")
    )
    return [
        {"name": "matched_milestones_completed", "passed": complete},
        {
            "name": "metrics_are_finite_and_training_time_increases",
            "passed": finite and increasing_times,
        },
        {"name": "both_implementations_reduce_exploitability", "passed": improving},
        {
            "name": "final_exploitabilities_agree_within_0_1_chips",
            "passed": final_difference <= FINAL_EXPLOITABILITY_TOLERANCE,
            "absolute_difference": final_difference,
            "absolute_tolerance": FINAL_EXPLOITABILITY_TOLERANCE,
        },
    ]


def _write_results(
    runs_path: Path,
    summary_path: Path,
    results: list[DeepCFRBenchmarkResult],
) -> None:
    """Write every repetition and one robust summary per implementation."""
    run_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for result in results:
        common = {
            "benchmark_id": BENCHMARK_ID,
            "game": "leduc",
            "implementation": result.implementation,
            "seed": BENCHMARK_SEED,
            "outer_iterations": BENCHMARK_ITERATIONS,
            "traversals": _total_traversals(),
            "optimizer_steps": _total_optimizer_steps(),
        }
        for measurement in result.repeats:
            run_records.append(
                {
                    **common,
                    "repeat": measurement.repeat,
                    "seconds": measurement.seconds,
                    "traversal_seconds": measurement.traversal_seconds,
                    "advantage_training_seconds": measurement.advantage_training_seconds,
                    "strategy_training_seconds": measurement.strategy_training_seconds,
                    "other_seconds": measurement.other_seconds,
                    "end_to_end_traversals_per_second": (_total_traversals() / measurement.seconds),
                    "collection_traversals_per_second": (
                        _total_traversals() / measurement.traversal_seconds
                    ),
                    "memory_metric": result.memory_metric,
                    "memory_sampling_interval_seconds": MEMORY_SAMPLING_INTERVAL_SECONDS,
                    "peak_memory_mb": measurement.peak_memory_mb,
                    "expected_value_player_zero": measurement.expected_value_player_zero,
                    "exploitability": measurement.exploitability,
                    "nash_conv": measurement.nash_conv,
                }
            )
        summary_records.append(_summary_record(common, result))
    write_csv(runs_path, _RUN_FIELDS, run_records)
    write_csv(summary_path, _SUMMARY_FIELDS, summary_records)


def _summary_record(
    common: dict[str, object],
    result: DeepCFRBenchmarkResult,
) -> dict[str, object]:
    """Summarise one implementation without hiding run-to-run variation."""
    measurements = result.repeats
    total_times = tuple(item.seconds for item in measurements)
    traversal_times = tuple(item.traversal_seconds for item in measurements)
    advantage_times = tuple(item.advantage_training_seconds for item in measurements)
    strategy_times = tuple(item.strategy_training_seconds for item in measurements)
    other_times = tuple(item.other_seconds for item in measurements)
    end_to_end_throughput = tuple(_total_traversals() / value for value in total_times)
    collection_throughput = tuple(_total_traversals() / value for value in traversal_times)
    memory_values = tuple(item.peak_memory_mb for item in measurements)
    final = measurements[-1]
    return {
        **common,
        "repeats": len(measurements),
        "median_seconds": median(total_times),
        "median_absolute_deviation_seconds": median_absolute_deviation(total_times),
        "median_traversal_seconds": median(traversal_times),
        "median_advantage_training_seconds": median(advantage_times),
        "median_strategy_training_seconds": median(strategy_times),
        "median_other_seconds": median(other_times),
        "end_to_end_traversals_per_second": _total_traversals() / median(total_times),
        "median_absolute_deviation_end_to_end_traversals_per_second": (
            median_absolute_deviation(end_to_end_throughput)
        ),
        "collection_traversals_per_second": _total_traversals() / median(traversal_times),
        "median_absolute_deviation_collection_traversals_per_second": (
            median_absolute_deviation(collection_throughput)
        ),
        "memory_metric": result.memory_metric,
        "memory_sampling_interval_seconds": MEMORY_SAMPLING_INTERVAL_SECONDS,
        "median_peak_memory_mb": median(memory_values),
        "median_absolute_deviation_memory_mb": median_absolute_deviation(memory_values),
        "expected_value_player_zero": final.expected_value_player_zero,
        "exploitability": final.exploitability,
        "nash_conv": final.nash_conv,
    }


def _benchmark_checks(results: list[DeepCFRBenchmarkResult]) -> list[dict[str, object]]:
    """Validate work, timing, memory, and deterministic repeated quality metrics."""
    complete = tuple(result.implementation for result in results) == tuple(
        name for name, _ in _IMPLEMENTATIONS
    ) and all(len(result.repeats) == REPEATS for result in results)
    measurements = tuple(item for result in results for item in result.repeats)
    finite = all(
        all(
            isfinite(value) and value >= 0.0
            for value in (
                item.seconds,
                item.traversal_seconds,
                item.advantage_training_seconds,
                item.strategy_training_seconds,
                item.peak_memory_mb,
                item.exploitability,
                item.nash_conv,
            )
        )
        and item.seconds > 0.0
        and item.traversal_seconds > 0.0
        and item.peak_memory_mb > 0.0
        for item in measurements
    )
    deterministic_quality = all(
        len(
            {
                (
                    item.expected_value_player_zero,
                    item.exploitability,
                    item.nash_conv,
                )
                for item in result.repeats
            }
        )
        == 1
        for result in results
    )
    return [
        {
            "name": "identical_fixed_workload_completed",
            "passed": complete,
            "traversals_per_implementation_repeat": _total_traversals(),
            "optimizer_steps_per_implementation_repeat": _total_optimizer_steps(),
            "repeats": REPEATS,
        },
        {
            "name": "timing_memory_and_exact_metrics_are_finite",
            "passed": finite,
        },
        {
            "name": "fresh_process_repetitions_reproduce_each_strategy_metric",
            "passed": deterministic_quality,
        },
    ]


def _expect_message(connection: Connection, expected_kind: str) -> None:
    message = connection.recv()
    if message[0] == "error":
        raise RuntimeError(f"Deep CFR benchmark worker failed: {message[1]}")
    if message[0] != expected_kind:
        raise RuntimeError(f"Deep CFR benchmark worker did not report {expected_kind}")
