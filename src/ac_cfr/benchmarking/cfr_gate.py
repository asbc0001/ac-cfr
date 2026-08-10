"""Correctness and performance gate for CFR and CFR+."""

import cProfile
import csv
import io
import json
import platform
import pstats
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Final

import numpy as np
import psutil
from numpy.typing import NDArray

from ac_cfr.benchmarking.harness import BenchmarkResult, run_tabular_benchmark
from ac_cfr.evaluation.metrics import StrategyMetrics, evaluate_strategy
from ac_cfr.evaluation.plotting import plot_cfr_gate_results
from ac_cfr.evaluation.self_play import evaluate_duplicate_self_play
from ac_cfr.games.base import GameId
from ac_cfr.games.tabular import create_tabular_game
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.persistence.files import atomic_text_writer
from ac_cfr.solvers import CFR, CFRPlus, NaiveCFR, NaiveCFRPlus

BENCHMARK_ID = "cfr_cfr_plus"
REPEATS = 5
MEMORY_SAMPLING_INTERVAL_SECONDS = 0.01
AVERAGING_DELAY = 10
PROFILE_ITERATIONS = 100
DETERMINISTIC_MILESTONES: Final = (1, 3)
DETERMINISTIC_ABSOLUTE_TOLERANCE = 1e-12
SELF_PLAY_DUPLICATE_PAIRS = 20_000
SELF_PLAY_SEED = 20260810


@dataclass(frozen=True, slots=True)
class GateWorkload:
    """Frozen correctness and timing workload for one poker game."""

    game: str
    convergence_milestones: tuple[int, ...]
    benchmark_iterations: int
    exploitability_limit: float
    value_error_limit: float | None
    equivalence_gap_limit: float


WORKLOADS: Final = (
    GateWorkload(
        game="kuhn",
        convergence_milestones=(10, 100, 1_000, 10_000),
        benchmark_iterations=10_000,
        exploitability_limit=1e-3,
        value_error_limit=1e-4,
        equivalence_gap_limit=2e-4,
    ),
    GateWorkload(
        game="leduc",
        convergence_milestones=(10, 100, 1_000, 5_000),
        benchmark_iterations=5_000,
        exploitability_limit=5e-3,
        value_error_limit=None,
        equivalence_gap_limit=1e-3,
    ),
)

_ALGORITHMS: Final = (
    ("cfr", "naive_cfr", "cfr"),
    ("cfr_plus", "naive_cfr_plus", "cfr_plus"),
)

_CONVERGENCE_FIELDS: Final = (
    "benchmark_id",
    "game",
    "algorithm",
    "implementation",
    "solver",
    "iteration",
    "elapsed_training_seconds",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
)
_BENCHMARK_RUN_FIELDS: Final = (
    "benchmark_id",
    "game",
    "algorithm",
    "implementation",
    "solver",
    "iterations",
    "traversals",
    "repeat",
    "seconds",
    "traversals_per_second",
    "memory_metric",
    "memory_sampling_interval_seconds",
    "peak_memory_mb",
)
_BENCHMARK_SUMMARY_FIELDS: Final = (
    "benchmark_id",
    "game",
    "algorithm",
    "implementation",
    "solver",
    "iterations",
    "traversals",
    "repeats",
    "median_seconds",
    "median_absolute_deviation_seconds",
    "traversals_per_second",
    "median_absolute_deviation_traversals_per_second",
    "memory_metric",
    "memory_sampling_interval_seconds",
    "median_peak_memory_mb",
    "median_absolute_deviation_memory_mb",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
)


def run_cfr_gate(
    output_directory: Path = Path("results") / BENCHMARK_ID,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run the fixed CFR/CFR+ gate and return its machine-readable result index."""
    output_directory.mkdir(parents=True, exist_ok=True)
    convergence_records: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    final_metrics: dict[tuple[str, str, str], StrategyMetrics] = {}
    final_policies: dict[tuple[str, str, str], NDArray[np.float64]] = {}

    for workload in WORKLOADS:
        for algorithm, reference_solver, optimised_solver in _ALGORITHMS:
            _report_progress(
                progress_callback,
                f"correctness: {workload.game} {algorithm}",
            )
            pair_records, pair_checks, pair_metrics, pair_policies = _run_correctness_pair(
                workload,
                algorithm,
                reference_solver,
                optimised_solver,
            )
            convergence_records.extend(pair_records)
            checks.extend(pair_checks)
            final_metrics.update(pair_metrics)
            final_policies.update(pair_policies)

    checks.extend(_quality_checks(final_metrics))
    _report_progress(progress_callback, "correctness: duplicate-deal self-play")
    checks.extend(_self_play_checks(final_policies))
    convergence_path = output_directory / "convergence.csv"
    _write_csv(convergence_path, _CONVERGENCE_FIELDS, convergence_records)

    benchmark_results: list[tuple[str, str, BenchmarkResult]] = []
    for workload in WORKLOADS:
        for algorithm, reference_solver, optimised_solver in _ALGORITHMS:
            for implementation, solver in (
                ("reference", reference_solver),
                ("optimised", optimised_solver),
            ):
                _report_progress(
                    progress_callback,
                    f"benchmark: {workload.game} {implementation} {algorithm}",
                )
                benchmark_results.append(
                    (
                        algorithm,
                        implementation,
                        run_tabular_benchmark(
                            game=workload.game,
                            solver_id=solver,
                            iterations=workload.benchmark_iterations,
                            repeats=REPEATS,
                            averaging_delay=AVERAGING_DELAY if algorithm == "cfr_plus" else 0,
                            memory_sampling_interval_seconds=(MEMORY_SAMPLING_INTERVAL_SECONDS),
                        ),
                    )
                )

    benchmark_runs_path = output_directory / "benchmark_runs.csv"
    benchmark_summary_path = output_directory / "benchmark_summary.csv"
    _write_benchmark_results(benchmark_runs_path, benchmark_summary_path, benchmark_results)

    profile_directory = output_directory / "profiles"
    profile_paths: list[Path] = []
    for algorithm, reference_solver, optimised_solver in _ALGORITHMS:
        for implementation, solver in (
            ("reference", reference_solver),
            ("optimised", optimised_solver),
        ):
            _report_progress(progress_callback, f"profile: leduc {implementation} {algorithm}")
            profile_paths.append(
                _write_profile(
                    profile_directory,
                    solver_id=solver,
                    averaging_delay=AVERAGING_DELAY if algorithm == "cfr_plus" else 0,
                )
            )

    plot_directory = output_directory / "plots"
    plot_paths = plot_cfr_gate_results(
        convergence_path,
        benchmark_summary_path,
        plot_directory,
    )
    gate_passed = all(bool(check["passed"]) for check in checks)
    gate_record = {
        "about": (
            "Machine-readable configuration, environment, checks, and artefact index for the "
            "CFR/CFR+ correctness and performance gate."
        ),
        "benchmark_id": BENCHMARK_ID,
        "passed": gate_passed,
        "metric_definitions": {
            "expected_value_player_zero": (
                "Player zero's expected chip result under both average strategies."
            ),
            "exploitability": (
                "Average amount either player can gain by switching to an exact best response."
            ),
            "nash_conv": (
                "Sum of both players' gains from switching individually to exact best responses."
            ),
            "traversal": "One root-to-tree traversal for one traversing player.",
            "median_absolute_deviation": (
                "Median distance from the median, used as a robust measure of run variation."
            ),
            "peak_memory": (
                "Highest sampled memory total across the benchmark process and its children."
            ),
        },
        "configuration": {
            "repeats": REPEATS,
            "deterministic_algorithms": True,
            "random_seed": None,
            "early_stopping": False,
            "traversals_per_iteration": 2,
            "timed_region": "solver.train only",
            "memory_sampling_interval_seconds": MEMORY_SAMPLING_INTERVAL_SECONDS,
            "cfr_plus_averaging_delay": AVERAGING_DELAY,
            "profile_iterations": PROFILE_ITERATIONS,
            "deterministic_milestones": list(DETERMINISTIC_MILESTONES),
            "deterministic_absolute_tolerance": DETERMINISTIC_ABSOLUTE_TOLERANCE,
            "self_play_duplicate_pairs": SELF_PLAY_DUPLICATE_PAIRS,
            "self_play_hands_per_policy": 2 * SELF_PLAY_DUPLICATE_PAIRS,
            "self_play_seed": SELF_PLAY_SEED,
            "self_play_confidence_level": 0.99,
            "self_play_confidence_interval_method": "normal",
            "workloads": [asdict(workload) for workload in WORKLOADS],
        },
        "environment": _environment_record(),
        "checks": checks,
        "files": {
            "convergence": convergence_path.name,
            "benchmark_runs": benchmark_runs_path.name,
            "benchmark_summary": benchmark_summary_path.name,
            "profiles": [str(path.relative_to(output_directory)) for path in profile_paths],
            "plots": [str(path.relative_to(output_directory)) for path in plot_paths],
        },
        "file_descriptions": {
            "convergence": "Exact strategy-quality measurements at declared milestones.",
            "benchmark_runs": "Every individual timing and peak-memory repetition.",
            "benchmark_summary": "Median performance, variation, memory, and final quality.",
            "profiles": "Automatically generated cProfile CPU-time reports in Markdown.",
            "plots": "Convergence and fixed-workload engineering comparisons.",
        },
    }
    gate_path = output_directory / "gate.json"
    _write_json(gate_path, gate_record)
    if not gate_passed:
        raise RuntimeError(f"CFR/CFR+ gate failed; see {gate_path}")
    return gate_path


def _run_correctness_pair(
    workload: GateWorkload,
    algorithm: str,
    reference_solver_id: str,
    optimised_solver_id: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, str, str], StrategyMetrics],
    dict[tuple[str, str, str], NDArray[np.float64]],
]:
    """Compare one reference/optimised pair and collect milestone evidence."""
    tabular_game = create_tabular_game(GameId(workload.game))
    delay = AVERAGING_DELAY if algorithm == "cfr_plus" else 0
    _create_solver(tabular_game.tree, optimised_solver_id, delay).train(1)
    reference_solver = _create_reference_solver(
        tabular_game.tree,
        reference_solver_id,
        delay,
    )
    optimised_solver = _create_optimised_solver(
        tabular_game.tree,
        optimised_solver_id,
        delay,
    )
    solvers: dict[str, NaiveCFR | CFR] = {
        "reference": reference_solver,
        "optimised": optimised_solver,
    }
    elapsed = {"reference": 0.0, "optimised": 0.0}
    records: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    final_metrics: dict[tuple[str, str, str], StrategyMetrics] = {}
    final_policies: dict[tuple[str, str, str], NDArray[np.float64]] = {}
    milestones = sorted({*DETERMINISTIC_MILESTONES, *workload.convergence_milestones})
    previous_iteration = 0
    for milestone in milestones:
        increment = milestone - previous_iteration
        for implementation, solver in solvers.items():
            start_time = perf_counter()
            solver.train(increment)
            elapsed[implementation] += perf_counter() - start_time

        if milestone in DETERMINISTIC_MILESTONES:
            differences = _solver_differences(reference_solver, optimised_solver)
            checks.append(
                {
                    "name": f"{workload.game}_{algorithm}_iteration_{milestone}_updates",
                    "passed": max(differences.values()) <= DETERMINISTIC_ABSOLUTE_TOLERANCE,
                    "tolerance": DETERMINISTIC_ABSOLUTE_TOLERANCE,
                    **differences,
                }
            )

        if milestone in workload.convergence_milestones:
            for implementation, solver in solvers.items():
                metrics = evaluate_strategy(tabular_game.tree, solver.average_policy())
                solver_id = (
                    reference_solver_id if implementation == "reference" else optimised_solver_id
                )
                records.append(
                    {
                        "benchmark_id": BENCHMARK_ID,
                        "game": workload.game,
                        "algorithm": algorithm,
                        "implementation": implementation,
                        "solver": solver_id,
                        "iteration": milestone,
                        "elapsed_training_seconds": elapsed[implementation],
                        "expected_value_player_zero": metrics.expected_values[0],
                        "exploitability": metrics.exploitability,
                        "nash_conv": metrics.nash_conv,
                    }
                )
                if milestone == workload.convergence_milestones[-1]:
                    final_metrics[(workload.game, algorithm, implementation)] = metrics
                    final_policies[(workload.game, algorithm, implementation)] = (
                        solver.average_policy()
                    )
        previous_iteration = milestone
    return records, checks, final_metrics, final_policies


def _quality_checks(
    metrics: dict[tuple[str, str, str], StrategyMetrics],
) -> list[dict[str, object]]:
    """Evaluate final strategy quality and implementation-equivalence limits."""
    checks: list[dict[str, object]] = []
    for workload in WORKLOADS:
        for algorithm, _, _ in _ALGORITHMS:
            reference = metrics[(workload.game, algorithm, "reference")]
            optimised = metrics[(workload.game, algorithm, "optimised")]
            exploitability_gap = abs(reference.exploitability - optimised.exploitability)
            value_gap = abs(reference.expected_values[0] - optimised.expected_values[0])
            value_gap_limit = reference.nash_conv + optimised.nash_conv
            value_error_limit = workload.value_error_limit
            if value_error_limit is None:
                value_errors: tuple[float, ...] = ()
                known_value_passed = True
            else:
                value_errors = (
                    abs(reference.expected_values[0] + 1 / 18),
                    abs(optimised.expected_values[0] + 1 / 18),
                )
                known_value_passed = max(value_errors) <= value_error_limit
            passed = (
                max(reference.exploitability, optimised.exploitability)
                <= workload.exploitability_limit
                and exploitability_gap <= workload.equivalence_gap_limit
                and value_gap <= value_gap_limit
                and known_value_passed
            )
            checks.append(
                {
                    "name": f"{workload.game}_{algorithm}_strategy_quality",
                    "passed": passed,
                    "iteration": workload.convergence_milestones[-1],
                    "exploitability_limit": workload.exploitability_limit,
                    "reference_exploitability": reference.exploitability,
                    "optimised_exploitability": optimised.exploitability,
                    "equivalence_gap_limit": workload.equivalence_gap_limit,
                    "exploitability_gap": exploitability_gap,
                    "reference_player_zero_value": reference.expected_values[0],
                    "optimised_player_zero_value": optimised.expected_values[0],
                    "value_gap": value_gap,
                    "value_gap_limit": value_gap_limit,
                    "known_value_error_limit": workload.value_error_limit,
                    "known_value_errors": list(value_errors),
                }
            )
    return checks


def _self_play_checks(
    policies: dict[tuple[str, str, str], NDArray[np.float64]],
) -> list[dict[str, object]]:
    """Check that duplicate-deal self-play results remain statistically neutral."""
    checks: list[dict[str, object]] = []
    for workload in WORKLOADS:
        tree = create_tabular_game(GameId(workload.game)).tree
        for algorithm, _, _ in _ALGORITHMS:
            result = evaluate_duplicate_self_play(
                tree,
                policies[(workload.game, algorithm, "optimised")],
                duplicate_pairs=SELF_PLAY_DUPLICATE_PAIRS,
                seed=SELF_PLAY_SEED,
            )
            checks.append(
                {
                    "name": f"{workload.game}_{algorithm}_duplicate_self_play",
                    "passed": result.includes_zero,
                    "game": workload.game,
                    "algorithm": algorithm,
                    "implementation": "optimised",
                    "duplicate_pairs": result.duplicate_pairs,
                    "hands": 2 * result.duplicate_pairs,
                    "seed": result.seed,
                    "confidence_level": 0.99,
                    "confidence_interval_method": "normal",
                    "mean_chips": result.mean_chips,
                    "standard_error_chips": result.standard_error_chips,
                    "confidence_interval_low": result.confidence_interval_low,
                    "confidence_interval_high": result.confidence_interval_high,
                }
            )
    return checks


def _solver_differences(reference: NaiveCFR, optimised: CFR) -> dict[str, float]:
    """Return maximum table and policy differences between paired solvers."""
    return {
        "maximum_regret_difference": _maximum_difference(
            _flatten_table(reference.regret_sum),
            _flatten_table(optimised.regret_sum),
        ),
        "maximum_strategy_sum_difference": _maximum_difference(
            _flatten_table(reference.strategy_sum),
            _flatten_table(optimised.strategy_sum),
        ),
        "maximum_current_policy_difference": _maximum_difference(
            reference.current_policy(),
            optimised.current_policy(),
        ),
        "maximum_average_policy_difference": _maximum_difference(
            reference.average_policy(),
            optimised.average_policy(),
        ),
    }


def _write_benchmark_results(
    runs_path: Path,
    summary_path: Path,
    results: list[tuple[str, str, BenchmarkResult]],
) -> None:
    """Write individual benchmark runs and their robust summary statistics."""
    run_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for algorithm, implementation, result in results:
        common = {
            "benchmark_id": BENCHMARK_ID,
            "game": result.game,
            "algorithm": algorithm,
            "implementation": implementation,
            "solver": result.solver,
            "iterations": result.iterations,
            "traversals": result.traversals,
        }
        for repeat in result.repeat_results:
            run_records.append(
                {
                    **common,
                    "repeat": repeat.repeat,
                    "seconds": repeat.seconds,
                    "traversals_per_second": repeat.traversals_per_second,
                    "memory_metric": result.memory_metric,
                    "memory_sampling_interval_seconds": (result.memory_sampling_interval_seconds),
                    "peak_memory_mb": repeat.peak_memory_mb,
                }
            )
        throughput_values = tuple(repeat.traversals_per_second for repeat in result.repeat_results)
        summary_records.append(
            {
                **common,
                "repeats": result.repeats,
                "median_seconds": result.median_seconds,
                "median_absolute_deviation_seconds": (result.median_absolute_deviation_seconds),
                "traversals_per_second": result.traversals_per_second,
                "median_absolute_deviation_traversals_per_second": (
                    _median_absolute_deviation(throughput_values)
                ),
                "memory_metric": result.memory_metric,
                "memory_sampling_interval_seconds": (result.memory_sampling_interval_seconds),
                "median_peak_memory_mb": result.median_peak_memory_mb,
                "median_absolute_deviation_memory_mb": (result.median_absolute_deviation_memory_mb),
                "expected_value_player_zero": result.expected_value_player_zero,
                "exploitability": result.exploitability,
                "nash_conv": result.nash_conv,
            }
        )
    _write_csv(runs_path, _BENCHMARK_RUN_FIELDS, run_records)
    _write_csv(summary_path, _BENCHMARK_SUMMARY_FIELDS, summary_records)


def _write_profile(
    profile_directory: Path,
    *,
    solver_id: str,
    averaging_delay: int,
) -> Path:
    """Run a separate CPU profile and write its annotated Markdown output."""
    tabular_game = create_tabular_game(GameId.LEDUC)
    if solver_id in ("cfr", "cfr_plus"):
        _create_solver(tabular_game.tree, solver_id, averaging_delay).train(1)
    solver = _create_solver(tabular_game.tree, solver_id, averaging_delay)
    profiler = cProfile.Profile()
    profiler.runcall(solver.train, PROFILE_ITERATIONS)
    profile_text = io.StringIO()
    statistics = pstats.Stats(profiler, stream=profile_text).strip_dirs().sort_stats("cumulative")
    statistics.print_stats(25)
    path = profile_directory / f"leduc_{solver_id}.md"
    with atomic_text_writer(path) as profile_file:
        profile_file.write(
            f"# CPU profile: Leduc {solver_id}\n\n"
            "This file is generated automatically with Python's `cProfile`. It shows where "
            "the solver spent CPU time during a separate diagnostic run and is not part of "
            "the formal benchmark timing.\n\n"
            f"- **Solver:** `{solver_id}`\n"
            "- **Game:** Leduc\n"
            f"- **Iterations:** {PROFILE_ITERATIONS}\n"
            "- **Rows:** top 25 functions, sorted by cumulative time\n\n"
            "`ncalls` is the number of calls. `tottime` is time spent inside the function "
            "itself. `cumtime` includes functions it called. Each `percall` column divides "
            "the adjacent time by its relevant call count. Times are in seconds.\n\n"
            "## Raw cProfile output\n\n"
            "```text\n"
        )
        profile_file.write(profile_text.getvalue())
        profile_file.write("```\n")
    return path


def _environment_record() -> dict[str, object]:
    """Capture the software and hardware context needed to interpret results."""
    process = psutil.Process()
    is_wsl2 = "microsoft" in platform.release().lower()
    wsl_config_paths = (
        sorted(str(path) for path in Path("/mnt/c/Users").glob("*/.wslconfig")) if is_wsl2 else []
    )
    return {
        "code_revision": _code_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "available_cpu_count": len(process.cpu_affinity())
        if hasattr(process, "cpu_affinity")
        else None,
        "total_memory_bytes": psutil.virtual_memory().total,
        "wsl2": is_wsl2,
        "wsl_config_paths": wsl_config_paths,
        "numpy": version("numpy"),
        "numba": version("numba"),
        "psutil": version("psutil"),
        "matplotlib": version("matplotlib"),
        "executable": sys.executable,
    }


def _code_revision() -> str:
    """Return the current commit hash with a marker for uncommitted changes."""
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{revision}-dirty" if dirty else revision


def _create_solver(
    tree: IndexedGameTree,
    solver_id: str,
    averaging_delay: int,
) -> NaiveCFR | CFR:
    """Construct the requested reference or optimised solver."""
    if solver_id == "naive_cfr":
        return NaiveCFR(tree)
    if solver_id == "naive_cfr_plus":
        return NaiveCFRPlus(tree, averaging_delay=averaging_delay)
    if solver_id == "cfr":
        return CFR(tree)
    return CFRPlus(tree, averaging_delay=averaging_delay)


def _create_reference_solver(
    tree: IndexedGameTree,
    solver_id: str,
    averaging_delay: int,
) -> NaiveCFR:
    """Construct a reference solver with the requested algorithm."""
    if solver_id == "naive_cfr":
        return NaiveCFR(tree)
    return NaiveCFRPlus(tree, averaging_delay=averaging_delay)


def _create_optimised_solver(
    tree: IndexedGameTree,
    solver_id: str,
    averaging_delay: int,
) -> CFR:
    """Construct an optimised solver with the requested algorithm."""
    if solver_id == "cfr":
        return CFR(tree)
    return CFRPlus(tree, averaging_delay=averaging_delay)


def _flatten_table(values: tuple[tuple[float, ...], ...]) -> np.ndarray:
    """Flatten information-set rows into stable action order."""
    return np.fromiter((value for row in values for value in row), dtype=np.float64)


def _maximum_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Return the greatest elementwise absolute difference between two arrays."""
    return float(np.max(np.abs(first - second), initial=0.0))


def _median_absolute_deviation(values: tuple[float, ...]) -> float:
    """Return the median distance from the sample median."""
    centre = median(values)
    return median(abs(value - centre) for value in values)


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    records: list[dict[str, object]],
) -> None:
    """Atomically replace a CSV file using a fixed column order."""
    with atomic_text_writer(path) as results_file:
        writer = csv.DictWriter(results_file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, values: dict[str, object]) -> None:
    """Atomically write deterministic, human-readable JSON."""
    with atomic_text_writer(path) as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    """Send a progress message when the caller supplied a callback."""
    if callback is not None:
        callback(message)
