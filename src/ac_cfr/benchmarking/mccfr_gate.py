"""Fixed-workload performance benchmark and completion gate for Leduc MCCFR."""

import cProfile
import csv
import io
import json
import platform
import pstats
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Final

import psutil

from ac_cfr.benchmarking.harness import BenchmarkResult, run_tabular_benchmark
from ac_cfr.evaluation.plotting import plot_mccfr_performance
from ac_cfr.games.base import GameId
from ac_cfr.games.tabular import create_tabular_game
from ac_cfr.persistence.files import atomic_text_writer
from ac_cfr.solvers import MCCFR, NaiveMCCFR

BENCHMARK_ID = "mccfr"
BENCHMARK_ITERATIONS = 500_000
BENCHMARK_SEED = 0
REPEATS = 5
MEMORY_SAMPLING_INTERVAL_SECONDS = 0.01
PROFILE_ITERATIONS = 100_000

_IMPLEMENTATIONS: Final = (
    ("reference", "naive_mccfr"),
    ("optimised", "mccfr"),
)
_BENCHMARK_RUN_FIELDS: Final = (
    "benchmark_id",
    "game",
    "implementation",
    "solver",
    "seed",
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
    "implementation",
    "solver",
    "seed",
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


def run_mccfr_gate(
    output_directory: Path = Path("results") / BENCHMARK_ID,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run the fixed MCCFR benchmark and assemble its completion evidence."""
    validation_path = output_directory / "validation.json"
    validation = _load_validation(validation_path)
    benchmark_results: list[tuple[str, BenchmarkResult]] = []
    for implementation, solver_id in _IMPLEMENTATIONS:
        _report_progress(progress_callback, f"benchmark: leduc {implementation} mccfr")
        benchmark_results.append(
            (
                implementation,
                run_tabular_benchmark(
                    game=GameId.LEDUC.value,
                    solver_id=solver_id,
                    iterations=BENCHMARK_ITERATIONS,
                    repeats=REPEATS,
                    memory_sampling_interval_seconds=MEMORY_SAMPLING_INTERVAL_SECONDS,
                ),
            )
        )

    runs_path = output_directory / "benchmark_runs.csv"
    summary_path = output_directory / "benchmark_summary.csv"
    _write_benchmark_results(runs_path, summary_path, benchmark_results)

    profile_directory = output_directory / "profiles"
    profile_paths: list[Path] = []
    for implementation, solver_id in _IMPLEMENTATIONS:
        _report_progress(progress_callback, f"profile: leduc {implementation} mccfr")
        profile_paths.append(_write_profile(profile_directory, solver_id))

    performance_path = output_directory / "plots" / "implementation_performance.png"
    plot_mccfr_performance(summary_path, performance_path)
    checks = [
        *_validation_checks(validation),
        {
            "name": "fixed_workload_benchmark_completed",
            "passed": _benchmark_results_are_complete(benchmark_results),
            "implementations": [implementation for implementation, _ in benchmark_results],
        },
        {
            "name": "representative_profiles_completed",
            "passed": all(path.stat().st_size > 0 for path in profile_paths),
            "iterations": PROFILE_ITERATIONS,
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    gate_path = output_directory / "gate.json"
    _write_json(
        gate_path,
        {
            "about": (
                "Machine-readable configuration, environment, checks, and file index for the "
                "Leduc MCCFR completion gate. Strategy-quality evidence remains in validation.json."
            ),
            "benchmark_id": BENCHMARK_ID,
            "passed": passed,
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
                "traversal": "One sampled root traversal for one traversing player.",
                "median_absolute_deviation": (
                    "Median distance from the median, used as a robust measure of run variation."
                ),
                "peak_memory": (
                    "Highest sampled memory total across the benchmark process and its children."
                ),
            },
            "configuration": {
                "strategy_quality_validation": validation["configuration"],
                "fixed_performance_benchmark": {
                    "game": GameId.LEDUC.value,
                    "implementations": [solver_id for _, solver_id in _IMPLEMENTATIONS],
                    "iterations": BENCHMARK_ITERATIONS,
                    "traversals": 2 * BENCHMARK_ITERATIONS,
                    "traversals_per_iteration": 2,
                    "seed": BENCHMARK_SEED,
                    "repeats": REPEATS,
                    "timed_region": "solver.train only",
                    "numba_warm_up_before_timing": True,
                    "early_stopping": False,
                    "memory_sampling_interval_seconds": MEMORY_SAMPLING_INTERVAL_SECONDS,
                },
                "profiles": {
                    "iterations": PROFILE_ITERATIONS,
                    "traversals": 2 * PROFILE_ITERATIONS,
                    "separate_from_formal_timing": True,
                },
            },
            "environment": _environment_record(),
            "checks": checks,
            "files": {
                "validation": validation_path.name,
                "convergence": "convergence.csv",
                "benchmark_runs": runs_path.name,
                "benchmark_summary": summary_path.name,
                "plots": [
                    "plots/convergence.png",
                    str(performance_path.relative_to(output_directory)),
                ],
                "profiles": [str(path.relative_to(output_directory)) for path in profile_paths],
            },
            "file_descriptions": {
                "validation": "Multi-seed correctness, convergence, export, and self-play checks.",
                "convergence": "Exact strategy-quality measurements by seed and milestone.",
                "benchmark_runs": "Every individual timing and peak-memory repetition.",
                "benchmark_summary": "Median performance, variation, memory, and final quality.",
                "plots": "Convergence and fixed-workload engineering comparisons.",
                "profiles": "Automatically generated cProfile CPU-time reports in Markdown.",
            },
        },
    )
    if not passed:
        raise RuntimeError(f"MCCFR gate failed; see {gate_path}")
    return gate_path


def _load_validation(path: Path) -> dict[str, object]:
    """Load the completed strategy-quality validation required by this gate."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("MCCFR validation metadata is unreadable") from error
    if not isinstance(value, dict) or value.get("validation_id") != BENCHMARK_ID:
        raise ValueError("MCCFR validation metadata does not match this gate")
    if value.get("passed") is not True:
        raise ValueError("MCCFR strategy-quality validation has not passed")
    return value


def _validation_checks(validation: dict[str, object]) -> list[dict[str, object]]:
    """Return the individual passed checks from the strategy-quality validation."""
    checks = validation.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("MCCFR validation checks are missing")
    typed_checks: list[dict[str, object]] = []
    for check in checks:
        if not isinstance(check, dict) or check.get("passed") is not True:
            raise ValueError("MCCFR validation contains an invalid or failed check")
        typed_checks.append({**check, "source": "validation.json"})
    return typed_checks


def _write_benchmark_results(
    runs_path: Path,
    summary_path: Path,
    results: list[tuple[str, BenchmarkResult]],
) -> None:
    """Write raw repetitions and robust summaries for both implementations."""
    run_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for implementation, result in results:
        common = {
            "benchmark_id": BENCHMARK_ID,
            "game": result.game,
            "implementation": implementation,
            "solver": result.solver,
            "seed": BENCHMARK_SEED,
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
                    "memory_sampling_interval_seconds": result.memory_sampling_interval_seconds,
                    "peak_memory_mb": repeat.peak_memory_mb,
                }
            )
        throughput = tuple(repeat.traversals_per_second for repeat in result.repeat_results)
        summary_records.append(
            {
                **common,
                "repeats": result.repeats,
                "median_seconds": result.median_seconds,
                "median_absolute_deviation_seconds": result.median_absolute_deviation_seconds,
                "traversals_per_second": result.traversals_per_second,
                "median_absolute_deviation_traversals_per_second": (
                    _median_absolute_deviation(throughput)
                ),
                "memory_metric": result.memory_metric,
                "memory_sampling_interval_seconds": result.memory_sampling_interval_seconds,
                "median_peak_memory_mb": result.median_peak_memory_mb,
                "median_absolute_deviation_memory_mb": (result.median_absolute_deviation_memory_mb),
                "expected_value_player_zero": result.expected_value_player_zero,
                "exploitability": result.exploitability,
                "nash_conv": result.nash_conv,
            }
        )
    _write_csv(runs_path, _BENCHMARK_RUN_FIELDS, run_records)
    _write_csv(summary_path, _BENCHMARK_SUMMARY_FIELDS, summary_records)


def _write_profile(profile_directory: Path, solver_id: str) -> Path:
    """Run a separate representative CPU profile and write it as Markdown."""
    tree = create_tabular_game(GameId.LEDUC).tree
    if solver_id == "mccfr":
        MCCFR(tree, seed=BENCHMARK_SEED).train(1)
        solver: NaiveMCCFR | MCCFR = MCCFR(tree, seed=BENCHMARK_SEED)
    else:
        solver = NaiveMCCFR(tree, seed=BENCHMARK_SEED)
    profiler = cProfile.Profile()
    profiler.runcall(solver.train, PROFILE_ITERATIONS)
    profile_text = io.StringIO()
    statistics = pstats.Stats(profiler, stream=profile_text).strip_dirs().sort_stats("cumulative")
    statistics.print_stats(25)
    path = profile_directory / f"leduc_{solver_id}.md"
    with atomic_text_writer(path) as profile_file:
        profile_file.write(
            f"# CPU profile: Leduc {solver_id}\n\n"
            "This file is generated automatically with Python's `cProfile`. It records a "
            "separate diagnostic run and does not affect the formal benchmark timing.\n\n"
            f"- **Solver:** `{solver_id}`\n"
            "- **Game:** Leduc\n"
            f"- **Iterations:** {PROFILE_ITERATIONS:,}\n"
            "- **Traversals:** "
            f"{2 * PROFILE_ITERATIONS:,}\n"
            "- **Rows:** top 25 functions, sorted by cumulative time\n\n"
            "`ncalls` is the call count. `tottime` is time inside a function; `cumtime` also "
            "includes functions it called. Times are seconds. Numba-compiled operations are not "
            "individually visible to `cProfile`.\n\n"
            "## Raw cProfile output\n\n"
            "```text\n"
        )
        profile_file.write(profile_text.getvalue())
        profile_file.write("```\n")
    return path


def _benchmark_results_are_complete(results: list[tuple[str, BenchmarkResult]]) -> bool:
    """Check that both fixed workloads produced finite positive measurements."""
    if tuple(implementation for implementation, _ in results) != tuple(
        implementation for implementation, _ in _IMPLEMENTATIONS
    ):
        return False
    return all(
        result.game == GameId.LEDUC.value
        and result.iterations == BENCHMARK_ITERATIONS
        and result.traversals == 2 * BENCHMARK_ITERATIONS
        and result.repeats == REPEATS
        and result.median_seconds > 0.0
        and result.traversals_per_second > 0.0
        and result.median_peak_memory_mb > 0.0
        and len(result.repeat_results) == REPEATS
        for _, result in results
    )


def _environment_record() -> dict[str, object]:
    """Capture the hardware and software context needed to interpret timings."""
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
        "available_cpu_count": (
            len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else None
        ),
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
    """Return the current commit hash and mark uncommitted code."""
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(("git", "diff", "--quiet"), check=False).returncode != 0
    return f"{revision}-dirty" if dirty else revision


def _median_absolute_deviation(values: tuple[float, ...]) -> float:
    """Return the median distance from the sample median."""
    centre = median(values)
    return median(abs(value - centre) for value in values)


def _write_csv(path: Path, fields: tuple[str, ...], records: list[dict[str, object]]) -> None:
    """Atomically write CSV records using a fixed column order."""
    with atomic_text_writer(path) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, values: dict[str, object]) -> None:
    """Atomically write deterministic, readable JSON."""
    with atomic_text_writer(path) as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    """Send a progress message when the caller supplied one."""
    if callback is not None:
        callback(message)
