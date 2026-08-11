"""Representative CPU and PyTorch profiling for Leduc Deep CFR."""

import cProfile
import io
import pstats
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Final

from torch.profiler import ProfilerActivity, profile

from ac_cfr.benchmarking.deep_cfr_benchmark import warm_up_deep_cfr_solver
from ac_cfr.benchmarking.harness import environment_record, report_progress
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.persistence.files import atomic_text_writer, write_json
from ac_cfr.solvers import DeepCFR, NaiveDeepCFR
from ac_cfr.solvers.naive_deep_cfr import NetworkTrainingMetrics
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig

PROFILE_ID = "deep_cfr"
PROFILE_ITERATIONS = 3
TRAVERSALS_PER_PLAYER = 500
PROFILE_ADVANTAGE_TRAINING_STEPS = 30
PROFILE_STRATEGY_TRAINING_STEPS = 40
TORCH_PROFILE_ITERATIONS = 1
TORCH_PROFILE_TRAVERSALS_PER_PLAYER = 250
TORCH_PROFILE_ADVANTAGE_TRAINING_STEPS = 20
TORCH_PROFILE_STRATEGY_TRAINING_STEPS = 20
PROFILE_SEED = 20260811

_IMPLEMENTATIONS: Final = (
    ("reference", NaiveDeepCFR),
    ("optimised", DeepCFR),
)


def run_deep_cfr_profiling(
    output_directory: Path = Path("results") / PROFILE_ID,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Profile both Leduc implementations and write compact diagnostic evidence."""
    profile_directory = output_directory / "profiles"
    profile_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    files: dict[str, dict[str, str]] = {}

    for implementation, solver_type in _IMPLEMENTATIONS:
        report_progress(progress_callback, f"warm-up: {implementation} Deep CFR")
        warm_up_deep_cfr_solver(
            solver_type,
            _tree(),
            _runtime_config(),
            seed=PROFILE_SEED,
        )

        report_progress(progress_callback, f"cProfile: {implementation} Deep CFR")
        cpu_path, cpu_record = _write_cpu_profile(profile_directory, implementation, solver_type)
        records.append(cpu_record)

        report_progress(progress_callback, f"torch.profiler: {implementation} Deep CFR")
        torch_path, torch_record = _write_torch_profile(
            profile_directory,
            implementation,
            solver_type,
        )
        records.append(torch_record)
        files[implementation] = {
            "cprofile": str(cpu_path.relative_to(output_directory)),
            "torch_profiler": str(torch_path.relative_to(output_directory)),
        }

    result_path = output_directory / "profiling.json"
    write_json(
        result_path,
        {
            "about": (
                "Machine-readable configuration, environment, and file index for separate "
                "Leduc Deep CFR diagnostic profiles. These runs are not formal timings."
            ),
            "profile_id": PROFILE_ID,
            "configuration": {
                "game": "leduc",
                "implementations": [implementation for implementation, _ in _IMPLEMENTATIONS],
                "cprofile_workload": _workload_record(_profile_config()),
                "torch_profiler_workload": _workload_record(_torch_profile_config()),
                "warm_up_before_each_profile": True,
                "profiles_run_separately": True,
                "formal_timing": False,
            },
            "environment": environment_record("torch", "numpy", "numba", "psutil", device="cpu"),
            "runs": records,
            "files": files,
            "file_descriptions": {
                "cprofile": "Python and native-call CPU time ranked by cumulative time.",
                "torch_profiler": "PyTorch operator CPU time ranked by self time.",
            },
        },
    )
    return result_path


def _profile_config() -> DeepCFRTrainingConfig:
    """Return the identical fixed workload used by both implementations."""
    return DeepCFRTrainingConfig(
        iterations=PROFILE_ITERATIONS,
        traversals_per_player=TRAVERSALS_PER_PLAYER,
        advantage_reservoir_capacity=100_000,
        strategy_reservoir_capacity=100_000,
        advantage_training_steps=PROFILE_ADVANTAGE_TRAINING_STEPS,
        strategy_training_steps=PROFILE_STRATEGY_TRAINING_STEPS,
        training_batch_size=512,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=PROFILE_SEED,
    )


def _torch_profile_config() -> DeepCFRTrainingConfig:
    """Return a bounded workload sufficient to rank PyTorch operators."""
    return DeepCFRTrainingConfig(
        iterations=TORCH_PROFILE_ITERATIONS,
        traversals_per_player=TORCH_PROFILE_TRAVERSALS_PER_PLAYER,
        advantage_reservoir_capacity=100_000,
        strategy_reservoir_capacity=100_000,
        advantage_training_steps=TORCH_PROFILE_ADVANTAGE_TRAINING_STEPS,
        strategy_training_steps=TORCH_PROFILE_STRATEGY_TRAINING_STEPS,
        training_batch_size=512,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=PROFILE_SEED,
    )


def _write_cpu_profile(
    profile_directory: Path,
    implementation: str,
    solver_type: type[NaiveDeepCFR],
) -> tuple[Path, dict[str, object]]:
    """Run one cProfile diagnostic and save its most costly call paths."""
    solver = solver_type(_tree(), _profile_config(), _runtime_config())
    profiler = cProfile.Profile()
    started = perf_counter()
    profiler.runcall(solver.train, PROFILE_ITERATIONS)
    elapsed_seconds = perf_counter() - started
    output = io.StringIO()
    statistics = pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative")
    statistics.print_stats(30)
    path = profile_directory / f"leduc_{implementation}_deep_cfr_cprofile.md"
    with atomic_text_writer(path) as profile_file:
        profile_file.write(
            f"# cProfile: {implementation} Leduc Deep CFR\n\n"
            "This automatically generated diagnostic records Python and native-call CPU time. "
            "It uses a separate warmed run and is not a formal runtime measurement.\n\n"
            f"- Outer iterations: {PROFILE_ITERATIONS}\n"
            f"- Traversals: {2 * TRAVERSALS_PER_PLAYER * PROFILE_ITERATIONS:,}\n"
            f"- Optimizer steps: "
            f"{_training_steps(solver.training_metrics, solver.config):,}\n\n"
            "```text\n"
        )
        profile_file.write(output.getvalue())
        profile_file.write("```\n")
    return path, {
        "implementation": implementation,
        "profiler": "cProfile",
        "profiled_seconds": elapsed_seconds,
        "optimizer_steps": _training_steps(solver.training_metrics, solver.config),
    }


def _write_torch_profile(
    profile_directory: Path,
    implementation: str,
    solver_type: type[NaiveDeepCFR],
) -> tuple[Path, dict[str, object]]:
    """Run one torch.profiler diagnostic and save its leading operators."""
    solver = solver_type(_tree(), _torch_profile_config(), _runtime_config())
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        solver.train(TORCH_PROFILE_ITERATIONS)
    table = profiler.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=30,
    )
    path = profile_directory / f"leduc_{implementation}_deep_cfr_torch_profiler.md"
    with atomic_text_writer(path) as profile_file:
        profile_file.write(
            f"# torch.profiler: {implementation} Leduc Deep CFR\n\n"
            "This automatically generated diagnostic records PyTorch operator CPU time. It "
            "uses a separate warmed run and is not a formal runtime measurement.\n\n"
            f"- Outer iterations: {TORCH_PROFILE_ITERATIONS}\n"
            f"- Traversals: "
            f"{2 * TORCH_PROFILE_TRAVERSALS_PER_PLAYER * TORCH_PROFILE_ITERATIONS:,}\n"
            f"- Optimizer steps: "
            f"{_training_steps(solver.training_metrics, solver.config):,}\n\n"
            "```text\n"
        )
        profile_file.write(table)
        profile_file.write("\n```\n")
    return path, {
        "implementation": implementation,
        "profiler": "torch.profiler",
        "profiled_self_cpu_seconds": profiler.key_averages().self_cpu_time_total / 1_000_000,
        "optimizer_steps": _training_steps(solver.training_metrics, solver.config),
    }


def _training_steps(
    metrics: tuple[NetworkTrainingMetrics, ...],
    config: DeepCFRTrainingConfig,
) -> int:
    """Count completed optimizer steps from the recorded network roles."""
    return sum(
        (
            config.advantage_training_steps
            if metric.network_role == "advantage"
            else config.strategy_training_steps
        )
        for metric in metrics
    )


def _workload_record(config: DeepCFRTrainingConfig) -> dict[str, object]:
    """Add explicit traversal totals to one serialised training configuration."""
    return {
        **config.to_dict(),
        "traversals_per_outer_iteration": 2 * config.traversals_per_player,
        "total_traversals": 2 * config.traversals_per_player * config.iterations,
        "total_advantage_optimizer_steps": (
            2 * config.iterations * config.advantage_training_steps
        ),
        "total_strategy_optimizer_steps": config.strategy_training_steps,
    }


def _tree() -> IndexedGameTree:
    """Compile the shared production Leduc tree used by every profile."""
    return compile_game_tree(LeducGame(), LeducConfig())


def _runtime_config(*, inference_batch_size: int = 512) -> DeepCFRRuntimeConfig:
    """Return the explicit CPU execution settings used while profiling."""
    return DeepCFRRuntimeConfig(
        inference_batch_size=inference_batch_size,
        cpu_threads=1,
        device="cpu",
    )
