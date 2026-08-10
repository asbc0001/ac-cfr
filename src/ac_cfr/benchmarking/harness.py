"""Repeated fixed-workload timings for tabular poker solvers."""

from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
from statistics import median
from time import perf_counter

import psutil

from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.base import GameId
from ac_cfr.games.tabular import create_tabular_game
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.solvers.cfr import CFR
from ac_cfr.solvers.cfr_plus import CFRPlus
from ac_cfr.solvers.naive_cfr import NaiveCFR
from ac_cfr.solvers.naive_cfr_plus import NaiveCFRPlus
from ac_cfr.training.runner import SOLVER_IDS

DEFAULT_MEMORY_SAMPLING_INTERVAL_SECONDS = 0.01
_OPTIMISED_SOLVERS = ("cfr", "cfr_plus")


@dataclass(frozen=True, slots=True)
class BenchmarkRepeat:
    """One isolated timing and process-tree memory measurement."""

    repeat: int
    seconds: float
    traversals_per_second: float
    peak_memory_mb: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Repeated timing summary for one exact solver workload."""

    game: str
    solver: str
    iterations: int
    traversals: int
    repeats: int
    median_seconds: float
    median_absolute_deviation_seconds: float
    traversals_per_second: float
    memory_metric: str
    memory_sampling_interval_seconds: float
    median_peak_memory_mb: float
    median_absolute_deviation_memory_mb: float
    expected_value_player_zero: float
    exploitability: float
    nash_conv: float
    repeat_results: tuple[BenchmarkRepeat, ...]


def run_tabular_benchmark(
    *,
    game: str,
    solver_id: str,
    iterations: int,
    repeats: int,
    averaging_delay: int = 0,
    memory_sampling_interval_seconds: float = DEFAULT_MEMORY_SAMPLING_INTERVAL_SECONDS,
) -> BenchmarkResult:
    """Measure repeated fresh-process runs without setup or evaluation in the timer."""
    if solver_id not in SOLVER_IDS:
        raise ValueError(f"solver must be one of: {', '.join(SOLVER_IDS)}")
    if game not in (GameId.KUHN.value, GameId.LEDUC.value):
        raise ValueError("game must be kuhn or leduc")
    _validate_positive_integer("iterations", iterations)
    _validate_positive_integer("repeats", repeats)
    _validate_averaging_delay(solver_id, averaging_delay)
    if (
        isinstance(memory_sampling_interval_seconds, bool)
        or not isinstance(memory_sampling_interval_seconds, (int, float))
        or memory_sampling_interval_seconds <= 0.0
    ):
        raise ValueError("memory_sampling_interval_seconds must be positive")

    memory_metric = _preferred_memory_metric()
    repeat_results: list[BenchmarkRepeat] = []
    final_metrics: tuple[float, float, float] | None = None
    traversals = 2 * iterations
    for repeat in range(1, repeats + 1):
        seconds, peak_memory_mb, final_metrics = _run_isolated_workload(
            game=game,
            solver_id=solver_id,
            iterations=iterations,
            averaging_delay=averaging_delay,
            memory_metric=memory_metric,
            memory_sampling_interval_seconds=float(memory_sampling_interval_seconds),
        )
        repeat_results.append(
            BenchmarkRepeat(
                repeat=repeat,
                seconds=seconds,
                traversals_per_second=traversals / seconds,
                peak_memory_mb=peak_memory_mb,
            )
        )

    assert final_metrics is not None
    timing_values = tuple(result.seconds for result in repeat_results)
    memory_values = tuple(result.peak_memory_mb for result in repeat_results)
    median_seconds = median(timing_values)
    median_memory = median(memory_values)
    expected_value, exploitability, nash_conv = final_metrics
    return BenchmarkResult(
        game=game,
        solver=solver_id,
        iterations=iterations,
        traversals=traversals,
        repeats=repeats,
        median_seconds=median_seconds,
        median_absolute_deviation_seconds=_median_absolute_deviation(timing_values),
        traversals_per_second=traversals / median_seconds,
        memory_metric=memory_metric,
        memory_sampling_interval_seconds=float(memory_sampling_interval_seconds),
        median_peak_memory_mb=median_memory,
        median_absolute_deviation_memory_mb=_median_absolute_deviation(memory_values),
        expected_value_player_zero=expected_value,
        exploitability=exploitability,
        nash_conv=nash_conv,
        repeat_results=tuple(repeat_results),
    )


def _run_isolated_workload(
    *,
    game: str,
    solver_id: str,
    iterations: int,
    averaging_delay: int,
    memory_metric: str,
    memory_sampling_interval_seconds: float,
) -> tuple[float, float, tuple[float, float, float]]:
    """Run one workload in a fresh process while sampling its peak memory."""
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_benchmark_worker,
        args=(child_connection, game, solver_id, iterations, averaging_delay),
    )
    process.start()
    child_connection.close()
    try:
        _expect_message(parent_connection, "ready")
        measured_process = psutil.Process(process.pid)
        parent_connection.send("start")
        peak_memory_bytes = 0
        while not parent_connection.poll(memory_sampling_interval_seconds):
            peak_memory_bytes = max(
                peak_memory_bytes,
                _process_tree_memory_bytes(measured_process, memory_metric),
            )
            if not process.is_alive():
                raise RuntimeError("benchmark worker stopped before returning a result")
        peak_memory_bytes = max(
            peak_memory_bytes,
            _process_tree_memory_bytes(measured_process, memory_metric),
        )
        trained_message = parent_connection.recv()
        if trained_message[0] == "error":
            raise RuntimeError(f"benchmark worker failed: {trained_message[1]}")
        if trained_message[0] != "trained":
            raise RuntimeError("benchmark worker returned an unexpected training message")
        seconds = float(trained_message[1])

        parent_connection.send("evaluate")
        result_message = parent_connection.recv()
        if result_message[0] == "error":
            raise RuntimeError(f"benchmark worker failed: {result_message[1]}")
        if result_message[0] != "result":
            raise RuntimeError("benchmark worker returned an unexpected result message")
        metrics = tuple(float(value) for value in result_message[1:])
    finally:
        parent_connection.close()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"benchmark worker exited with status {process.exitcode}")
    if len(metrics) != 3:
        raise RuntimeError("benchmark worker returned incomplete strategy metrics")
    return seconds, peak_memory_bytes / (1024 * 1024), metrics


def _benchmark_worker(
    connection: Connection,
    game: str,
    solver_id: str,
    iterations: int,
    averaging_delay: int,
) -> None:
    """Train and evaluate one solver under parent-process timing control."""
    try:
        tabular_game = create_tabular_game(GameId(game))
        if solver_id in _OPTIMISED_SOLVERS:
            _create_solver(tabular_game.tree, solver_id, averaging_delay).train(1)
        solver = _create_solver(tabular_game.tree, solver_id, averaging_delay)
        connection.send(("ready",))
        if connection.recv() != "start":
            raise RuntimeError("benchmark worker did not receive the start command")

        start_time = perf_counter()
        solver.train(iterations)
        elapsed_seconds = perf_counter() - start_time
        connection.send(("trained", elapsed_seconds))
        if connection.recv() != "evaluate":
            raise RuntimeError("benchmark worker did not receive the evaluation command")

        metrics = evaluate_strategy(tabular_game.tree, solver.average_policy())
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


def _expect_message(connection: Connection, expected_kind: str) -> None:
    """Receive one worker message and validate its type."""
    message = connection.recv()
    if message[0] == "error":
        raise RuntimeError(f"benchmark worker failed: {message[1]}")
    if message[0] != expected_kind:
        raise RuntimeError(f"benchmark worker did not report {expected_kind}")


def _preferred_memory_metric() -> str:
    """Choose the most informative memory measure available on this platform."""
    memory = psutil.Process().memory_full_info()
    if hasattr(memory, "pss"):
        return "pss"
    if hasattr(memory, "uss"):
        return "uss"
    return "rss"


def _process_tree_memory_bytes(process: psutil.Process, metric: str) -> int:
    """Sum a memory measure across a process and its current descendants."""
    total = 0
    for member in (process, *process.children(recursive=True)):
        try:
            memory = member.memory_full_info()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        total += int(getattr(memory, metric))
    return total


def _create_solver(
    tree: IndexedGameTree,
    solver_id: str,
    averaging_delay: int,
) -> NaiveCFR | CFR:
    """Construct the requested tabular solver for an isolated workload."""
    if solver_id == "naive_cfr":
        return NaiveCFR(tree)
    if solver_id == "naive_cfr_plus":
        return NaiveCFRPlus(tree, averaging_delay=averaging_delay)
    if solver_id == "cfr":
        return CFR(tree)
    return CFRPlus(tree, averaging_delay=averaging_delay)


def _validate_averaging_delay(solver_id: str, averaging_delay: int) -> None:
    if isinstance(averaging_delay, bool) or not isinstance(averaging_delay, int):
        raise TypeError("averaging_delay must be an integer")
    if averaging_delay < 0:
        raise ValueError("averaging_delay must not be negative")
    if solver_id in ("naive_cfr", "cfr") and averaging_delay != 0:
        raise ValueError("averaging_delay applies only to CFR+ solvers")


def _median_absolute_deviation(values: tuple[float, ...]) -> float:
    centre = median(values)
    return median(abs(value - centre) for value in values)


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
