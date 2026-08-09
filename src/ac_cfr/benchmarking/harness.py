"""Repeated implementation-level timings for tabular poker solvers."""

from dataclasses import dataclass
from statistics import median
from time import perf_counter

from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.base import GameId
from ac_cfr.games.tabular import create_tabular_game
from ac_cfr.solvers.naive_cfr import NaiveCFR
from ac_cfr.solvers.naive_cfr_plus import NaiveCFRPlus
from ac_cfr.training.runner import SOLVER_IDS


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
    exploitability: float


def run_tabular_benchmark(
    *,
    game: str,
    solver_id: str,
    iterations: int,
    repeats: int,
    averaging_delay: int = 0,
) -> BenchmarkResult:
    """Time repeated identical training runs without evaluation or file I/O."""
    if solver_id not in SOLVER_IDS:
        raise ValueError(f"solver must be one of: {', '.join(SOLVER_IDS)}")
    _validate_positive_integer("iterations", iterations)
    _validate_positive_integer("repeats", repeats)
    if isinstance(averaging_delay, bool) or not isinstance(averaging_delay, int):
        raise TypeError("averaging_delay must be an integer")
    if averaging_delay < 0:
        raise ValueError("averaging_delay must not be negative")
    if solver_id == "naive_cfr" and averaging_delay != 0:
        raise ValueError("averaging_delay applies only to naive_cfr_plus")

    tabular_game = create_tabular_game(GameId(game))
    timings: list[float] = []
    final_solver: NaiveCFR | None = None
    for _ in range(repeats):
        if solver_id == "naive_cfr":
            solver = NaiveCFR(tabular_game.tree)
        else:
            solver = NaiveCFRPlus(tabular_game.tree, averaging_delay=averaging_delay)
        start_time = perf_counter()
        solver.train(iterations)
        timings.append(perf_counter() - start_time)
        final_solver = solver

    assert final_solver is not None
    median_seconds = median(timings)
    absolute_deviations = tuple(abs(value - median_seconds) for value in timings)
    traversals = 2 * iterations
    metrics = evaluate_strategy(tabular_game.tree, final_solver.average_policy())
    return BenchmarkResult(
        game=game,
        solver=solver_id,
        iterations=iterations,
        traversals=traversals,
        repeats=repeats,
        median_seconds=median_seconds,
        median_absolute_deviation_seconds=median(absolute_deviations),
        traversals_per_second=traversals / median_seconds,
        exploitability=metrics.exploitability,
    )


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
