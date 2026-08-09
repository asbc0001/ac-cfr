"""Exact strategy-quality metrics for Kuhn and Leduc poker."""

from dataclasses import dataclass

from ac_cfr.evaluation.best_response import ExactEvaluator, Policy
from ac_cfr.games.tree import IndexedGameTree


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """Exact Kuhn or Leduc poker strategy-quality measurements."""

    expected_values: tuple[float, float]
    best_response_values: tuple[float, float]
    improvements: tuple[float, float]
    nash_conv: float
    exploitability: float


def evaluate_strategy(tree: IndexedGameTree, policy: Policy) -> StrategyMetrics:
    """Return exact value, best-response, NashConv, and exploitability metrics."""
    player_zero_value, player_zero_best_response, player_one_best_response = ExactEvaluator(
        tree
    ).profile_values(policy)
    expected_values = (player_zero_value, -player_zero_value)
    best_response_values = (player_zero_best_response, player_one_best_response)
    improvements = (
        _non_negative_improvement(player_zero_best_response - player_zero_value),
        _non_negative_improvement(player_one_best_response + player_zero_value),
    )
    nash_conv = sum(improvements)
    return StrategyMetrics(
        expected_values=expected_values,
        best_response_values=best_response_values,
        improvements=improvements,
        nash_conv=nash_conv,
        exploitability=nash_conv / 2.0,
    )


def _non_negative_improvement(improvement: float) -> float:
    if improvement >= 0.0:
        return improvement
    if improvement >= -1e-12:
        return 0.0
    raise RuntimeError("best-response value is below the supplied policy value")
