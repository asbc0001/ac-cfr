import numpy as np
import pytest
from numpy.typing import NDArray

from ac_cfr.evaluation import evaluate_strategy
from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.solvers import NaiveCFR, NaiveCFRPlus

# Kuhn's cheap known-value gate uses 0.1% chip exploitability and 0.01% value error.
KUHN_EXPLOITABILITY_LIMIT = 1e-3
KUHN_VALUE_ERROR_LIMIT = 1e-4
# Leduc's larger slow gate permits at most 0.5% of the one-chip utility unit.
LEDUC_EXPLOITABILITY_LIMIT = 5e-3


def test_naive_cfr_first_iteration_uses_frozen_alternating_updates() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    solver = NaiveCFR(tree)

    assert np.all(solver.current_policy() == 0.5)
    assert np.all(solver.average_policy() == 0.5)

    solver.train(1)

    assert solver.iteration == 1
    assert solver.regret_sum[0] == pytest.approx((-1 / 8, 1 / 8))
    assert solver.regret_sum[3] == pytest.approx((-1 / 6, 1 / 6))
    assert solver.regret_sum[8] == pytest.approx((-1 / 12, 1 / 12))
    assert solver.strategy_sum[1] == pytest.approx((1 / 2, 1 / 2))
    assert solver.strategy_sum[0] == pytest.approx((0.0, 1.0))
    assert solver.strategy_sum[2] == pytest.approx((0.0, 0.0))
    assert _information_set_policy(tree, solver.average_policy(), 0) == pytest.approx((0.0, 1.0))
    assert _information_set_policy(tree, solver.average_policy(), 2) == pytest.approx((0.5, 0.5))


def test_naive_cfr_plus_clips_aggregated_regrets_and_delays_averaging() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    cfr = NaiveCFR(tree)
    cfr_plus = NaiveCFRPlus(tree, averaging_delay=1)
    cfr.train(1)
    cfr_plus.train(1)

    expected_clipped_regrets = tuple(
        tuple(max(regret, 0.0) for regret in information_set) for information_set in cfr.regret_sum
    )
    assert np.asarray(cfr_plus.regret_sum) == pytest.approx(np.asarray(expected_clipped_regrets))
    # This information set has opposing history contributions, so clipping it earlier differs.
    assert cfr_plus.regret_sum[8] == pytest.approx((0.0, 1 / 12))
    assert all(value == 0.0 for row in cfr_plus.strategy_sum for value in row)
    assert np.all(cfr_plus.average_policy() == 0.5)

    cfr_plus.train(1)
    assert cfr_plus.iteration == 2
    assert any(value > 0.0 for row in cfr_plus.strategy_sum for value in row)


def test_naive_solver_configuration_rejects_invalid_counts() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    solver = NaiveCFR(tree)

    with pytest.raises(TypeError, match="iterations"):
        solver.train(True)
    with pytest.raises(ValueError, match="iterations"):
        solver.train(-1)
    with pytest.raises(ValueError, match="averaging_delay"):
        NaiveCFRPlus(tree, averaging_delay=-1)


def test_reference_cfr_solvers_converge_on_kuhn() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    solvers = (
        NaiveCFR(tree),
        # A short nonzero delay ensures this gate exercises delayed averaging.
        NaiveCFRPlus(tree, averaging_delay=10),
    )

    for solver in solvers:
        previous_value_error = float("inf")
        previous_exploitability = float("inf")
        for milestone in (10, 100, 1_000, 10_000):
            solver.train(milestone - solver.iteration)
            metrics = evaluate_strategy(tree, solver.average_policy())
            value_error = abs(metrics.expected_values[0] + 1 / 18)
            assert value_error < previous_value_error
            assert metrics.exploitability < previous_exploitability
            previous_value_error = value_error
            previous_exploitability = metrics.exploitability

        assert metrics.expected_values[0] == pytest.approx(-1 / 18, abs=KUHN_VALUE_ERROR_LIMIT)
        assert metrics.exploitability <= KUHN_EXPLOITABILITY_LIMIT


def test_reference_cfr_solvers_reduce_exploitability_on_leduc() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    solvers = (
        NaiveCFR(tree),
        NaiveCFRPlus(tree, averaging_delay=10),
    )

    for solver in solvers:
        previous_exploitability = float("inf")
        for milestone in (10, 50, 100):
            solver.train(milestone - solver.iteration)
            metrics = evaluate_strategy(tree, solver.average_policy())
            assert metrics.exploitability < previous_exploitability
            previous_exploitability = metrics.exploitability


@pytest.mark.slow
def test_reference_cfr_solvers_reach_low_exact_leduc_exploitability() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    solvers = (
        NaiveCFR(tree),
        NaiveCFRPlus(tree, averaging_delay=10),
    )
    final_metrics = []

    for solver in solvers:
        previous_exploitability = float("inf")
        for milestone in (250, 1_000, 5_000):
            solver.train(milestone - solver.iteration)
            metrics = evaluate_strategy(tree, solver.average_policy())
            assert metrics.exploitability < previous_exploitability
            previous_exploitability = metrics.exploitability
        final_metrics.append(metrics)

    assert all(metrics.exploitability <= LEDUC_EXPLOITABILITY_LIMIT for metrics in final_metrics)
    value_difference = abs(
        final_metrics[0].expected_values[0] - final_metrics[1].expected_values[0]
    )
    assert value_difference <= sum(metrics.nash_conv for metrics in final_metrics)


def _information_set_policy(
    tree: IndexedGameTree,
    policy: NDArray[np.float64],
    information_set_id: int,
) -> NDArray[np.float64]:
    action_start = int(tree.information_set_action_offsets[information_set_id])
    action_count = int(tree.information_set_action_counts[information_set_id])
    return policy[action_start : action_start + action_count]
