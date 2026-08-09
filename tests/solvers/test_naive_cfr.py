import numpy as np
import pytest
from numpy.typing import NDArray

from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.solvers import NaiveCFR, NaiveCFRPlus


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


def _information_set_policy(
    tree: IndexedGameTree,
    policy: NDArray[np.float64],
    information_set_id: int,
) -> NDArray[np.float64]:
    action_start = int(tree.information_set_action_offsets[information_set_id])
    action_count = int(tree.information_set_action_counts[information_set_id])
    return policy[action_start : action_start + action_count]
