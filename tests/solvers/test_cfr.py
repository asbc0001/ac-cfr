import numpy as np

from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
from ac_cfr.solvers import CFR, CFRPlus, NaiveCFR, NaiveCFRPlus


def test_dense_solvers_match_reference_updates_on_kuhn_and_leduc() -> None:
    games = ((KuhnGame(), KuhnConfig()), (LeducGame(), LeducConfig()))
    for game, configuration in games:
        tree = compile_game_tree(game, configuration)
        solver_pairs = (
            (NaiveCFR(tree), CFR(tree)),
            (
                NaiveCFRPlus(tree, averaging_delay=1),
                CFRPlus(tree, averaging_delay=1),
            ),
        )
        for reference, optimised in solver_pairs:
            for milestone in (1, 3):
                iterations = milestone - reference.iteration
                reference.train(iterations)
                optimised.train(iterations)

                assert optimised.iteration == reference.iteration
                np.testing.assert_allclose(
                    _flatten(optimised.regret_sum),
                    _flatten(reference.regret_sum),
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    _flatten(optimised.strategy_sum),
                    _flatten(reference.strategy_sum),
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    optimised.current_policy(),
                    reference.current_policy(),
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    optimised.average_policy(),
                    reference.average_policy(),
                    rtol=0.0,
                    atol=1e-12,
                )


def test_dense_solver_reuses_flat_float64_training_buffers() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    solver = CFR(tree)
    buffers = (
        solver._regret_sum,
        solver._strategy_sum,
        solver._current_policy,
        solver._regret_delta,
        solver._node_values,
        solver._reach_player_zero,
        solver._reach_player_one,
        solver._reach_chance,
    )
    buffer_ids = tuple(id(buffer) for buffer in buffers)

    solver.train(2)

    assert all(buffer.dtype == np.float64 and buffer.ndim == 1 for buffer in buffers)
    assert tuple(id(buffer) for buffer in buffers) == buffer_ids


def test_dense_solver_preserves_updates_beyond_zero_probability_actions() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    action_count = len(tree.information_set_actions)
    regrets = np.tile((1.0, 0.0), action_count // 2)
    strategy_sum = np.zeros(action_count, dtype=np.float64)
    reference = NaiveCFR(tree)
    optimised = CFR(tree)
    for solver in (reference, optimised):
        solver.restore_training_state(
            iteration=0,
            regret_sum=regrets,
            strategy_sum=strategy_sum,
        )
        assert 0.0 in solver.current_policy()
        solver.train(1)

    np.testing.assert_allclose(
        _flatten(optimised.regret_sum),
        _flatten(reference.regret_sum),
        rtol=0.0,
        atol=1e-12,
    )


def _flatten(table: tuple[tuple[float, ...], ...]) -> np.ndarray:
    return np.fromiter((value for row in table for value in row), dtype=np.float64)
