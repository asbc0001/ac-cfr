import numpy as np
import pytest
from numpy.random import default_rng

from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import compile_game_tree
from ac_cfr.solvers import MCCFR, NaiveMCCFR


def test_dense_mccfr_matches_reference_draws_and_training_chunks() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    seed_deriver = SeedDeriver(2026)
    reference = NaiveMCCFR(tree, seed=2026)
    reference._chance_rng = default_rng(  # pyright: ignore[reportAttributeAccessIssue]
        seed_deriver.derive(RngStream.CHANCE)
    )
    reference._policy_rng = default_rng(  # pyright: ignore[reportAttributeAccessIssue]
        seed_deriver.derive(RngStream.POLICY)
    )
    uninterrupted = MCCFR(tree, seed=2026)
    chunked = MCCFR(tree, seed=2026)
    different_seed = MCCFR(tree, seed=2027)

    reference.train(10)
    uninterrupted.train(10)
    chunked.train(4)
    chunked.train(6)
    different_seed.train(10)

    assert uninterrupted.iteration == chunked.iteration == 10
    assert uninterrupted.regret_sum == chunked.regret_sum
    assert uninterrupted.strategy_sum == chunked.strategy_sum
    assert uninterrupted.regret_sum != different_seed.regret_sum
    assert uninterrupted.strategy_sum != different_seed.strategy_sum
    np.testing.assert_allclose(
        _flatten(uninterrupted.regret_sum),
        _flatten(reference.regret_sum),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _flatten(uninterrupted.strategy_sum),
        _flatten(reference.strategy_sum),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        uninterrupted.current_policy(),
        reference.current_policy(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        uninterrupted.average_policy(),
        reference.average_policy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_dense_mccfr_reuses_flat_typed_buffers_and_returns_valid_policies() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    solver = MCCFR(tree, seed=0)
    buffers = (
        solver._regret_sum,
        solver._strategy_sum,
        solver._current_policy,
        solver._sampled_opponent_actions,
        solver._frame_edge_starts,
        solver._frame_information_sets,
        solver._frame_action_counts,
        solver._frame_next_actions,
        solver._frame_node_values,
        solver._frame_action_values,
        solver._frame_strategies,
    )
    buffer_ids = tuple(id(buffer) for buffer in buffers)

    solver.train(2)

    assert tuple(id(buffer) for buffer in buffers) == buffer_ids
    assert all(buffer.flags.c_contiguous for buffer in buffers)
    assert solver._regret_sum.dtype == solver._strategy_sum.dtype == np.float64
    assert solver._sampled_opponent_actions.dtype == np.int8
    for policy in (solver.current_policy(), solver.average_policy()):
        for offset, count in zip(
            tree.information_set_action_offsets,
            tree.information_set_action_counts,
            strict=True,
        ):
            start = int(offset)
            np.testing.assert_allclose(np.sum(policy[start : start + int(count)]), 1.0)


def test_dense_mccfr_rejects_unsupported_games_and_invalid_counts() -> None:
    leduc_tree = compile_game_tree(LeducGame(), LeducConfig())
    solver = MCCFR(leduc_tree, seed=0)

    with pytest.raises(TypeError, match="iterations"):
        solver.train(True)
    with pytest.raises(ValueError, match="iterations"):
        solver.train(-1)
    with pytest.raises(TypeError, match="root_seed"):
        MCCFR(leduc_tree, seed=True)
    with pytest.raises(ValueError, match="only Leduc"):
        MCCFR(compile_game_tree(KuhnGame(), KuhnConfig()), seed=0)

    solver.train(1)
    regrets_before_restore = solver._regret_sum.copy()
    with pytest.raises(ValueError, match="RNG state"):
        solver.restore_training_state(
            iteration=99,
            regret_sum=solver._regret_sum,
            strategy_sum=solver._strategy_sum,
            rng_state={},
        )
    assert solver.iteration == 1
    assert np.array_equal(solver._regret_sum, regrets_before_restore)


def _flatten(table: tuple[tuple[float, ...], ...]) -> np.ndarray:
    return np.fromiter((value for row in table for value in row), dtype=np.float64)
