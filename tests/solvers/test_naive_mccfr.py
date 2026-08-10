from random import Random

import numpy as np
import pytest
from numpy.typing import DTypeLike

from ac_cfr.games.base import Action, GameId, NodeType
from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.solvers import NaiveMCCFR


class _ScriptedRandom(Random):
    def __init__(self, draws: list[float]) -> None:
        super().__init__(0)
        self._draws = iter(draws)
        self.call_count = 0

    def random(self) -> float:
        self.call_count += 1
        return next(self._draws)


def test_external_sampling_updates_and_averages_in_sequence() -> None:
    solver = NaiveMCCFR(_small_sampling_tree(), seed=7)
    solver._chance_rng = _ScriptedRandom([0.5, 0.5])
    solver._policy_rng = _ScriptedRandom([0.75])

    assert np.all(solver.current_policy() == 0.5)
    assert np.all(solver.average_policy() == 0.5)

    solver.train(1)

    assert solver.iteration == 1
    # The sampled 0.75-probability chance branch is not an explicit regret multiplier.
    np.testing.assert_allclose(solver.regret_sum, ((-2.0, 2.0), (1.0, -1.0)))
    # Both players acted as the sampled opponent once; repeated members add twice.
    np.testing.assert_allclose(solver.strategy_sum, ((0.0, 1.0), (1.0, 1.0)))
    # Player 1's regrets above use the negative of Player 0's stored terminal utility.
    assert solver.current_policy() == pytest.approx((0.0, 1.0, 1.0, 0.0))
    assert solver.average_policy() == pytest.approx((0.0, 1.0, 0.5, 0.5))
    assert solver._chance_rng.call_count == 2
    # One sampled opponent action is reused at both members of its information set.
    assert solver._policy_rng.call_count == 1


def test_seeded_leduc_training_is_reproducible_and_seed_dependent() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    first = NaiveMCCFR(tree, seed=2026)
    second = NaiveMCCFR(tree, seed=2026)
    different_seed = NaiveMCCFR(tree, seed=2027)

    for solver in (first, second, different_seed):
        solver.train(5)

    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum
    assert first.regret_sum != different_seed.regret_sum
    assert first.strategy_sum != different_seed.strategy_sum


def test_naive_mccfr_rejects_unsupported_games_and_invalid_counts() -> None:
    leduc_tree = compile_game_tree(LeducGame(), LeducConfig())
    solver = NaiveMCCFR(leduc_tree, seed=0)

    with pytest.raises(TypeError, match="iterations"):
        solver.train(True)
    with pytest.raises(ValueError, match="iterations"):
        solver.train(-1)
    with pytest.raises(TypeError, match="root_seed"):
        NaiveMCCFR(leduc_tree, seed=True)
    with pytest.raises(ValueError, match="only Leduc"):
        NaiveMCCFR(compile_game_tree(KuhnGame(), KuhnConfig()), seed=0)


def _small_sampling_tree() -> IndexedGameTree:
    return IndexedGameTree(
        game_id=GameId.LEDUC,
        node_types=_array(
            (
                NodeType.CHANCE,
                NodeType.TERMINAL,
                NodeType.PLAYER,
                NodeType.PLAYER,
                NodeType.TERMINAL,
                NodeType.TERMINAL,
                NodeType.PLAYER,
                NodeType.TERMINAL,
                NodeType.TERMINAL,
            ),
            np.uint8,
        ),
        current_players=_array((-1, -1, 0, 1, -1, -1, 1, -1, -1), np.int8),
        information_set_ids=_array((-1, -1, 0, 1, -1, -1, 1, -1, -1), np.int32),
        child_offsets=_array((0, 2, 2, 4, 6, 6, 6, 8, 8), np.int32),
        child_counts=_array((2, 0, 2, 2, 0, 0, 2, 0, 0), np.uint8),
        children=_array((1, 2, 3, 6, 4, 5, 7, 8), np.int32),
        edge_labels=_array(
            (
                0,
                1,
                Action.CHECK_CALL,
                Action.BET_RAISE,
                Action.CHECK_CALL,
                Action.BET_RAISE,
                Action.CHECK_CALL,
                Action.BET_RAISE,
            ),
            np.int16,
        ),
        chance_probabilities=_array((0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), np.float64),
        chance_multiplicities=_array((1, 1, 0, 0, 0, 0, 0, 0), np.uint8),
        terminal_utilities=_array((0.0, 99.0, 0.0, 0.0, 1.0, 3.0, 0.0, 5.0, 7.0), np.float64),
        depths=_array((0, 1, 1, 2, 3, 3, 2, 3, 3), np.uint8),
        chance_nodes=_array((0,), np.int32),
        player_nodes=_array((2, 3, 6), np.int32),
        terminal_nodes=_array((1, 4, 5, 7, 8), np.int32),
        depth_offsets=_array((0, 1, 3, 5, 9), np.int32),
        nodes_by_depth=_array((0, 1, 2, 3, 6, 4, 5, 7, 8), np.int32),
        information_set_players=_array((0, 1), np.int8),
        information_set_action_offsets=_array((0, 2), np.int32),
        information_set_action_counts=_array((2, 2), np.uint8),
        information_set_actions=_array(
            (Action.CHECK_CALL, Action.BET_RAISE, Action.CHECK_CALL, Action.BET_RAISE),
            np.uint8,
        ),
        information_set_encoding_offsets=_array((0, 1), np.int32),
        information_set_encoding_counts=_array((1, 1), np.uint8),
        information_set_encodings=_array((0, 1), np.int16),
        information_set_member_offsets=_array((0, 1), np.int32),
        information_set_member_counts=_array((1, 2), np.int32),
        information_set_members=_array((2, 3, 6), np.int32),
    )


def _array(values: tuple[object, ...], dtype: DTypeLike) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array
