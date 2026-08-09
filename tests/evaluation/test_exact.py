from collections.abc import Iterator
from itertools import product

import numpy as np
import pytest
from numpy.typing import NDArray

from ac_cfr.evaluation import ExactEvaluator, evaluate_strategy
from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree


def test_kuhn_equilibrium_has_known_value_and_zero_exploitability() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    policy = np.array(
        [
            # Each pair follows the tree's stable information-set ID and action order.
            2 / 3,
            1 / 3,
            1.0,
            0.0,
            1.0,
            0.0,
            2 / 3,
            1 / 3,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            2 / 3,
            1 / 3,
            1 / 3,
            2 / 3,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )

    metrics = evaluate_strategy(tree, policy)

    assert metrics.expected_values == pytest.approx((-1 / 18, 1 / 18))
    assert metrics.best_response_values == pytest.approx((-1 / 18, 1 / 18))
    assert metrics.improvements == pytest.approx((0.0, 0.0), abs=1e-12)
    assert metrics.nash_conv == pytest.approx(0.0, abs=1e-12)
    assert metrics.exploitability == pytest.approx(0.0, abs=1e-12)


def test_kuhn_best_response_matches_enumeration_of_information_set_policies() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    evaluator = ExactEvaluator(tree)
    uniform_policy = _uniform_policy(tree)

    for player in range(2):
        enumerated_value = max(
            evaluator.expected_value(candidate, player)
            for candidate in _pure_responses(tree, uniform_policy, player)
        )
        assert evaluator.best_response_value(uniform_policy, player) == pytest.approx(
            enumerated_value
        )


def test_leduc_uniform_policy_has_exact_reproducible_metrics() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    metrics = evaluate_strategy(tree, _uniform_policy(tree))

    assert metrics.expected_values == pytest.approx((-5 / 64, 5 / 64))
    assert metrics.best_response_values == pytest.approx((167 / 80, 383 / 144))
    assert metrics.nash_conv == pytest.approx(sum(metrics.improvements))
    assert metrics.exploitability == pytest.approx(metrics.nash_conv / 2)


def test_policy_validation_rejects_invalid_distributions() -> None:
    tree = compile_game_tree(KuhnGame(), KuhnConfig())
    evaluator = ExactEvaluator(tree)
    policy = _uniform_policy(tree)

    with pytest.raises(ValueError, match="exactly"):
        evaluator.expected_value(policy[:-1])
    policy[0] = -0.5
    with pytest.raises(ValueError, match="non-negative"):
        evaluator.expected_value(policy)
    policy[0] = 0.25
    with pytest.raises(ValueError, match="sum to 1"):
        evaluator.expected_value(policy)


def _uniform_policy(tree: IndexedGameTree) -> NDArray[np.float64]:
    policy_size = int(
        tree.information_set_action_offsets[-1] + tree.information_set_action_counts[-1]
    )
    policy = np.empty(policy_size, dtype=np.float64)
    for information_set_id in range(tree.information_set_count):
        action_start = int(tree.information_set_action_offsets[information_set_id])
        action_count = int(tree.information_set_action_counts[information_set_id])
        policy[action_start : action_start + action_count] = 1.0 / action_count
    return policy


def _pure_responses(
    tree: IndexedGameTree,
    opponent_policy: NDArray[np.float64],
    player: int,
) -> Iterator[NDArray[np.float64]]:
    information_sets = np.flatnonzero(tree.information_set_players == player)
    action_counts = [int(tree.information_set_action_counts[index]) for index in information_sets]
    for choices in product(*(range(count) for count in action_counts)):
        policy = opponent_policy.copy()
        for information_set_id_value, choice in zip(information_sets, choices, strict=True):
            information_set_id = int(information_set_id_value)
            action_start = int(tree.information_set_action_offsets[information_set_id])
            action_count = int(tree.information_set_action_counts[information_set_id])
            policy[action_start : action_start + action_count] = 0.0
            policy[action_start + choice] = 1.0
        yield policy
