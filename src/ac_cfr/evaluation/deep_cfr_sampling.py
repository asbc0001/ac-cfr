"""Exact Leduc gate for exploratory opponent sampling in Deep CFR."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.base import NodeType
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.solvers.naive_deep_cfr import (
    exploratory_opponent_probabilities,
    opponent_importance_ratio,
)


@dataclass(frozen=True, slots=True)
class ExploratorySamplingValidation:
    """Errors and weight-health measurements from one fixed-policy Leduc gate."""

    sample_count: int
    epsilon: float
    seed: int
    advantage_reach_weighted_rmse: float
    strategy_reach_weighted_rmse: float
    expected_information_sets: int
    observed_information_sets: int
    maximum_importance_ratio: float
    effective_sample_size: float
    weighted_sample_count: int

    @property
    def effective_sample_fraction(self) -> float:
        """Return effective samples divided by the number of weighted samples."""
        if self.weighted_sample_count == 0:
            return 0.0
        return self.effective_sample_size / self.weighted_sample_count

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-safe validation measurements."""
        return {
            "sample_count": self.sample_count,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "advantage_reach_weighted_rmse": self.advantage_reach_weighted_rmse,
            "strategy_reach_weighted_rmse": self.strategy_reach_weighted_rmse,
            "expected_information_sets": self.expected_information_sets,
            "observed_information_sets": self.observed_information_sets,
            "maximum_importance_ratio": self.maximum_importance_ratio,
            "effective_sample_size": self.effective_sample_size,
            "effective_sample_fraction": self.effective_sample_fraction,
            "weighted_sample_count": self.weighted_sample_count,
        }


@dataclass(slots=True)
class _WeightedEstimates:
    """Accumulate ratio-weighted targets, visitation mass, and diagnostics."""

    numerators: NDArray[np.float64]
    denominators: NDArray[np.float64]
    weights: list[float]
    ratios: list[float]


def validate_exploratory_sampling_estimators(
    *,
    sample_count: int = 100_000,
    epsilon: float = 0.1,
    seed: int = 20260814,
) -> ExploratorySamplingValidation:
    """Compare corrected sampled estimates with full-tree Leduc expectations."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if epsilon not in (0.0, 0.1):
        raise ValueError("epsilon must be zero or 0.1")

    tree = compile_game_tree(LeducGame(), LeducConfig())
    # Dense, non-uniform policies exercise every legal action without making the
    # validation depend on trained network weights.
    profiles = (_fixed_policy(tree, 0), _fixed_policy(tree, 1))
    rng = Random(seed)

    advantage_squared_error = 0.0
    advantage_error_weight = 0.0
    expected_information_sets = 0
    observed_information_sets = 0
    all_weights: list[float] = []
    all_ratios: list[float] = []
    for traverser in (0, 1):
        exact = _exact_advantages(tree, profiles[0], traverser)
        sampled = _sample_advantages(
            tree,
            profiles[0],
            traverser,
            sample_count,
            epsilon,
            rng,
        )
        squared_error, error_weight, expected, observed = _reach_weighted_error(
            tree, exact, sampled, traverser
        )
        advantage_squared_error += squared_error
        advantage_error_weight += error_weight
        expected_information_sets += expected
        observed_information_sets += observed
        all_weights.extend(sampled.weights)
        all_ratios.extend(sampled.ratios)

    exact_strategy = _exact_average_strategy(tree, profiles)
    sampled_strategy = _sample_average_strategy(
        tree,
        profiles,
        sample_count,
        epsilon,
        rng,
    )
    strategy_squared_error, strategy_error_weight, expected, observed = _reach_weighted_error(
        tree, exact_strategy, sampled_strategy, None
    )
    expected_information_sets += expected
    observed_information_sets += observed
    all_weights.extend(sampled_strategy.weights)
    all_ratios.extend(sampled_strategy.ratios)
    # ESS describes the sampling correction alone. Linear-CFR iteration weights
    # belong to the learning objective and would confound this diagnostic.
    weight_values = np.asarray(all_weights, dtype=np.float64)
    weight_sum = float(weight_values.sum())
    squared_weight_sum = float(np.square(weight_values).sum())

    return ExploratorySamplingValidation(
        sample_count=sample_count,
        epsilon=epsilon,
        seed=seed,
        advantage_reach_weighted_rmse=float(
            (advantage_squared_error / advantage_error_weight) ** 0.5
            if advantage_error_weight > 0.0
            else float("inf")
        ),
        strategy_reach_weighted_rmse=float(
            (strategy_squared_error / strategy_error_weight) ** 0.5
            if strategy_error_weight > 0.0
            else float("inf")
        ),
        expected_information_sets=expected_information_sets,
        observed_information_sets=observed_information_sets,
        maximum_importance_ratio=max(all_ratios, default=1.0),
        effective_sample_size=(
            weight_sum * weight_sum / squared_weight_sum if squared_weight_sum > 0.0 else 0.0
        ),
        weighted_sample_count=len(all_weights),
    )


def _fixed_policy(tree: IndexedGameTree, profile: int) -> NDArray[np.float64]:
    policy = np.zeros((tree.information_set_count, LEDUC_ACTION_COUNT), dtype=np.float64)
    for information_set_id in range(tree.information_set_count):
        start = int(tree.information_set_action_offsets[information_set_id])
        count = int(tree.information_set_action_counts[information_set_id])
        actions = tree.information_set_actions[start : start + count]
        raw = np.asarray(
            [1.0 + ((information_set_id + int(action) + profile) % 3) for action in actions],
            dtype=np.float64,
        )
        policy[information_set_id, actions] = raw / raw.sum()
    return policy


def _node_values(
    tree: IndexedGameTree,
    policy: NDArray[np.float64],
    traverser: int,
) -> NDArray[np.float64]:
    values = np.zeros(tree.node_count, dtype=np.float64)
    for node_id in range(tree.node_count - 1, -1, -1):
        node_type = NodeType(tree.node_types[node_id])
        if node_type is NodeType.TERMINAL:
            utility = float(tree.terminal_utilities[node_id])
            values[node_id] = utility if traverser == 0 else -utility
            continue
        start = int(tree.child_offsets[node_id])
        count = int(tree.child_counts[node_id])
        children = tree.children[start : start + count]
        if node_type is NodeType.CHANCE:
            probabilities = tree.chance_probabilities[start : start + count]
        else:
            information_set_id = int(tree.information_set_ids[node_id])
            actions = tree.edge_labels[start : start + count]
            probabilities = policy[information_set_id, actions]
        values[node_id] = float(np.dot(probabilities, values[children]))
    return values


def _exact_advantages(
    tree: IndexedGameTree,
    policy: NDArray[np.float64],
    traverser: int,
) -> _WeightedEstimates:
    node_values = _node_values(tree, policy, traverser)
    estimates = _empty_estimates(tree)

    def visit(node_id: int, external_reach: float) -> None:
        node_type = NodeType(tree.node_types[node_id])
        if node_type is NodeType.TERMINAL:
            return
        start = int(tree.child_offsets[node_id])
        count = int(tree.child_counts[node_id])
        if node_type is NodeType.CHANCE:
            for position in range(count):
                visit(
                    int(tree.children[start + position]),
                    external_reach * float(tree.chance_probabilities[start + position]),
                )
            return
        acting_player = int(tree.current_players[node_id])
        information_set_id = int(tree.information_set_ids[node_id])
        actions = tree.edge_labels[start : start + count]
        if acting_player == traverser:
            # Counterfactual reach includes chance and opponent reach, but excludes
            # the traverser's own policy.
            estimates.denominators[information_set_id] += external_reach
            for position, action in enumerate(actions):
                estimates.numerators[information_set_id, action] += external_reach * (
                    node_values[int(tree.children[start + position])] - node_values[node_id]
                )
                visit(int(tree.children[start + position]), external_reach)
            return
        for position, action in enumerate(actions):
            visit(
                int(tree.children[start + position]),
                external_reach * policy[information_set_id, action],
            )

    visit(0, 1.0)
    return estimates


def _sample_advantages(
    tree: IndexedGameTree,
    policy: NDArray[np.float64],
    traverser: int,
    sample_count: int,
    epsilon: float,
    rng: Random,
) -> _WeightedEstimates:
    estimates = _empty_estimates(tree)

    def traverse(
        node_id: int,
        prefix_weight: float,
        sampled_actions: dict[int, int],
    ) -> float:
        node_type = NodeType(tree.node_types[node_id])
        if node_type is NodeType.TERMINAL:
            utility = float(tree.terminal_utilities[node_id])
            return utility if traverser == 0 else -utility
        start = int(tree.child_offsets[node_id])
        count = int(tree.child_counts[node_id])
        if node_type is NodeType.CHANCE:
            probabilities = tree.chance_probabilities[start : start + count]
            position = _sample_position(probabilities, rng)
            return traverse(int(tree.children[start + position]), prefix_weight, sampled_actions)

        acting_player = int(tree.current_players[node_id])
        information_set_id = int(tree.information_set_ids[node_id])
        actions = tree.edge_labels[start : start + count]
        local_strategy = policy[information_set_id, actions]
        if acting_player != traverser:
            position = sampled_actions.get(information_set_id)
            if position is None:
                behaviour = exploratory_opponent_probabilities(local_strategy, epsilon)
                position = _sample_position(behaviour, rng)
                sampled_actions[information_set_id] = position
            ratio = opponent_importance_ratio(local_strategy, position, epsilon)
            estimates.ratios.append(ratio)
            # The prefix ratio corrects descendant visitation. The same local
            # ratio corrects this sampled action's return on the way back up.
            child_value = traverse(
                int(tree.children[start + position]),
                prefix_weight * ratio,
                sampled_actions,
            )
            return child_value if epsilon == 0.0 else ratio * child_value

        action_values = np.asarray(
            [
                traverse(int(tree.children[start + position]), prefix_weight, sampled_actions)
                for position in range(count)
            ],
            dtype=np.float64,
        )
        node_value = float(np.dot(local_strategy, action_values))
        estimates.denominators[information_set_id] += prefix_weight
        estimates.weights.append(prefix_weight)
        for position, action in enumerate(actions):
            estimates.numerators[information_set_id, action] += prefix_weight * (
                action_values[position] - node_value
            )
        return node_value

    for _ in range(sample_count):
        traverse(0, 1.0, {})
    return estimates


def _exact_average_strategy(
    tree: IndexedGameTree,
    profiles: tuple[NDArray[np.float64], ...],
) -> _WeightedEstimates:
    estimates = _empty_estimates(tree)
    for iteration, policy in enumerate(profiles, start=1):
        for player in (0, 1):

            def visit(
                node_id: int,
                reach: float,
                *,
                current_iteration: int = iteration,
                current_policy: NDArray[np.float64] = policy,
                current_player: int = player,
            ) -> None:
                node_type = NodeType(tree.node_types[node_id])
                if node_type is NodeType.TERMINAL:
                    return
                start = int(tree.child_offsets[node_id])
                count = int(tree.child_counts[node_id])
                if node_type is NodeType.CHANCE:
                    for position in range(count):
                        visit(
                            int(tree.children[start + position]),
                            reach * float(tree.chance_probabilities[start + position]),
                        )
                    return
                acting_player = int(tree.current_players[node_id])
                information_set_id = int(tree.information_set_ids[node_id])
                actions = tree.edge_labels[start : start + count]
                if acting_player == current_player:
                    # This is the policy owner's reach contribution to Linear
                    # CFR's iteration-weighted average strategy.
                    weight = current_iteration * reach
                    estimates.denominators[information_set_id] += weight
                    estimates.numerators[information_set_id] += (
                        weight * current_policy[information_set_id]
                    )
                    for position, action in enumerate(actions):
                        visit(
                            int(tree.children[start + position]),
                            reach * current_policy[information_set_id, action],
                        )
                    return
                for position in range(count):
                    visit(int(tree.children[start + position]), reach)

            visit(0, 1.0)
    return estimates


def _sample_average_strategy(
    tree: IndexedGameTree,
    profiles: tuple[NDArray[np.float64], ...],
    sample_count: int,
    epsilon: float,
    rng: Random,
) -> _WeightedEstimates:
    estimates = _empty_estimates(tree)
    for iteration, policy in enumerate(profiles, start=1):
        for player in (0, 1):

            def traverse(
                node_id: int,
                prefix_weight: float,
                sampled_actions: dict[int, int],
                *,
                current_iteration: int = iteration,
                current_policy: NDArray[np.float64] = policy,
                current_player: int = player,
            ) -> None:
                node_type = NodeType(tree.node_types[node_id])
                if node_type is NodeType.TERMINAL:
                    return
                start = int(tree.child_offsets[node_id])
                count = int(tree.child_counts[node_id])
                if node_type is NodeType.CHANCE:
                    probabilities = tree.chance_probabilities[start : start + count]
                    position = _sample_position(probabilities, rng)
                    traverse(int(tree.children[start + position]), prefix_weight, sampled_actions)
                    return
                acting_player = int(tree.current_players[node_id])
                information_set_id = int(tree.information_set_ids[node_id])
                actions = tree.edge_labels[start : start + count]
                if acting_player == current_player:
                    # Store the true policy. Prefix importance weights correct only
                    # the changed probability of reaching this information set.
                    weight = current_iteration * prefix_weight
                    estimates.denominators[information_set_id] += weight
                    estimates.numerators[information_set_id] += (
                        weight * current_policy[information_set_id]
                    )
                    estimates.weights.append(prefix_weight)
                    local_strategy = current_policy[information_set_id, actions]
                    position = sampled_actions.get(information_set_id)
                    if position is None:
                        behaviour = exploratory_opponent_probabilities(local_strategy, epsilon)
                        position = _sample_position(behaviour, rng)
                        sampled_actions[information_set_id] = position
                    ratio = opponent_importance_ratio(local_strategy, position, epsilon)
                    estimates.ratios.append(ratio)
                    traverse(
                        int(tree.children[start + position]),
                        prefix_weight * ratio,
                        sampled_actions,
                    )
                    return
                for position in range(count):
                    traverse(int(tree.children[start + position]), prefix_weight, sampled_actions)

            for _ in range(sample_count):
                traverse(0, 1.0, {})
    return estimates


def _empty_estimates(tree: IndexedGameTree) -> _WeightedEstimates:
    return _WeightedEstimates(
        numerators=np.zeros((tree.information_set_count, LEDUC_ACTION_COUNT), dtype=np.float64),
        denominators=np.zeros(tree.information_set_count, dtype=np.float64),
        weights=[],
        ratios=[],
    )


def _reach_weighted_error(
    tree: IndexedGameTree,
    exact: _WeightedEstimates,
    sampled: _WeightedEstimates,
    player: int | None,
) -> tuple[float, float, int, int]:
    squared_error = 0.0
    error_weight = 0.0
    expected_information_sets = 0
    observed_information_sets = 0
    for information_set_id in range(tree.information_set_count):
        if player is not None and int(tree.information_set_players[information_set_id]) != player:
            continue
        if exact.denominators[information_set_id] <= 0.0:
            continue
        expected_information_sets += 1
        if sampled.denominators[information_set_id] <= 0.0:
            continue
        observed_information_sets += 1
        start = int(tree.information_set_action_offsets[information_set_id])
        count = int(tree.information_set_action_counts[information_set_id])
        actions = tree.information_set_actions[start : start + count]
        exact_values = (
            exact.numerators[information_set_id, actions] / exact.denominators[information_set_id]
        )
        sampled_values = (
            sampled.numerators[information_set_id, actions]
            / sampled.denominators[information_set_id]
        )
        action_errors = np.square(exact_values - sampled_values)
        squared_error += exact.denominators[information_set_id] * float(action_errors.sum())
        error_weight += exact.denominators[information_set_id] * len(actions)
    return squared_error, error_weight, expected_information_sets, observed_information_sets


def _sample_position(
    probabilities: tuple[float, ...] | NDArray[np.float64],
    rng: Random,
) -> int:
    draw = rng.random()
    cumulative = 0.0
    for position, probability in enumerate(probabilities):
        cumulative += float(probability)
        if draw < cumulative:
            return position
    return len(probabilities) - 1
