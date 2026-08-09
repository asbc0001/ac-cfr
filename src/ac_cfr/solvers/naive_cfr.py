"""Readable full-tree counterfactual regret minimisation for Kuhn and Leduc poker."""

from math import fsum, isclose

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.base import GameId, NodeType
from ac_cfr.games.tree import IndexedGameTree

_SUPPORTED_GAMES = (GameId.KUHN, GameId.LEDUC)


class NaiveCFR:
    """Straightforward alternating full-tree CFR correctness reference."""

    def __init__(self, tree: IndexedGameTree) -> None:
        if not isinstance(tree, IndexedGameTree):
            raise TypeError("tree must be an IndexedGameTree")
        if tree.game_id not in _SUPPORTED_GAMES:
            raise ValueError("tabular CFR supports only Kuhn and Leduc")

        self._tree = tree
        self._regret_sum = self._empty_table()
        self._strategy_sum = self._empty_table()
        self._current_policy = self._regret_matched_policy()
        self._iteration = 0

    @property
    def iteration(self) -> int:
        """Return the number of completed outer iterations."""
        return self._iteration

    @property
    def regret_sum(self) -> tuple[tuple[float, ...], ...]:
        """Return a read-only snapshot of cumulative regrets by information set."""
        return tuple(tuple(regrets) for regrets in self._regret_sum)

    @property
    def strategy_sum(self) -> tuple[tuple[float, ...], ...]:
        """Return a read-only snapshot of reach-weighted strategy sums."""
        return tuple(tuple(strategy) for strategy in self._strategy_sum)

    def train(self, iterations: int) -> None:
        """Run a number of Player-0-then-Player-1 outer iterations."""
        _validate_non_negative_integer("iterations", iterations)
        for _ in range(iterations):
            next_iteration = self._iteration + 1
            averaging_weight = self._averaging_weight(next_iteration)
            self._run_player_pass(traverser=0, averaging_weight=averaging_weight)
            self._run_player_pass(traverser=1, averaging_weight=averaging_weight)
            self._iteration = next_iteration

    def current_policy(self) -> NDArray[np.float64]:
        """Return the current regret-matched policy in flat tree action order."""
        return self._flatten_policy(self._current_policy)

    def average_policy(self) -> NDArray[np.float64]:
        """Return the normalised reach-weighted policy used for play and evaluation."""
        average_policy = tuple(_normalise(weights) for weights in self._strategy_sum)
        return self._flatten_policy(average_policy)

    def _run_player_pass(self, traverser: int, averaging_weight: float) -> None:
        # This immutable tuple is the only policy read during the complete pass.
        frozen_policy = self._current_policy
        regret_delta = self._empty_table()
        strategy_delta = self._empty_table()
        recorded_average_reach: dict[int, float] = {}
        self._traverse(
            node_id=0,
            traverser=traverser,
            reach_player_zero=1.0,
            reach_player_one=1.0,
            reach_chance=1.0,
            frozen_policy=frozen_policy,
            regret_delta=regret_delta,
            strategy_delta=strategy_delta,
            recorded_average_reach=recorded_average_reach,
            averaging_weight=averaging_weight,
        )
        self._apply_regret_delta(regret_delta)
        self._apply_strategy_delta(strategy_delta)
        self._current_policy = self._regret_matched_policy()

    def _traverse(
        self,
        node_id: int,
        traverser: int,
        reach_player_zero: float,
        reach_player_one: float,
        reach_chance: float,
        frozen_policy: tuple[tuple[float, ...], ...],
        regret_delta: list[list[float]],
        strategy_delta: list[list[float]],
        recorded_average_reach: dict[int, float],
        averaging_weight: float,
    ) -> float:
        tree = self._tree
        node_type = NodeType(tree.node_types[node_id])
        if node_type is NodeType.TERMINAL:
            utility = float(tree.terminal_utilities[node_id])
            return utility if traverser == 0 else -utility

        edge_start = int(tree.child_offsets[node_id])
        edge_count = int(tree.child_counts[node_id])
        if node_type is NodeType.CHANCE:
            node_value = 0.0
            for action_position in range(edge_count):
                edge_id = edge_start + action_position
                probability = float(tree.chance_probabilities[edge_id])
                node_value += probability * self._traverse(
                    node_id=int(tree.children[edge_id]),
                    traverser=traverser,
                    reach_player_zero=reach_player_zero,
                    reach_player_one=reach_player_one,
                    reach_chance=reach_chance * probability,
                    frozen_policy=frozen_policy,
                    regret_delta=regret_delta,
                    strategy_delta=strategy_delta,
                    recorded_average_reach=recorded_average_reach,
                    averaging_weight=averaging_weight,
                )
            return node_value

        acting_player = int(tree.current_players[node_id])
        information_set_id = int(tree.information_set_ids[node_id])
        strategy = frozen_policy[information_set_id]
        if acting_player != traverser:
            own_reach = reach_player_zero if acting_player == 0 else reach_player_one
            self._record_average_strategy(
                information_set_id,
                own_reach,
                strategy,
                strategy_delta,
                recorded_average_reach,
                averaging_weight,
            )

        action_values: list[float] = []
        for action_position, probability in enumerate(strategy):
            child_reach_zero = reach_player_zero
            child_reach_one = reach_player_one
            if acting_player == 0:
                child_reach_zero *= probability
            else:
                child_reach_one *= probability
            action_values.append(
                self._traverse(
                    node_id=int(tree.children[edge_start + action_position]),
                    traverser=traverser,
                    reach_player_zero=child_reach_zero,
                    reach_player_one=child_reach_one,
                    reach_chance=reach_chance,
                    frozen_policy=frozen_policy,
                    regret_delta=regret_delta,
                    strategy_delta=strategy_delta,
                    recorded_average_reach=recorded_average_reach,
                    averaging_weight=averaging_weight,
                )
            )

        node_value = fsum(
            probability * action_value
            for probability, action_value in zip(strategy, action_values, strict=True)
        )
        if acting_player == traverser:
            opponent_reach = reach_player_one if traverser == 0 else reach_player_zero
            counterfactual_reach = reach_chance * opponent_reach
            for action_position, action_value in enumerate(action_values):
                regret_delta[information_set_id][action_position] += counterfactual_reach * (
                    action_value - node_value
                )
        return node_value

    def _record_average_strategy(
        self,
        information_set_id: int,
        own_reach: float,
        strategy: tuple[float, ...],
        strategy_delta: list[list[float]],
        recorded_average_reach: dict[int, float],
        averaging_weight: float,
    ) -> None:
        previous_reach = recorded_average_reach.get(information_set_id)
        if previous_reach is not None:
            if not isclose(previous_reach, own_reach, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError("information-set members have inconsistent player reach")
            return

        recorded_average_reach[information_set_id] = own_reach
        for action_position, probability in enumerate(strategy):
            strategy_delta[information_set_id][action_position] = (
                averaging_weight * own_reach * probability
            )

    def _apply_regret_delta(self, regret_delta: list[list[float]]) -> None:
        for information_set_id, delta in enumerate(regret_delta):
            for action_position, value in enumerate(delta):
                self._regret_sum[information_set_id][action_position] += value

    def _apply_strategy_delta(self, strategy_delta: list[list[float]]) -> None:
        for information_set_id, delta in enumerate(strategy_delta):
            for action_position, value in enumerate(delta):
                self._strategy_sum[information_set_id][action_position] += value

    def _averaging_weight(self, iteration: int) -> float:
        return 1.0

    def _regret_matched_policy(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            _normalise(tuple(max(regret, 0.0) for regret in regrets))
            for regrets in self._regret_sum
        )

    def _empty_table(self) -> list[list[float]]:
        return [
            [0.0] * int(action_count) for action_count in self._tree.information_set_action_counts
        ]

    def _flatten_policy(
        self,
        policy: tuple[tuple[float, ...], ...],
    ) -> NDArray[np.float64]:
        return np.fromiter(
            (probability for strategy in policy for probability in strategy),
            dtype=np.float64,
            count=sum(len(strategy) for strategy in policy),
        )


def _normalise(weights: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    total = fsum(weights)
    if total > 0.0:
        return tuple(weight / total for weight in weights)
    probability = 1.0 / len(weights)
    return tuple(probability for _ in weights)


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
