"""Reference external-sampling MCCFR for Leduc poker."""

from math import fsum
from random import Random

import numpy as np
from numpy.typing import NDArray

from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import GameId, NodeType
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.solvers.naive_cfr import _normalise, _validate_non_negative_integer


class NaiveMCCFR:
    """Straightforward external-sampling MCCFR correctness reference."""

    def __init__(self, tree: IndexedGameTree, *, seed: int) -> None:
        if not isinstance(tree, IndexedGameTree):
            raise TypeError("tree must be an IndexedGameTree")
        if tree.game_id is not GameId.LEDUC:
            raise ValueError("external-sampling MCCFR currently supports only Leduc")

        seed_deriver = SeedDeriver(seed)
        self._tree = tree
        self._seed = seed
        self._chance_rng = seed_deriver.python_rng(RngStream.CHANCE)
        self._policy_rng = seed_deriver.python_rng(RngStream.POLICY)
        self._regret_sum = self._empty_table()
        self._strategy_sum = self._empty_table()
        self._iteration = 0

    @property
    def iteration(self) -> int:
        """Return the number of completed outer iterations."""
        return self._iteration

    @property
    def seed(self) -> int:
        """Return the reproducible root seed used by this solver."""
        return self._seed

    @property
    def regret_sum(self) -> tuple[tuple[float, ...], ...]:
        """Return a read-only snapshot of cumulative sampled regrets."""
        return tuple(tuple(regrets) for regrets in self._regret_sum)

    @property
    def strategy_sum(self) -> tuple[tuple[float, ...], ...]:
        """Return a read-only snapshot of sampled opponent-strategy sums."""
        return tuple(tuple(strategy) for strategy in self._strategy_sum)

    def train(self, iterations: int) -> None:
        """Run sampled Player-0-then-Player-1 outer iterations."""
        _validate_non_negative_integer("iterations", iterations)
        for _ in range(iterations):
            self._run_player_traversal(traverser=0)
            self._run_player_traversal(traverser=1)
            self._iteration += 1

    def current_policy(self) -> NDArray[np.float64]:
        """Return the current regret-matched policy in flat tree action order."""
        return self._flatten_policy(self._regret_matched_policy())

    def average_policy(self) -> NDArray[np.float64]:
        """Return the normalised sampled average policy used for evaluation."""
        average_policy = tuple(_normalise(weights) for weights in self._strategy_sum)
        return self._flatten_policy(average_policy)

    def _run_player_traversal(self, traverser: int) -> float:
        """Run one sampled traversal with a fresh opponent-action cache."""
        sampled_opponent_actions: dict[int, int] = {}
        return self._traverse(
            node_id=0,
            traverser=traverser,
            sampled_opponent_actions=sampled_opponent_actions,
        )

    def _traverse(
        self,
        node_id: int,
        traverser: int,
        sampled_opponent_actions: dict[int, int],
    ) -> float:
        """Sample chance/opponent nodes and fully explore traverser actions."""
        tree = self._tree
        node_type = NodeType(tree.node_types[node_id])
        if node_type is NodeType.TERMINAL:
            utility = float(tree.terminal_utilities[node_id])
            return utility if traverser == 0 else -utility

        edge_start = int(tree.child_offsets[node_id])
        edge_count = int(tree.child_counts[node_id])
        if node_type is NodeType.CHANCE:
            probabilities = tree.chance_probabilities[edge_start : edge_start + edge_count]
            action_position = _sample_position(probabilities, self._chance_rng)
            return self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                sampled_opponent_actions=sampled_opponent_actions,
            )

        acting_player = int(tree.current_players[node_id])
        information_set_id = int(tree.information_set_ids[node_id])
        strategy = _normalise([max(regret, 0.0) for regret in self._regret_sum[information_set_id]])
        if acting_player != traverser:
            for action_position, probability in enumerate(strategy):
                self._strategy_sum[information_set_id][action_position] += probability

            action_position = sampled_opponent_actions.get(information_set_id)
            if action_position is None:
                action_position = _sample_position(strategy, self._policy_rng)
                sampled_opponent_actions[information_set_id] = action_position
            return self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                sampled_opponent_actions=sampled_opponent_actions,
            )

        action_values = [
            self._traverse(
                node_id=int(tree.children[edge_start + action_position]),
                traverser=traverser,
                sampled_opponent_actions=sampled_opponent_actions,
            )
            for action_position in range(edge_count)
        ]
        node_value = fsum(
            probability * action_value
            for probability, action_value in zip(strategy, action_values, strict=True)
        )
        for action_position, action_value in enumerate(action_values):
            self._regret_sum[information_set_id][action_position] += action_value - node_value
        return node_value

    def _regret_matched_policy(self) -> tuple[tuple[float, ...], ...]:
        """Build every current information-set strategy from positive regrets."""
        return tuple(
            _normalise([max(regret, 0.0) for regret in regrets]) for regrets in self._regret_sum
        )

    def _empty_table(self) -> list[list[float]]:
        return [
            [0.0] * int(action_count) for action_count in self._tree.information_set_action_counts
        ]

    def _flatten_policy(
        self,
        policy: tuple[tuple[float, ...], ...],
    ) -> NDArray[np.float64]:
        """Flatten policy rows into stable tree action order."""
        return np.fromiter(
            (probability for strategy in policy for probability in strategy),
            dtype=np.float64,
            count=sum(len(strategy) for strategy in policy),
        )


def _sample_position(probabilities: tuple[float, ...] | NDArray[np.float64], rng: Random) -> int:
    """Sample one position while avoiding RNG consumption for deterministic rows."""
    positive_positions = [
        position for position, probability in enumerate(probabilities) if probability > 0.0
    ]
    if len(positive_positions) == 1:
        return positive_positions[0]

    draw = rng.random()
    cumulative_probability = 0.0
    for position, probability in enumerate(probabilities):
        cumulative_probability += float(probability)
        if draw < cumulative_probability:
            return position
    return len(probabilities) - 1
