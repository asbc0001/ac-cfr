"""Dense full-tree counterfactual regret minimisation for Kuhn and Leduc poker."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ac_cfr.games.base import GameId, NodeType
from ac_cfr.games.tree import IndexedGameTree

_SUPPORTED_GAMES = (GameId.KUHN, GameId.LEDUC)
_CHANCE_NODE = int(NodeType.CHANCE)
_TERMINAL_NODE = int(NodeType.TERMINAL)


class CFR:
    """Alternating full-tree CFR using flat reusable NumPy storage."""

    _CLIP_REGRETS = False

    def __init__(self, tree: IndexedGameTree) -> None:
        if not isinstance(tree, IndexedGameTree):
            raise TypeError("tree must be an IndexedGameTree")
        if tree.game_id not in _SUPPORTED_GAMES:
            raise ValueError("tabular CFR supports only Kuhn and Leduc")

        self._tree = tree
        action_count = len(tree.information_set_actions)
        self._regret_sum = np.zeros(action_count, dtype=np.float64)
        self._strategy_sum = np.zeros(action_count, dtype=np.float64)
        self._current_policy = np.empty(action_count, dtype=np.float64)
        self._regret_delta = np.empty(action_count, dtype=np.float64)
        self._node_values = np.empty(tree.node_count, dtype=np.float64)
        self._reach_player_zero = np.empty(tree.node_count, dtype=np.float64)
        self._reach_player_one = np.empty(tree.node_count, dtype=np.float64)
        self._reach_chance = np.empty(tree.node_count, dtype=np.float64)
        self._information_set_representatives = tree.information_set_members[
            tree.information_set_member_offsets
        ]
        self._iteration = 0
        self._update_current_policy()

    @property
    def iteration(self) -> int:
        """Return the number of completed outer iterations."""
        return self._iteration

    @property
    def regret_sum(self) -> tuple[tuple[float, ...], ...]:
        """Return a read-only snapshot of cumulative regrets by information set."""
        return self._table_snapshot(self._regret_sum)

    @property
    def strategy_sum(self) -> tuple[tuple[float, ...], ...]:
        """Return a read-only snapshot of reach-weighted strategy sums."""
        return self._table_snapshot(self._strategy_sum)

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
        """Return a copy of the current regret-matched policy."""
        return self._current_policy.copy()

    def average_policy(self) -> NDArray[np.float64]:
        """Return the normalised reach-weighted policy used for play and evaluation."""
        average_policy = np.empty_like(self._strategy_sum)
        _normalise_information_sets(
            self._strategy_sum,
            average_policy,
            self._tree.information_set_action_offsets,
            self._tree.information_set_action_counts,
        )
        return average_policy

    def restore_training_state(
        self,
        *,
        iteration: int,
        regret_sum: NDArray[np.float64],
        strategy_sum: NDArray[np.float64],
    ) -> None:
        """Restore fully validated tabular state from a compatible checkpoint."""
        _validate_non_negative_integer("iteration", iteration)
        restored_regrets = self._validated_array(regret_sum, "regret_sum", non_negative=False)
        restored_strategies = self._validated_array(
            strategy_sum,
            "strategy_sum",
            non_negative=True,
        )
        self._validate_restored_regrets(restored_regrets)

        self._iteration = iteration
        self._regret_sum[:] = restored_regrets
        self._strategy_sum[:] = restored_strategies
        self._update_current_policy()

    def _run_player_pass(self, traverser: int, averaging_weight: float) -> None:
        tree = self._tree
        _run_player_pass_kernel(
            tree.node_types,
            tree.current_players,
            tree.information_set_ids,
            tree.child_offsets,
            tree.child_counts,
            tree.children,
            tree.chance_probabilities,
            tree.terminal_utilities,
            tree.information_set_players,
            tree.information_set_action_offsets,
            tree.information_set_action_counts,
            self._information_set_representatives,
            self._regret_sum,
            self._strategy_sum,
            self._current_policy,
            self._regret_delta,
            self._node_values,
            self._reach_player_zero,
            self._reach_player_one,
            self._reach_chance,
            traverser,
            averaging_weight,
            self._CLIP_REGRETS,
        )

    def _update_current_policy(self) -> None:
        _normalise_information_sets(
            self._regret_sum,
            self._current_policy,
            self._tree.information_set_action_offsets,
            self._tree.information_set_action_counts,
            True,
        )

    def _averaging_weight(self, iteration: int) -> float:
        return 1.0

    def _validate_restored_regrets(self, regrets: NDArray[np.float64]) -> None:
        return

    def _validated_array(
        self,
        values: NDArray[np.float64],
        name: str,
        *,
        non_negative: bool,
    ) -> NDArray[np.float64]:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if values.shape != self._regret_sum.shape:
            raise ValueError(f"{name} has an incompatible shape")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        if non_negative and np.any(values < 0.0):
            raise ValueError(f"{name} must not contain negative values")
        return np.asarray(values, dtype=np.float64)

    def _table_snapshot(
        self,
        values: NDArray[np.float64],
    ) -> tuple[tuple[float, ...], ...]:
        tree = self._tree
        rows: list[tuple[float, ...]] = []
        for offset, count in zip(
            tree.information_set_action_offsets,
            tree.information_set_action_counts,
            strict=True,
        ):
            action_start = int(offset)
            action_end = action_start + int(count)
            rows.append(tuple(values[action_start:action_end].tolist()))
        return tuple(rows)


@njit(cache=True)
def _run_player_pass_kernel(
    node_types: NDArray[np.uint8],
    current_players: NDArray[np.int8],
    information_set_ids: NDArray[np.int32],
    child_offsets: NDArray[np.int32],
    child_counts: NDArray[np.uint8],
    children: NDArray[np.int32],
    chance_probabilities: NDArray[np.float64],
    terminal_utilities: NDArray[np.float64],
    information_set_players: NDArray[np.int8],
    action_offsets: NDArray[np.int32],
    action_counts: NDArray[np.uint8],
    information_set_representatives: NDArray[np.int32],
    regret_sum: NDArray[np.float64],
    strategy_sum: NDArray[np.float64],
    current_policy: NDArray[np.float64],
    regret_delta: NDArray[np.float64],
    node_values: NDArray[np.float64],
    reach_zero: NDArray[np.float64],
    reach_one: NDArray[np.float64],
    reach_chance: NDArray[np.float64],
    traverser: int,
    averaging_weight: float,
    clip_regrets: bool,
) -> None:
    reach_zero[0] = 1.0
    reach_one[0] = 1.0
    reach_chance[0] = 1.0

    for node_id in range(len(node_types)):
        node_type = node_types[node_id]
        if node_type == _TERMINAL_NODE:
            continue
        edge_start = child_offsets[node_id]
        edge_count = child_counts[node_id]
        if node_type == _CHANCE_NODE:
            for action_position in range(edge_count):
                edge_id = edge_start + action_position
                child_id = children[edge_id]
                reach_zero[child_id] = reach_zero[node_id]
                reach_one[child_id] = reach_one[node_id]
                reach_chance[child_id] = reach_chance[node_id] * chance_probabilities[edge_id]
            continue

        information_set_id = information_set_ids[node_id]
        action_start = action_offsets[information_set_id]
        acting_player = current_players[node_id]
        for action_position in range(edge_count):
            child_id = children[edge_start + action_position]
            probability = current_policy[action_start + action_position]
            reach_zero[child_id] = reach_zero[node_id]
            reach_one[child_id] = reach_one[node_id]
            reach_chance[child_id] = reach_chance[node_id]
            if acting_player == 0:
                reach_zero[child_id] *= probability
            else:
                reach_one[child_id] *= probability

    regret_delta.fill(0.0)
    utility_sign = 1.0 if traverser == 0 else -1.0
    for node_id in range(len(node_types) - 1, -1, -1):
        node_type = node_types[node_id]
        if node_type == _TERMINAL_NODE:
            node_values[node_id] = utility_sign * terminal_utilities[node_id]
            continue
        edge_start = child_offsets[node_id]
        edge_count = child_counts[node_id]
        if node_type == _CHANCE_NODE:
            node_value = 0.0
            for action_position in range(edge_count):
                edge_id = edge_start + action_position
                node_value += chance_probabilities[edge_id] * node_values[children[edge_id]]
            node_values[node_id] = node_value
            continue

        information_set_id = information_set_ids[node_id]
        action_start = action_offsets[information_set_id]
        node_value = 0.0
        for action_position in range(edge_count):
            edge_id = edge_start + action_position
            node_value += (
                current_policy[action_start + action_position] * node_values[children[edge_id]]
            )
        node_values[node_id] = node_value

        acting_player = current_players[node_id]
        if acting_player != traverser:
            continue
        opponent_reach = reach_one[node_id] if traverser == 0 else reach_zero[node_id]
        counterfactual_reach = reach_chance[node_id] * opponent_reach
        for action_position in range(edge_count):
            child_id = children[edge_start + action_position]
            regret_delta[action_start + action_position] += counterfactual_reach * (
                node_values[child_id] - node_value
            )

    for action_id in range(len(regret_sum)):
        updated_regret = regret_sum[action_id] + regret_delta[action_id]
        regret_sum[action_id] = max(updated_regret, 0.0) if clip_regrets else updated_regret

    average_player = 1 - traverser
    player_reach = reach_zero if average_player == 0 else reach_one
    for information_set_id in range(len(information_set_players)):
        if information_set_players[information_set_id] != average_player:
            continue
        representative = information_set_representatives[information_set_id]
        weighted_reach = averaging_weight * player_reach[representative]
        action_start = action_offsets[information_set_id]
        action_count = action_counts[information_set_id]
        for action_position in range(action_count):
            action_id = action_start + action_position
            strategy_sum[action_id] += weighted_reach * current_policy[action_id]

    _normalise_information_sets(regret_sum, current_policy, action_offsets, action_counts, True)


@njit(cache=True)
def _normalise_information_sets(
    source: NDArray[np.float64],
    destination: NDArray[np.float64],
    action_offsets: NDArray[np.int32],
    action_counts: NDArray[np.uint8],
    positive_only: bool = False,
) -> None:
    for information_set_id in range(len(action_offsets)):
        action_start = action_offsets[information_set_id]
        action_count = action_counts[information_set_id]
        action_end = action_start + action_count
        total = 0.0
        for action_id in range(action_start, action_end):
            value = max(source[action_id], 0.0) if positive_only else source[action_id]
            destination[action_id] = value
            total += value
        if total > 0.0:
            destination[action_start:action_end] /= total
        else:
            destination[action_start:action_end] = 1.0 / action_count


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
