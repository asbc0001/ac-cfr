"""Dense external-sampling MCCFR for Leduc poker."""

import numpy as np
from numba import njit
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import GameId, NodeType
from ac_cfr.games.tree import IndexedGameTree
from ac_cfr.solvers.cfr import _normalise_information_sets, _validate_non_negative_integer

_CHANCE_NODE = int(NodeType.CHANCE)
_TERMINAL_NODE = int(NodeType.TERMINAL)


class MCCFR:
    """External-sampling MCCFR using dense storage and compiled traversal."""

    def __init__(self, tree: IndexedGameTree, *, seed: int) -> None:
        if not isinstance(tree, IndexedGameTree):
            raise TypeError("tree must be an IndexedGameTree")
        if tree.game_id is not GameId.LEDUC:
            raise ValueError("external-sampling MCCFR currently supports only Leduc")

        seed_deriver = SeedDeriver(seed)
        action_count = len(tree.information_set_actions)
        maximum_action_count = int(np.max(tree.information_set_action_counts))
        self._tree = tree
        self._seed = seed
        self._chance_rng = default_rng(seed_deriver.derive(RngStream.CHANCE))
        self._policy_rng = default_rng(seed_deriver.derive(RngStream.POLICY))
        self._regret_sum = np.zeros(action_count, dtype=np.float64)
        self._strategy_sum = np.zeros(action_count, dtype=np.float64)
        self._current_policy = np.empty(action_count, dtype=np.float64)
        self._sampled_opponent_actions = np.empty(tree.information_set_count, dtype=np.int8)
        frame_count = tree.max_depth + 1
        self._frame_edge_starts = np.empty(frame_count, dtype=np.int32)
        self._frame_information_sets = np.empty(frame_count, dtype=np.int32)
        self._frame_action_counts = np.empty(frame_count, dtype=np.uint8)
        self._frame_next_actions = np.empty(frame_count, dtype=np.uint8)
        self._frame_node_values = np.empty(frame_count, dtype=np.float64)
        self._frame_action_values = np.empty((frame_count, maximum_action_count), dtype=np.float64)
        self._frame_strategies = np.empty((frame_count, maximum_action_count), dtype=np.float64)
        self._iteration = 0
        self._update_current_policy()

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
        return self._table_snapshot(self._regret_sum)

    @property
    def strategy_sum(self) -> tuple[tuple[float, ...], ...]:
        """Return a read-only snapshot of sampled opponent-strategy sums."""
        return self._table_snapshot(self._strategy_sum)

    def train(self, iterations: int) -> None:
        """Run sampled Player-0-then-Player-1 outer iterations."""
        _validate_non_negative_integer("iterations", iterations)
        if iterations == 0:
            return

        tree = self._tree
        _train_kernel(
            tree.node_types,
            tree.current_players,
            tree.information_set_ids,
            tree.child_offsets,
            tree.child_counts,
            tree.children,
            tree.chance_probabilities,
            tree.terminal_utilities,
            tree.information_set_action_offsets,
            tree.information_set_action_counts,
            self._regret_sum,
            self._strategy_sum,
            self._current_policy,
            self._sampled_opponent_actions,
            self._frame_edge_starts,
            self._frame_information_sets,
            self._frame_action_counts,
            self._frame_next_actions,
            self._frame_node_values,
            self._frame_action_values,
            self._frame_strategies,
            self._chance_rng,
            self._policy_rng,
            iterations,
        )
        self._iteration += iterations

    def current_policy(self) -> NDArray[np.float64]:
        """Return a copy of the current regret-matched policy."""
        self._update_current_policy()
        return self._current_policy.copy()

    def average_policy(self) -> NDArray[np.float64]:
        """Return the normalised sampled average policy used for evaluation."""
        average_policy = np.empty_like(self._strategy_sum)
        _normalise_information_sets(
            self._strategy_sum,
            average_policy,
            self._tree.information_set_action_offsets,
            self._tree.information_set_action_counts,
        )
        return average_policy

    def _update_current_policy(self) -> None:
        """Rebuild every information-set strategy through regret matching."""
        _normalise_information_sets(
            self._regret_sum,
            self._current_policy,
            self._tree.information_set_action_offsets,
            self._tree.information_set_action_counts,
            True,
        )

    def _table_snapshot(
        self,
        values: NDArray[np.float64],
    ) -> tuple[tuple[float, ...], ...]:
        """Copy one flat action array into read-only information-set rows."""
        rows: list[tuple[float, ...]] = []
        for offset, count in zip(
            self._tree.information_set_action_offsets,
            self._tree.information_set_action_counts,
            strict=True,
        ):
            action_start = int(offset)
            action_end = action_start + int(count)
            rows.append(tuple(values[action_start:action_end].tolist()))
        return tuple(rows)


@njit(cache=True)
def _train_kernel(
    node_types: NDArray[np.uint8],
    current_players: NDArray[np.int8],
    information_set_ids: NDArray[np.int32],
    child_offsets: NDArray[np.int32],
    child_counts: NDArray[np.uint8],
    children: NDArray[np.int32],
    chance_probabilities: NDArray[np.float64],
    terminal_utilities: NDArray[np.float64],
    action_offsets: NDArray[np.int32],
    action_counts: NDArray[np.uint8],
    regret_sum: NDArray[np.float64],
    strategy_sum: NDArray[np.float64],
    current_policy: NDArray[np.float64],
    sampled_opponent_actions: NDArray[np.int8],
    frame_edge_starts: NDArray[np.int32],
    frame_information_sets: NDArray[np.int32],
    frame_action_counts: NDArray[np.uint8],
    frame_next_actions: NDArray[np.uint8],
    frame_node_values: NDArray[np.float64],
    frame_action_values: NDArray[np.float64],
    frame_strategies: NDArray[np.float64],
    chance_rng: Generator,
    policy_rng: Generator,
    iterations: int,
) -> None:
    """Run sequential sampled updates without crossing the Python boundary."""
    for _ in range(iterations):
        for traverser in range(2):
            sampled_opponent_actions.fill(-1)
            _run_traversal_kernel(
                traverser,
                node_types,
                current_players,
                information_set_ids,
                child_offsets,
                child_counts,
                children,
                chance_probabilities,
                terminal_utilities,
                action_offsets,
                action_counts,
                regret_sum,
                strategy_sum,
                current_policy,
                sampled_opponent_actions,
                frame_edge_starts,
                frame_information_sets,
                frame_action_counts,
                frame_next_actions,
                frame_node_values,
                frame_action_values,
                frame_strategies,
                chance_rng,
                policy_rng,
            )


@njit(cache=True)
def _run_traversal_kernel(
    traverser: int,
    node_types: NDArray[np.uint8],
    current_players: NDArray[np.int8],
    information_set_ids: NDArray[np.int32],
    child_offsets: NDArray[np.int32],
    child_counts: NDArray[np.uint8],
    children: NDArray[np.int32],
    chance_probabilities: NDArray[np.float64],
    terminal_utilities: NDArray[np.float64],
    action_offsets: NDArray[np.int32],
    action_counts: NDArray[np.uint8],
    regret_sum: NDArray[np.float64],
    strategy_sum: NDArray[np.float64],
    current_policy: NDArray[np.float64],
    sampled_opponent_actions: NDArray[np.int8],
    frame_edge_starts: NDArray[np.int32],
    frame_information_sets: NDArray[np.int32],
    frame_action_counts: NDArray[np.uint8],
    frame_next_actions: NDArray[np.uint8],
    frame_node_values: NDArray[np.float64],
    frame_action_values: NDArray[np.float64],
    frame_strategies: NDArray[np.float64],
    chance_rng: Generator,
    policy_rng: Generator,
) -> float:
    """Execute one sampled traversal using a reusable explicit stack."""
    node_id = 0
    active_frames = 0
    while True:
        node_type = node_types[node_id]
        if node_type == _TERMINAL_NODE:
            returned_value = terminal_utilities[node_id]
            if traverser == 1:
                returned_value = -returned_value
        else:
            edge_start = int(child_offsets[node_id])
            edge_count = int(child_counts[node_id])
            if node_type == _CHANCE_NODE:
                action_position = int(
                    _sample_position(
                        chance_probabilities,
                        edge_start,
                        edge_count,
                        chance_rng,
                    )
                )
                node_id = children[edge_start + action_position]
                continue

            information_set_id = int(information_set_ids[node_id])
            action_start = int(action_offsets[information_set_id])
            action_count = int(action_counts[information_set_id])
            _regret_match_row(regret_sum, current_policy, action_start, action_count)
            if current_players[node_id] != traverser:
                for action_position in range(action_count):
                    action_id = action_start + action_position
                    strategy_sum[action_id] += current_policy[action_id]

                action_position = int(sampled_opponent_actions[information_set_id])
                if action_position < 0:
                    action_position = int(
                        _sample_position(
                            current_policy,
                            action_start,
                            action_count,
                            policy_rng,
                        )
                    )
                    sampled_opponent_actions[information_set_id] = action_position
                node_id = children[edge_start + action_position]
                continue

            frame_id = active_frames
            frame_edge_starts[frame_id] = edge_start
            frame_information_sets[frame_id] = information_set_id
            frame_action_counts[frame_id] = action_count
            frame_next_actions[frame_id] = 0
            frame_node_values[frame_id] = 0.0
            for action_position in range(action_count):
                frame_strategies[frame_id, action_position] = current_policy[
                    action_start + action_position
                ]
            active_frames += 1
            node_id = children[edge_start]
            continue

        while active_frames > 0:
            frame_id = active_frames - 1
            action_position = frame_next_actions[frame_id]
            frame_action_values[frame_id, action_position] = returned_value
            frame_node_values[frame_id] += (
                frame_strategies[frame_id, action_position] * returned_value
            )
            action_position += 1
            frame_next_actions[frame_id] = action_position
            action_count = frame_action_counts[frame_id]
            if action_position < action_count:
                node_id = children[frame_edge_starts[frame_id] + action_position]
                break

            node_value = frame_node_values[frame_id]
            information_set_id = frame_information_sets[frame_id]
            action_start = action_offsets[information_set_id]
            for completed_action in range(action_count):
                regret_sum[action_start + completed_action] += (
                    frame_action_values[frame_id, completed_action] - node_value
                )
            returned_value = node_value
            active_frames -= 1
        if active_frames == 0:
            return returned_value


@njit(cache=True)
def _regret_match_row(
    regret_sum: NDArray[np.float64],
    current_policy: NDArray[np.float64],
    action_start: int,
    action_count: int,
) -> None:
    """Regret-match one flat information-set row in place."""
    positive_total = 0.0
    for action_position in range(action_count):
        action_id = action_start + action_position
        positive_regret = max(regret_sum[action_id], 0.0)
        current_policy[action_id] = positive_regret
        positive_total += positive_regret

    if positive_total > 0.0:
        for action_position in range(action_count):
            action_id = action_start + action_position
            current_policy[action_id] /= positive_total
    else:
        uniform_probability = 1.0 / action_count
        for action_position in range(action_count):
            current_policy[action_start + action_position] = uniform_probability


@njit(cache=True)
def _sample_position(
    probabilities: NDArray[np.float64],
    start: int,
    count: int,
    rng: Generator,
) -> int:
    """Sample a row position without consuming randomness for deterministic rows."""
    positive_count = 0
    only_positive_position = 0
    for position in range(count):
        if probabilities[start + position] > 0.0:
            positive_count += 1
            only_positive_position = position
    if positive_count == 1:
        return only_positive_position

    draw = rng.random()
    cumulative_probability = 0.0
    for position in range(count):
        cumulative_probability += probabilities[start + position]
        if draw < cumulative_probability:
            return position
    return count - 1
