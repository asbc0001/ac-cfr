"""Exact expected-value and best-response evaluation for Kuhn and Leduc poker."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.base import GameId, NodeType, validate_player
from ac_cfr.games.tree import IndexedGameTree

Policy = Sequence[float] | NDArray[np.float64]
_SUPPORTED_GAMES = (GameId.KUHN, GameId.LEDUC)


class ExactEvaluator:
    """Evaluate complete tabular policies on a shared indexed game tree.

    Instances reuse mutable NumPy work arrays and must not be shared between
    concurrent evaluations. Create one evaluator per concurrent caller.
    """

    def __init__(self, tree: IndexedGameTree) -> None:
        if not isinstance(tree, IndexedGameTree):
            raise TypeError("tree must be an IndexedGameTree")
        if tree.game_id not in _SUPPORTED_GAMES:
            raise ValueError("exact evaluation supports only Kuhn and Leduc")

        self._tree = tree
        self._node_values = np.empty(tree.node_count, dtype=np.float64)
        self._counterfactual_reach = np.empty(tree.node_count, dtype=np.float64)
        self._information_set_depths = self._validate_information_set_depths()
        self._information_sets_by_player_and_depth = tuple(
            tuple(
                tuple(
                    information_set_id
                    for information_set_id in range(tree.information_set_count)
                    if tree.information_set_players[information_set_id] == player
                    and self._information_set_depths[information_set_id] == depth
                )
                for depth in range(tree.max_depth + 1)
            )
            for player in range(2)
        )
        maximum_action_count = int(tree.information_set_action_counts.max(initial=0))
        self._action_values = np.empty(maximum_action_count, dtype=np.float64)
        self._policy_size = self._calculate_policy_size()

    @property
    def policy_size(self) -> int:
        """Return the required number of flattened action probabilities."""
        return self._policy_size

    def expected_value(self, policy: Policy, player: int = 0) -> float:
        """Return one player's exact utility under a complete policy profile."""
        validate_player(player)
        probabilities = self._validate_policy(policy)
        return self._expected_value(probabilities, player)

    def best_response_value(self, policy: Policy, player: int) -> float:
        """Return one player's exact value when best responding to the opponent."""
        validate_player(player)
        probabilities = self._validate_policy(policy)
        return self._best_response_value(probabilities, player)

    def profile_values(self, policy: Policy) -> tuple[float, float, float]:
        """Return player-zero value followed by both players' best-response values."""
        probabilities = self._validate_policy(policy)
        return (
            self._expected_value(probabilities, player=0),
            self._best_response_value(probabilities, player=0),
            self._best_response_value(probabilities, player=1),
        )

    def _expected_value(self, policy: NDArray[np.float64], player: int) -> float:
        tree = self._tree
        values = self._node_values
        utility_sign = 1.0 if player == 0 else -1.0
        values[tree.terminal_nodes] = utility_sign * tree.terminal_utilities[tree.terminal_nodes]

        for depth in range(tree.max_depth - 1, -1, -1):
            depth_start = int(tree.depth_offsets[depth])
            depth_end = int(tree.depth_offsets[depth + 1])
            for node_id_value in tree.nodes_by_depth[depth_start:depth_end]:
                node_id = int(node_id_value)
                edge_start = int(tree.child_offsets[node_id])
                edge_end = edge_start + int(tree.child_counts[node_id])
                if tree.node_types[node_id] == NodeType.CHANCE:
                    values[node_id] = np.dot(
                        tree.chance_probabilities[edge_start:edge_end],
                        values[tree.children[edge_start:edge_end]],
                    )
                elif tree.node_types[node_id] == NodeType.PLAYER:
                    information_set_id = int(tree.information_set_ids[node_id])
                    action_start = int(tree.information_set_action_offsets[information_set_id])
                    action_end = action_start + int(
                        tree.information_set_action_counts[information_set_id]
                    )
                    values[node_id] = np.dot(
                        policy[action_start:action_end],
                        values[tree.children[edge_start:edge_end]],
                    )
        return float(values[0])

    def _best_response_value(
        self,
        policy: NDArray[np.float64],
        player: int,
    ) -> float:
        tree = self._tree
        reach = self._counterfactual_reach
        reach.fill(0.0)
        reach[0] = 1.0

        for node_id in range(tree.node_count):
            edge_start = int(tree.child_offsets[node_id])
            edge_end = edge_start + int(tree.child_counts[node_id])
            node_type = tree.node_types[node_id]
            if node_type == NodeType.CHANCE:
                reach[tree.children[edge_start:edge_end]] = (
                    reach[node_id] * tree.chance_probabilities[edge_start:edge_end]
                )
            elif node_type == NodeType.PLAYER:
                children = tree.children[edge_start:edge_end]
                if tree.current_players[node_id] == player:
                    reach[children] = reach[node_id]
                else:
                    information_set_id = int(tree.information_set_ids[node_id])
                    action_start = int(tree.information_set_action_offsets[information_set_id])
                    action_end = action_start + int(
                        tree.information_set_action_counts[information_set_id]
                    )
                    reach[children] = reach[node_id] * policy[action_start:action_end]

        values = self._node_values
        utility_sign = 1.0 if player == 0 else -1.0
        values[tree.terminal_nodes] = utility_sign * tree.terminal_utilities[tree.terminal_nodes]

        for depth in range(tree.max_depth - 1, -1, -1):
            depth_start = int(tree.depth_offsets[depth])
            depth_end = int(tree.depth_offsets[depth + 1])
            for node_id_value in tree.nodes_by_depth[depth_start:depth_end]:
                node_id = int(node_id_value)
                edge_start = int(tree.child_offsets[node_id])
                edge_end = edge_start + int(tree.child_counts[node_id])
                node_type = tree.node_types[node_id]
                if node_type == NodeType.CHANCE:
                    values[node_id] = np.dot(
                        tree.chance_probabilities[edge_start:edge_end],
                        values[tree.children[edge_start:edge_end]],
                    )
                elif node_type == NodeType.PLAYER and tree.current_players[node_id] != player:
                    information_set_id = int(tree.information_set_ids[node_id])
                    action_start = int(tree.information_set_action_offsets[information_set_id])
                    action_end = action_start + int(
                        tree.information_set_action_counts[information_set_id]
                    )
                    values[node_id] = np.dot(
                        policy[action_start:action_end],
                        values[tree.children[edge_start:edge_end]],
                    )

            for information_set_id in self._information_sets_by_player_and_depth[player][depth]:
                self._evaluate_best_response_information_set(information_set_id, values, reach)

        return float(values[0])

    def _evaluate_best_response_information_set(
        self,
        information_set_id: int,
        values: NDArray[np.float64],
        reach: NDArray[np.float64],
    ) -> None:
        tree = self._tree
        member_start = int(tree.information_set_member_offsets[information_set_id])
        member_end = member_start + int(tree.information_set_member_counts[information_set_id])
        members = tree.information_set_members[member_start:member_end]
        action_count = int(tree.information_set_action_counts[information_set_id])
        action_values = self._action_values[:action_count]
        action_values.fill(0.0)

        for node_id_value in members:
            node_id = int(node_id_value)
            edge_start = int(tree.child_offsets[node_id])
            children = tree.children[edge_start : edge_start + action_count]
            action_values += reach[node_id] * values[children]

        best_action_position = int(np.argmax(action_values))
        for node_id_value in members:
            node_id = int(node_id_value)
            edge_id = int(tree.child_offsets[node_id]) + best_action_position
            values[node_id] = values[int(tree.children[edge_id])]

    def _validate_policy(self, policy: Policy) -> NDArray[np.float64]:
        try:
            probabilities = np.array(policy, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as error:
            raise TypeError("policy must contain real probabilities") from error
        if probabilities.ndim != 1 or len(probabilities) != self._policy_size:
            raise ValueError(f"policy must contain exactly {self._policy_size} probabilities")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("policy probabilities must be finite")
        if np.any(probabilities < 0.0):
            raise ValueError("policy probabilities must be non-negative")

        tree = self._tree
        for information_set_id in range(tree.information_set_count):
            action_start = int(tree.information_set_action_offsets[information_set_id])
            action_end = action_start + int(tree.information_set_action_counts[information_set_id])
            if not np.isclose(
                probabilities[action_start:action_end].sum(),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("policy probabilities must sum to 1 at every information set")
        return probabilities

    def _validate_information_set_depths(self) -> NDArray[np.uint8]:
        tree = self._tree
        depths = np.empty(tree.information_set_count, dtype=np.uint8)
        for information_set_id in range(tree.information_set_count):
            member_start = int(tree.information_set_member_offsets[information_set_id])
            member_end = member_start + int(tree.information_set_member_counts[information_set_id])
            members = tree.information_set_members[member_start:member_end]
            member_depths = tree.depths[members]
            if len(member_depths) == 0 or np.any(member_depths != member_depths[0]):
                raise ValueError("information-set members must share one tree depth")
            depths[information_set_id] = member_depths[0]
        return depths

    def _calculate_policy_size(self) -> int:
        tree = self._tree
        if tree.information_set_count == 0:
            return 0
        last_information_set = tree.information_set_count - 1
        return int(
            tree.information_set_action_offsets[last_information_set]
            + tree.information_set_action_counts[last_information_set]
        )
