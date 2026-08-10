"""Shared dense indexed-tree compiler for small tabular games."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.base import (
    Action,
    DeterministicIdRegistry,
    ExtensiveFormGame,
    ExtensiveFormState,
    GameConfiguration,
    GameId,
    NodeType,
)

NO_PLAYER = -1
NO_INFORMATION_SET = -1


@dataclass(frozen=True, slots=True)
class IndexedGameTree:
    """Read-only flat arrays describing one complete deterministic game tree."""

    game_id: GameId
    node_types: NDArray[np.uint8]
    current_players: NDArray[np.int8]
    information_set_ids: NDArray[np.int32]
    child_offsets: NDArray[np.int32]
    child_counts: NDArray[np.uint8]
    children: NDArray[np.int32]
    edge_labels: NDArray[np.int16]
    chance_probabilities: NDArray[np.float64]
    chance_multiplicities: NDArray[np.uint8]
    terminal_utilities: NDArray[np.float64]
    depths: NDArray[np.uint8]
    chance_nodes: NDArray[np.int32]
    player_nodes: NDArray[np.int32]
    terminal_nodes: NDArray[np.int32]
    depth_offsets: NDArray[np.int32]
    nodes_by_depth: NDArray[np.int32]
    information_set_players: NDArray[np.int8]
    information_set_action_offsets: NDArray[np.int32]
    information_set_action_counts: NDArray[np.uint8]
    information_set_actions: NDArray[np.uint8]
    information_set_encoding_offsets: NDArray[np.int32]
    information_set_encoding_counts: NDArray[np.uint8]
    information_set_encodings: NDArray[np.int16]
    information_set_member_offsets: NDArray[np.int32]
    information_set_member_counts: NDArray[np.int32]
    information_set_members: NDArray[np.int32]

    @property
    def node_count(self) -> int:
        """Return the number of indexed tree nodes."""
        return len(self.node_types)

    @property
    def information_set_count(self) -> int:
        """Return the number of indexed player information sets."""
        return len(self.information_set_players)

    @property
    def max_depth(self) -> int:
        """Return the greatest root-relative node depth."""
        return len(self.depth_offsets) - 2


@dataclass(slots=True)
class _InformationSetRecord:
    """Mutable information-set data collected while compiling the tree."""

    player: int
    encoding: tuple[int, ...]
    legal_actions: tuple[Action, ...]
    members: list[int]


class _TreeCompiler:
    """Compile game states into stable dense node and information-set arrays."""

    def __init__(self, game_id: GameId) -> None:
        self.game_id = game_id
        self.node_types: list[NodeType] = []
        self.current_players: list[int] = []
        self.information_set_ids: list[int] = []
        self.terminal_utilities: list[float] = []
        self.depths: list[int] = []
        self.edges_by_node: list[list[tuple[int, int, float, int]]] = []
        self.information_set_registry: DeterministicIdRegistry[tuple[int, tuple[int, ...]]] = (
            DeterministicIdRegistry()
        )
        self.information_sets: list[_InformationSetRecord] = []

    def compile(self, root: ExtensiveFormState) -> IndexedGameTree:
        """Visit every state from the root and pack the collected structure."""
        self._visit(root, depth=0)
        return self._build_tree()

    def _visit(self, state: ExtensiveFormState, depth: int) -> int:
        """Append one state recursively and return its assigned node ID."""
        node_id = len(self.node_types)
        node_type = state.node_type
        self.node_types.append(node_type)
        self.current_players.append(NO_PLAYER)
        self.information_set_ids.append(NO_INFORMATION_SET)
        self.terminal_utilities.append(0.0)
        self.depths.append(depth)
        self.edges_by_node.append([])

        if node_type is NodeType.TERMINAL:
            self.terminal_utilities[node_id] = state.utility(0)
            return node_id
        if node_type is NodeType.CHANCE:
            for outcome in state.chance_outcomes():
                child_id = self._visit(state.apply_action(outcome.outcome), depth + 1)
                self.edges_by_node[node_id].append(
                    (child_id, outcome.outcome, outcome.probability, outcome.multiplicity)
                )
            return node_id

        information_state = state.information_state()
        player = information_state.player
        key = player, information_state.encoding
        information_set_id = self.information_set_registry.assign(key)
        if information_set_id == len(self.information_sets):
            self.information_sets.append(
                _InformationSetRecord(
                    player,
                    information_state.encoding,
                    information_state.legal_actions,
                    [node_id],
                )
            )
        else:
            record = self.information_sets[information_set_id]
            if record.legal_actions != information_state.legal_actions:
                raise ValueError("information-set members have inconsistent legal actions")
            record.members.append(node_id)

        self.current_players[node_id] = player
        self.information_set_ids[node_id] = information_set_id
        for action in information_state.legal_actions:
            child_id = self._visit(state.apply_action(action), depth + 1)
            self.edges_by_node[node_id].append((child_id, int(action), 0.0, 0))
        return node_id

    def _build_tree(self) -> IndexedGameTree:
        """Pack collected Python records into immutable typed arrays."""
        child_offsets: list[int] = []
        child_counts: list[int] = []
        children: list[int] = []
        edge_labels: list[int] = []
        probabilities: list[float] = []
        multiplicities: list[int] = []
        for edges in self.edges_by_node:
            child_offsets.append(len(children))
            child_counts.append(len(edges))
            for child, label, probability, multiplicity in edges:
                children.append(child)
                edge_labels.append(label)
                probabilities.append(probability)
                multiplicities.append(multiplicity)

        nodes_by_type = {
            node_type: [node for node, value in enumerate(self.node_types) if value is node_type]
            for node_type in NodeType
        }
        depth_offsets, nodes_by_depth = _group_nodes_by_depth(self.depths)
        action_offsets, action_counts, actions = _flatten_information_set_actions(
            self.information_sets
        )
        encoding_offsets, encoding_counts, encodings = _flatten_information_set_encodings(
            self.information_sets
        )
        member_offsets, member_counts, members = _flatten_information_set_members(
            self.information_sets
        )

        return IndexedGameTree(
            game_id=self.game_id,
            node_types=_read_only(self.node_types, np.uint8),
            current_players=_read_only(self.current_players, np.int8),
            information_set_ids=_read_only(self.information_set_ids, np.int32),
            child_offsets=_read_only(child_offsets, np.int32),
            child_counts=_read_only(child_counts, np.uint8),
            children=_read_only(children, np.int32),
            edge_labels=_read_only(edge_labels, np.int16),
            chance_probabilities=_read_only(probabilities, np.float64),
            chance_multiplicities=_read_only(multiplicities, np.uint8),
            terminal_utilities=_read_only(self.terminal_utilities, np.float64),
            depths=_read_only(self.depths, np.uint8),
            chance_nodes=_read_only(nodes_by_type[NodeType.CHANCE], np.int32),
            player_nodes=_read_only(nodes_by_type[NodeType.PLAYER], np.int32),
            terminal_nodes=_read_only(nodes_by_type[NodeType.TERMINAL], np.int32),
            depth_offsets=_read_only(depth_offsets, np.int32),
            nodes_by_depth=_read_only(nodes_by_depth, np.int32),
            information_set_players=_read_only(
                [record.player for record in self.information_sets], np.int8
            ),
            information_set_action_offsets=_read_only(action_offsets, np.int32),
            information_set_action_counts=_read_only(action_counts, np.uint8),
            information_set_actions=_read_only(actions, np.uint8),
            information_set_encoding_offsets=_read_only(encoding_offsets, np.int32),
            information_set_encoding_counts=_read_only(encoding_counts, np.uint8),
            information_set_encodings=_read_only(encodings, np.int16),
            information_set_member_offsets=_read_only(member_offsets, np.int32),
            information_set_member_counts=_read_only(member_counts, np.int32),
            information_set_members=_read_only(members, np.int32),
        )


def compile_game_tree(
    game: ExtensiveFormGame,
    configuration: GameConfiguration,
) -> IndexedGameTree:
    """Compile one complete small game in deterministic depth-first order."""
    if game.game_id != configuration.game_id:
        raise ValueError("game and configuration identifiers do not match")
    compiler = _TreeCompiler(game.game_id)
    return compiler.compile(game.initial_state(configuration))


def _group_nodes_by_depth(depths: list[int]) -> tuple[list[int], list[int]]:
    """Flatten node IDs by depth and return offsets into that ordering."""
    grouped_nodes = [[] for _ in range(max(depths) + 1)]
    for node_id, depth in enumerate(depths):
        grouped_nodes[depth].append(node_id)

    offsets: list[int] = []
    flattened: list[int] = []
    for nodes in grouped_nodes:
        offsets.append(len(flattened))
        flattened.extend(nodes)
    offsets.append(len(flattened))
    return offsets, flattened


def _flatten_information_set_actions(
    records: list[_InformationSetRecord],
) -> tuple[list[int], list[int], list[Action]]:
    """Pack legal-action rows and return each row's offset and length."""
    offsets: list[int] = []
    counts: list[int] = []
    actions: list[Action] = []
    for record in records:
        offsets.append(len(actions))
        counts.append(len(record.legal_actions))
        actions.extend(record.legal_actions)
    return offsets, counts, actions


def _flatten_information_set_members(
    records: list[_InformationSetRecord],
) -> tuple[list[int], list[int], list[int]]:
    """Pack member-node rows and return each row's offset and length."""
    offsets: list[int] = []
    counts: list[int] = []
    members: list[int] = []
    for record in records:
        offsets.append(len(members))
        counts.append(len(record.members))
        members.extend(record.members)
    return offsets, counts, members


def _flatten_information_set_encodings(
    records: list[_InformationSetRecord],
) -> tuple[list[int], list[int], list[int]]:
    """Pack information-state encodings and return row offsets and lengths."""
    offsets: list[int] = []
    counts: list[int] = []
    encodings: list[int] = []
    for record in records:
        offsets.append(len(encodings))
        counts.append(len(record.encoding))
        encodings.extend(record.encoding)
    return offsets, counts, encodings


def _read_only[ScalarType: np.generic](
    values: Sequence[object],
    dtype: type[ScalarType],
) -> NDArray[ScalarType]:
    """Copy values into a compact immutable array backed by bytes."""
    packed_values = np.asarray(values, dtype=dtype).tobytes()
    return np.frombuffer(packed_values, dtype=dtype)
