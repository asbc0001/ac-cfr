import numpy as np
import pytest

from ac_cfr.games.base import NodeType
from ac_cfr.games.kuhn import KuhnConfig, KuhnGame
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import NO_INFORMATION_SET, NO_PLAYER, compile_game_tree


def test_dense_trees_are_complete_stable_and_read_only() -> None:
    cases = (
        (KuhnGame(), KuhnConfig(), 55, 12, 4),
        (LeducGame(), LeducConfig(), 9451, 936, 10),
    )
    for game, configuration, expected_nodes, expected_information_sets, expected_depth in cases:
        first = compile_game_tree(game, configuration)
        second = compile_game_tree(game, configuration)

        assert first.node_count == expected_nodes
        assert first.information_set_count == expected_information_sets
        assert first.max_depth == expected_depth
        assert np.array_equal(first.children, second.children)
        assert np.array_equal(first.information_set_ids, second.information_set_ids)
        assert len(first.children) == first.node_count - 1
        assert sorted(first.children.tolist()) == list(range(1, first.node_count))
        assert len(first.nodes_by_depth) == first.node_count
        assert set(first.nodes_by_depth.tolist()) == set(range(first.node_count))

        with pytest.raises(ValueError, match="read-only"):
            first.node_types[0] = NodeType.PLAYER
        with pytest.raises(ValueError):
            first.node_types.setflags(write=True)


def test_dense_tree_edges_information_sets_and_precomputation_are_consistent() -> None:
    for game, configuration in ((KuhnGame(), KuhnConfig()), (LeducGame(), LeducConfig())):
        tree = compile_game_tree(game, configuration)
        assert tree.node_types.dtype == np.uint8
        assert tree.children.dtype == np.int32
        assert tree.terminal_utilities.dtype == np.float64

        for node_id in range(tree.node_count):
            node_type = NodeType(tree.node_types[node_id])
            edge_start = tree.child_offsets[node_id]
            edge_end = edge_start + tree.child_counts[node_id]
            if node_type is NodeType.CHANCE:
                assert sum(tree.chance_probabilities[edge_start:edge_end]) == pytest.approx(1.0)
                assert np.all(tree.chance_multiplicities[edge_start:edge_end] == 1)
                assert tree.current_players[node_id] == NO_PLAYER
            elif node_type is NodeType.PLAYER:
                information_set_id = tree.information_set_ids[node_id]
                action_start = tree.information_set_action_offsets[information_set_id]
                action_end = action_start + tree.information_set_action_counts[information_set_id]
                assert np.array_equal(
                    tree.edge_labels[edge_start:edge_end],
                    tree.information_set_actions[action_start:action_end],
                )
                assert np.all(tree.chance_probabilities[edge_start:edge_end] == 0.0)
            else:
                assert tree.child_counts[node_id] == 0
                assert tree.current_players[node_id] == NO_PLAYER
                assert tree.information_set_ids[node_id] == NO_INFORMATION_SET

        for information_set_id in range(tree.information_set_count):
            member_start = tree.information_set_member_offsets[information_set_id]
            member_end = member_start + tree.information_set_member_counts[information_set_id]
            members = tree.information_set_members[member_start:member_end]
            assert np.all(tree.information_set_ids[members] == information_set_id)
            assert np.all(
                tree.current_players[members] == tree.information_set_players[information_set_id]
            )
