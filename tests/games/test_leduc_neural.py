import numpy as np

from ac_cfr.games.base import ACTION_ORDER, Action
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.leduc_neural import (
    LEDUC_ACTION_COUNT,
    LEDUC_NEURAL_STATE_SIZE,
    build_leduc_neural_data,
)
from ac_cfr.games.tree import compile_game_tree


def test_leduc_neural_inputs_are_fixed_lossless_and_match_legal_actions() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    neural_data = build_leduc_neural_data(tree)

    assert neural_data.states.shape == (tree.information_set_count, LEDUC_NEURAL_STATE_SIZE)
    assert neural_data.action_masks.shape == (tree.information_set_count, LEDUC_ACTION_COUNT)
    assert neural_data.states.dtype == np.float32
    assert neural_data.action_masks.dtype == np.bool
    assert len(np.unique(neural_data.states, axis=0)) == tree.information_set_count
    assert not neural_data.states.flags.writeable
    assert not neural_data.action_masks.flags.writeable

    for information_set_id in range(tree.information_set_count):
        action_offset = int(tree.information_set_action_offsets[information_set_id])
        action_count = int(tree.information_set_action_counts[information_set_id])
        legal_actions = {
            Action(int(action))
            for action in tree.information_set_actions[action_offset : action_offset + action_count]
        }
        assert neural_data.action_masks[information_set_id].tolist() == [
            action in legal_actions for action in ACTION_ORDER
        ]
