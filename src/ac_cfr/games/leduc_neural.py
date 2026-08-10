"""Fixed neural-network inputs for canonical Leduc information states."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ac_cfr.common.config import StateEncodingId
from ac_cfr.games.base import (
    ACTION_ORDER,
    Action,
    GameId,
    InformationState,
    validate_legal_actions,
)
from ac_cfr.games.leduc import LEDUC_DECK
from ac_cfr.games.tree import IndexedGameTree

LEDUC_NEURAL_STATE_ENCODING_ID = StateEncodingId.LEDUC_NEURAL
LEDUC_NEURAL_STATE_SIZE = 37
LEDUC_ACTION_COUNT = len(ACTION_ORDER)
# Every feature is already a binary one-hot indicator, so no further scaling is applied.
LEDUC_NEURAL_INPUT_SCALING = "binary_0_1"

_FIRST_ROUND_ACTION_SLOTS = 4
_SECOND_ROUND_ACTION_SLOTS = 3
_HISTORY_SEPARATOR = len(Action)


@dataclass(frozen=True, slots=True)
class LeducNeuralData:
    """Precomputed neural inputs and legal-action masks by information-set ID."""

    states: NDArray[np.float32]
    action_masks: NDArray[np.bool]


def encode_leduc_information_state(information_state: InformationState) -> NDArray[np.float32]:
    """Encode one player-visible Leduc decision as a fixed 37-value vector.

    The stable layout contains one-hot player, private-card, optional public-card,
    round and seven ordered action-slot fields. An unused card or action slot is
    all zero, so no hidden opponent card is needed.
    """
    if not isinstance(information_state, InformationState):
        raise TypeError("information_state must be an InformationState")
    if information_state.game_id is not GameId.LEDUC:
        raise ValueError("information_state must belong to Leduc")

    private_card, public_card, round_index, histories = _parse_encoding(information_state.encoding)
    state = np.zeros(LEDUC_NEURAL_STATE_SIZE, dtype=np.float32)
    offset = 0

    state[offset + information_state.player] = 1.0
    offset += 2
    state[offset + private_card] = 1.0
    offset += len(LEDUC_DECK)
    if public_card >= 0:
        state[offset + public_card] = 1.0
    offset += len(LEDUC_DECK)
    state[offset + round_index] = 1.0
    offset += 2

    history_slots: tuple[Action | None, ...] = histories[0] + (None,) * (
        _FIRST_ROUND_ACTION_SLOTS - len(histories[0])
    )
    if round_index == 1:
        history_slots += histories[1] + (None,) * (_SECOND_ROUND_ACTION_SLOTS - len(histories[1]))
    else:
        history_slots += (None,) * _SECOND_ROUND_ACTION_SLOTS

    for slot, action in enumerate(history_slots):
        if action is not None:
            state[offset + slot * LEDUC_ACTION_COUNT + int(action)] = 1.0
    state.setflags(write=False)
    return state


def leduc_action_mask(legal_actions: tuple[Action, ...]) -> NDArray[np.bool]:
    """Return a mask aligned with the global fold, check/call, bet/raise order."""
    validate_legal_actions(legal_actions)
    mask = np.zeros(LEDUC_ACTION_COUNT, dtype=np.bool)
    for action in legal_actions:
        mask[int(action)] = True
    mask.setflags(write=False)
    return mask


def build_leduc_neural_data(tree: IndexedGameTree) -> LeducNeuralData:
    """Precompute every Leduc information-set input and action mask."""
    if not isinstance(tree, IndexedGameTree):
        raise TypeError("tree must be an IndexedGameTree")
    if tree.game_id is not GameId.LEDUC:
        raise ValueError("tree must describe Leduc")

    states = np.empty((tree.information_set_count, LEDUC_NEURAL_STATE_SIZE), dtype=np.float32)
    action_masks = np.empty((tree.information_set_count, LEDUC_ACTION_COUNT), dtype=np.bool)
    for information_set_id in range(tree.information_set_count):
        encoding_offset = int(tree.information_set_encoding_offsets[information_set_id])
        encoding_count = int(tree.information_set_encoding_counts[information_set_id])
        action_offset = int(tree.information_set_action_offsets[information_set_id])
        action_count = int(tree.information_set_action_counts[information_set_id])
        information_state = InformationState(
            game_id=GameId.LEDUC,
            player=int(tree.information_set_players[information_set_id]),
            encoding=tuple(
                int(value)
                for value in tree.information_set_encodings[
                    encoding_offset : encoding_offset + encoding_count
                ]
            ),
            legal_actions=tuple(
                Action(int(value))
                for value in tree.information_set_actions[
                    action_offset : action_offset + action_count
                ]
            ),
        )
        states[information_set_id] = encode_leduc_information_state(information_state)
        action_masks[information_set_id] = leduc_action_mask(information_state.legal_actions)

    states.setflags(write=False)
    action_masks.setflags(write=False)
    return LeducNeuralData(states=states, action_masks=action_masks)


def _parse_encoding(
    encoding: tuple[int, ...],
) -> tuple[int, int, int, tuple[tuple[Action, ...], ...]]:
    """Validate and split the canonical variable-length Leduc encoding."""
    if len(encoding) < 4:
        raise ValueError("Leduc information-state encoding is incomplete")
    private_card, public_card, round_index = encoding[:3]
    if private_card not in LEDUC_DECK:
        raise ValueError("Leduc private card is invalid")
    if public_card != -1 and public_card not in LEDUC_DECK:
        raise ValueError("Leduc public card is invalid")
    if public_card == private_card:
        raise ValueError("Leduc private and public cards must differ")
    if round_index not in (0, 1):
        raise ValueError("Leduc round index is invalid")
    if (round_index == 0) != (public_card == -1):
        raise ValueError("Leduc public card and round index are inconsistent")

    histories: list[tuple[Action, ...]] = []
    current_history: list[Action] = []
    for value in encoding[3:]:
        if value == _HISTORY_SEPARATOR:
            histories.append(tuple(current_history))
            current_history.clear()
            continue
        try:
            current_history.append(Action(value))
        except ValueError as error:
            raise ValueError("Leduc action history is invalid") from error
    if current_history or len(histories) != round_index + 1:
        raise ValueError("Leduc action-history rounds are malformed")
    if len(histories[0]) > _FIRST_ROUND_ACTION_SLOTS or (
        round_index == 1 and len(histories[1]) > _SECOND_ROUND_ACTION_SLOTS
    ):
        raise ValueError("Leduc action history exceeds the canonical limits")
    return private_card, public_card, round_index, tuple(histories)
