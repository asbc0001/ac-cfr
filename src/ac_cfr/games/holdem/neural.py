"""Fixed neural inputs for player-visible Hold'em information states."""

import numpy as np
from numpy.typing import NDArray

from ac_cfr.common.config import StateEncodingId
from ac_cfr.games.base import ACTION_ORDER, Action, GameId, InformationState, validate_legal_actions
from ac_cfr.games.holdem.cards import RANK_COUNT, SUIT_COUNT
from ac_cfr.games.holdem.engine import Street

HOLD_EM_NEURAL_STATE_ENCODING_ID = StateEncodingId.HOLD_EM
HOLD_EM_ACTION_COUNT = len(ACTION_ORDER)
HOLD_EM_NEURAL_STATE_SIZE = 201
HOLD_EM_NEURAL_INPUT_SCALING = (
    "binary_0_1;contributions_div_64;round_commitments_div_16;betting_level_div_4"
)

_CARD_SLOTS = 7
_MAX_BETS_PER_ROUND = 4
_MAX_HISTORY_ACTIONS = 5
_STREET_COUNT = len(Street)


def encode_holdem_information_state(
    information_state: InformationState,
) -> NDArray[np.float32]:
    """Encode one acting player's observable Hold'em decision as 201 values.

    Cards use separate rank and canonical-suit one-hot fields. Position, streets,
    cap and action history are categorical; exact chip-unit fields use fixed scales.
    Missing turn/river cards and unused history slots remain all zero.
    """
    if not isinstance(information_state, InformationState):
        raise TypeError("information_state must be an InformationState")
    if information_state.game_id is not GameId.HOLD_EM:
        raise ValueError("information_state must belong to Hold'em")

    parsed = _parse_encoding(information_state)
    state = np.zeros(HOLD_EM_NEURAL_STATE_SIZE, dtype=np.float32)
    offset = 0

    state[offset + information_state.player] = 1.0
    offset += 2
    state[offset + parsed.button_player] = 1.0
    offset += 2
    state[offset + parsed.start_street] = 1.0
    offset += _STREET_COUNT
    state[offset + parsed.street] = 1.0
    offset += _STREET_COUNT
    state[offset + parsed.max_bets_per_round - 1] = 1.0
    offset += _MAX_BETS_PER_ROUND

    for card in (*parsed.hole_cards, *parsed.board_cards):
        if card >= 0:
            rank, suit = divmod(card, SUIT_COUNT)
            state[offset + rank] = 1.0
            state[offset + RANK_COUNT + suit] = 1.0
        offset += RANK_COUNT + SUIT_COUNT

    state[offset : offset + 2] = np.asarray(parsed.contributions, dtype=np.float32) / 64.0
    offset += 2
    state[offset : offset + 2] = np.asarray(parsed.round_commitments, dtype=np.float32) / 16.0
    offset += 2
    state[offset] = parsed.betting_level / _MAX_BETS_PER_ROUND
    offset += 1
    state[offset] = float(parsed.live_big_blind)
    offset += 1

    for round_index in range(_STREET_COUNT):
        history = (
            parsed.round_histories[round_index] if round_index < len(parsed.round_histories) else ()
        )
        for action_index, action in enumerate(history):
            state[offset + action_index * HOLD_EM_ACTION_COUNT + int(action)] = 1.0
        offset += _MAX_HISTORY_ACTIONS * HOLD_EM_ACTION_COUNT

    if offset != HOLD_EM_NEURAL_STATE_SIZE:
        raise RuntimeError("Hold'em neural-state layout is inconsistent")
    state.setflags(write=False)
    return state


def holdem_action_mask(legal_actions: tuple[Action, ...]) -> NDArray[np.bool]:
    """Return a mask aligned with fold, check/call and bet/raise."""
    validate_legal_actions(legal_actions)
    mask = np.zeros(HOLD_EM_ACTION_COUNT, dtype=np.bool)
    for action in legal_actions:
        mask[int(action)] = True
    mask.setflags(write=False)
    return mask


class _ParsedHoldemInformation:
    """Validated fields extracted from the compact information-state tuple."""

    __slots__ = (
        "betting_level",
        "board_cards",
        "button_player",
        "contributions",
        "hole_cards",
        "live_big_blind",
        "max_bets_per_round",
        "round_commitments",
        "round_histories",
        "start_street",
        "street",
    )

    def __init__(
        self,
        *,
        start_street: int,
        max_bets_per_round: int,
        button_player: int,
        street: int,
        hole_cards: tuple[int, int],
        board_cards: tuple[int, int, int, int, int],
        contributions: tuple[int, int],
        round_commitments: tuple[int, int],
        betting_level: int,
        live_big_blind: bool,
        round_histories: tuple[tuple[Action, ...], ...],
    ) -> None:
        self.start_street = start_street
        self.max_bets_per_round = max_bets_per_round
        self.button_player = button_player
        self.street = street
        self.hole_cards = hole_cards
        self.board_cards = board_cards
        self.contributions = contributions
        self.round_commitments = round_commitments
        self.betting_level = betting_level
        self.live_big_blind = live_big_blind
        self.round_histories = round_histories


def _parse_encoding(information_state: InformationState) -> _ParsedHoldemInformation:
    encoding = information_state.encoding
    if len(encoding) < 21:
        raise ValueError("Hold'em information-state encoding is incomplete")
    start_street, max_bets, button, encoded_player, street = encoding[:5]
    holes = (encoding[5], encoding[6])
    board_count = encoding[7]
    board = encoding[8:13]
    contributions = (encoding[13], encoding[14])
    commitments = (encoding[15], encoding[16])
    betting_level, raw_live_big_blind, history_count = encoding[17:20]

    if encoded_player != information_state.player:
        raise ValueError("Hold'em encoded player is inconsistent")
    if start_street not in (int(Street.PREFLOP), int(Street.FLOP)):
        raise ValueError("Hold'em start street is invalid")
    if street not in range(_STREET_COUNT) or street < start_street:
        raise ValueError("Hold'em street is invalid")
    if button not in (0, 1):
        raise ValueError("Hold'em button player is invalid")
    if not 1 <= max_bets <= _MAX_BETS_PER_ROUND:
        raise ValueError("Hold'em cap exceeds the neural encoding")
    if board_count not in (0, 3, 4, 5) or board_count != sum(card >= 0 for card in board):
        raise ValueError("Hold'em board-card count is inconsistent")
    if any(card < -1 for card in board):
        raise ValueError("Hold'em missing-card marker is invalid")
    expected_board_count = (0, 3, 4, 5)[street]
    if board_count != expected_board_count:
        raise ValueError("Hold'em board cards do not match the street")
    visible_cards = (*holes, *(card for card in board if card >= 0))
    if any(not 0 <= card < RANK_COUNT * SUIT_COUNT for card in visible_cards):
        raise ValueError("Hold'em visible card is invalid")
    if len(set(visible_cards)) != len(visible_cards):
        raise ValueError("Hold'em visible cards must be distinct")
    if any(value < 0 for value in (*contributions, *commitments)) or any(
        value > limit
        for value, limit in zip(
            (*contributions, *commitments),
            (64, 64, 16, 16),
            strict=True,
        )
    ):
        raise ValueError("Hold'em commitments exceed the neural scaling")
    if not 0 <= betting_level <= max_bets or raw_live_big_blind not in (0, 1):
        raise ValueError("Hold'em betting state is invalid")
    if history_count != street - start_street + 1:
        raise ValueError("Hold'em history count is invalid")

    histories: list[tuple[Action, ...]] = []
    position = 20
    for _ in range(history_count):
        if position >= len(encoding):
            raise ValueError("Hold'em action histories are incomplete")
        action_count = encoding[position]
        position += 1
        if not 0 <= action_count <= _MAX_HISTORY_ACTIONS or position + action_count > len(encoding):
            raise ValueError("Hold'em action history exceeds the neural encoding")
        try:
            history = tuple(Action(value) for value in encoding[position : position + action_count])
        except ValueError as error:
            raise ValueError("Hold'em action history contains an invalid action") from error
        histories.append(history)
        position += action_count
    if position != len(encoding):
        raise ValueError("Hold'em information-state encoding has trailing values")

    return _ParsedHoldemInformation(
        start_street=start_street,
        max_bets_per_round=max_bets,
        button_player=button,
        street=street,
        hole_cards=holes,
        board_cards=(board[0], board[1], board[2], board[3], board[4]),
        contributions=contributions,
        round_commitments=commitments,
        betting_level=betting_level,
        live_big_blind=bool(raw_live_big_blind),
        round_histories=tuple(histories),
    )
