"""Deterministic player-visible information encoding for Hold'em."""

from dataclasses import dataclass

from ac_cfr.games.base import Action, GameId, InformationState, PlayerId
from ac_cfr.games.holdem.canonicalisation import canonicalise_visible_cards
from ac_cfr.games.holdem.cards import validate_card

_MISSING_CARD = -1
_BOARD_CARD_COUNT = 5


@dataclass(frozen=True, slots=True)
class HoldemVisibleState:
    """Validated cards and street visible to one acting Hold'em player."""

    street: int
    hole_cards: tuple[int, int]
    board_cards: tuple[int, ...]


def build_holdem_information_state(
    *,
    player: PlayerId,
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
    start_street: int,
    street: int,
    button_player: PlayerId,
    max_bets_per_round: int,
    contributions: tuple[int, int],
    round_commitments: tuple[int, int],
    betting_level: int,
    live_big_blind: bool,
    round_histories: tuple[tuple[Action, ...], ...],
    legal_actions: tuple[Action, ...],
) -> InformationState:
    """Build one encoding without accepting or observing the opponent's cards."""
    canonical_holes, canonical_board = canonicalise_visible_cards(hole_cards, board_cards)
    padded_board = (*canonical_board, *(_MISSING_CARD,) * (_BOARD_CARD_COUNT - len(board_cards)))
    history_encoding = tuple(
        value
        for history in round_histories
        for value in (len(history), *(int(action) for action in history))
    )

    return InformationState(
        game_id=GameId.HOLD_EM,
        player=player,
        encoding=(
            start_street,
            max_bets_per_round,
            button_player,
            player,
            street,
            *canonical_holes,
            len(board_cards),
            *padded_board,
            *contributions,
            *round_commitments,
            betting_level,
            int(live_big_blind),
            len(round_histories),
            *history_encoding,
        ),
        legal_actions=legal_actions,
    )


def parse_holdem_visible_state(information_state: InformationState) -> HoldemVisibleState:
    """Read validated visible cards without exposing any underlying game state."""
    if not isinstance(information_state, InformationState):
        raise TypeError("information_state must be an InformationState")
    if information_state.game_id is not GameId.HOLD_EM:
        raise ValueError("information_state must belong to Hold'em")

    encoding = information_state.encoding
    if len(encoding) < 13:
        raise ValueError("Hold'em information-state encoding is incomplete")
    if encoding[3] != information_state.player:
        raise ValueError("Hold'em encoded player is inconsistent")

    street = encoding[4]
    if street not in range(4):
        raise ValueError("Hold'em street is invalid")
    board_count = encoding[7]
    expected_board_count = (0, 3, 4, 5)[street]
    if board_count != expected_board_count:
        raise ValueError("Hold'em board cards do not match the street")

    hole_cards = (encoding[5], encoding[6])
    padded_board = encoding[8:13]
    if any(card != _MISSING_CARD for card in padded_board[board_count:]):
        raise ValueError("Hold'em missing board-card markers are invalid")
    board_cards = padded_board[:board_count]
    visible_cards = (*hole_cards, *board_cards)
    for card in visible_cards:
        validate_card(card)
    if len(set(visible_cards)) != len(visible_cards):
        raise ValueError("Hold'em visible cards must be distinct")

    return HoldemVisibleState(
        street=street,
        hole_cards=hole_cards,
        board_cards=board_cards,
    )
