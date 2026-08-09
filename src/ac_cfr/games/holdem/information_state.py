"""Deterministic player-visible information encoding for Hold'em."""

from ac_cfr.games.base import Action, GameId, InformationState, PlayerId
from ac_cfr.games.holdem.canonicalisation import canonicalise_visible_cards

_MISSING_CARD = -1
_BOARD_CARD_COUNT = 5


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
