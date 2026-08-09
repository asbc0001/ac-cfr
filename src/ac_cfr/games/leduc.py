"""Canonical two-player Leduc poker rules and physical-card encoding."""

from dataclasses import dataclass, field
from enum import IntEnum

from ac_cfr.games.base import GameId, UtilityUnit


class LeducRank(IntEnum):
    """Leduc ranks ordered from weakest to strongest."""

    JACK = 0
    QUEEN = 1
    KING = 2


class LeducSuit(IntEnum):
    """Physical suits retained in the canonical Leduc game."""

    FIRST = 0
    SECOND = 1


LEDUC_SUIT_COUNT = len(LeducSuit)


def encode_card(rank: LeducRank, suit: LeducSuit) -> int:
    """Encode one physical Leduc card as a compact rank-major integer."""
    if not isinstance(rank, LeducRank):
        raise TypeError("rank must be a LeducRank")
    if not isinstance(suit, LeducSuit):
        raise TypeError("suit must be a LeducSuit")
    return int(rank) * LEDUC_SUIT_COUNT + int(suit)


def card_rank(card: int) -> LeducRank:
    """Return the rank stored in a valid compact Leduc card."""
    _validate_card(card)
    return LeducRank(card // LEDUC_SUIT_COUNT)


def card_suit(card: int) -> LeducSuit:
    """Return the physical suit stored in a valid compact Leduc card."""
    _validate_card(card)
    return LeducSuit(card % LEDUC_SUIT_COUNT)


LEDUC_DECK = tuple(encode_card(rank, suit) for rank in LeducRank for suit in LeducSuit)


@dataclass(frozen=True, slots=True)
class LeducConfig:
    """Fixed OpenSpiel-compatible canonical Leduc rules."""

    game_id: GameId = field(default=GameId.LEDUC, init=False)
    utility_unit: UtilityUnit = field(default=UtilityUnit.CHIP, init=False)
    player_count: int = field(default=2, init=False)
    ante: int = field(default=1, init=False)
    private_cards_per_player: int = field(default=1, init=False)
    public_cards: int = field(default=1, init=False)
    betting_rounds: int = field(default=2, init=False)
    bet_sizes: tuple[int, int] = field(default=(2, 4), init=False)
    max_bets_per_round: int = field(default=2, init=False)
    starting_players: tuple[int, int] = field(default=(0, 0), init=False)
    suit_isomorphism: bool = field(default=False, init=False)


def _validate_card(card: int) -> None:
    if isinstance(card, bool) or not isinstance(card, int):
        raise TypeError("card must be an integer")
    if not 0 <= card < len(LEDUC_DECK):
        raise ValueError(f"card must be between 0 and {len(LEDUC_DECK) - 1}")
