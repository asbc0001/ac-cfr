"""Compact physical-card encoding for a standard 52-card deck."""

from collections.abc import Sequence
from enum import IntEnum


class Rank(IntEnum):
    """Card ranks ordered from deuce to ace."""

    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


class Suit(IntEnum):
    """Physical card suits in stable encoding order."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


RANK_COUNT = len(Rank)
SUIT_COUNT = len(Suit)
CARD_COUNT = RANK_COUNT * SUIT_COUNT
DECK = tuple(range(CARD_COUNT))

_RANK_CHARACTERS = "23456789TJQKA"
_SUIT_CHARACTERS = "cdhs"


def encode_card(rank: Rank, suit: Suit) -> int:
    """Encode one physical card as a compact rank-major integer."""
    if not isinstance(rank, Rank):
        raise TypeError("rank must be a Rank")
    if not isinstance(suit, Suit):
        raise TypeError("suit must be a Suit")
    return int(rank) * SUIT_COUNT + int(suit)


def card_rank(card: int) -> Rank:
    """Return the rank of a validated compact card."""
    validate_card(card)
    return Rank(card // SUIT_COUNT)


def card_suit(card: int) -> Suit:
    """Return the suit of a validated compact card."""
    validate_card(card)
    return Suit(card % SUIT_COUNT)


def card_rank_bit(card: int) -> int:
    """Return the card's rank as one bit in a 13-bit mask."""
    validate_card(card)
    return 1 << (card // SUIT_COUNT)


def card_to_string(card: int) -> str:
    """Return the conventional two-character form used by test oracles."""
    validate_card(card)
    return f"{_RANK_CHARACTERS[card // SUIT_COUNT]}{_SUIT_CHARACTERS[card % SUIT_COUNT]}"


def validate_card(card: int) -> None:
    """Reject a value that is not one physical card in the deck."""
    if isinstance(card, bool) or not isinstance(card, int):
        raise TypeError("card must be an integer")
    if not 0 <= card < CARD_COUNT:
        raise ValueError(f"card must be between 0 and {CARD_COUNT - 1}")


def validate_cards(cards: Sequence[int], expected_count: int) -> tuple[int, ...]:
    """Return a validated fixed-size sequence of distinct physical cards."""
    if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
        raise TypeError("cards must be a sequence")
    card_tuple = tuple(cards)
    if len(card_tuple) != expected_count:
        raise ValueError(f"expected exactly {expected_count} cards")
    for card in card_tuple:
        validate_card(card)
    if len(set(card_tuple)) != expected_count:
        raise ValueError("cards must be physically distinct")
    return card_tuple


def validate_holdem_cards(
    hole_cards: Sequence[int],
    board_cards: Sequence[int],
) -> tuple[int, int, int, int, int, int, int]:
    """Validate the exact two-hole-card and five-board-card runtime contract."""
    holes = validate_cards(hole_cards, 2)
    board = validate_cards(board_cards, 5)
    cards = (*holes, *board)
    if len(set(cards)) != 7:
        raise ValueError("hole and board cards must be physically distinct")
    return cards[0], cards[1], cards[2], cards[3], cards[4], cards[5], cards[6]
