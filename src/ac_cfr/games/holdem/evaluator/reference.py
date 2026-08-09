"""Clear reference evaluator and canonical five-card strength classes."""

from collections.abc import Iterator, Sequence
from enum import IntEnum
from functools import cache
from itertools import combinations

from ac_cfr.games.holdem.cards import (
    RANK_COUNT,
    SUIT_COUNT,
    Rank,
    Suit,
    encode_card,
    validate_cards,
    validate_holdem_cards,
)

type HandStrength = tuple[int, ...]


class HandCategory(IntEnum):
    """Five-card categories ordered from weakest to strongest."""

    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


def score_five_cards(cards: Sequence[int]) -> HandStrength:
    """Return an inspectable category-and-kicker tuple for five cards."""
    validated_cards = validate_cards(cards, 5)
    return _score_five_cards_unchecked(validated_cards)


def evaluate_seven_cards_reference(
    hole_cards: Sequence[int],
    board_cards: Sequence[int],
) -> int:
    """Evaluate all 21 five-card subsets and return a canonical rank."""
    cards = validate_holdem_cards(hole_cards, board_cards)
    return _evaluate_seven_cards_unchecked(cards)


@cache
def strength_classes() -> tuple[HandStrength, ...]:
    """Generate all 7,462 valid strength tuples from representative hands."""
    strengths: set[HandStrength] = set()
    for rank_counts in _rank_count_vectors(total=5):
        cards = _representative_cards(rank_counts)
        strengths.add(_score_five_cards_unchecked(cards))

    for rank_mask in range(1 << RANK_COUNT):
        if rank_mask.bit_count() != 5:
            continue
        flush_cards = tuple(
            encode_card(Rank(rank), Suit.CLUBS)
            for rank in range(RANK_COUNT)
            if rank_mask & (1 << rank)
        )
        strengths.add(_score_five_cards_unchecked(flush_cards))
    return tuple(sorted(strengths, reverse=True))


@cache
def _strength_class_ranks() -> dict[HandStrength, int]:
    return {strength: rank for rank, strength in enumerate(strength_classes(), start=1)}


def _evaluate_seven_cards_unchecked(
    cards: tuple[int, int, int, int, int, int, int],
) -> int:
    strongest = max(_score_five_cards_unchecked(subset) for subset in combinations(cards, 5))
    return _strength_class_ranks()[strongest]


def _score_five_cards_unchecked(cards: Sequence[int]) -> HandStrength:
    rank_counts = [0] * RANK_COUNT
    suit = cards[0] % SUIT_COUNT
    flush = True
    rank_mask = 0
    for card in cards:
        rank = card // SUIT_COUNT
        rank_counts[rank] += 1
        rank_mask |= 1 << rank
        flush = flush and card % SUIT_COUNT == suit

    straight_high = _straight_high_rank(rank_mask)
    if flush and straight_high >= 0:
        return HandCategory.STRAIGHT_FLUSH, straight_high

    ranks_by_count = sorted(
        ((count, rank) for rank, count in enumerate(rank_counts) if count), reverse=True
    )
    if ranks_by_count[0][0] == 4:
        return HandCategory.FOUR_OF_A_KIND, ranks_by_count[0][1], ranks_by_count[1][1]
    if ranks_by_count[0][0] == 3 and ranks_by_count[1][0] == 2:
        return HandCategory.FULL_HOUSE, ranks_by_count[0][1], ranks_by_count[1][1]

    descending_ranks = tuple(rank for rank in range(RANK_COUNT - 1, -1, -1) if rank_counts[rank])
    if flush:
        return HandCategory.FLUSH, *descending_ranks
    if straight_high >= 0:
        return HandCategory.STRAIGHT, straight_high
    if ranks_by_count[0][0] == 3:
        kickers = tuple(rank for rank in descending_ranks if rank != ranks_by_count[0][1])
        return HandCategory.THREE_OF_A_KIND, ranks_by_count[0][1], *kickers

    pairs = tuple(rank for rank in descending_ranks if rank_counts[rank] == 2)
    if len(pairs) == 2:
        kicker = next(rank for rank in descending_ranks if rank_counts[rank] == 1)
        return HandCategory.TWO_PAIR, *pairs, kicker
    if len(pairs) == 1:
        kickers = tuple(rank for rank in descending_ranks if rank_counts[rank] == 1)
        return HandCategory.PAIR, pairs[0], *kickers
    return HandCategory.HIGH_CARD, *descending_ranks


def _straight_high_rank(rank_mask: int) -> int:
    wheel_mask = (1 << int(Rank.ACE)) | 0b1111
    best_high = int(Rank.FIVE) if rank_mask & wheel_mask == wheel_mask else -1
    for high_rank in range(int(Rank.ACE), int(Rank.SIX) - 1, -1):
        straight_mask = 0b11111 << (high_rank - 4)
        if rank_mask & straight_mask == straight_mask:
            return high_rank
    return best_high


def _rank_count_vectors(total: int) -> Iterator[tuple[int, ...]]:
    counts = [0] * RANK_COUNT

    def visit(position: int, remaining: int) -> Iterator[tuple[int, ...]]:
        if position == RANK_COUNT:
            if remaining == 0:
                yield tuple(counts)
            return
        for count in range(min(4, remaining) + 1):
            counts[position] = count
            yield from visit(position + 1, remaining - count)

    yield from visit(0, total)


def _representative_cards(rank_counts: Sequence[int]) -> tuple[int, ...]:
    suit_counts = [0] * SUIT_COUNT
    cards: list[int] = []
    for rank, count in enumerate(rank_counts):
        available_suits = sorted(range(SUIT_COUNT), key=lambda suit: (suit_counts[suit], suit))
        for suit in available_suits[:count]:
            cards.append(rank * SUIT_COUNT + suit)
            suit_counts[suit] += 1
    return tuple(cards)
