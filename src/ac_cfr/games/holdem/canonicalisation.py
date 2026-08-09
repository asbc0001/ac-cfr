"""Suit-isomorphic canonicalisation for player-visible Hold'em cards."""

from collections.abc import Sequence

from ac_cfr.games.holdem.cards import RANK_COUNT, SUIT_COUNT, validate_card


def canonicalise_visible_cards(
    hole_cards: Sequence[int],
    board_cards: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, ...]]:
    """Return a deterministic representative under every global suit renaming.

    Suits are ordered by their complete observable rank pattern across the hole,
    flop, turn, and river groups. This avoids inspecting either hidden hole card.
    """
    holes = tuple(hole_cards)
    board = tuple(board_cards)
    if len(holes) != 2:
        raise ValueError("hole_cards must contain exactly two cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board_cards must contain zero, three, four, or five cards")

    visible_cards = (*holes, *board)
    for card in visible_cards:
        validate_card(card)
    if len(set(visible_cards)) != len(visible_cards):
        raise ValueError("visible cards must be physically distinct")

    groups = (holes, board[:3], board[3:4], board[4:5])
    suit_signatures = [0] * SUIT_COUNT
    for group_index, cards in enumerate(groups):
        for card in cards:
            rank, suit = divmod(card, SUIT_COUNT)
            suit_signatures[suit] |= 1 << (group_index * RANK_COUNT + rank)

    ordered_suits = sorted(
        range(SUIT_COUNT),
        key=suit_signatures.__getitem__,
        reverse=True,
    )
    canonical_suits = [0] * SUIT_COUNT
    for canonical_suit, physical_suit in enumerate(ordered_suits):
        canonical_suits[physical_suit] = canonical_suit

    def canonical_card(card: int) -> int:
        rank, suit = divmod(card, SUIT_COUNT)
        return rank * SUIT_COUNT + canonical_suits[suit]

    canonical_holes = tuple(sorted(canonical_card(card) for card in holes))
    canonical_flop = tuple(sorted(canonical_card(card) for card in board[:3]))
    canonical_board = (*canonical_flop, *(canonical_card(card) for card in board[3:]))
    assert len(canonical_holes) == 2
    assert all(0 <= card < RANK_COUNT * SUIT_COUNT for card in canonical_holes)
    return (canonical_holes[0], canonical_holes[1]), canonical_board
