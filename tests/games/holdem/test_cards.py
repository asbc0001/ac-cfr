import pytest

from ac_cfr.games.holdem.cards import (
    CARD_COUNT,
    DECK,
    Rank,
    Suit,
    card_rank,
    card_rank_bit,
    card_suit,
    encode_card,
    validate_holdem_cards,
)


def test_card_encoding_round_trips_all_52_physical_cards() -> None:
    assert len(DECK) == len(set(DECK)) == CARD_COUNT == 52
    for rank in Rank:
        for suit in Suit:
            card = encode_card(rank, suit)
            assert card_rank(card) is rank
            assert card_suit(card) is suit
            assert card_rank_bit(card) == 1 << int(rank)


def test_holdem_card_boundary_requires_exact_distinct_valid_cards() -> None:
    assert validate_holdem_cards((0, 1), (2, 3, 4, 5, 6)) == tuple(range(7))

    with pytest.raises(ValueError, match="exactly 2"):
        validate_holdem_cards((0,), (1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="distinct"):
        validate_holdem_cards((0, 1), (1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="between"):
        validate_holdem_cards((0, 52), (1, 2, 3, 4, 5))
    with pytest.raises(TypeError, match="integer"):
        validate_holdem_cards((0, True), (1, 2, 3, 4, 5))
