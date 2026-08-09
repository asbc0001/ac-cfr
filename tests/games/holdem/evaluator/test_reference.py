from phevaluator import evaluate_cards

from ac_cfr.games.holdem.cards import Rank, Suit, card_to_string, encode_card
from ac_cfr.games.holdem.evaluator.reference import (
    HandCategory,
    score_five_cards,
    strength_classes,
)


def _card(rank: Rank, suit: Suit) -> int:
    return encode_card(rank, suit)


def test_reference_evaluator_covers_categories_kickers_and_independent_oracle() -> None:
    cases = (
        (HandCategory.STRAIGHT_FLUSH, "As Ks Qs Js Ts"),
        (HandCategory.FOUR_OF_A_KIND, "Ac Ad Ah As Kc"),
        (HandCategory.FULL_HOUSE, "Kc Kd Kh Qs Qc"),
        (HandCategory.FLUSH, "As Js 8s 4s 2s"),
        (HandCategory.STRAIGHT, "6c 5d 4h 3s 2c"),
        (HandCategory.THREE_OF_A_KIND, "Qc Qd Qh 9s 2c"),
        (HandCategory.TWO_PAIR, "Jc Jd 8h 8s Ac"),
        (HandCategory.PAIR, "Tc Td Ah Ks 3c"),
        (HandCategory.HIGH_CARD, "Ac Jd 8h 5s 2c"),
    )
    card_lookup = {
        card_to_string(encode_card(rank, suit)): encode_card(rank, suit)
        for rank in Rank
        for suit in Suit
    }
    classes = strength_classes()
    for category, text in cases:
        card_strings = text.split()
        cards = tuple(card_lookup[value] for value in card_strings)
        strength = score_five_cards(cards)
        assert strength[0] is category
        assert classes.index(strength) + 1 == evaluate_cards(*card_strings)

    six_high = score_five_cards(
        (
            _card(Rank.SIX, Suit.CLUBS),
            _card(Rank.FIVE, Suit.DIAMONDS),
            _card(Rank.FOUR, Suit.HEARTS),
            _card(Rank.THREE, Suit.SPADES),
            _card(Rank.TWO, Suit.CLUBS),
        )
    )
    wheel = score_five_cards(
        (
            _card(Rank.ACE, Suit.CLUBS),
            _card(Rank.TWO, Suit.DIAMONDS),
            _card(Rank.THREE, Suit.HEARTS),
            _card(Rank.FOUR, Suit.SPADES),
            _card(Rank.FIVE, Suit.CLUBS),
        )
    )
    assert six_high > wheel


def test_strength_classes_are_generated_in_canonical_order() -> None:
    classes = strength_classes()
    assert len(classes) == len(set(classes)) == 7_462
    assert classes[0] == (HandCategory.STRAIGHT_FLUSH, Rank.ACE)
    assert classes[-1] == (
        HandCategory.HIGH_CARD,
        Rank.SEVEN,
        Rank.FIVE,
        Rank.FOUR,
        Rank.THREE,
        Rank.TWO,
    )
