from dataclasses import FrozenInstanceError

import pytest

from ac_cfr.games.base import GameId, UtilityUnit
from ac_cfr.games.leduc import (
    LEDUC_DECK,
    LeducConfig,
    LeducRank,
    LeducSuit,
    card_rank,
    card_suit,
    encode_card,
)


def test_canonical_leduc_rules_and_utility_scale_are_fixed() -> None:
    configuration = LeducConfig()

    assert configuration.game_id is GameId.LEDUC
    assert configuration.utility_unit is UtilityUnit.CHIP
    assert configuration.player_count == 2
    assert configuration.ante == 1
    assert configuration.private_cards_per_player == 1
    assert configuration.public_cards == 1
    assert configuration.betting_rounds == 2
    assert configuration.bet_sizes == (2, 4)
    assert configuration.max_bets_per_round == 2
    assert configuration.starting_players == (0, 0)
    assert configuration.suit_isomorphism is False


def test_leduc_deck_retains_all_six_physical_cards() -> None:
    assert len(LEDUC_DECK) == 6
    assert len(set(LEDUC_DECK)) == 6
    assert tuple(range(6)) == LEDUC_DECK

    decoded_cards = {(card_rank(card), card_suit(card)) for card in LEDUC_DECK}
    expected_cards = {(rank, suit) for rank in LeducRank for suit in LeducSuit}
    assert decoded_cards == expected_cards


def test_leduc_card_encoding_round_trips() -> None:
    for rank in LeducRank:
        for suit in LeducSuit:
            card = encode_card(rank, suit)
            assert card_rank(card) is rank
            assert card_suit(card) is suit


def test_leduc_card_encoding_requires_explicit_rank_and_suit_types() -> None:
    for invalid_card in (-1, 6):
        with pytest.raises(ValueError):
            card_rank(invalid_card)
        with pytest.raises(ValueError):
            card_suit(invalid_card)

    with pytest.raises(TypeError):
        encode_card(0, LeducSuit.FIRST)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        encode_card(LeducRank.JACK, 0)  # type: ignore[arg-type]


def test_canonical_leduc_configuration_cannot_be_modified_or_overridden() -> None:
    configuration = LeducConfig()

    with pytest.raises(FrozenInstanceError):
        configuration.ante = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        LeducConfig(ante=2)  # type: ignore[call-arg]
