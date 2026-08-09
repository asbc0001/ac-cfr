from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from ac_cfr.games.base import GameId, UtilityUnit
from ac_cfr.games.kuhn import KUHN_DECK, KuhnCard, KuhnConfig


def test_canonical_kuhn_rules_and_utility_scale_are_fixed() -> None:
    configuration = KuhnConfig()

    assert configuration.game_id is GameId.KUHN
    assert configuration.utility_unit is UtilityUnit.CHIP
    assert configuration.player_count == 2
    assert configuration.ante == 1
    assert configuration.private_cards_per_player == 1
    assert configuration.betting_rounds == 1
    assert configuration.bet_size == 1
    assert configuration.max_bets_per_round == 1
    assert configuration.starting_player == 0
    assert configuration.player_zero_equilibrium_value == Fraction(-1, 18)
    assert KUHN_DECK == (KuhnCard.JACK, KuhnCard.QUEEN, KuhnCard.KING)
    assert tuple(int(card) for card in KUHN_DECK) == (0, 1, 2)


def test_canonical_kuhn_configuration_cannot_be_modified_or_overridden() -> None:
    configuration = KuhnConfig()

    with pytest.raises(FrozenInstanceError):
        configuration.ante = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        KuhnConfig(ante=2)  # type: ignore[call-arg]
