from dataclasses import replace

from ac_cfr.games.base import Action
from ac_cfr.games.holdem.canonicalisation import canonicalise_visible_cards
from ac_cfr.games.holdem.engine import HoldemConfig, HoldemGame, HoldemState


def _deal_modified(cards: tuple[int, ...]) -> HoldemState:
    assert len(cards) == 7
    state = HoldemGame().initial_state(HoldemConfig.modified())
    for card in cards:
        state = state.apply_action(card)
    return state


def _rename_suits(
    cards: tuple[int, ...], permutation: tuple[int, int, int, int]
) -> tuple[int, ...]:
    return tuple((card // 4) * 4 + permutation[card % 4] for card in cards)


def test_global_suit_renamings_have_identical_information_encodings() -> None:
    cards = (48, 45, 42, 39, 32, 29, 22)
    renamed = _rename_suits(cards, (2, 0, 3, 1))
    original_state = _deal_modified(cards)
    renamed_state = _deal_modified(renamed)

    assert original_state.information_state() == renamed_state.information_state()
    assert canonicalise_visible_cards((45, 39), (32, 29, 22)) == canonicalise_visible_cards(
        _rename_suits((45, 39), (2, 0, 3, 1)),
        _rename_suits((32, 29, 22), (2, 0, 3, 1)),
    )


def test_information_state_excludes_opponent_cards_and_preserves_public_history() -> None:
    first = _deal_modified((0, 4, 8, 12, 16, 20, 24))
    different_opponent = _deal_modified((1, 4, 9, 12, 16, 20, 24))
    assert first.hole_cards[0] != different_opponent.hole_cards[0]
    assert first.information_state() == different_opponent.information_state()

    facing_bet = first.apply_action(Action.CHECK_CALL).apply_action(Action.BET_RAISE)
    assert facing_bet.current_player == first.current_player
    assert facing_bet.information_state().encoding != first.information_state().encoding
    assert facing_bet.information_state().legal_actions != first.information_state().legal_actions

    button_one_configuration = HoldemConfig(
        start_street=first.configuration.start_street,
        max_bets_per_round=2,
        button_player=1,
        synthetic_flop_start=True,
    )
    other_position = replace(first, configuration=button_one_configuration)
    assert other_position.information_state().encoding != first.information_state().encoding
