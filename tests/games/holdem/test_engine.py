from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from ac_cfr.games.base import Action, NodeType
from ac_cfr.games.holdem.engine import HoldemConfig, HoldemGame, HoldemState, Street


def _deal_cards(state: HoldemState, cards: tuple[int, ...]) -> HoldemState:
    for card in cards:
        state = state.apply_action(card)
    return state


def _modified_flop(
    deal: tuple[int, int, int, int, int, int, int] = (0, 4, 8, 12, 16, 20, 24),
) -> HoldemState:
    return _deal_cards(HoldemGame().initial_state(HoldemConfig.modified()), deal)


def test_configurations_create_exact_full_and_synthetic_starting_states() -> None:
    full = HoldemGame().initial_state(HoldemConfig.full())
    assert full.node_type is NodeType.CHANCE
    assert full.street is Street.PREFLOP
    assert full.contributions == full.round_commitments == (1, 2)
    assert full.pot == 3
    assert full.betting_level == 1
    assert full.live_big_blind
    assert full.configuration.base_unit == Fraction(1, 2)
    assert HoldemConfig(
        small_blind=0.25,
        big_blind=0.5,
        small_bet=0.5,
        big_bet=1,
    ).base_unit == Fraction(1, 4)

    modified = HoldemGame().initial_state(HoldemConfig.modified())
    assert modified.street is Street.FLOP
    assert modified.contributions == (2, 2)
    assert modified.round_commitments == (0, 0)
    assert modified.pot == 4
    assert modified.betting_level == 0
    assert not modified.live_big_blind
    assert modified.round_histories == ((),)

    with pytest.raises(ValueError, match="explicitly"):
        HoldemConfig(start_street=Street.FLOP)
    with pytest.raises(ValueError, match="positive"):
        HoldemConfig(max_bets_per_round=0)
    with pytest.raises(ValueError, match="ratios"):
        HoldemConfig(big_bet=3)


def test_physical_dealing_is_uniform_without_replacement_and_position_aware() -> None:
    root = HoldemGame().initial_state(HoldemConfig.modified())
    outcomes = root.chance_outcomes()
    assert len(outcomes) == 52
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    assert all(outcome.probability == pytest.approx(1 / 52) for outcome in outcomes)
    assert all(outcome.multiplicity == 1 for outcome in outcomes)

    state = root.apply_action(0)
    assert len(state.chance_outcomes()) == 51
    assert all(outcome.outcome != 0 for outcome in state.chance_outcomes())
    with pytest.raises(ValueError, match="already"):
        state.apply_action(0)

    dealt = _modified_flop()
    assert dealt.current_player == 1
    assert dealt.hole_cards == ((0, 8), (4, 12))
    assert dealt.board_cards == (16, 20, 24)
    assert dealt.dealt_card_mask.bit_count() == 7
    with pytest.raises(FrozenInstanceError):
        dealt.betting_level = 3  # type: ignore[misc]

    button_one = _deal_cards(
        HoldemGame().initial_state(HoldemConfig(button_player=1)),
        (0, 4, 8, 12),
    )
    assert button_one.hole_cards == ((4, 12), (0, 8))
    assert button_one.current_player == 1


def test_live_big_blind_and_four_level_preflop_cap_are_exact() -> None:
    state = _deal_cards(
        HoldemGame().initial_state(HoldemConfig.full()),
        (0, 4, 8, 12),
    )
    assert state.current_player == 0
    assert state.amount_to_call == 1
    assert state.legal_actions() == (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE)

    big_blind_option = state.apply_action(Action.CHECK_CALL)
    assert big_blind_option.current_player == 1
    assert big_blind_option.amount_to_call == 0
    assert big_blind_option.legal_actions() == (Action.CHECK_CALL, Action.BET_RAISE)
    flop_chance = big_blind_option.apply_action(Action.CHECK_CALL)
    assert flop_chance.node_type is NodeType.CHANCE
    assert flop_chance.street is Street.FLOP
    flop = _deal_cards(flop_chance, (16, 20, 24))
    assert flop.current_player == 1
    turn_chance = flop.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL)
    turn = turn_chance.apply_action(28)
    river_chance = turn.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL)
    river = river_chance.apply_action(32)
    showdown = river.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL)
    assert showdown.utility(0) == -showdown.utility(1)
    assert showdown.dealt_card_mask.bit_count() == 9

    capped = (
        state.apply_action(Action.BET_RAISE)
        .apply_action(Action.BET_RAISE)
        .apply_action(Action.BET_RAISE)
    )
    assert capped.betting_level == 4
    assert capped.round_commitments == (8, 6)
    assert capped.legal_actions() == (Action.FOLD, Action.CHECK_CALL)
    with pytest.raises(ValueError, match="illegal action"):
        capped.apply_action(Action.BET_RAISE)


def test_modified_cap_street_progression_and_bet_sizes_reset_cleanly() -> None:
    flop = _modified_flop()
    raised = flop.apply_action(Action.BET_RAISE).apply_action(Action.BET_RAISE)
    assert raised.betting_level == 2
    assert raised.legal_actions() == (Action.FOLD, Action.CHECK_CALL)
    turn_chance = raised.apply_action(Action.CHECK_CALL)
    assert turn_chance.street is Street.TURN
    assert len(turn_chance.chance_outcomes()) == 45
    assert all(
        outcome.probability == pytest.approx(1 / 45) for outcome in turn_chance.chance_outcomes()
    )
    assert turn_chance.round_commitments == (0, 0)
    assert turn_chance.betting_level == 0
    assert turn_chance.contributions == (6, 6)
    assert not turn_chance.live_big_blind

    turn = turn_chance.apply_action(28)
    assert turn.current_player == 1
    river_chance = turn.apply_action(Action.BET_RAISE).apply_action(Action.CHECK_CALL)
    assert river_chance.street is Street.RIVER
    assert river_chance.contributions == (10, 10)
    river = river_chance.apply_action(32)
    assert river.current_player == 1
    assert river.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL).is_terminal


def test_fold_and_showdown_utilities_use_exact_net_chip_accounting() -> None:
    flop = _modified_flop()
    folded = flop.apply_action(Action.BET_RAISE).apply_action(Action.FOLD)
    assert folded.contributions == (2, 4)
    assert folded.utility(0) == -1.0
    assert folded.utility(1) == 1.0
    assert folded.utility(0) == -folded.utility(1)

    winning_flop = _modified_flop((48, 44, 49, 45, 0, 5, 10))
    winning_turn = (
        winning_flop.apply_action(Action.CHECK_CALL)
        .apply_action(Action.CHECK_CALL)
        .apply_action(15)
    )
    winning_river = (
        winning_turn.apply_action(Action.CHECK_CALL)
        .apply_action(Action.CHECK_CALL)
        .apply_action(20)
    )
    showdown = winning_river.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL)
    assert showdown.showdown
    assert showdown.utility(0) == 1.0
    assert showdown.utility(1) == -1.0

    tied_flop = _modified_flop((1, 2, 5, 6, 32, 36, 40))
    tied_turn = (
        tied_flop.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL).apply_action(44)
    )
    tied_river = (
        tied_turn.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL).apply_action(48)
    )
    tied_showdown = tied_river.apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL)
    assert tied_showdown.utility(0) == tied_showdown.utility(1) == 0.0
    with pytest.raises(ValueError, match="terminal"):
        tied_showdown.apply_action(Action.CHECK_CALL)
