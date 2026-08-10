from dataclasses import FrozenInstanceError

import pytest

from ac_cfr.games.base import Action, NodeType
from ac_cfr.games.leduc import (
    LEDUC_DECK,
    LEDUC_PRIVATE_DEALS,
    LeducConfig,
    LeducGame,
    LeducState,
    card_rank,
    compare_leduc_hands,
)


def _dealt_state(deal_id: int = 0) -> LeducState:
    return LeducGame().initial_state(LeducConfig()).apply_action(deal_id)


def test_leduc_physical_chance_dealing_and_round_progression() -> None:
    root = LeducGame().initial_state(LeducConfig())
    private_outcomes = root.chance_outcomes()

    assert len(LEDUC_PRIVATE_DEALS) == len(private_outcomes) == 30
    assert len(set(LEDUC_PRIVATE_DEALS)) == 30
    assert all(first != second for first, second in LEDUC_PRIVATE_DEALS)
    assert all(outcome.probability == pytest.approx(1 / 30) for outcome in private_outcomes)
    assert all(outcome.multiplicity == 1 for outcome in private_outcomes)

    public_chance = (
        root.apply_action(0).apply_action(Action.CHECK_CALL).apply_action(Action.CHECK_CALL)
    )
    assert public_chance.node_type is NodeType.CHANCE
    assert public_chance.round_index == 1
    assert public_chance.round_commitments == (0, 0)
    assert public_chance.aggressive_actions == 0
    public_outcomes = public_chance.chance_outcomes()
    assert len(public_outcomes) == 4
    assert public_chance.private_cards is not None
    assert all(outcome.outcome not in public_chance.private_cards for outcome in public_outcomes)
    assert all(outcome.probability == pytest.approx(1 / 4) for outcome in public_outcomes)
    assert all(outcome.multiplicity == 1 for outcome in public_outcomes)

    second_round = public_chance.apply_action(public_outcomes[0].outcome)
    assert second_round.current_player == 0
    assert second_round.legal_actions() == (Action.CHECK_CALL, Action.BET_RAISE)

    same_card_deals = [
        deal_id
        for deal_id, private_cards in enumerate(LEDUC_PRIVATE_DEALS)
        if private_cards[0] == 0
    ]
    assert (
        root.apply_action(same_card_deals[0]).information_state()
        == root.apply_action(same_card_deals[1]).information_state()
    )

    bet_call_chance = (
        root.apply_action(0).apply_action(Action.BET_RAISE).apply_action(Action.CHECK_CALL)
    )
    bet_call_round = bet_call_chance.apply_action(public_outcomes[0].outcome)
    assert second_round.information_state().encoding != bet_call_round.information_state().encoding


def test_leduc_strict_betting_cap_contributions_and_terminal_utility() -> None:
    state = _dealt_state()
    assert state.contributions == (1, 1)
    assert state.legal_actions() == (Action.CHECK_CALL, Action.BET_RAISE)
    with pytest.raises(ValueError, match="illegal action"):
        state.apply_action(Action.FOLD)

    state = state.apply_action(Action.BET_RAISE)
    assert state.contributions == (3, 1)
    assert state.legal_actions() == (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE)
    state = state.apply_action(Action.BET_RAISE)
    assert state.contributions == (3, 5)
    assert state.legal_actions() == (Action.FOLD, Action.CHECK_CALL)
    with pytest.raises(ValueError, match="illegal action"):
        state.apply_action(Action.BET_RAISE)

    public_chance = state.apply_action(Action.CHECK_CALL)
    assert public_chance.contributions == (5, 5)
    second_round = public_chance.apply_action(public_chance.chance_outcomes()[0].outcome)
    folded = second_round.apply_action(Action.BET_RAISE).apply_action(Action.FOLD)
    assert folded.contributions == (9, 5)
    assert folded.utility(0) == 5.0
    assert folded.utility(1) == -5.0
    with pytest.raises(ValueError, match="terminal"):
        folded.apply_action(Action.CHECK_CALL)
    with pytest.raises(FrozenInstanceError):
        state.aggressive_actions = 0  # type: ignore[misc]

    tied_private_deal = LEDUC_PRIVATE_DEALS.index((0, 1))
    tied_public_chance = (
        LeducGame()
        .initial_state(LeducConfig())
        .apply_action(tied_private_deal)
        .apply_action(Action.CHECK_CALL)
        .apply_action(Action.CHECK_CALL)
    )
    tied_showdown = (
        tied_public_chance.apply_action(2)
        .apply_action(Action.CHECK_CALL)
        .apply_action(Action.CHECK_CALL)
    )
    assert tied_showdown.utility(0) == tied_showdown.utility(1) == 0.0


def test_leduc_specialised_showdown_comparison_is_exhaustive() -> None:
    for private_cards in LEDUC_PRIVATE_DEALS:
        for public_card in LEDUC_DECK:
            if public_card in private_cards:
                continue
            first_rank = card_rank(private_cards[0])
            second_rank = card_rank(private_cards[1])
            public_rank = card_rank(public_card)
            expected = (
                (first_rank == public_rank, first_rank) > (second_rank == public_rank, second_rank)
            ) - (
                (first_rank == public_rank, first_rank) < (second_rank == public_rank, second_rank)
            )
            assert compare_leduc_hands(private_cards, public_card) == expected


def test_leduc_exhaustive_state_invariants_and_information_boundary() -> None:
    root = LeducGame().initial_state(LeducConfig())
    information_actions: dict[tuple[int, tuple[int, ...]], tuple[Action, ...]] = {}
    player_nodes = terminal_nodes = 0
    stack = [root]

    while stack:
        state = stack.pop()
        assert state.contributions[0] >= 1 and state.contributions[1] >= 1
        if state.is_terminal:
            terminal_nodes += 1
            assert state.utility(0) == -state.utility(1)
            continue
        if state.is_chance_node:
            outcomes = state.chance_outcomes()
            assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
            stack.extend(state.apply_action(outcome.outcome) for outcome in outcomes)
            continue

        player_nodes += 1
        information_state = state.information_state()
        assert state.private_cards is not None
        assert information_state.encoding[0] == state.private_cards[information_state.player]
        key = information_state.player, information_state.encoding
        assert information_actions.setdefault(key, state.legal_actions()) == state.legal_actions()
        stack.extend(state.apply_action(action) for action in state.legal_actions())

    assert player_nodes == 3780
    assert terminal_nodes == 5520
    assert len(information_actions) == 936
