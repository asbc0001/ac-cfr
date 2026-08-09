from dataclasses import FrozenInstanceError

import pytest

from ac_cfr.games.base import Action, NodeType
from ac_cfr.games.kuhn import KUHN_DEALS, KuhnConfig, KuhnGame


def test_kuhn_chance_deals_all_ordered_physical_cards_once() -> None:
    root = KuhnGame().initial_state(KuhnConfig())
    outcomes = root.chance_outcomes()

    assert root.node_type is NodeType.CHANCE
    assert len(KUHN_DEALS) == len(outcomes) == 6
    assert len(set(KUHN_DEALS)) == 6
    assert all(first != second for first, second in KUHN_DEALS)
    assert tuple(outcome.outcome for outcome in outcomes) == tuple(range(6))
    assert all(outcome.probability == pytest.approx(1 / 6) for outcome in outcomes)
    assert all(outcome.multiplicity == 1 for outcome in outcomes)


def test_kuhn_exhaustive_actions_information_and_terminal_utilities() -> None:
    root = KuhnGame().initial_state(KuhnConfig())
    same_card_deals = [
        deal_id for deal_id, private_cards in enumerate(KUHN_DEALS) if private_cards[0] == 0
    ]
    assert (
        root.apply_action(same_card_deals[0]).information_state()
        == root.apply_action(same_card_deals[1]).information_state()
    )

    terminal_histories: dict[tuple[Action, ...], int] = {}
    information_actions: dict[tuple[int, tuple[int, ...]], tuple[Action, ...]] = {}

    stack = [root]
    while stack:
        state = stack.pop()
        if state.is_terminal:
            utility_zero = state.utility(0)
            assert utility_zero in {-2.0, -1.0, 1.0, 2.0}
            assert state.utility(1) == -utility_zero
            terminal_histories[state.action_history] = (
                terminal_histories.get(state.action_history, 0) + 1
            )
            continue
        if state.is_chance_node:
            stack.extend(state.apply_action(outcome.outcome) for outcome in state.chance_outcomes())
            continue

        information_state = state.information_state()
        assert state.private_cards is not None
        assert information_state.encoding[0] == int(state.private_cards[information_state.player])
        key = information_state.player, information_state.encoding
        assert information_actions.setdefault(key, state.legal_actions()) == state.legal_actions()
        stack.extend(state.apply_action(action) for action in state.legal_actions())

    assert terminal_histories == {
        (Action.CHECK_CALL, Action.CHECK_CALL): 6,
        (Action.BET_RAISE, Action.FOLD): 6,
        (Action.BET_RAISE, Action.CHECK_CALL): 6,
        (Action.CHECK_CALL, Action.BET_RAISE, Action.FOLD): 6,
        (Action.CHECK_CALL, Action.BET_RAISE, Action.CHECK_CALL): 6,
    }
    assert len(information_actions) == 12


def test_kuhn_rejects_invalid_transitions_and_keeps_states_immutable() -> None:
    root = KuhnGame().initial_state(KuhnConfig())
    dealt = root.apply_action(0)
    facing_bet = dealt.apply_action(Action.BET_RAISE)

    with pytest.raises(ValueError, match="illegal action"):
        dealt.apply_action(Action.FOLD)
    with pytest.raises(ValueError, match="illegal action"):
        facing_bet.apply_action(Action.BET_RAISE)
    with pytest.raises(TypeError):
        root.apply_action(True)
    with pytest.raises(ValueError, match="terminal"):
        facing_bet.apply_action(Action.FOLD).apply_action(Action.CHECK_CALL)
    with pytest.raises(FrozenInstanceError):
        dealt.action_history = ()  # type: ignore[misc]
