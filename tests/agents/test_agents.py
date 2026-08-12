from math import inf, nan

import pytest

from ac_cfr.agents import (
    BaselineAgent,
    PlayableAgent,
    RuleBasedAgent,
    Strategy,
    normalise_strategy,
    validate_strategy,
)
from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import Action, GameId, InformationState
from ac_cfr.games.holdem.cards import Rank, Suit, card_to_string, encode_card
from ac_cfr.games.holdem.engine import Street
from ac_cfr.games.holdem.information_state import build_holdem_information_state

_LEGAL_ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE)
_INFORMATION_STATE = InformationState(GameId.LEDUC, 0, (1, 2, 0), _LEGAL_ACTIONS)
_CARD_LOOKUP = {
    card_to_string(encode_card(rank, suit)): encode_card(rank, suit)
    for rank in Rank
    for suit in Suit
}


class _FixedAgent(PlayableAgent):
    __slots__ = ("_strategy",)

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def get_strategy(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
    ) -> Strategy:
        return self._strategy


def _cards(text: str) -> tuple[int, ...]:
    """Parse concise test-only card notation."""
    return tuple(_CARD_LOOKUP[value] for value in text.split())


def test_strategy_normalisation_and_strict_validation() -> None:
    assert normalise_strategy((2, 3, 5), _LEGAL_ACTIONS) == pytest.approx((0.2, 0.3, 0.5))
    assert normalise_strategy((0, 0, 0), _LEGAL_ACTIONS, uniform_if_zero=True) == pytest.approx(
        (1 / 3, 1 / 3, 1 / 3)
    )
    assert validate_strategy((0.2, 0.3, 0.5), _LEGAL_ACTIONS) == (0.2, 0.3, 0.5)

    invalid_strategies = ((0.5, 0.5), (0.2, -0.1, 0.9), (nan, 0.0, 1.0), (inf, 0.0, 0.0))
    for invalid_strategy in invalid_strategies:
        with pytest.raises((TypeError, ValueError)):
            validate_strategy(invalid_strategy, _LEGAL_ACTIONS)
    with pytest.raises(ValueError, match="positive total mass"):
        normalise_strategy((0, 0, 0), _LEGAL_ACTIONS)


def test_baseline_returns_only_uniform_legal_probabilities() -> None:
    strategy = BaselineAgent().get_strategy(_INFORMATION_STATE, _LEGAL_ACTIONS)
    assert strategy == pytest.approx((1 / 3, 1 / 3, 1 / 3))

    with pytest.raises(ValueError, match="match"):
        BaselineAgent().get_strategy(_INFORMATION_STATE, _LEGAL_ACTIONS[:2])


@pytest.mark.parametrize(
    ("hole_cards", "board_cards", "legal_actions", "expected_action"),
    (
        (
            "Ac 7d",
            "2h 4s 9c",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.FOLD,
        ),
        (
            "9c 7d",
            "9h Js 2c",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.FOLD,
        ),
        (
            "Ac Jd",
            "Ah 9s 2c",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.CHECK_CALL,
        ),
        (
            "Qc Qd",
            "Jh 9s 2c",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.CHECK_CALL,
        ),
        (
            "8c 8d",
            "Jh 9s 2c",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.FOLD,
        ),
        (
            "2c 3d",
            "Ah Ad As",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.FOLD,
        ),
        (
            "As 7s",
            "2s 4s 9d",
            (Action.CHECK_CALL, Action.BET_RAISE),
            Action.CHECK_CALL,
        ),
        (
            "8c 7d",
            "6h 5s Kc",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.CHECK_CALL,
        ),
        (
            "Ah Kd",
            "As Kc 2h",
            (Action.CHECK_CALL, Action.BET_RAISE),
            Action.BET_RAISE,
        ),
        (
            "Ah Kd",
            "As Kc 2h 7c",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.CHECK_CALL,
        ),
        (
            "Ac Qd",
            "2s 4s 9s Js",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.FOLD,
        ),
        (
            "8c 7d",
            "6h 5s 4c Kc 2d",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.BET_RAISE,
        ),
        (
            "Ah Ad",
            "As Kc Kd",
            (Action.FOLD, Action.CHECK_CALL),
            Action.CHECK_CALL,
        ),
        (
            "As 7s",
            "2s 4s 9d Jc Kd",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.FOLD,
        ),
        (
            "2c 3h",
            "Ah Ad As Kc Kd",
            (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE),
            Action.FOLD,
        ),
    ),
    ids=(
        "weak-folds",
        "bottom-pair-folds",
        "top-pair-calls",
        "overpair-calls",
        "underpair-folds",
        "board-only-trips-fold",
        "flush-draw-checks",
        "open-ended-draw-calls",
        "two-pair-bets",
        "turn-two-pair-calls",
        "turn-ignores-board-only-draw",
        "river-straight-raises",
        "capped-full-house-calls",
        "river-ignores-draw",
        "river-ignores-board-only-full-house",
    ),
)
def test_rule_based_agent_applies_frozen_visible_hand_rules(
    hole_cards: str,
    board_cards: str,
    legal_actions: tuple[Action, ...],
    expected_action: Action,
) -> None:
    parsed_holes = _cards(hole_cards)
    assert len(parsed_holes) == 2
    information_state = _holdem_information_state(
        (parsed_holes[0], parsed_holes[1]),
        _cards(board_cards),
        legal_actions,
    )
    agent = RuleBasedAgent()

    first_strategy = agent.get_strategy(information_state, legal_actions)
    second_strategy = agent.get_strategy(information_state, legal_actions)

    assert first_strategy == second_strategy
    assert first_strategy == tuple(
        1.0 if action is expected_action else 0.0 for action in legal_actions
    )


def test_rule_based_agent_rejects_non_holdem_information() -> None:
    with pytest.raises(ValueError, match="Hold'em only"):
        RuleBasedAgent().get_strategy(_INFORMATION_STATE, _LEGAL_ACTIONS)


def test_seeded_sampling_is_reproducible_mixed_and_degenerate() -> None:
    agent = _FixedAgent((0.2, 0.3, 0.5))
    first_rng = SeedDeriver(2026).python_rng(RngStream.POLICY)
    second_rng = SeedDeriver(2026).python_rng(RngStream.POLICY)
    first_actions = [
        agent.sample_action(_INFORMATION_STATE, _LEGAL_ACTIONS, first_rng) for _ in range(20)
    ]
    second_actions = [
        agent.sample_action(_INFORMATION_STATE, _LEGAL_ACTIONS, second_rng) for _ in range(20)
    ]
    assert first_actions == second_actions
    assert len(set(first_actions)) > 1
    assert set(first_actions) <= set(_LEGAL_ACTIONS)

    deterministic_rng = SeedDeriver(1).python_rng(RngStream.POLICY)
    saved_state = deterministic_rng.getstate()
    assert (
        _FixedAgent((0.0, 1.0, 0.0)).sample_action(
            _INFORMATION_STATE, _LEGAL_ACTIONS, deterministic_rng
        )
        is Action.CHECK_CALL
    )
    assert deterministic_rng.getstate() == saved_state


def test_sampling_rejects_stale_actions_and_invalid_agent_output() -> None:
    rng = SeedDeriver(1).python_rng(RngStream.POLICY)
    with pytest.raises(ValueError, match="match"):
        _FixedAgent((0.5, 0.5)).sample_action(
            _INFORMATION_STATE,
            _LEGAL_ACTIONS[:2],
            rng,
        )
    with pytest.raises(ValueError, match="sum to 1"):
        _FixedAgent((0.2, 0.2, 0.2)).sample_action(
            _INFORMATION_STATE,
            _LEGAL_ACTIONS,
            rng,
        )


def _holdem_information_state(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
    legal_actions: tuple[Action, ...],
) -> InformationState:
    """Build one canonical modified-HULHE decision for baseline tests."""
    street = {3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}[len(board_cards)]
    facing_bet = Action.FOLD in legal_actions
    histories = tuple(() for _ in range(int(street) - int(Street.FLOP))) + (
        (Action.BET_RAISE,) if facing_bet else (),
    )
    return build_holdem_information_state(
        player=0,
        hole_cards=hole_cards,
        board_cards=board_cards,
        start_street=int(Street.FLOP),
        street=int(street),
        button_player=0,
        max_bets_per_round=2,
        contributions=(0, 1) if facing_bet else (0, 0),
        round_commitments=(0, 1) if facing_bet else (0, 0),
        betting_level=1 if facing_bet else 0,
        live_big_blind=False,
        round_histories=histories,
        legal_actions=legal_actions,
    )
