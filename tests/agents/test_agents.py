from math import inf, nan

import pytest

from ac_cfr.agents import (
    BaselineAgent,
    PlayableAgent,
    Strategy,
    normalise_strategy,
    validate_strategy,
)
from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import Action, GameId, InformationState

_LEGAL_ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE)
_INFORMATION_STATE = InformationState(GameId.LEDUC, 0, (1, 2, 0), _LEGAL_ACTIONS)


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
    assert validate_strategy(strategy, _LEGAL_ACTIONS) == strategy

    with pytest.raises(ValueError, match="match"):
        BaselineAgent().get_strategy(_INFORMATION_STATE, _LEGAL_ACTIONS[:2])


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
