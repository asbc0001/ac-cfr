"""Shared contract and probability handling for frozen playable agents."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import fsum, isclose, isfinite
from numbers import Real
from random import Random

from ac_cfr.games.base import Action, InformationState, validate_legal_actions

Strategy = tuple[float, ...]
StrategyValue = int | float


class PlayableAgent(ABC):
    """Frozen policy queried through player-visible information only."""

    __slots__ = ()

    @abstractmethod
    def get_strategy(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
    ) -> Strategy:
        """Return probabilities aligned with the supplied legal actions."""
        raise NotImplementedError

    def sample_action(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
        rng: Random,
    ) -> Action:
        """Sample one legal action from this agent's mixed strategy."""
        _validate_request(information_state, legal_actions)
        if not isinstance(rng, Random):
            raise TypeError("rng must be a random.Random instance")
        strategy = validate_strategy(
            self.get_strategy(information_state, legal_actions), legal_actions
        )

        for action, probability in zip(legal_actions, strategy, strict=True):
            if probability == 1.0:
                return action

        draw = rng.random()
        cumulative_probability = 0.0
        for action, probability in zip(legal_actions, strategy, strict=True):
            cumulative_probability += probability
            if draw < cumulative_probability:
                return action
        return legal_actions[-1]


def normalise_strategy(
    weights: Sequence[StrategyValue],
    legal_actions: tuple[Action, ...],
    *,
    uniform_if_zero: bool = False,
) -> Strategy:
    """Convert finite non-negative legal-action weights to probabilities.

    ``uniform_if_zero`` is explicit because only defined policy boundaries, such
    as an unvisited tabular information set, may use a uniform zero-mass fallback.
    """
    validate_legal_actions(legal_actions)
    values = _validate_values(weights, len(legal_actions), "strategy weights")
    maximum = max(values)
    if maximum == 0.0:
        if not uniform_if_zero:
            raise ValueError("strategy weights must have positive total mass")
        probability = 1.0 / len(legal_actions)
        return tuple(probability for _ in legal_actions)

    scaled_values = tuple(value / maximum for value in values)
    total = fsum(scaled_values)
    strategy = tuple(value / total for value in scaled_values)
    return validate_strategy(strategy, legal_actions)


def validate_strategy(
    strategy: Sequence[StrategyValue],
    legal_actions: tuple[Action, ...],
) -> Strategy:
    """Validate an already-normalised distribution in legal-action order."""
    validate_legal_actions(legal_actions)
    probabilities = _validate_values(strategy, len(legal_actions), "strategy")
    if not isclose(fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("strategy probabilities must sum to 1")
    return probabilities


def _validate_request(
    information_state: InformationState,
    legal_actions: tuple[Action, ...],
) -> None:
    if not isinstance(information_state, InformationState):
        raise TypeError("information_state must be an InformationState")
    validate_legal_actions(legal_actions)
    if legal_actions != information_state.legal_actions:
        raise ValueError("legal_actions must match the information state")


def _validate_values(
    values: Sequence[StrategyValue],
    expected_count: int,
    name: str,
) -> Strategy:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) != expected_count:
        raise ValueError(f"{name} must contain one value per legal action")

    parsed_values: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"every {name} value must be a real number")
        parsed_value = float(value)
        if not isfinite(parsed_value):
            raise ValueError(f"every {name} value must be finite")
        if parsed_value < 0.0:
            raise ValueError(f"every {name} value must be non-negative")
        parsed_values.append(parsed_value)
    return tuple(parsed_values)
