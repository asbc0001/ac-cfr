"""Sample schemas shared by Deep CFR reservoir implementations."""

from dataclasses import dataclass
from math import fsum, isclose, isfinite
from numbers import Real

from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, LEDUC_NEURAL_STATE_SIZE

DEEP_CFR_RESERVOIR_SCHEMA_VERSION = 1

NeuralState = tuple[float, ...]
ActionMask = tuple[bool, ...]
ActionValues = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AdvantageSample:
    """One traverser advantage target stored in a player's reservoir."""

    state: NeuralState
    action_mask: ActionMask
    iteration: int
    advantages: ActionValues

    def __post_init__(self) -> None:
        _validate_common_sample(self.state, self.action_mask, self.iteration)
        _validate_action_values("advantages", self.advantages, self.action_mask)


@dataclass(frozen=True, slots=True)
class StrategySample:
    """One opponent strategy target stored in the shared reservoir."""

    player: int
    state: NeuralState
    action_mask: ActionMask
    iteration: int
    strategy: ActionValues

    def __post_init__(self) -> None:
        if isinstance(self.player, bool) or not isinstance(self.player, int):
            raise TypeError("player must be an integer")
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        _validate_common_sample(self.state, self.action_mask, self.iteration)
        _validate_action_values("strategy", self.strategy, self.action_mask, normalised=True)


def _validate_common_sample(state: NeuralState, action_mask: ActionMask, iteration: int) -> None:
    if not isinstance(state, tuple) or len(state) != LEDUC_NEURAL_STATE_SIZE:
        raise ValueError(f"state must contain {LEDUC_NEURAL_STATE_SIZE} values")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in state):
        raise TypeError("state values must be real numbers")
    if any(not isfinite(float(value)) for value in state):
        raise ValueError("state values must be finite")
    if not isinstance(action_mask, tuple) or len(action_mask) != LEDUC_ACTION_COUNT:
        raise ValueError(f"action_mask must contain {LEDUC_ACTION_COUNT} values")
    if any(not isinstance(value, bool) for value in action_mask):
        raise TypeError("action_mask values must be booleans")
    if not any(action_mask):
        raise ValueError("action_mask must contain a legal action")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("iteration must be an integer")
    if iteration < 1:
        raise ValueError("iteration must be positive")


def _validate_action_values(
    name: str,
    values: ActionValues,
    action_mask: ActionMask,
    *,
    normalised: bool = False,
) -> None:
    if not isinstance(values, tuple) or len(values) != LEDUC_ACTION_COUNT:
        raise ValueError(f"{name} must contain {LEDUC_ACTION_COUNT} values")
    for value, is_legal in zip(values, action_mask, strict=True):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} values must be real numbers")
        if not isfinite(float(value)):
            raise ValueError(f"{name} values must be finite")
        if not is_legal and value != 0.0:
            raise ValueError(f"illegal-action {name} values must be zero")
        if normalised and value < 0.0:
            raise ValueError("strategy values must be non-negative")
    if normalised and not isclose(
        fsum(float(value) for value in values), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("strategy values must sum to one")
