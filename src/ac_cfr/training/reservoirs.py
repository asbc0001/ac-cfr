"""Sample schemas and reference/packed uniform storage for Deep CFR."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose, isfinite
from numbers import Real
from random import Random

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, LEDUC_NEURAL_STATE_SIZE

DEEP_CFR_RESERVOIR_SCHEMA_VERSION = 1

NeuralState = tuple[float, ...]
ActionMask = tuple[bool, ...]
ActionValues = tuple[float, ...]


class UniformReservoir[SampleT]:
    """Keep a bounded uniform sample of every item admitted so far."""

    def __init__(self, capacity: int, rng: Random) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not isinstance(rng, Random):
            raise TypeError("rng must be a random.Random instance")
        self._capacity = capacity
        self._rng = rng
        self._samples: list[SampleT] = []
        self._samples_seen = 0

    @property
    def capacity(self) -> int:
        """Return the maximum number of retained samples."""
        return self._capacity

    @property
    def samples_seen(self) -> int:
        """Return the total number of samples offered to the reservoir."""
        return self._samples_seen

    @property
    def samples(self) -> tuple[SampleT, ...]:
        """Return an immutable snapshot of the retained samples."""
        return tuple(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: SampleT) -> None:
        """Admit one sample using standard uniform reservoir replacement."""
        self._samples_seen += 1
        if len(self._samples) < self._capacity:
            self._samples.append(sample)
            return

        replacement_index = self._rng.randrange(self._samples_seen)
        if replacement_index < self._capacity:
            self._samples[replacement_index] = sample

    def training_state(self) -> dict[str, object]:
        """Return occupancy and random state needed for exact continuation."""
        return {
            "samples_seen": self._samples_seen,
            "rng_state": self._rng.getstate(),
        }

    def restore_training_state(
        self,
        *,
        samples: tuple[SampleT, ...],
        samples_seen: int,
        rng_state: object,
    ) -> None:
        """Restore validated contents and sampling state without partial mutation."""
        if not isinstance(samples, tuple):
            raise TypeError("samples must be a tuple")
        if isinstance(samples_seen, bool) or not isinstance(samples_seen, int):
            raise TypeError("samples_seen must be an integer")
        if samples_seen < 0 or len(samples) != min(samples_seen, self._capacity):
            raise ValueError("reservoir occupancy is inconsistent")
        if not isinstance(rng_state, tuple):
            raise ValueError("reservoir RNG state is invalid")
        restored_rng = Random()
        try:
            restored_rng.setstate(rng_state)
        except (TypeError, ValueError) as error:
            raise ValueError("reservoir RNG state is invalid") from error

        self._samples = list(samples)
        self._samples_seen = samples_seen
        self._rng = restored_rng


class PackedAdvantageReservoir:
    """Store one player's uniform advantage reservoir in contiguous arrays."""

    def __init__(self, capacity: int, rng: Random) -> None:
        _validate_packed_reservoir_arguments(capacity, rng)
        self._capacity = capacity
        self._rng = rng
        self._size = 0
        self._samples_seen = 0
        self._states = np.empty((capacity, LEDUC_NEURAL_STATE_SIZE), dtype=np.float32)
        self._action_masks = np.empty((capacity, LEDUC_ACTION_COUNT), dtype=np.bool)
        self._iterations = np.empty(capacity, dtype=np.uint32)
        self._advantages = np.empty((capacity, LEDUC_ACTION_COUNT), dtype=np.float32)

    @property
    def capacity(self) -> int:
        """Return the maximum number of retained samples."""
        return self._capacity

    @property
    def samples_seen(self) -> int:
        """Return the total number of samples offered to the reservoir."""
        return self._samples_seen

    @property
    def resident_bytes(self) -> int:
        """Return bytes allocated by the packed sample arrays."""
        return sum(
            values.nbytes
            for values in (
                self._states,
                self._action_masks,
                self._iterations,
                self._advantages,
            )
        )

    @property
    def arrays(
        self,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool],
        NDArray[np.uint32],
        NDArray[np.float32],
    ]:
        """Return occupied views used directly by batched network training."""
        return (
            self._states[: self._size],
            self._action_masks[: self._size],
            self._iterations[: self._size],
            self._advantages[: self._size],
        )

    @property
    def samples(self) -> tuple[AdvantageSample, ...]:
        """Materialise compatible samples for checkpoints and inspection."""
        states, masks, iterations, advantages = self.arrays
        return tuple(
            AdvantageSample(
                state=tuple(float(value) for value in state),
                action_mask=tuple(bool(value) for value in mask),
                iteration=int(iteration),
                advantages=tuple(float(value) for value in values),
            )
            for state, mask, iteration, values in zip(
                states, masks, iterations, advantages, strict=True
            )
        )

    def __len__(self) -> int:
        return self._size

    def add(self, sample: AdvantageSample) -> None:
        """Admit one validated sample using uniform reservoir replacement."""
        if not isinstance(sample, AdvantageSample):
            raise TypeError("sample must be an AdvantageSample")
        self.add_values(sample.state, sample.action_mask, sample.iteration, sample.advantages)

    def add_values(
        self,
        state: NeuralState | NDArray[np.float32],
        action_mask: ActionMask | NDArray[np.bool],
        iteration: int,
        advantages: ActionValues | NDArray[np.float32] | NDArray[np.float64],
    ) -> None:
        """Admit prevalidated traversal values without constructing an object."""
        _validate_packed_iteration(iteration)
        index = self._admission_index()
        if index is None:
            return
        self._states[index] = state
        self._action_masks[index] = action_mask
        self._iterations[index] = iteration
        self._advantages[index] = advantages

    def training_state(self) -> dict[str, object]:
        """Return occupancy and random state needed for exact continuation."""
        return {"samples_seen": self._samples_seen, "rng_state": self._rng.getstate()}

    def restore_training_state(
        self,
        *,
        samples: tuple[AdvantageSample, ...],
        samples_seen: int,
        rng_state: object,
    ) -> None:
        """Restore validated contents and sampling state without random admission."""
        restored_rng = _validated_restore_state(
            samples=samples,
            samples_seen=samples_seen,
            capacity=self._capacity,
            rng_state=rng_state,
            sample_type=AdvantageSample,
        )
        self._size = len(samples)
        self._samples_seen = samples_seen
        self._rng = restored_rng
        for index, sample in enumerate(samples):
            self._states[index] = sample.state
            self._action_masks[index] = sample.action_mask
            self._iterations[index] = sample.iteration
            self._advantages[index] = sample.advantages

    def _admission_index(self) -> int | None:
        self._samples_seen += 1
        if self._size < self._capacity:
            index = self._size
            self._size += 1
            return index
        replacement_index = self._rng.randrange(self._samples_seen)
        return replacement_index if replacement_index < self._capacity else None


class PackedStrategyReservoir:
    """Store the shared uniform strategy reservoir in contiguous arrays."""

    def __init__(self, capacity: int, rng: Random) -> None:
        _validate_packed_reservoir_arguments(capacity, rng)
        self._capacity = capacity
        self._rng = rng
        self._size = 0
        self._samples_seen = 0
        self._players = np.empty(capacity, dtype=np.int8)
        self._states = np.empty((capacity, LEDUC_NEURAL_STATE_SIZE), dtype=np.float32)
        self._action_masks = np.empty((capacity, LEDUC_ACTION_COUNT), dtype=np.bool)
        self._iterations = np.empty(capacity, dtype=np.uint32)
        self._strategies = np.empty((capacity, LEDUC_ACTION_COUNT), dtype=np.float32)

    @property
    def capacity(self) -> int:
        """Return the maximum number of retained samples."""
        return self._capacity

    @property
    def samples_seen(self) -> int:
        """Return the total number of samples offered to the reservoir."""
        return self._samples_seen

    @property
    def resident_bytes(self) -> int:
        """Return bytes allocated by the packed sample arrays."""
        return sum(
            values.nbytes
            for values in (
                self._players,
                self._states,
                self._action_masks,
                self._iterations,
                self._strategies,
            )
        )

    @property
    def arrays(
        self,
    ) -> tuple[
        NDArray[np.int8],
        NDArray[np.float32],
        NDArray[np.bool],
        NDArray[np.uint32],
        NDArray[np.float32],
    ]:
        """Return occupied views used directly by batched network training."""
        return (
            self._players[: self._size],
            self._states[: self._size],
            self._action_masks[: self._size],
            self._iterations[: self._size],
            self._strategies[: self._size],
        )

    @property
    def samples(self) -> tuple[StrategySample, ...]:
        """Materialise compatible samples for checkpoints and inspection."""
        players, states, masks, iterations, strategies = self.arrays
        return tuple(
            StrategySample(
                player=int(player),
                state=tuple(float(value) for value in state),
                action_mask=tuple(bool(value) for value in mask),
                iteration=int(iteration),
                strategy=_normalised_strategy_tuple(values, mask),
            )
            for player, state, mask, iteration, values in zip(
                players, states, masks, iterations, strategies, strict=True
            )
        )

    def __len__(self) -> int:
        return self._size

    def add(self, sample: StrategySample) -> None:
        """Admit one validated sample using uniform reservoir replacement."""
        if not isinstance(sample, StrategySample):
            raise TypeError("sample must be a StrategySample")
        self.add_values(
            sample.player,
            sample.state,
            sample.action_mask,
            sample.iteration,
            sample.strategy,
        )

    def add_values(
        self,
        player: int,
        state: NeuralState | NDArray[np.float32],
        action_mask: ActionMask | NDArray[np.bool],
        iteration: int,
        strategy: ActionValues | NDArray[np.float32] | NDArray[np.float64],
    ) -> None:
        """Admit prevalidated traversal values without constructing an object."""
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        _validate_packed_iteration(iteration)
        index = self._admission_index()
        if index is None:
            return
        self._players[index] = player
        self._states[index] = state
        self._action_masks[index] = action_mask
        self._iterations[index] = iteration
        self._strategies[index] = strategy

    def training_state(self) -> dict[str, object]:
        """Return occupancy and random state needed for exact continuation."""
        return {"samples_seen": self._samples_seen, "rng_state": self._rng.getstate()}

    def restore_training_state(
        self,
        *,
        samples: tuple[StrategySample, ...],
        samples_seen: int,
        rng_state: object,
    ) -> None:
        """Restore validated contents and sampling state without random admission."""
        restored_rng = _validated_restore_state(
            samples=samples,
            samples_seen=samples_seen,
            capacity=self._capacity,
            rng_state=rng_state,
            sample_type=StrategySample,
        )
        self._size = len(samples)
        self._samples_seen = samples_seen
        self._rng = restored_rng
        for index, sample in enumerate(samples):
            self._players[index] = sample.player
            self._states[index] = sample.state
            self._action_masks[index] = sample.action_mask
            self._iterations[index] = sample.iteration
            self._strategies[index] = sample.strategy

    def _admission_index(self) -> int | None:
        self._samples_seen += 1
        if self._size < self._capacity:
            index = self._size
            self._size += 1
            return index
        replacement_index = self._rng.randrange(self._samples_seen)
        return replacement_index if replacement_index < self._capacity else None


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


def _validate_packed_reservoir_arguments(capacity: int, rng: Random) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if not isinstance(rng, Random):
        raise TypeError("rng must be a random.Random instance")


def _normalised_strategy_tuple(
    values: NDArray[np.float32],
    action_mask: NDArray[np.bool],
) -> ActionValues:
    """Recover an exactly normalised inspection/checkpoint view from float32 storage."""
    total = fsum(float(value) for value, legal in zip(values, action_mask, strict=True) if legal)
    if not isfinite(total) or total <= 0.0:
        raise ValueError("packed strategy must contain positive finite legal probability")
    return tuple(
        float(value) / total if legal else 0.0
        for value, legal in zip(values, action_mask, strict=True)
    )


def _validate_packed_iteration(iteration: int) -> None:
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("iteration must be an integer")
    if not 1 <= iteration <= np.iinfo(np.uint32).max:
        raise ValueError("iteration is outside packed uint32 range")


def _validated_restore_state[SampleT](
    *,
    samples: tuple[SampleT, ...],
    samples_seen: int,
    capacity: int,
    rng_state: object,
    sample_type: type[SampleT],
) -> Random:
    if not isinstance(samples, tuple) or any(
        not isinstance(sample, sample_type) for sample in samples
    ):
        raise TypeError("samples contain an incompatible sample type")
    if isinstance(samples_seen, bool) or not isinstance(samples_seen, int):
        raise TypeError("samples_seen must be an integer")
    if samples_seen < 0 or len(samples) != min(samples_seen, capacity):
        raise ValueError("reservoir occupancy is inconsistent")
    if not isinstance(rng_state, tuple):
        raise ValueError("reservoir RNG state is invalid")
    restored_rng = Random()
    try:
        restored_rng.setstate(rng_state)
    except (TypeError, ValueError) as error:
        raise ValueError("reservoir RNG state is invalid") from error
    return restored_rng
