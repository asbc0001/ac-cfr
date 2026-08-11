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
PackedStates = NDArray[np.float16] | NDArray[np.float32]
_DEFAULT_PACKED_STATE_DTYPE = np.dtype(np.float32)


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

    def __init__(
        self,
        capacity: int,
        rng: Random,
        *,
        state_size: int = LEDUC_NEURAL_STATE_SIZE,
        action_count: int = LEDUC_ACTION_COUNT,
        state_dtype: np.dtype[np.floating] = _DEFAULT_PACKED_STATE_DTYPE,
    ) -> None:
        _validate_packed_reservoir_arguments(capacity, rng)
        _validate_packed_layout(state_size, action_count, state_dtype)
        self._capacity = capacity
        self._rng = rng
        self._size = 0
        self._samples_seen = 0
        self._states = np.empty((capacity, state_size), dtype=state_dtype)
        self._action_masks = np.empty((capacity, action_count), dtype=np.bool)
        self._iterations = np.empty(capacity, dtype=np.uint32)
        self._advantages = np.empty((capacity, action_count), dtype=np.float32)

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
    def bytes_per_sample(self) -> int:
        """Return the exact allocated bytes for one packed row."""
        return self.resident_bytes // self._capacity

    @property
    def state_size(self) -> int:
        """Return the stored neural-state width."""
        return int(self._states.shape[1])

    @property
    def action_count(self) -> int:
        """Return the canonical action width."""
        return int(self._action_masks.shape[1])

    @property
    def arrays(
        self,
    ) -> tuple[
        PackedStates,
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

    def restore_arrays(
        self,
        *,
        states: PackedStates,
        action_masks: NDArray[np.bool],
        iterations: NDArray[np.uint32],
        advantages: NDArray[np.float32],
        samples_seen: int,
        rng_state: object,
    ) -> None:
        """Restore validated packed contents without materialising sample objects."""
        restored_rng = _validated_packed_arrays(
            states=states,
            action_masks=action_masks,
            iterations=iterations,
            targets=advantages,
            samples_seen=samples_seen,
            capacity=self._capacity,
            expected_state_dtype=self._states.dtype,
            state_size=self.state_size,
            action_count=self.action_count,
            rng_state=rng_state,
            normalised=False,
        )
        self._restore_array_values(
            states,
            action_masks,
            iterations,
            advantages,
            samples_seen,
            restored_rng,
        )

    def _restore_array_values(
        self,
        states: PackedStates,
        action_masks: NDArray[np.bool],
        iterations: NDArray[np.uint32],
        advantages: NDArray[np.float32],
        samples_seen: int,
        rng: Random,
    ) -> None:
        size = len(states)
        self._states[:size] = states
        self._action_masks[:size] = action_masks
        self._iterations[:size] = iterations
        self._advantages[:size] = advantages
        self._size = size
        self._samples_seen = samples_seen
        self._rng = rng

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

    def __init__(
        self,
        capacity: int,
        rng: Random,
        *,
        state_size: int = LEDUC_NEURAL_STATE_SIZE,
        action_count: int = LEDUC_ACTION_COUNT,
        state_dtype: np.dtype[np.floating] = _DEFAULT_PACKED_STATE_DTYPE,
    ) -> None:
        _validate_packed_reservoir_arguments(capacity, rng)
        _validate_packed_layout(state_size, action_count, state_dtype)
        self._capacity = capacity
        self._rng = rng
        self._size = 0
        self._samples_seen = 0
        self._players = np.empty(capacity, dtype=np.int8)
        self._states = np.empty((capacity, state_size), dtype=state_dtype)
        self._action_masks = np.empty((capacity, action_count), dtype=np.bool)
        self._iterations = np.empty(capacity, dtype=np.uint32)
        self._strategies = np.empty((capacity, action_count), dtype=np.float32)

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
    def bytes_per_sample(self) -> int:
        """Return the exact allocated bytes for one packed row."""
        return self.resident_bytes // self._capacity

    @property
    def state_size(self) -> int:
        """Return the stored neural-state width."""
        return int(self._states.shape[1])

    @property
    def action_count(self) -> int:
        """Return the canonical action width."""
        return int(self._action_masks.shape[1])

    @property
    def arrays(
        self,
    ) -> tuple[
        NDArray[np.int8],
        PackedStates,
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

    def restore_arrays(
        self,
        *,
        players: NDArray[np.int8],
        states: PackedStates,
        action_masks: NDArray[np.bool],
        iterations: NDArray[np.uint32],
        strategies: NDArray[np.float32],
        samples_seen: int,
        rng_state: object,
    ) -> None:
        """Restore validated packed contents without materialising sample objects."""
        if players.shape != (len(states),) or players.dtype != np.int8:
            raise ValueError("packed players have an incompatible layout")
        if np.any((players < 0) | (players > 1)):
            raise ValueError("packed players must be zero or one")
        restored_rng = _validated_packed_arrays(
            states=states,
            action_masks=action_masks,
            iterations=iterations,
            targets=strategies,
            samples_seen=samples_seen,
            capacity=self._capacity,
            expected_state_dtype=self._states.dtype,
            state_size=self.state_size,
            action_count=self.action_count,
            rng_state=rng_state,
            normalised=True,
        )
        size = len(states)
        self._players[:size] = players
        self._states[:size] = states
        self._action_masks[:size] = action_masks
        self._iterations[:size] = iterations
        self._strategies[:size] = strategies
        self._size = size
        self._samples_seen = samples_seen
        self._rng = restored_rng

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
    if not isinstance(state, tuple) or not state:
        raise ValueError("state must contain at least one value")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in state):
        raise TypeError("state values must be real numbers")
    if any(not isfinite(float(value)) for value in state):
        raise ValueError("state values must be finite")
    if not isinstance(action_mask, tuple) or not action_mask:
        raise ValueError("action_mask must contain at least one value")
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
    if not isinstance(values, tuple) or len(values) != len(action_mask):
        raise ValueError(f"{name} must match the action mask")
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


def _validate_packed_layout(
    state_size: int,
    action_count: int,
    state_dtype: np.dtype[np.floating],
) -> None:
    for name, value in (("state_size", state_size), ("action_count", action_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if np.dtype(state_dtype) not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("state_dtype must be float16 or float32")


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


def _validated_packed_arrays(
    *,
    states: PackedStates,
    action_masks: NDArray[np.bool],
    iterations: NDArray[np.uint32],
    targets: NDArray[np.float32],
    samples_seen: int,
    capacity: int,
    expected_state_dtype: np.dtype[np.floating],
    state_size: int,
    action_count: int,
    rng_state: object,
    normalised: bool,
) -> Random:
    sample_count = len(states)
    if states.shape != (sample_count, state_size) or states.dtype != expected_state_dtype:
        raise ValueError("packed states have an incompatible layout")
    if action_masks.shape != (sample_count, action_count) or action_masks.dtype != np.bool:
        raise ValueError("packed action masks have an incompatible layout")
    if iterations.shape != (sample_count,) or iterations.dtype != np.uint32:
        raise ValueError("packed iterations have an incompatible layout")
    if targets.shape != (sample_count, action_count) or targets.dtype != np.float32:
        raise ValueError("packed targets have an incompatible layout")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(targets)):
        raise ValueError("packed floating-point values must be finite")
    if sample_count and (
        not np.all(np.any(action_masks, axis=1))
        or np.any(targets[~action_masks] != 0.0)
        or np.any(iterations == 0)
    ):
        raise ValueError("packed samples violate the Deep CFR schema")
    if normalised and sample_count:
        legal_targets = np.where(action_masks, targets, 0.0)
        if np.any(legal_targets < 0.0) or not np.allclose(
            legal_targets.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("packed strategies must be normalised")
    if (
        isinstance(samples_seen, bool)
        or not isinstance(samples_seen, int)
        or sample_count != min(samples_seen, capacity)
    ):
        raise ValueError("packed reservoir occupancy is inconsistent")
    if not isinstance(rng_state, tuple):
        raise ValueError("reservoir RNG state is invalid")
    restored_rng = Random()
    try:
        restored_rng.setstate(rng_state)
    except (TypeError, ValueError) as error:
        raise ValueError("reservoir RNG state is invalid") from error
    return restored_rng
