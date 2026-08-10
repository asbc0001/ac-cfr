"""Validated Deep CFR training configuration."""

from dataclasses import asdict, dataclass
from math import isfinite

from ac_cfr.common.config import ModelConfigId, OptimizerId, StateEncodingId
from ac_cfr.common.rng import SeedDeriver


@dataclass(frozen=True, slots=True)
class DeepCFRTrainingConfig:
    """Algorithmic work and memory limits for one Leduc Deep CFR run."""

    iterations: int
    traversals_per_player: int
    advantage_reservoir_capacity: int
    strategy_reservoir_capacity: int
    advantage_training_epochs: int
    strategy_training_epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    snapshot_iterations: tuple[int, ...] = ()
    model_config_id: ModelConfigId = ModelConfigId.LEDUC_DEEP_CFR
    state_encoding_id: StateEncodingId = StateEncodingId.LEDUC_NEURAL
    optimizer_id: OptimizerId = OptimizerId.ADAM

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "traversals_per_player",
            "advantage_reservoir_capacity",
            "strategy_reservoir_capacity",
            "advantage_training_epochs",
            "strategy_training_epochs",
            "batch_size",
        ):
            _validate_positive_integer(name, getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        SeedDeriver(self.seed)
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise TypeError("learning_rate must be a real number")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(self.snapshot_iterations, tuple):
            raise TypeError("snapshot_iterations must be a tuple")
        if tuple(sorted(set(self.snapshot_iterations))) != self.snapshot_iterations:
            raise ValueError("snapshot_iterations must be sorted and unique")
        for iteration in self.snapshot_iterations:
            _validate_positive_integer("snapshot iteration", iteration)
            if iteration > self.iterations:
                raise ValueError("snapshot iterations must not exceed the training budget")
        if self.model_config_id is not ModelConfigId.LEDUC_DEEP_CFR:
            raise ValueError("model_config_id must select the Leduc Deep CFR network")
        if self.state_encoding_id is not StateEncodingId.LEDUC_NEURAL:
            raise ValueError("state_encoding_id must select the Leduc neural encoding")
        if self.optimizer_id is not OptimizerId.ADAM:
            raise ValueError("optimizer_id must select Adam")

    def to_dict(self) -> dict[str, object]:
        """Return stable configuration values suitable for checkpoint metadata."""
        values = asdict(self)
        values["snapshot_iterations"] = list(self.snapshot_iterations)
        return values


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
