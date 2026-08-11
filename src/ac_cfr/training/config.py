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
    advantage_training_steps: int
    strategy_training_steps: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    max_gradient_norm: float | None
    dropout_probability: float
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
            "advantage_training_steps",
            "strategy_training_steps",
            "batch_size",
        ):
            _validate_positive_integer(name, getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        SeedDeriver(self.seed)
        _validate_positive_real("learning_rate", self.learning_rate)
        if isinstance(self.validation_fraction, bool) or not isinstance(
            self.validation_fraction, (int, float)
        ):
            raise TypeError("validation_fraction must be a real number")
        if not isfinite(self.validation_fraction) or not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be finite and between zero and one")
        if self.max_gradient_norm is not None:
            _validate_positive_real("max_gradient_norm", self.max_gradient_norm)
        if isinstance(self.dropout_probability, bool) or not isinstance(
            self.dropout_probability, (int, float)
        ):
            raise TypeError("dropout_probability must be a real number")
        if not isfinite(self.dropout_probability) or not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be finite and between zero and one")
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
        values["model_config_id"] = self.model_config_id.value
        values["state_encoding_id"] = self.state_encoding_id.value
        values["optimizer_id"] = self.optimizer_id.value
        return values

    @classmethod
    def from_dict(cls, values: object) -> "DeepCFRTrainingConfig":
        """Reconstruct an exactly validated checkpointed configuration."""
        if not isinstance(values, dict) or set(values) != {
            "iterations",
            "traversals_per_player",
            "advantage_reservoir_capacity",
            "strategy_reservoir_capacity",
            "advantage_training_steps",
            "strategy_training_steps",
            "batch_size",
            "learning_rate",
            "validation_fraction",
            "max_gradient_norm",
            "dropout_probability",
            "seed",
            "snapshot_iterations",
            "model_config_id",
            "state_encoding_id",
            "optimizer_id",
        }:
            raise ValueError("Deep CFR training configuration fields are incompatible")
        parsed = values.copy()
        snapshots = parsed["snapshot_iterations"]
        if not isinstance(snapshots, list):
            raise ValueError("snapshot_iterations must be stored as a list")
        try:
            parsed["snapshot_iterations"] = tuple(snapshots)
            parsed["model_config_id"] = ModelConfigId(parsed["model_config_id"])
            parsed["state_encoding_id"] = StateEncodingId(parsed["state_encoding_id"])
            parsed["optimizer_id"] = OptimizerId(parsed["optimizer_id"])
            return cls(**parsed)
        except (TypeError, ValueError) as error:
            raise ValueError("Deep CFR training configuration is invalid") from error


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_positive_real(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
