"""Reconstructable PyTorch networks used by Deep CFR."""

from dataclasses import dataclass
from math import isfinite

from torch import Tensor, nn

from ac_cfr.common.config import ModelConfigId
from ac_cfr.games.leduc_neural import (
    LEDUC_ACTION_COUNT,
    LEDUC_NEURAL_INPUT_SCALING,
    LEDUC_NEURAL_STATE_SIZE,
)


@dataclass(frozen=True, slots=True)
class DeepCFRNetworkConfig:
    """Complete architecture needed to reconstruct a Deep CFR network."""

    model_config_id: ModelConfigId
    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int
    input_scaling: str
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.model_config_id, ModelConfigId):
            raise TypeError("model_config_id must be a ModelConfigId")
        for name, value in (("input_size", self.input_size), ("output_size", self.output_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.hidden_sizes, tuple) or not self.hidden_sizes:
            raise ValueError("hidden_sizes must be a non-empty tuple")
        for hidden_size in self.hidden_sizes:
            if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
                raise TypeError("hidden sizes must be integers")
            if hidden_size < 1:
                raise ValueError("hidden sizes must be positive")
        if not isinstance(self.input_scaling, str) or not self.input_scaling:
            raise ValueError("input_scaling must be a non-empty string")
        if isinstance(self.dropout_probability, bool) or not isinstance(
            self.dropout_probability, (int, float)
        ):
            raise TypeError("dropout_probability must be a real number")
        if not isfinite(self.dropout_probability) or not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be finite and between zero and one")

    def to_dict(self) -> dict[str, object]:
        """Return portable architecture metadata for checkpoints and snapshots."""
        return {
            "model_config_id": self.model_config_id.value,
            "input_size": self.input_size,
            "hidden_sizes": list(self.hidden_sizes),
            "output_size": self.output_size,
            "input_scaling": self.input_scaling,
            "dropout_probability": self.dropout_probability,
        }


LEDUC_DEEP_CFR_NETWORK = DeepCFRNetworkConfig(
    model_config_id=ModelConfigId.LEDUC_DEEP_CFR,
    input_size=LEDUC_NEURAL_STATE_SIZE,
    hidden_sizes=(64, 64, 64),
    output_size=LEDUC_ACTION_COUNT,
    input_scaling=LEDUC_NEURAL_INPUT_SCALING,
)

_NETWORK_CONFIGS = {
    ModelConfigId.LEDUC_DEEP_CFR: LEDUC_DEEP_CFR_NETWORK,
}


class DeepCFRNetwork(nn.Module):
    """Feed-forward network returning one raw value for every canonical action."""

    def __init__(self, config: DeepCFRNetworkConfig) -> None:
        super().__init__()
        if not isinstance(config, DeepCFRNetworkConfig):
            raise TypeError("config must be a DeepCFRNetworkConfig")
        self.config = config

        layers: list[nn.Module] = []
        layer_input_size = config.input_size
        for hidden_size in config.hidden_sizes:
            layers.extend((nn.Linear(layer_input_size, hidden_size), nn.ReLU()))
            if config.dropout_probability > 0.0:
                layers.append(nn.Dropout(config.dropout_probability))
            layer_input_size = hidden_size
        layers.append(nn.Linear(layer_input_size, config.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, states: Tensor) -> Tensor:
        """Return raw advantages or policy logits for a batch of states."""
        return self.layers(states)


def deep_cfr_network_config(
    model_config_id: ModelConfigId,
    *,
    dropout_probability: float = 0.0,
) -> DeepCFRNetworkConfig:
    """Return the exact architecture selected by stable configuration values."""
    if not isinstance(model_config_id, ModelConfigId):
        raise TypeError("model_config_id must be a ModelConfigId")
    base_config = _NETWORK_CONFIGS[model_config_id]
    return DeepCFRNetworkConfig(
        model_config_id=base_config.model_config_id,
        input_size=base_config.input_size,
        hidden_sizes=base_config.hidden_sizes,
        output_size=base_config.output_size,
        input_scaling=base_config.input_scaling,
        dropout_probability=dropout_probability,
    )


def build_deep_cfr_network(
    model_config_id: ModelConfigId,
    *,
    dropout_probability: float = 0.0,
) -> DeepCFRNetwork:
    """Construct the exact network selected by stable configuration values."""
    return DeepCFRNetwork(
        deep_cfr_network_config(
            model_config_id,
            dropout_probability=dropout_probability,
        )
    )
