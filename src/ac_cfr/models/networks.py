"""Reconstructable PyTorch networks used by Deep CFR."""

from dataclasses import dataclass

from torch import Tensor, nn

from ac_cfr.common.config import ModelConfigId
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, LEDUC_NEURAL_STATE_SIZE


@dataclass(frozen=True, slots=True)
class DeepCFRNetworkConfig:
    """Complete architecture needed to reconstruct a Deep CFR network."""

    model_config_id: ModelConfigId
    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int

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


LEDUC_DEEP_CFR_NETWORK = DeepCFRNetworkConfig(
    model_config_id=ModelConfigId.LEDUC_DEEP_CFR,
    input_size=LEDUC_NEURAL_STATE_SIZE,
    hidden_sizes=(64, 64, 64),
    output_size=LEDUC_ACTION_COUNT,
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
            layer_input_size = hidden_size
        layers.append(nn.Linear(layer_input_size, config.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, states: Tensor) -> Tensor:
        """Return raw advantages or policy logits for a batch of states."""
        return self.layers(states)


def build_deep_cfr_network(model_config_id: ModelConfigId) -> DeepCFRNetwork:
    """Construct the exact network selected by a stable model identifier."""
    if not isinstance(model_config_id, ModelConfigId):
        raise TypeError("model_config_id must be a ModelConfigId")
    return DeepCFRNetwork(_NETWORK_CONFIGS[model_config_id])
