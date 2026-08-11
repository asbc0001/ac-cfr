import torch
from torch import nn

from ac_cfr.common.config import ModelConfigId
from ac_cfr.games.leduc_neural import (
    LEDUC_ACTION_COUNT,
    LEDUC_NEURAL_INPUT_SCALING,
    LEDUC_NEURAL_STATE_SIZE,
)
from ac_cfr.models import (
    LEDUC_DEEP_CFR_NETWORK,
    LEDUC_DEEP_CFR_SMALL_NETWORK,
    build_deep_cfr_network,
)


def test_leduc_network_is_reconstructable_and_has_stable_dimensions() -> None:
    network = build_deep_cfr_network(ModelConfigId.LEDUC_DEEP_CFR)
    output = network(torch.zeros((4, LEDUC_NEURAL_STATE_SIZE), dtype=torch.float32))

    assert LEDUC_DEEP_CFR_NETWORK.hidden_sizes == (64, 64, 64)
    assert LEDUC_DEEP_CFR_NETWORK.input_scaling == LEDUC_NEURAL_INPUT_SCALING
    assert LEDUC_DEEP_CFR_NETWORK.dropout_probability == 0.0
    assert output.shape == (4, LEDUC_ACTION_COUNT)
    assert output.dtype is torch.float32
    assert sum(isinstance(layer, nn.Linear) for layer in network.layers) == 4

    dropout_network = build_deep_cfr_network(
        ModelConfigId.LEDUC_DEEP_CFR,
        dropout_probability=0.25,
    )
    assert sum(isinstance(layer, nn.Dropout) for layer in dropout_network.layers) == 3

    small_network = build_deep_cfr_network(ModelConfigId.LEDUC_DEEP_CFR_SMALL)
    assert LEDUC_DEEP_CFR_SMALL_NETWORK.hidden_sizes == (32, 32, 32)
    assert sum(parameter.numel() for parameter in small_network.parameters()) < sum(
        parameter.numel() for parameter in network.parameters()
    )
