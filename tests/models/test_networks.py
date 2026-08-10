import torch
from torch import nn

from ac_cfr.common.config import ModelConfigId
from ac_cfr.games.leduc_neural import LEDUC_ACTION_COUNT, LEDUC_NEURAL_STATE_SIZE
from ac_cfr.models import LEDUC_DEEP_CFR_NETWORK, build_deep_cfr_network


def test_leduc_network_is_reconstructable_and_has_stable_dimensions() -> None:
    network = build_deep_cfr_network(ModelConfigId.LEDUC_DEEP_CFR)
    output = network(torch.zeros((4, LEDUC_NEURAL_STATE_SIZE), dtype=torch.float32))

    assert LEDUC_DEEP_CFR_NETWORK.hidden_sizes == (64, 64, 64)
    assert output.shape == (4, LEDUC_ACTION_COUNT)
    assert output.dtype is torch.float32
    assert sum(isinstance(layer, nn.Linear) for layer in network.layers) == 4
