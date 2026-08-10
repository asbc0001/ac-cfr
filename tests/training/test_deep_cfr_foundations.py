import pytest

from ac_cfr.common.config import ModelConfigId, StateEncodingId
from ac_cfr.games.leduc_neural import LEDUC_NEURAL_STATE_SIZE
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import (
    DEEP_CFR_RESERVOIR_SCHEMA_VERSION,
    AdvantageSample,
    StrategySample,
)

_STATE = (0.0,) * LEDUC_NEURAL_STATE_SIZE
_MASK = (False, True, True)


def test_deep_cfr_configuration_and_sample_schemas_are_explicit() -> None:
    config = DeepCFRTrainingConfig(
        iterations=100,
        traversals_per_player=50,
        advantage_reservoir_capacity=10_000,
        strategy_reservoir_capacity=10_000,
        advantage_training_epochs=4,
        strategy_training_epochs=8,
        batch_size=128,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=2026,
        snapshot_iterations=(25, 100),
    )
    advantage_sample = AdvantageSample(_STATE, _MASK, 3, (0.0, -0.5, 0.5))
    strategy_sample = StrategySample(1, _STATE, _MASK, 3, (0.0, 0.25, 0.75))

    assert config.model_config_id is ModelConfigId.LEDUC_DEEP_CFR
    assert config.state_encoding_id is StateEncodingId.LEDUC_NEURAL
    assert config.validation_fraction == 0.1
    assert config.max_gradient_norm == 10.0
    assert config.dropout_probability == 0.0
    assert config.to_dict()["snapshot_iterations"] == [25, 100]
    assert advantage_sample.iteration == strategy_sample.iteration == 3
    assert DEEP_CFR_RESERVOIR_SCHEMA_VERSION == 1

    with pytest.raises(ValueError, match="illegal-action"):
        StrategySample(1, _STATE, _MASK, 3, (0.1, 0.2, 0.7))
