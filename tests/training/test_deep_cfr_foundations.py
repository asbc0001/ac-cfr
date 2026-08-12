from pathlib import Path

import pytest

from ac_cfr.benchmarking.deep_cfr_sensitivity import deep_cfr_sensitivity_cases
from ac_cfr.common.config import (
    DeepCFRImplementationId,
    GameConfigurationId,
    ModelConfigId,
    StateEncodingId,
)
from ac_cfr.games.leduc_neural import LEDUC_NEURAL_STATE_SIZE
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_config import load_deep_cfr_run_config
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
        advantage_training_steps=4,
        strategy_training_steps=8,
        advantage_batch_size=128,
        strategy_batch_size=128,
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


def test_deep_cfr_toml_is_strict_and_cli_values_override_the_preset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = Path(__file__).parents[2] / "configs" / "deep_cfr" / "leduc_baseline.toml"
    config = load_deep_cfr_run_config(
        preset,
        run_id="configured_deep_cfr",
        overrides={
            "iterations": 12,
            "implementation": DeepCFRImplementationId.REFERENCE.value,
            "snapshot_iterations": (2, 12),
            "inference_batch_size": 256,
            "model_config_id": ModelConfigId.LEDUC_DEEP_CFR_SMALL.value,
        },
    )

    assert config.implementation is DeepCFRImplementationId.REFERENCE
    assert config.training.iterations == 12
    assert config.training.snapshot_iterations == (2, 12)
    assert config.training.model_config_id is ModelConfigId.LEDUC_DEEP_CFR_SMALL
    assert config.training.advantage_batch_size == 512
    assert config.training.strategy_batch_size == 512
    assert config.runtime.inference_batch_size == 256
    assert config.runtime.cpu_threads == 1
    assert config.to_dict()["runtime"] == {
        "inference_batch_size": 256,
        "cpu_threads": 1,
        "device": "cpu",
        "traversal_workers": 1,
        "storage_budget_bytes": None,
    }

    with pytest.raises(ValueError, match="unknown"):
        load_deep_cfr_run_config(
            preset,
            run_id="invalid_override",
            overrides={"invented_setting": 1},
        )

    invalid_preset = tmp_path / "invalid.toml"
    invalid_preset.write_text(
        f"{preset.read_text(encoding='utf-8')}\ninvented_setting = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime configuration fields"):
        load_deep_cfr_run_config(invalid_preset, run_id="invalid_preset")

    monkeypatch.setattr(
        "ac_cfr.training.deep_cfr_config.effective_cpu_count",
        lambda: 13.6,
    )
    holdem = load_deep_cfr_run_config(
        preset.parents[0] / "modified_hulhe_calibration.toml",
        run_id="modified_hulhe_calibration",
    )
    assert holdem.training.game_configuration_id is GameConfigurationId.MODIFIED_HULHE
    assert holdem.training.traversals_per_player == 10_000
    assert holdem.training.advantage_training_steps == 16_000
    assert holdem.training.strategy_training_steps == 16_000
    assert holdem.training.advantage_batch_size == 10_000
    assert holdem.training.strategy_batch_size == 10_000
    assert holdem.training.advantage_reservoir_capacity == 10_000_000
    assert holdem.training.strategy_reservoir_capacity == 10_000_000
    assert holdem.runtime.device == "cuda"
    assert holdem.runtime.traversal_workers == 10
    assert holdem.runtime.storage_budget_bytes == 200_000_000_000
    assert holdem.checkpoint_retention == 2
    assert type(holdem).from_dict(holdem.to_dict()) == holdem

    cloud_smoke = load_deep_cfr_run_config(
        preset.parents[0] / "modified_hulhe_cloud_smoke.toml",
        run_id="modified_hulhe_cloud_smoke",
        overrides={"traversal_workers": 1},
    )
    assert cloud_smoke.training.iterations == 2
    assert cloud_smoke.training.traversals_per_player == 100
    assert cloud_smoke.runtime.device == "cuda"
    assert cloud_smoke.runtime.storage_budget_bytes == 200_000_000_000


def test_deep_cfr_sensitivity_cases_change_one_declared_factor() -> None:
    config_directory = Path(__file__).parents[2] / "configs" / "deep_cfr"
    preset = config_directory / "leduc_baseline.toml"
    baseline = load_deep_cfr_run_config(preset, run_id="sensitivity_test").training
    cases = deep_cfr_sensitivity_cases(baseline)

    assert tuple(case.changed_factor for case in cases) == (
        "none",
        "traversals_per_player",
        "advantage_training_steps",
        "advantage_training_steps",
        "advantage_training_steps",
        "model_config_id",
    )
    baseline_values = baseline.to_dict()
    for case in cases[1:]:
        changed = {
            name for name, value in case.config.to_dict().items() if value != baseline_values[name]
        }
        assert changed == {case.changed_factor}

    selected = load_deep_cfr_run_config(
        config_directory / "leduc_selected.toml",
        run_id="selected_test",
    )
    assert selected.training.iterations == 100
    assert selected.training.advantage_training_steps == 200
