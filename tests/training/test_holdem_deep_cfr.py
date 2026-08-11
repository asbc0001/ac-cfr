from pathlib import Path

import pytest

from ac_cfr.agents import NeuralAgent
from ac_cfr.common.config import (
    DeepCFRImplementationId,
    GameConfigurationId,
    ModelConfigId,
    StateEncodingId,
)
from ac_cfr.games.holdem.engine import HoldemConfig, HoldemGame, HoldemState
from ac_cfr.persistence.deep_cfr_checkpoints import load_deep_cfr_checkpoint
from ac_cfr.persistence.deep_cfr_snapshots import load_deep_cfr_snapshot
from ac_cfr.solvers import DeepCFR
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig, start_deep_cfr_training


def test_modified_hulhe_optimised_pipeline_preserves_inference_metadata(
    tmp_path: Path,
) -> None:
    training = DeepCFRTrainingConfig(
        iterations=1,
        traversals_per_player=2,
        advantage_reservoir_capacity=1_000,
        strategy_reservoir_capacity=1_000,
        advantage_training_steps=1,
        strategy_training_steps=1,
        advantage_batch_size=4,
        strategy_batch_size=4,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=1.0,
        dropout_probability=0.0,
        seed=2026,
        game_configuration_id=GameConfigurationId.MODIFIED_HULHE,
        model_config_id=ModelConfigId.MODIFIED_HULHE_DEEP_CFR,
        state_encoding_id=StateEncodingId.HOLD_EM,
    )
    config = DeepCFRRunConfig(
        run_id="modified_hulhe_integration",
        implementation=DeepCFRImplementationId.OPTIMISED,
        checkpoint_interval=1,
        training=training,
        runtime=DeepCFRRuntimeConfig(
            inference_batch_size=2,
            cpu_threads=1,
            device="cpu",
        ),
    )

    outcome = start_deep_cfr_training(config, runs_root=tmp_path)
    game = HoldemConfig.modified()
    loaded_checkpoint = load_deep_cfr_checkpoint(outcome.latest_checkpoint, game)
    assert type(loaded_checkpoint.solver) is DeepCFR
    assert loaded_checkpoint.metadata["game_version"] == "modified_hulhe"
    assert loaded_checkpoint.solver.advantage_reservoirs[0].bytes_per_sample == 421

    snapshot = load_deep_cfr_snapshot(outcome.snapshot_paths[-1], game)
    assert snapshot.metadata.model_config_id == "modified_hulhe_deep_cfr"
    assert snapshot.metadata.state_encoding == "holdem"
    assert snapshot.metadata.architecture_config["input_size"] == 201
    assert snapshot.metadata.architecture_config["hidden_sizes"] == [512] * 7
    assert snapshot.metadata.architecture_config["output_size"] == 3
    state = _first_player_state()
    strategy = NeuralAgent(snapshot).get_strategy(state.information_state(), state.legal_actions())
    assert sum(strategy) == pytest.approx(1.0)
    assert len(strategy) == len(state.legal_actions())


def _first_player_state() -> HoldemState:
    state = HoldemGame().initial_state(HoldemConfig.modified())
    for card in (0, 4, 8, 12, 16, 20, 24):
        state = state.apply_action(card)
    return state
