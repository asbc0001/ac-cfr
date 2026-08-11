import csv
from math import isfinite
from pathlib import Path
from random import Random

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
from ac_cfr.training.deep_cfr_runner import (
    DeepCFRRunConfig,
    resume_deep_cfr_training,
    start_deep_cfr_training,
)

_LOSS_FIELDS = (
    "player_zero_advantage_training_loss",
    "player_zero_advantage_validation_loss",
    "player_one_advantage_training_loss",
    "player_one_advantage_validation_loss",
    "strategy_training_loss",
    "strategy_validation_loss",
)


def test_modified_hulhe_optimised_pipeline_resumes_and_loads_for_play(
    tmp_path: Path,
) -> None:
    training = DeepCFRTrainingConfig(
        iterations=2,
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
        snapshot_iterations=(1,),
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
    first_checkpoint = outcome.run_directory / "checkpoints" / "iter_1.pt"
    loaded_checkpoint = load_deep_cfr_checkpoint(first_checkpoint, game)
    assert type(loaded_checkpoint.solver) is DeepCFR
    assert loaded_checkpoint.solver.iteration == 1
    assert loaded_checkpoint.metadata["game_version"] == "modified_hulhe"
    assert loaded_checkpoint.solver.advantage_reservoirs[0].bytes_per_sample == 421

    resumed = resume_deep_cfr_training(first_checkpoint)
    assert resumed.final_iteration == 2
    with (outcome.run_directory / "metrics.csv").open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert [record["iteration"] for record in records] == ["1", "2"]
    for record in records:
        for field in _LOSS_FIELDS:
            assert isfinite(float(record[field]))

    snapshot_path = (
        outcome.run_directory / "strategy_snapshots" / "modified_hulhe_integration_iter_2.pt"
    )
    snapshot = load_deep_cfr_snapshot(snapshot_path, game)
    assert snapshot.metadata.model_config_id == "modified_hulhe_deep_cfr"
    assert snapshot.metadata.state_encoding == "holdem"
    assert snapshot.metadata.architecture_config["input_size"] == 201
    assert snapshot.metadata.architecture_config["hidden_sizes"] == [512] * 7
    assert snapshot.metadata.architecture_config["output_size"] == 3
    state = _first_player_state()
    agent = NeuralAgent(snapshot)
    strategy = agent.get_strategy(state.information_state(), state.legal_actions())
    assert all(isfinite(probability) for probability in strategy)
    assert sum(strategy) == pytest.approx(1.0)
    assert len(strategy) == len(state.legal_actions())
    assert state.legal_actions() == state.information_state().legal_actions
    _play_one_hand(agent)


def _first_player_state() -> HoldemState:
    state = HoldemGame().initial_state(HoldemConfig.modified())
    for card in (0, 4, 8, 12, 16, 20, 24):
        state = state.apply_action(card)
    return state


def _play_one_hand(agent: NeuralAgent) -> None:
    """Play one deterministic-deal self-play hand through the frozen policy."""
    state = HoldemGame().initial_state(HoldemConfig.modified())
    rng = Random(2026)
    while not state.is_terminal:
        if state.is_chance_node:
            state = state.apply_action(state.chance_outcomes()[0].outcome)
            continue
        information_state = state.information_state()
        action = agent.sample_action(information_state, state.legal_actions(), rng)
        state = state.apply_action(action)
    assert isfinite(state.utility(0))
    assert state.utility(0) == pytest.approx(-state.utility(1))
