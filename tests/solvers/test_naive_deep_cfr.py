from collections.abc import Callable
from random import Random

import numpy as np
import pytest
import torch

from ac_cfr.common.config import ModelConfigId
from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.leduc_neural import LEDUC_NEURAL_STATE_SIZE
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.models import build_deep_cfr_network
from ac_cfr.solvers.naive_deep_cfr import (
    NaiveDeepCFR,
    _train_network_tensors,
    deep_cfr_regret_matching,
    linear_cfr_loss,
)
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import UniformReservoir


class _ObservedNaiveDeepCFR(NaiveDeepCFR):
    def __init__(
        self,
        tree: IndexedGameTree,
        config: DeepCFRTrainingConfig,
        runtime: DeepCFRRuntimeConfig,
    ) -> None:
        super().__init__(tree, config, runtime)
        self.update_observations: list[tuple[int, bool, bool]] = []

    def _run_player_update(self, player: int, iteration: int) -> None:
        self.update_observations.append(
            (player, self.advantage_networks[0] is not None, self.advantage_networks[1] is not None)
        )
        super()._run_player_update(player, iteration)


def test_uniform_reservoir_admits_and_replaces_samples_uniformly() -> None:
    reservoir = UniformReservoir[str](2, Random(0))

    for sample in ("first", "second", "third", "fourth"):
        reservoir.add(sample)

    assert reservoir.samples_seen == 4
    assert reservoir.samples == ("first", "third")


def test_deep_cfr_strategy_and_linear_losses_follow_algorithm_rules() -> None:
    assert deep_cfr_regret_matching((-2.0, -1.0, 99.0), (True, True, False)) == (
        0.0,
        1.0,
        0.0,
    )
    assert deep_cfr_regret_matching((0.0, 0.0, 0.0), (False, True, True)) == (
        0.0,
        1.0,
        0.0,
    )
    assert deep_cfr_regret_matching((1.0, 3.0, 0.0), (True, True, False)) == pytest.approx(
        (0.25, 0.75, 0.0)
    )

    predictions = torch.zeros((2, 3))
    targets = torch.tensor(((1.0, 2.0, 100.0), (1.0, 1.0, 100.0)))
    masks = torch.tensor(((True, True, False), (True, True, False)))
    sample_iterations = torch.tensor((1.0, 4.0))
    advantage_loss = linear_cfr_loss(
        predictions,
        targets,
        masks,
        sample_iterations,
        current_iteration=4,
        strategy_targets=False,
    )
    strategy_loss = linear_cfr_loss(
        torch.tensor(((0.0, 0.0, 100.0),)),
        torch.tensor(((0.5, 0.5, 0.0),)),
        torch.tensor(((True, True, False),)),
        torch.tensor((4.0,)),
        current_iteration=4,
        strategy_targets=True,
    )

    assert float(advantage_loss) == pytest.approx(3.25)
    assert float(strategy_loss) == pytest.approx(0.0)


def test_naive_deep_cfr_updates_in_order_and_exports_frozen_policies() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    config = DeepCFRTrainingConfig(
        iterations=2,
        traversals_per_player=1,
        advantage_reservoir_capacity=100,
        strategy_reservoir_capacity=100,
        advantage_training_steps=1,
        strategy_training_steps=1,
        advantage_batch_size=128,
        strategy_batch_size=128,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=2026,
        snapshot_iterations=(1,),
    )
    solver = _ObservedNaiveDeepCFR(
        tree,
        config,
        DeepCFRRuntimeConfig(inference_batch_size=64, cpu_threads=1, device="cpu"),
    )
    player_zero_information_set = int(np.flatnonzero(tree.information_set_players == 0)[0])

    initial_strategy = solver.strategy_for_information_set(player_zero_information_set, 0)
    assert initial_strategy == (0.0, 1.0, 0.0)

    solver.train(1)
    first_player_zero_network = solver.advantage_networks[0]
    solver.train(1)
    phase_times = solver.recent_training_times

    assert solver.update_observations == [
        (0, False, False),
        (1, True, False),
        (0, True, True),
        (1, True, True),
    ]
    assert solver.advantage_networks[0] is not first_player_zero_network
    assert all(len(reservoir) > 0 for reservoir in solver.advantage_reservoirs)
    assert len(solver.strategy_reservoir) > 0
    assert set(solver.snapshot_networks) == {1}
    assert solver.final_strategy_network is not None
    assert solver.final_strategy_network is not solver.snapshot_networks[1]
    assert all(
        not parameter.requires_grad
        for network in (*solver.snapshot_networks.values(), solver.final_strategy_network)
        for parameter in network.parameters()
    )
    assert all(
        value == 0.0
        for reservoir in solver.advantage_reservoirs
        for sample in reservoir.samples
        for value, is_legal in zip(sample.advantages, sample.action_mask, strict=True)
        if not is_legal
    )
    assert all(
        sum(sample.strategy) == pytest.approx(1.0)
        and all(
            value == 0.0
            for value, legal in zip(sample.strategy, sample.action_mask, strict=True)
            if not legal
        )
        for sample in solver.strategy_reservoir.samples
    )
    assert len(solver.training_metrics) == 6
    assert all(metric.training_samples > 0 for metric in solver.training_metrics)
    assert all(
        np.isfinite(metric.training_loss)
        and (metric.validation_loss is None or np.isfinite(metric.validation_loss))
        for metric in solver.training_metrics
    )
    assert any(metric.validation_samples > 0 for metric in solver.training_metrics)
    assert phase_times.traversal_seconds > 0.0
    assert phase_times.advantage_training_seconds > 0.0
    assert phase_times.strategy_training_seconds > 0.0


def test_linear_cfr_loss_rejects_non_finite_network_output() -> None:
    with pytest.raises(FloatingPointError, match="predictions"):
        linear_cfr_loss(
            torch.tensor(((float("nan"), 0.0, 0.0),)),
            torch.zeros((1, 3)),
            torch.tensor(((True, True, False),)),
            torch.ones(1),
            current_iteration=1,
            strategy_targets=False,
        )


def test_network_training_uses_the_exact_fixed_update_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_steps = 7
    completed_steps = 0
    original_step = torch.optim.Adam.step

    def count_step(
        optimiser: torch.optim.Adam,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        nonlocal completed_steps
        completed_steps += 1
        return original_step(optimiser, closure)

    monkeypatch.setattr(torch.optim.Adam, "step", count_step)
    sample_count = 10
    _train_network_tensors(
        network=build_deep_cfr_network(ModelConfigId.LEDUC_DEEP_CFR),
        states=torch.zeros((sample_count, LEDUC_NEURAL_STATE_SIZE)),
        action_masks=torch.tensor(((True, True, False),) * sample_count),
        targets=torch.zeros((sample_count, 3)),
        sample_iterations=torch.ones(sample_count),
        current_iteration=1,
        training_steps=training_steps,
        batch_size=4,
        learning_rate=1e-3,
        data_seed=10,
        training_seed=11,
        strategy_targets=False,
        validation_fraction=0.2,
        max_gradient_norm=10.0,
    )

    assert completed_steps == training_steps
