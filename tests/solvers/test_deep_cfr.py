from random import Random

import numpy as np
import pytest
from numpy.typing import NDArray

from ac_cfr.games.leduc import LeducConfig, LeducGame
from ac_cfr.games.tree import IndexedGameTree, compile_game_tree
from ac_cfr.solvers.deep_cfr import DeepCFR, _InferenceRequest, _regret_match_batch
from ac_cfr.solvers.naive_deep_cfr import deep_cfr_regret_matching
from ac_cfr.training.config import DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import AdvantageSample, PackedAdvantageReservoir, UniformReservoir


class _ObservedDeepCFR(DeepCFR):
    """Record inference request sizes without changing solver behaviour."""

    def __init__(self, tree: IndexedGameTree, config: DeepCFRTrainingConfig) -> None:
        super().__init__(tree, config)
        self.inference_batch_sizes: list[int] = []

    def _batched_strategies(
        self,
        requests: list[_InferenceRequest],
    ) -> NDArray[np.float64]:
        self.inference_batch_sizes.append(len(requests))
        return super()._batched_strategies(requests)


def test_packed_reservoir_and_batched_regret_matching_preserve_reference_semantics() -> None:
    state = (0.0,) * 37
    mask = (False, True, True)
    samples = tuple(
        AdvantageSample(state, mask, 1, (0.0, float(index), -float(index))) for index in range(1, 5)
    )
    reference = UniformReservoir[AdvantageSample](2, Random(7))
    packed = PackedAdvantageReservoir(2, Random(7))
    for sample in samples:
        reference.add(sample)
        packed.add(sample)

    predictions = np.asarray(((-2.0, -1.0, 99.0), (0.0, 1.0, 3.0)), dtype=np.float32)
    masks = np.asarray(((True, True, False), (False, True, True)), dtype=np.bool)
    strategies = _regret_match_batch(predictions, masks)

    assert packed.samples == reference.samples
    assert packed.arrays[0].dtype == np.float32
    assert packed.arrays[1].dtype == np.bool
    assert packed.arrays[2].dtype == np.uint32
    assert packed.arrays[3].dtype == np.float32
    assert strategies[0] == pytest.approx(
        deep_cfr_regret_matching((-2.0, -1.0, 99.0), (True, True, False))
    )
    assert strategies[1] == pytest.approx(
        deep_cfr_regret_matching((0.0, 1.0, 3.0), (False, True, True))
    )


def test_optimised_deep_cfr_uses_packed_memory_and_exports_a_legal_policy() -> None:
    tree = compile_game_tree(LeducGame(), LeducConfig())
    config = DeepCFRTrainingConfig(
        iterations=1,
        traversals_per_player=4,
        advantage_reservoir_capacity=200,
        strategy_reservoir_capacity=400,
        advantage_training_steps=1,
        strategy_training_steps=1,
        batch_size=64,
        learning_rate=1e-3,
        validation_fraction=0.1,
        max_gradient_norm=10.0,
        dropout_probability=0.0,
        seed=2026,
    )
    solver = _ObservedDeepCFR(tree, config)

    solver.train(1)

    assert solver.iteration == 1
    assert max(solver.inference_batch_sizes) == config.traversals_per_player
    assert all(len(reservoir) > 0 for reservoir in solver.advantage_reservoirs)
    assert len(solver.strategy_reservoir) > 0
    assert all(reservoir.resident_bytes > 0 for reservoir in solver.advantage_reservoirs)
    assert solver.strategy_reservoir.resident_bytes > 0
    assert solver.final_strategy_network is not None
    assert all(
        sum(sample.strategy) == pytest.approx(1.0)
        and all(
            probability == 0.0
            for probability, legal in zip(sample.strategy, sample.action_mask, strict=True)
            if not legal
        )
        for sample in solver.strategy_reservoir.samples
    )
    assert len(solver.training_metrics) == 3
