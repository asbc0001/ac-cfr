from pathlib import Path

import pytest

from ac_cfr.agents import BaselineAgent
from ac_cfr.evaluation import (
    evaluate_holdem_duplicate_difference,
    evaluate_holdem_duplicate_match,
)
from ac_cfr.evaluation.holdem_h2h import PAIRED_BOOTSTRAP_METHOD
from ac_cfr.persistence.results import HoldemH2HResultStore


def test_modified_hulhe_duplicate_match_is_seeded_balanced_and_finite() -> None:
    first = evaluate_holdem_duplicate_match(
        BaselineAgent(),
        BaselineAgent(),
        duplicate_pairs=100,
        seed=20260811,
        confidence_level=0.95,
        bootstrap_resamples=500,
    )
    second = evaluate_holdem_duplicate_match(
        BaselineAgent(),
        BaselineAgent(),
        duplicate_pairs=100,
        seed=20260811,
        confidence_level=0.95,
        bootstrap_resamples=500,
    )

    assert first == second
    assert first.hands == 200
    assert first.confidence_interval_method == PAIRED_BOOTSTRAP_METHOD
    assert first.confidence_interval_low <= first.mbb_per_game <= first.confidence_interval_high
    assert first.includes_zero


def test_modified_hulhe_results_upsert_by_complete_protocol(tmp_path: Path) -> None:
    results_path = tmp_path / "h2h.csv"
    store = HoldemH2HResultStore(results_path)
    record: dict[str, object] = {
        "game": "holdem",
        "game_version": "modified_hulhe",
        "utility_unit": "chip",
        "solver": "optimised",
        "run_id": "shakedown",
        "strategy_snapshot_id": "later",
        "source_checkpoint_id": "checkpoint-2",
        "iteration": 2,
        "seed": 7,
        "opponent_id": "earlier",
        "opponent_snapshot_id": "earlier",
        "opponent_iteration": 1,
        "hands": 200,
        "paired_deals": 100,
        "mbb_per_game": 12.0,
        "confidence_level": 0.95,
        "confidence_interval_method": PAIRED_BOOTSTRAP_METHOD,
        "confidence_interval_low": -8.0,
        "confidence_interval_high": 32.0,
        "bootstrap_resamples": 500,
    }

    store.upsert(record)
    store.upsert({**record, "mbb_per_game": 10.0})

    reloaded = HoldemH2HResultStore(results_path)
    assert len(reloaded.records) == 1
    assert reloaded.records[0]["mbb_per_game"] == "10.0"

    with pytest.raises(ValueError, match="hand or iteration"):
        store.upsert({**record, "hands": 199})


def test_duplicate_difference_is_exactly_neutral_for_identical_focal_policies() -> None:
    difference = evaluate_holdem_duplicate_difference(
        BaselineAgent(),
        BaselineAgent(),
        BaselineAgent(),
        duplicate_pairs=20,
        seed=20260811,
        confidence_level=0.95,
        bootstrap_resamples=100,
    )

    assert difference.first_minus_second_mbb_per_game == 0.0
    assert difference.confidence_interval_low == 0.0
    assert difference.confidence_interval_high == 0.0
    assert difference.includes_zero
