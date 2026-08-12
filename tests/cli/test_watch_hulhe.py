import json
from pathlib import Path

import pytest

from ac_cfr.cli import watch_hulhe
from ac_cfr.evaluation.holdem_h2h import PAIRED_BOOTSTRAP_METHOD
from ac_cfr.persistence.results import HoldemH2HResultStore
from ac_cfr.training.deep_cfr_config import load_deep_cfr_run_config

_PROJECT_ROOT = Path(__file__).parents[2]
_MONITORING_CONFIG = (
    _PROJECT_ROOT / "configs" / "deep_cfr" / "modified_hulhe_production_monitoring.toml"
)
_PRODUCTION_CONFIG = _PROJECT_ROOT / "configs" / "deep_cfr" / "modified_hulhe_production.toml"


def test_production_monitoring_is_strict_and_matches_training_schedule(tmp_path: Path) -> None:
    monitoring = watch_hulhe.load_hulhe_monitoring_config(_MONITORING_CONFIG)
    training = load_deep_cfr_run_config(_PRODUCTION_CONFIG, run_id=monitoring.run_id)
    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(
        json.dumps({"code_revision": "test", "run_config": training.to_dict()}),
        encoding="utf-8",
    )

    loaded = watch_hulhe._load_and_validate_run_config(run_config_path, monitoring)

    assert loaded == training
    assert monitoring.representative_iterations == (
        1,
        2,
        5,
        10,
        20,
        40,
        60,
        80,
        100,
        120,
        140,
    )

    incompatible = tmp_path / "incompatible.toml"
    incompatible.write_text(
        _MONITORING_CONFIG.read_text(encoding="utf-8").replace(
            "representative_iterations = [1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 140]",
            "representative_iterations = [1, 3, 140]",
        ),
        encoding="utf-8",
    )
    invalid_monitoring = watch_hulhe.load_hulhe_monitoring_config(incompatible)
    with pytest.raises(ValueError, match="unscheduled"):
        watch_hulhe._load_and_validate_run_config(run_config_path, invalid_monitoring)


def test_monitoring_evaluates_available_protocol_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitoring = watch_hulhe.load_hulhe_monitoring_config(_MONITORING_CONFIG)
    training = load_deep_cfr_run_config(_PRODUCTION_CONFIG, run_id=monitoring.run_id)
    run_directory = tmp_path / monitoring.run_id
    snapshot_directory = run_directory / "strategy_snapshots"
    snapshot_directory.mkdir(parents=True)
    for iteration in (1, 2):
        (snapshot_directory / f"{monitoring.run_id}_iter_{iteration}.pt").touch()
    results_path = run_directory / "evaluation" / "production_h2h.csv"
    plot_path = run_directory / "evaluation" / "production_h2h.png"
    calls: list[tuple[str, ...]] = []

    def fake_evaluate(arguments: list[str]) -> int:
        calls.append(tuple(arguments))
        focal_path = Path(arguments[arguments.index("--snapshot") + 1])
        focal_id = focal_path.stem
        iteration = int(focal_id.rsplit("_iter_", maxsplit=1)[1])
        opponents: list[tuple[str, str, int]] = []
        if "--include-random" in arguments:
            opponents.append(("uniform_random", "", 0))
        if "--include-rule-based" in arguments:
            opponents.append(("rule_based_v1", "", 0))
        if "--anchor-snapshot" in arguments:
            anchor_path = Path(arguments[arguments.index("--anchor-snapshot") + 1])
            anchor_id = anchor_path.stem
            anchor_iteration = int(anchor_id.rsplit("_iter_", maxsplit=1)[1])
            opponents.append((anchor_id, anchor_id, anchor_iteration))
        store = HoldemH2HResultStore(results_path)
        for opponent_id, opponent_snapshot_id, opponent_iteration in opponents:
            store.upsert(
                {
                    "game": "holdem",
                    "game_version": "modified_hulhe",
                    "utility_unit": "chip",
                    "solver": "optimised",
                    "run_id": monitoring.run_id,
                    "strategy_snapshot_id": focal_id,
                    "source_checkpoint_id": focal_id,
                    "iteration": iteration,
                    "seed": monitoring.seed,
                    "opponent_id": opponent_id,
                    "opponent_snapshot_id": opponent_snapshot_id,
                    "opponent_iteration": opponent_iteration,
                    "hands": 2 * monitoring.duplicate_pairs,
                    "paired_deals": monitoring.duplicate_pairs,
                    "mbb_per_game": 1.0,
                    "confidence_level": monitoring.confidence_level,
                    "confidence_interval_method": PAIRED_BOOTSTRAP_METHOD,
                    "confidence_interval_low": -1.0,
                    "confidence_interval_high": 3.0,
                    "bootstrap_resamples": monitoring.bootstrap_resamples,
                }
            )
        return 0

    def fake_plot(_results_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

    monkeypatch.setattr(watch_hulhe, "evaluate_hulhe_main", fake_evaluate)
    monkeypatch.setattr(watch_hulhe, "_settle_after_snapshot", lambda *_: None)
    monkeypatch.setattr(watch_hulhe, "plot_modified_hulhe_h2h", fake_plot)

    assert not watch_hulhe._evaluate_available_snapshots(
        monitoring,
        training,
        run_directory,
        results_path=results_path,
        plot_path=plot_path,
    )
    assert len(calls) == 3
    assert len(HoldemH2HResultStore(results_path).records) == 5
    assert plot_path.exists()

    assert not watch_hulhe._evaluate_available_snapshots(
        monitoring,
        training,
        run_directory,
        results_path=results_path,
        plot_path=plot_path,
    )
    assert len(calls) == 3
