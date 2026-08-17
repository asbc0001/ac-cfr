"""CLI orchestration for modified-HULHE snapshot progression evaluation."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from ac_cfr.agents import (
    RULE_BASED_AGENT_ID,
    BaselineAgent,
    NeuralAgent,
    PlayableAgent,
    RuleBasedAgent,
)
from ac_cfr.common.config import GameConfigurationId
from ac_cfr.evaluation.holdem_h2h import HoldemDuplicateResult, evaluate_holdem_duplicate_match
from ac_cfr.games.base import GameId, UtilityUnit
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.persistence.deep_cfr_snapshots import (
    DeepCFRSnapshotMetadata,
    load_deep_cfr_snapshot,
)
from ac_cfr.persistence.results import HoldemH2HResultStore


@dataclass(frozen=True, slots=True)
class _SnapshotStrategy:
    """One validated playable snapshot and its provenance."""

    metadata: DeepCFRSnapshotMetadata
    agent: NeuralAgent


@dataclass(frozen=True, slots=True)
class _FixedOpponent:
    """One named non-snapshot comparison policy."""

    opponent_id: str
    agent: PlayableAgent


type _Opponent = _SnapshotStrategy | _FixedOpponent


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate snapshots against random, fixed anchors, and one another."""
    parser = argparse.ArgumentParser(
        description="Evaluate modified-HULHE average-strategy snapshots with duplicate deals."
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="progression snapshot; repeat to run the ordered snapshot round-robin",
    )
    parser.add_argument(
        "--anchor-snapshot",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="fixed snapshot opponent; repeat for multiple anchors",
    )
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--include-rule-based", action="store_true")
    parser.add_argument("--include-self-play", action="store_true")
    parser.add_argument("--duplicate-pairs", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--confidence-level", required=True, type=float)
    parser.add_argument("--bootstrap-resamples", required=True, type=int)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("runs/modified-hulhe-evaluation/h2h.csv"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)

    snapshots = _load_snapshots(arguments.snapshot, device=arguments.device)
    anchors = _load_snapshots(arguments.anchor_snapshot, device=arguments.device)
    _validate_snapshot_selection(snapshots, anchors)
    matches = _build_matches(
        snapshots,
        anchors,
        include_random=arguments.include_random,
        include_rule_based=arguments.include_rule_based,
        include_self_play=arguments.include_self_play,
    )
    if not matches:
        parser.error(
            "select a baseline or anchor for one snapshot, or provide at least two snapshots"
        )

    store = HoldemH2HResultStore(arguments.results)
    for focal, opponent in matches:
        result = evaluate_holdem_duplicate_match(
            focal.agent,
            opponent.agent,
            duplicate_pairs=arguments.duplicate_pairs,
            seed=arguments.seed,
            confidence_level=arguments.confidence_level,
            bootstrap_resamples=arguments.bootstrap_resamples,
        )
        _store_result(store, focal, opponent, result)
        opponent_id = _opponent_id(opponent)
        print(
            f"{focal.metadata.snapshot_id} vs {opponent_id}: "
            f"{result.mbb_per_game:.6g} mbb/g "
            f"[{result.confidence_interval_low:.6g}, "
            f"{result.confidence_interval_high:.6g}]"
        )
    print(f"results: {arguments.results}")
    return 0


def _load_snapshots(paths: list[Path], *, device: str) -> tuple[_SnapshotStrategy, ...]:
    """Load compatible frozen policies and order them by iteration and ID."""
    configuration = HoldemConfig.modified()
    strategies = tuple(
        _SnapshotStrategy(
            metadata=(
                loaded := load_deep_cfr_snapshot(
                    path,
                    configuration,
                    map_location=device,
                )
            ).metadata,
            agent=NeuralAgent(loaded),
        )
        for path in paths
    )
    return tuple(
        sorted(
            strategies,
            key=lambda strategy: (
                strategy.metadata.training_iteration,
                strategy.metadata.snapshot_id,
            ),
        )
    )


def _validate_snapshot_selection(
    snapshots: tuple[_SnapshotStrategy, ...],
    anchors: tuple[_SnapshotStrategy, ...],
) -> None:
    """Reject empty or duplicate snapshot selections."""
    if not snapshots:
        raise ValueError("at least one --snapshot is required")
    identifiers = [strategy.metadata.snapshot_id for strategy in (*snapshots, *anchors)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("snapshot and anchor identifiers must be unique")


def _build_matches(
    snapshots: tuple[_SnapshotStrategy, ...],
    anchors: tuple[_SnapshotStrategy, ...],
    *,
    include_random: bool,
    include_rule_based: bool,
    include_self_play: bool,
) -> tuple[tuple[_SnapshotStrategy, _Opponent], ...]:
    """Build baseline, self-play, fixed-anchor, and snapshot round-robin matches."""
    matches: list[tuple[_SnapshotStrategy, _Opponent]] = []
    if include_random:
        random_opponent = _FixedOpponent("uniform_random", BaselineAgent())
        matches.extend((snapshot, random_opponent) for snapshot in snapshots)
    if include_rule_based:
        rule_opponent = _FixedOpponent(RULE_BASED_AGENT_ID, RuleBasedAgent())
        matches.extend((snapshot, rule_opponent) for snapshot in snapshots)
    if include_self_play:
        matches.extend((snapshot, snapshot) for snapshot in snapshots)
    matches.extend((snapshot, anchor) for snapshot in snapshots for anchor in anchors)
    matches.extend((later, earlier) for earlier, later in combinations(snapshots, 2))
    return tuple(matches)


def _store_result(
    store: HoldemH2HResultStore,
    focal: _SnapshotStrategy,
    opponent: _Opponent,
    result: HoldemDuplicateResult,
) -> None:
    """Upsert one compact result linked to both snapshot metadata records."""
    focal_metadata = focal.metadata
    opponent_metadata = opponent.metadata if isinstance(opponent, _SnapshotStrategy) else None
    store.upsert(
        {
            "game": GameId.HOLD_EM.value,
            "game_version": GameConfigurationId.MODIFIED_HULHE.value,
            "utility_unit": UtilityUnit.CHIP.value,
            "solver": focal_metadata.solver,
            "run_id": focal_metadata.run_id,
            "strategy_snapshot_id": focal_metadata.snapshot_id,
            "source_checkpoint_id": focal_metadata.source_checkpoint_id,
            "iteration": focal_metadata.training_iteration,
            "seed": result.seed,
            "opponent_id": _opponent_id(opponent),
            "opponent_snapshot_id": opponent_metadata.snapshot_id if opponent_metadata else "",
            "opponent_iteration": (
                opponent_metadata.training_iteration if opponent_metadata else 0
            ),
            "hands": result.hands,
            "paired_deals": result.duplicate_pairs,
            "mbb_per_game": result.mbb_per_game,
            "confidence_level": result.confidence_level,
            "confidence_interval_method": result.confidence_interval_method,
            "confidence_interval_low": result.confidence_interval_low,
            "confidence_interval_high": result.confidence_interval_high,
            "bootstrap_resamples": result.bootstrap_resamples,
        }
    )


def _opponent_id(opponent: _Opponent) -> str:
    """Return the stable result identifier for any opponent type."""
    if isinstance(opponent, _SnapshotStrategy):
        return opponent.metadata.snapshot_id
    return opponent.opponent_id
