"""Command-line entry point for exact registered-strategy evaluation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ac_cfr.cli.evaluate_hulhe import main as holdem_h2h_main
from ac_cfr.evaluation.metrics import evaluate_strategy
from ac_cfr.games.base import UtilityUnit
from ac_cfr.persistence.registry import load_strategy_registry
from ac_cfr.persistence.results import EvaluationResultStore


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch exact evaluation or modified-HULHE duplicate-deal evaluation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["modified-hulhe"]:
        return holdem_h2h_main(arguments[1:])

    parser = argparse.ArgumentParser(description="Evaluate a registered Kuhn or Leduc strategy.")
    parser.add_argument("strategy_id")
    parser.add_argument(
        "--strategy-registry",
        type=Path,
        default=Path("configs/strategy_registry.json"),
    )
    parser.add_argument("--results", type=Path, default=Path("results/evaluations.csv"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parsed = parser.parse_args(arguments)

    registry = load_strategy_registry(
        parsed.strategy_registry,
        project_root=parsed.project_root,
    )
    resolved = registry.resolve(parsed.strategy_id)
    metrics = evaluate_strategy(resolved.tabular_game.tree, resolved.policy)
    entry = resolved.entry
    snapshot_metadata = resolved.snapshot_metadata
    EvaluationResultStore(parsed.results).upsert(
        {
            "game": entry.game,
            "game_version": entry.game_version,
            "utility_unit": UtilityUnit.CHIP.value,
            "solver": entry.algorithm,
            "run_id": snapshot_metadata.run_id if snapshot_metadata else entry.strategy_id,
            "strategy_snapshot_id": entry.snapshot_id or "",
            "source_checkpoint_id": (
                snapshot_metadata.source_checkpoint_id if snapshot_metadata else ""
            ),
            "iteration": entry.training_iteration,
            "seed": snapshot_metadata.seed if snapshot_metadata else 0,
            "expected_value_player_zero": metrics.expected_values[0],
            "exploitability": metrics.exploitability,
            "nash_conv": metrics.nash_conv,
        }
    )
    print(f"player-zero value: {metrics.expected_values[0]:.12g}")
    print(f"NashConv: {metrics.nash_conv:.12g}")
    print(f"exploitability: {metrics.exploitability:.12g}")
    return 0
