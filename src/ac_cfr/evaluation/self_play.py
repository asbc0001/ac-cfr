"""Duplicate-deal self-play checks for frozen Kuhn and Leduc policies."""

from dataclasses import dataclass
from math import sqrt
from random import Random

import numpy as np
from numpy.typing import NDArray

from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.evaluation.best_response import ExactEvaluator, Policy
from ac_cfr.games.base import GameId, NodeType
from ac_cfr.games.leduc import LEDUC_DECK, LEDUC_PRIVATE_DEALS
from ac_cfr.games.tree import IndexedGameTree

_SUPPORTED_GAMES = (GameId.KUHN, GameId.LEDUC)
_NORMAL_99_PERCENT_Z = 2.5758293035489004


@dataclass(frozen=True, slots=True)
class DuplicateSelfPlayResult:
    """Paired self-play mean and 99% normal confidence interval."""

    duplicate_pairs: int
    seed: int
    mean_chips: float
    standard_error_chips: float
    confidence_interval_low: float
    confidence_interval_high: float

    @property
    def includes_zero(self) -> bool:
        """Return whether neutral play lies inside the confidence interval."""
        return self.confidence_interval_low <= 0.0 <= self.confidence_interval_high


def evaluate_duplicate_self_play(
    tree: IndexedGameTree,
    policy: Policy,
    *,
    duplicate_pairs: int,
    seed: int,
) -> DuplicateSelfPlayResult:
    """Replay sampled deals with the focal policy in each seat."""
    if not isinstance(tree, IndexedGameTree):
        raise TypeError("tree must be an IndexedGameTree")
    if tree.game_id not in _SUPPORTED_GAMES:
        raise ValueError("duplicate self-play supports only indexed Kuhn and Leduc trees")
    _validate_non_negative_integer("duplicate_pairs", duplicate_pairs)
    if duplicate_pairs < 2:
        raise ValueError("duplicate_pairs must be at least 2")

    ExactEvaluator(tree).expected_value(policy)
    probabilities = np.asarray(policy, dtype=np.float64)
    seed_deriver = SeedDeriver(seed)
    chance_rng = seed_deriver.python_rng(RngStream.CHANCE)
    first_policy_rng = seed_deriver.python_rng(RngStream.POLICY, 0)
    second_policy_rng = seed_deriver.python_rng(RngStream.POLICY, 1)
    pair_scores = np.empty(duplicate_pairs, dtype=np.float64)

    for pair_index in range(duplicate_pairs):
        private_deal, public_card = _sample_complete_deal(tree, chance_rng)
        first_utility = _play_hand(
            tree,
            probabilities,
            private_deal,
            public_card,
            first_policy_rng,
        )
        second_utility = _play_hand(
            tree,
            probabilities,
            private_deal,
            public_card,
            second_policy_rng,
        )
        # The focal policy occupies Player 0 first and Player 1 in the replay.
        pair_scores[pair_index] = (first_utility - second_utility) / 2.0

    mean = float(np.mean(pair_scores))
    standard_error = float(np.std(pair_scores, ddof=1) / sqrt(duplicate_pairs))
    margin = _NORMAL_99_PERCENT_Z * standard_error
    return DuplicateSelfPlayResult(
        duplicate_pairs=duplicate_pairs,
        seed=seed,
        mean_chips=mean,
        standard_error_chips=standard_error,
        confidence_interval_low=mean - margin,
        confidence_interval_high=mean + margin,
    )


def _sample_complete_deal(tree: IndexedGameTree, rng: Random) -> tuple[int, int | None]:
    """Sample a complete physical Kuhn or Leduc deal for duplicate replay."""
    root_edge = int(tree.child_offsets[0]) + rng.randrange(int(tree.child_counts[0]))
    private_deal = int(tree.edge_labels[root_edge])
    if tree.game_id is GameId.KUHN:
        return private_deal, None

    private_cards = LEDUC_PRIVATE_DEALS[private_deal]
    available_public_cards = tuple(card for card in LEDUC_DECK if card not in private_cards)
    return private_deal, available_public_cards[rng.randrange(len(available_public_cards))]


def _play_hand(
    tree: IndexedGameTree,
    policy: NDArray[np.float64],
    private_deal: int,
    public_card: int | None,
    rng: Random,
) -> float:
    """Play one fixed deal by sampling actions from a complete policy."""
    node_id = 0
    while tree.node_types[node_id] != NodeType.TERMINAL:
        node_type = tree.node_types[node_id]
        edge_start = int(tree.child_offsets[node_id])
        edge_count = int(tree.child_counts[node_id])
        if node_type == NodeType.CHANCE:
            outcome = private_deal if node_id == 0 else public_card
            assert outcome is not None
            node_id = _chance_child(tree, edge_start, edge_count, outcome)
            continue

        information_set_id = int(tree.information_set_ids[node_id])
        action_start = int(tree.information_set_action_offsets[information_set_id])
        action_count = int(tree.information_set_action_counts[information_set_id])
        action_position = _sample_action_position(
            policy[action_start : action_start + action_count], rng
        )
        node_id = int(tree.children[edge_start + action_position])
    return float(tree.terminal_utilities[node_id])


def _chance_child(
    tree: IndexedGameTree,
    edge_start: int,
    edge_count: int,
    outcome: int,
) -> int:
    """Return the child matching a preselected physical chance outcome."""
    for edge_id in range(edge_start, edge_start + edge_count):
        if tree.edge_labels[edge_id] == outcome:
            return int(tree.children[edge_id])
    raise RuntimeError("sampled chance outcome is unavailable at the reached node")


def _sample_action_position(probabilities: NDArray[np.float64], rng: Random) -> int:
    """Sample a stable action position from cumulative probabilities."""
    draw = rng.random()
    cumulative_probability = 0.0
    for action_position, probability in enumerate(probabilities):
        cumulative_probability += float(probability)
        if draw < cumulative_probability:
            return action_position
    return len(probabilities) - 1


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
