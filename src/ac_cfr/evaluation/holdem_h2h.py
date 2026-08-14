"""Seeded duplicate-deal head-to-head evaluation for modified HULHE."""

from dataclasses import dataclass
from random import Random

import numpy as np
from numpy.typing import NDArray

from ac_cfr.agents.base import PlayableAgent
from ac_cfr.common.config import GameConfigurationId
from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.base import NodeType
from ac_cfr.games.holdem.cards import DECK
from ac_cfr.games.holdem.engine import HoldemConfig, HoldemGame

PAIRED_BOOTSTRAP_METHOD = "seeded_paired_bootstrap_percentile"
_MAX_BOOTSTRAP_INDICES = 1_000_000


@dataclass(frozen=True, slots=True)
class HoldemDuplicateDifferenceResult:
    """Paired same-deal difference between two focal policies versus one opponent."""

    duplicate_pairs: int
    seed: int
    first_minus_second_mbb_per_game: float
    confidence_level: float
    confidence_interval_method: str
    confidence_interval_low: float
    confidence_interval_high: float
    bootstrap_resamples: int

    @property
    def includes_zero(self) -> bool:
        """Return whether zero difference lies inside the interval."""
        return self.confidence_interval_low <= 0.0 <= self.confidence_interval_high


@dataclass(frozen=True, slots=True)
class HoldemDuplicateResult:
    """One balanced H2H estimate from independent duplicate-deal pairs."""

    duplicate_pairs: int
    hands: int
    seed: int
    mean_chips_per_game: float
    mbb_per_game: float
    confidence_level: float
    confidence_interval_method: str
    confidence_interval_low: float
    confidence_interval_high: float
    bootstrap_resamples: int

    @property
    def includes_zero(self) -> bool:
        """Return whether neutral play lies inside the confidence interval."""
        return self.confidence_interval_low <= 0.0 <= self.confidence_interval_high


def evaluate_holdem_duplicate_match(
    focal_agent: PlayableAgent,
    opponent_agent: PlayableAgent,
    *,
    duplicate_pairs: int,
    seed: int,
    confidence_level: float,
    bootstrap_resamples: int,
    configuration: HoldemConfig | None = None,
) -> HoldemDuplicateResult:
    """Evaluate two frozen agents on seeded deals with seats swapped once per pair."""
    _validate_agent("focal_agent", focal_agent)
    _validate_agent("opponent_agent", opponent_agent)
    resolved_configuration = _validated_protocol(
        duplicate_pairs,
        bootstrap_resamples,
        confidence_level,
        configuration,
    )

    seed_deriver = SeedDeriver(seed)
    pair_scores = _duplicate_pair_scores(
        focal_agent,
        opponent_agent,
        duplicate_pairs,
        seed_deriver,
        resolved_configuration,
    )

    mean_chips = float(np.mean(pair_scores))
    mbb_scale = 1_000.0 / float(resolved_configuration.small_bet)
    bootstrap_rng = np.random.default_rng(seed_deriver.derive(RngStream.BOOTSTRAP))
    interval_low, interval_high = _paired_bootstrap_interval(
        pair_scores * mbb_scale,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        rng=bootstrap_rng,
    )
    return HoldemDuplicateResult(
        duplicate_pairs=duplicate_pairs,
        hands=2 * duplicate_pairs,
        seed=seed,
        mean_chips_per_game=mean_chips,
        mbb_per_game=mean_chips * mbb_scale,
        confidence_level=confidence_level,
        confidence_interval_method=PAIRED_BOOTSTRAP_METHOD,
        confidence_interval_low=interval_low,
        confidence_interval_high=interval_high,
        bootstrap_resamples=bootstrap_resamples,
    )


def evaluate_holdem_duplicate_difference(
    first_focal_agent: PlayableAgent,
    second_focal_agent: PlayableAgent,
    opponent_agent: PlayableAgent,
    *,
    duplicate_pairs: int,
    seed: int,
    confidence_level: float,
    bootstrap_resamples: int,
    configuration: HoldemConfig | None = None,
) -> HoldemDuplicateDifferenceResult:
    """Compare two focal policies versus one opponent on identical duplicate deals."""
    for name, agent in (
        ("first_focal_agent", first_focal_agent),
        ("second_focal_agent", second_focal_agent),
        ("opponent_agent", opponent_agent),
    ):
        _validate_agent(name, agent)
    resolved_configuration = _validated_protocol(
        duplicate_pairs,
        bootstrap_resamples,
        confidence_level,
        configuration,
    )
    seed_deriver = SeedDeriver(seed)
    first_scores = _duplicate_pair_scores(
        first_focal_agent,
        opponent_agent,
        duplicate_pairs,
        seed_deriver,
        resolved_configuration,
    )
    second_scores = _duplicate_pair_scores(
        second_focal_agent,
        opponent_agent,
        duplicate_pairs,
        seed_deriver,
        resolved_configuration,
    )
    mbb_scale = 1_000.0 / float(resolved_configuration.small_bet)
    differences = (first_scores - second_scores) * mbb_scale
    bootstrap_rng = np.random.default_rng(seed_deriver.derive(RngStream.BOOTSTRAP))
    interval_low, interval_high = _paired_bootstrap_interval(
        differences,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        rng=bootstrap_rng,
    )
    return HoldemDuplicateDifferenceResult(
        duplicate_pairs=duplicate_pairs,
        seed=seed,
        first_minus_second_mbb_per_game=float(np.mean(differences)),
        confidence_level=confidence_level,
        confidence_interval_method=PAIRED_BOOTSTRAP_METHOD,
        confidence_interval_low=interval_low,
        confidence_interval_high=interval_high,
        bootstrap_resamples=bootstrap_resamples,
    )


def _duplicate_pair_scores(
    focal_agent: PlayableAgent,
    opponent_agent: PlayableAgent,
    duplicate_pairs: int,
    seed_deriver: SeedDeriver,
    configuration: HoldemConfig,
) -> NDArray[np.float64]:
    """Return one balanced score for every independently seeded duplicate pair."""
    chance_rng = seed_deriver.python_rng(RngStream.CHANCE)
    pair_scores = np.empty(duplicate_pairs, dtype=np.float64)
    for pair_index in range(duplicate_pairs):
        complete_deal = tuple(chance_rng.sample(DECK, 9))
        first_utility = _play_complete_deal(
            complete_deal,
            (focal_agent, opponent_agent),
            _policy_rngs(seed_deriver, pair_index, replay=False),
            configuration,
        )
        replay_player_zero_utility = _play_complete_deal(
            complete_deal,
            (opponent_agent, focal_agent),
            _policy_rngs(seed_deriver, pair_index, replay=True),
            configuration,
        )
        pair_scores[pair_index] = (first_utility - replay_player_zero_utility) / 2.0
    return pair_scores


def _play_complete_deal(
    cards: tuple[int, ...],
    agents: tuple[PlayableAgent, PlayableAgent],
    policy_rngs: tuple[Random, Random],
    configuration: HoldemConfig,
) -> float:
    """Play one preselected physical deal and return Player 0's chip utility."""
    state = HoldemGame().initial_state(configuration)
    card_index = 0
    while state.node_type is not NodeType.TERMINAL:
        if state.node_type is NodeType.CHANCE:
            state = state.apply_action(cards[card_index])
            card_index += 1
            continue
        player = state.current_player
        assert player is not None
        information_state = state.information_state()
        legal_actions = state.legal_actions()
        action = agents[player].sample_action(
            information_state,
            legal_actions,
            policy_rngs[player],
        )
        state = state.apply_action(action)
    return state.utility(0)


def _policy_rngs(
    seed_deriver: SeedDeriver,
    pair_index: int,
    *,
    replay: bool,
) -> tuple[Random, Random]:
    """Return independent policy streams for both seats of one component hand."""
    first_index = pair_index * 4 + (2 if replay else 0)
    return (
        seed_deriver.python_rng(RngStream.POLICY, first_index),
        seed_deriver.python_rng(RngStream.POLICY, first_index + 1),
    )


def _paired_bootstrap_interval(
    pair_scores: NDArray[np.float64],
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return a bounded-memory percentile interval over duplicate-pair means."""
    sample_count = len(pair_scores)
    batch_size = max(1, min(bootstrap_resamples, _MAX_BOOTSTRAP_INDICES // sample_count))
    means = np.empty(bootstrap_resamples, dtype=np.float64)
    for start in range(0, bootstrap_resamples, batch_size):
        stop = min(start + batch_size, bootstrap_resamples)
        indices = rng.integers(0, sample_count, size=(stop - start, sample_count))
        means[start:stop] = np.mean(pair_scores[indices], axis=1)
    tail_probability = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(means, (tail_probability, 1.0 - tail_probability))
    return float(low), float(high)


def _validated_protocol(
    duplicate_pairs: int,
    bootstrap_resamples: int,
    confidence_level: float,
    configuration: HoldemConfig | None,
) -> HoldemConfig:
    """Validate one duplicate-deal protocol and return its game configuration."""
    _validate_positive_integer("duplicate_pairs", duplicate_pairs)
    _validate_positive_integer("bootstrap_resamples", bootstrap_resamples)
    if duplicate_pairs < 2:
        raise ValueError("duplicate_pairs must be at least 2")
    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    if not isinstance(confidence_level, float) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be a float between 0 and 1")
    resolved = configuration or HoldemConfig.modified()
    if not isinstance(resolved, HoldemConfig):
        raise TypeError("configuration must be a HoldemConfig")
    if resolved.configuration_id is not GameConfigurationId.MODIFIED_HULHE:
        raise ValueError("duplicate H2H evaluation requires canonical modified HULHE")
    return resolved


def _validate_agent(name: str, agent: PlayableAgent) -> None:
    if not isinstance(agent, PlayableAgent):
        raise TypeError(f"{name} must be a PlayableAgent")


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
