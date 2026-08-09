"""Canonical two-player Kuhn poker rules and compact card encoding."""

from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction

from ac_cfr.games.base import GameId, UtilityUnit


class KuhnCard(IntEnum):
    """Kuhn cards ordered from weakest to strongest."""

    JACK = 0
    QUEEN = 1
    KING = 2


KUHN_DECK = tuple(KuhnCard)


@dataclass(frozen=True, slots=True)
class KuhnConfig:
    """Fixed canonical Kuhn rules."""

    game_id: GameId = field(default=GameId.KUHN, init=False)
    utility_unit: UtilityUnit = field(default=UtilityUnit.CHIP, init=False)
    player_count: int = field(default=2, init=False)
    ante: int = field(default=1, init=False)
    private_cards_per_player: int = field(default=1, init=False)
    betting_rounds: int = field(default=1, init=False)
    bet_size: int = field(default=1, init=False)
    max_bets_per_round: int = field(default=1, init=False)
    starting_player: int = field(default=0, init=False)
    player_zero_equilibrium_value: Fraction = field(
        default=Fraction(-1, 18),
        init=False,
    )
