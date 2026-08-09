"""Canonical two-player Kuhn poker engine and compact card encoding."""

from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from typing import ClassVar

from ac_cfr.common.config import GameConfigurationId, StateEncodingId
from ac_cfr.games.base import (
    Action,
    ChanceOutcome,
    ExtensiveFormGame,
    ExtensiveFormState,
    GameConfiguration,
    GameId,
    InformationState,
    NodeType,
    PlayerId,
    UtilityUnit,
    require_legal_action,
    validate_player,
)


class KuhnCard(IntEnum):
    """Kuhn cards ordered from weakest to strongest."""

    JACK = 0
    QUEEN = 1
    KING = 2


KUHN_DECK = tuple(KuhnCard)
KUHN_DEALS = tuple(
    (first, second) for first in KUHN_DECK for second in KUHN_DECK if first != second
)

_DEAL_OUTCOMES = tuple(
    ChanceOutcome(outcome=deal_id, probability=1.0 / len(KUHN_DEALS))
    for deal_id in range(len(KUHN_DEALS))
)
_OPEN_ACTIONS = (Action.CHECK_CALL, Action.BET_RAISE)
_FACING_BET_ACTIONS = (Action.FOLD, Action.CHECK_CALL)
_TERMINAL_HISTORIES = {
    (Action.CHECK_CALL, Action.CHECK_CALL),
    (Action.BET_RAISE, Action.FOLD),
    (Action.BET_RAISE, Action.CHECK_CALL),
    (Action.CHECK_CALL, Action.BET_RAISE, Action.FOLD),
    (Action.CHECK_CALL, Action.BET_RAISE, Action.CHECK_CALL),
}


@dataclass(frozen=True, slots=True)
class KuhnConfig:
    """Fixed canonical Kuhn rules."""

    game_id: GameId = field(default=GameId.KUHN, init=False)
    configuration_id: GameConfigurationId = field(
        default=GameConfigurationId.KUHN,
        init=False,
    )
    state_encoding_id: StateEncodingId = field(default=StateEncodingId.KUHN, init=False)
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


@dataclass(frozen=True, slots=True)
class KuhnState(ExtensiveFormState):
    """Immutable state for one canonical Kuhn hand."""

    configuration: KuhnConfig
    private_cards: tuple[KuhnCard, KuhnCard] | None = None
    action_history: tuple[Action, ...] = ()

    _OPEN_HISTORIES: ClassVar[set[tuple[Action, ...]]] = {
        (),
        (Action.CHECK_CALL,),
    }

    @property
    def node_type(self) -> NodeType:
        """Return the kind of transition available from this state."""
        if self.private_cards is None:
            return NodeType.CHANCE
        if self.action_history in _TERMINAL_HISTORIES:
            return NodeType.TERMINAL
        return NodeType.PLAYER

    @property
    def current_player(self) -> PlayerId | None:
        """Return the acting player at a decision node."""
        if self.node_type is not NodeType.PLAYER:
            return None
        if self.action_history == (Action.CHECK_CALL, Action.BET_RAISE):
            return 0
        return len(self.action_history) % 2

    def legal_actions(self) -> tuple[Action, ...]:
        """Return contextually legal actions in canonical storage order."""
        if self.node_type is not NodeType.PLAYER:
            return ()
        if self.action_history in self._OPEN_HISTORIES:
            return _OPEN_ACTIONS
        return _FACING_BET_ACTIONS

    def chance_outcomes(self) -> tuple[ChanceOutcome, ...]:
        """Return all six ordered private-card deals at the root."""
        return _DEAL_OUTCOMES if self.node_type is NodeType.CHANCE else ()

    def apply_action(self, action: int) -> "KuhnState":
        """Return the state after a strict chance or player transition."""
        if self.node_type is NodeType.TERMINAL:
            raise ValueError("cannot act in a terminal state")
        if self.node_type is NodeType.CHANCE:
            deal_id = _require_outcome(action, len(KUHN_DEALS))
            return KuhnState(self.configuration, KUHN_DEALS[deal_id])

        parsed_action = require_legal_action(action, self.legal_actions())
        return KuhnState(
            self.configuration,
            self.private_cards,
            (*self.action_history, parsed_action),
        )

    def utility(self, player: PlayerId) -> float:
        """Return terminal net-chip utility for the requested player."""
        validate_player(player)
        if self.node_type is not NodeType.TERMINAL or self.private_cards is None:
            raise ValueError("utility is available only at terminal states")

        player_zero_utility = self._player_zero_utility()
        return player_zero_utility if player == 0 else -player_zero_utility

    def information_state(self) -> InformationState:
        """Return the acting player's private card and observed action history."""
        player = self.current_player
        if player is None or self.private_cards is None:
            raise ValueError("information state is available only at player nodes")
        return InformationState(
            game_id=GameId.KUHN,
            player=player,
            encoding=(
                int(self.private_cards[player]),
                *(int(action) for action in self.action_history),
            ),
            legal_actions=self.legal_actions(),
        )

    def _player_zero_utility(self) -> float:
        history = self.action_history
        if history == (Action.BET_RAISE, Action.FOLD):
            return 1.0
        if history == (Action.CHECK_CALL, Action.BET_RAISE, Action.FOLD):
            return -1.0

        assert self.private_cards is not None
        showdown_stake = 1.0 if history == (Action.CHECK_CALL, Action.CHECK_CALL) else 2.0
        return showdown_stake if self.private_cards[0] > self.private_cards[1] else -showdown_stake


class KuhnGame(ExtensiveFormGame):
    """Factory for canonical Kuhn states."""

    @property
    def game_id(self) -> GameId:
        """Return the canonical Kuhn identifier."""
        return GameId.KUHN

    def initial_state(self, configuration: GameConfiguration) -> KuhnState:
        """Create a chance root from the canonical Kuhn configuration."""
        if not isinstance(configuration, KuhnConfig):
            raise TypeError("configuration must be a KuhnConfig")
        return KuhnState(configuration)


def _require_outcome(outcome: int, outcome_count: int) -> int:
    if isinstance(outcome, bool) or not isinstance(outcome, int):
        raise TypeError("chance outcome must be an integer")
    if not 0 <= outcome < outcome_count:
        raise ValueError(f"unknown chance outcome: {outcome}")
    return outcome
