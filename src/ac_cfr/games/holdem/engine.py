"""Configurable heads-up fixed-limit Hold'em engine."""

from dataclasses import dataclass, field, replace
from enum import IntEnum
from fractions import Fraction
from math import isfinite
from numbers import Real

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
from ac_cfr.games.holdem.cards import CARD_COUNT, DECK, validate_card
from ac_cfr.games.holdem.evaluator import evaluate_holdem
from ac_cfr.games.holdem.information_state import build_holdem_information_state


class Street(IntEnum):
    """Hold'em streets in chronological order."""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


_OPEN_ACTIONS = (Action.CHECK_CALL, Action.BET_RAISE)
_CHECK_ONLY = (Action.CHECK_CALL,)
_FACING_WAGER_ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE)
_CAPPED_ACTIONS = (Action.FOLD, Action.CHECK_CALL)
_EMPTY_HOLE_CARDS: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
_CHANCE_OUTCOMES = {
    remaining_count: tuple(ChanceOutcome(card, 1.0 / remaining_count) for card in DECK)
    for remaining_count in range(CARD_COUNT - 9, CARD_COUNT + 1)
}

ChipAmount = Fraction | int | float


@dataclass(frozen=True, slots=True)
class HoldemConfig:
    """Exact rules and starting-state choice for one Hold'em hand."""

    start_street: Street = Street.PREFLOP
    max_bets_per_round: int = 4
    small_blind: ChipAmount = Fraction(1, 2)
    big_blind: ChipAmount = Fraction(1)
    small_bet: ChipAmount = Fraction(1)
    big_bet: ChipAmount = Fraction(2)
    button_player: PlayerId = 0
    synthetic_flop_start: bool = False
    game_id: GameId = field(default=GameId.HOLD_EM, init=False)
    state_encoding_id: StateEncodingId = field(default=StateEncodingId.HOLD_EM, init=False)
    utility_unit: UtilityUnit = field(default=UtilityUnit.CHIP, init=False)
    player_count: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.start_street, Street):
            raise TypeError("start_street must be a Street")
        if self.start_street not in (Street.PREFLOP, Street.FLOP):
            raise ValueError("a Hold'em hand must start preflop or on the flop")
        if isinstance(self.max_bets_per_round, bool) or not isinstance(
            self.max_bets_per_round, int
        ):
            raise TypeError("max_bets_per_round must be an integer")
        if self.max_bets_per_round < 1:
            raise ValueError("max_bets_per_round must be positive")
        validate_player(self.button_player)
        if not isinstance(self.synthetic_flop_start, bool):
            raise TypeError("synthetic_flop_start must be a bool")
        if self.synthetic_flop_start != (self.start_street is Street.FLOP):
            raise ValueError("flop starts must explicitly use the synthetic starting state")
        small_blind = _normalise_chip_amount("small_blind", self.small_blind)
        big_blind = _normalise_chip_amount("big_blind", self.big_blind)
        small_bet = _normalise_chip_amount("small_bet", self.small_bet)
        big_bet = _normalise_chip_amount("big_bet", self.big_bet)
        object.__setattr__(self, "small_blind", small_blind)
        object.__setattr__(self, "big_blind", big_blind)
        object.__setattr__(self, "small_bet", small_bet)
        object.__setattr__(self, "big_bet", big_bet)
        if (
            small_blind <= 0
            or small_blind * 2 != big_blind
            or big_blind != small_bet
            or big_bet != small_bet * 2
        ):
            raise ValueError("bet sizes must follow the canonical fixed-limit ratios")

    @classmethod
    def full(cls, *, max_bets_per_round: int = 4) -> "HoldemConfig":
        """Return the conventional preflop-start configuration."""
        return cls(max_bets_per_round=max_bets_per_round)

    @classmethod
    def modified(cls) -> "HoldemConfig":
        """Return the canonical flop-first, two-bet-cap configuration."""
        return cls(
            start_street=Street.FLOP,
            max_bets_per_round=2,
            synthetic_flop_start=True,
        )

    @property
    def base_unit(self) -> Fraction:
        """Return the public chip value represented by one internal unit."""
        assert isinstance(self.small_bet, Fraction)
        return self.small_bet / 2

    @property
    def configuration_id(self) -> GameConfigurationId | None:
        """Return the identifier only for an exact canonical configuration."""
        canonical_amounts = (
            self.small_blind == Fraction(1, 2)
            and self.big_blind == 1
            and self.small_bet == 1
            and self.big_bet == 2
        )
        if not canonical_amounts or self.button_player != 0:
            return None
        if self.start_street is Street.PREFLOP and self.max_bets_per_round == 4:
            return GameConfigurationId.HULHE
        if self.start_street is Street.FLOP and self.max_bets_per_round == 2:
            return GameConfigurationId.MODIFIED_HULHE
        return None

    @property
    def small_blind_units(self) -> int:
        """Return the small blind in exact internal units."""
        return 1

    @property
    def big_blind_units(self) -> int:
        """Return the big blind in exact internal units."""
        return 2

    @property
    def small_bet_units(self) -> int:
        """Return the small bet in exact internal units."""
        return 2

    @property
    def big_bet_units(self) -> int:
        """Return the big bet in exact internal units."""
        return 4


@dataclass(frozen=True, slots=True)
class HoldemState(ExtensiveFormState):
    """Compact immutable state for one on-demand Hold'em hand."""

    configuration: HoldemConfig
    hole_cards: tuple[tuple[int, ...], tuple[int, ...]] = _EMPTY_HOLE_CARDS
    board_cards: tuple[int, ...] = ()
    dealt_card_mask: int = 0
    street: Street = Street.PREFLOP
    round_histories: tuple[tuple[Action, ...], ...] = ((),)
    contributions: tuple[int, int] = (0, 0)
    round_commitments: tuple[int, int] = (0, 0)
    betting_level: int = 0
    live_big_blind: bool = False
    acting_player: PlayerId | None = None
    folded_player: PlayerId | None = None
    showdown: bool = False

    @property
    def node_type(self) -> NodeType:
        """Return the kind of transition available from this state."""
        if self.folded_player is not None or self.showdown:
            return NodeType.TERMINAL
        if self._needs_card:
            return NodeType.CHANCE
        return NodeType.PLAYER

    @property
    def current_player(self) -> PlayerId | None:
        """Return the acting player at a decision node."""
        return self.acting_player if self.node_type is NodeType.PLAYER else None

    @property
    def pot(self) -> int:
        """Return the pot in exact internal base units."""
        return self.contributions[0] + self.contributions[1]

    @property
    def amount_to_call(self) -> int:
        """Return the acting player's exact outstanding commitment."""
        if self.current_player is None:
            return 0
        return max(self.round_commitments) - self.round_commitments[self.current_player]

    def legal_actions(self) -> tuple[Action, ...]:
        """Return strict legal actions in canonical storage order."""
        if self.current_player is None:
            return ()
        if self.amount_to_call == 0:
            if self.betting_level < self.configuration.max_bets_per_round:
                return _OPEN_ACTIONS
            return _CHECK_ONLY
        if self.betting_level < self.configuration.max_bets_per_round:
            return _FACING_WAGER_ACTIONS
        return _CAPPED_ACTIONS

    def chance_outcomes(self) -> tuple[ChanceOutcome, ...]:
        """Return every remaining physical card with its uniform probability."""
        if self.node_type is not NodeType.CHANCE:
            return ()
        remaining_count = CARD_COUNT - self.dealt_card_mask.bit_count()
        return tuple(
            outcome
            for outcome in _CHANCE_OUTCOMES[remaining_count]
            if not self.dealt_card_mask & (1 << outcome.outcome)
        )

    def apply_action(self, action: int) -> "HoldemState":
        """Return the state after one strict chance or player transition."""
        if self.node_type is NodeType.TERMINAL:
            raise ValueError("cannot act in a terminal state")
        if self.node_type is NodeType.CHANCE:
            return self._apply_chance(action)
        parsed_action = require_legal_action(action, self.legal_actions())
        return self._apply_player_action(parsed_action)

    def utility(self, player: PlayerId) -> float:
        """Return terminal net-chip utility for the requested player."""
        validate_player(player)
        if self.node_type is not NodeType.TERMINAL:
            raise ValueError("utility is available only at terminal states")

        if self.folded_player is not None:
            player_zero_units = (
                self.contributions[1] if self.folded_player == 1 else -self.contributions[0]
            )
            player_zero_utility = Fraction(player_zero_units) * self.configuration.base_unit
        else:
            if len(self.board_cards) != 5 or any(len(cards) != 2 for cards in self.hole_cards):
                raise ValueError("showdown requires two complete seven-card hands")
            first_rank = evaluate_holdem(self.hole_cards[0], self.board_cards)
            second_rank = evaluate_holdem(self.hole_cards[1], self.board_cards)
            if first_rank < second_rank:
                player_zero_units = Fraction(self.contributions[1])
            elif first_rank > second_rank:
                player_zero_units = Fraction(-self.contributions[0])
            else:
                player_zero_units = Fraction(self.pot, 2) - self.contributions[0]
            player_zero_utility = player_zero_units * self.configuration.base_unit

        utility = float(player_zero_utility)
        return utility if player == 0 else -utility

    def information_state(self) -> InformationState:
        """Return only the acting player's observable, canonical information."""
        player = self.current_player
        if player is None or len(self.hole_cards[player]) != 2:
            raise ValueError("information state is available only at player nodes")
        hole_cards = self.hole_cards[player]
        assert len(hole_cards) == 2
        return build_holdem_information_state(
            player=player,
            hole_cards=(hole_cards[0], hole_cards[1]),
            board_cards=self.board_cards,
            start_street=int(self.configuration.start_street),
            street=int(self.street),
            button_player=self.configuration.button_player,
            max_bets_per_round=self.configuration.max_bets_per_round,
            contributions=self.contributions,
            round_commitments=self.round_commitments,
            betting_level=self.betting_level,
            live_big_blind=self.live_big_blind,
            round_histories=self.round_histories,
            legal_actions=self.legal_actions(),
        )

    @property
    def _needs_card(self) -> bool:
        """Return whether the next transition must deal a private or board card."""
        if len(self.hole_cards[0]) + len(self.hole_cards[1]) < 4:
            return True
        required_board_cards = (0, 3, 4, 5)[int(self.street)]
        return len(self.board_cards) < required_board_cards

    def _apply_chance(self, card: int) -> "HoldemState":
        """Deal one unused physical card and activate play when dealing finishes."""
        validate_card(card)
        if self.dealt_card_mask & (1 << card):
            raise ValueError(f"card has already been dealt: {card}")

        holes = self.hole_cards
        board = self.board_cards
        if len(holes[0]) + len(holes[1]) < 4:
            deal_index = len(holes[0]) + len(holes[1])
            player = (
                self.configuration.button_player
                if deal_index % 2 == 0
                else 1 - self.configuration.button_player
            )
            player_cards = (*holes[player], card)
            if len(player_cards) == 2:
                player_cards = tuple(sorted(player_cards))
            holes = _replace_player_cards(holes, player, player_cards)
        else:
            board = (*board, card)
            if len(board) == 3:
                board = tuple(sorted(board))

        state = replace(
            self,
            hole_cards=holes,
            board_cards=board,
            dealt_card_mask=self.dealt_card_mask | (1 << card),
        )
        if state._needs_card:
            return state
        return replace(state, acting_player=state._starting_player_for_street)

    def _apply_player_action(self, action: Action) -> "HoldemState":
        """Apply one legal betting action, including round and street closure."""
        assert self.acting_player is not None
        player = self.acting_player
        opponent = 1 - player
        history = (*self.round_histories[-1], action)
        histories = (*self.round_histories[:-1], history)

        if action is Action.FOLD:
            return replace(
                self,
                round_histories=histories,
                acting_player=None,
                folded_player=player,
            )

        amount_to_call = self.amount_to_call
        if action is Action.BET_RAISE:
            payment = amount_to_call + self._bet_size
            return replace(
                self,
                round_histories=histories,
                contributions=_add_to_player(self.contributions, player, payment),
                round_commitments=_add_to_player(self.round_commitments, player, payment),
                betting_level=self.betting_level + 1,
                live_big_blind=False,
                acting_player=opponent,
            )

        contributions = _add_to_player(self.contributions, player, amount_to_call)
        commitments = _add_to_player(self.round_commitments, player, amount_to_call)
        closes_round = self._passive_action_closes_round(amount_to_call)
        if not closes_round:
            return replace(
                self,
                round_histories=histories,
                contributions=contributions,
                round_commitments=commitments,
                live_big_blind=self.live_big_blind and amount_to_call > 0,
                acting_player=opponent,
            )
        if self.street is Street.RIVER:
            return replace(
                self,
                round_histories=histories,
                contributions=contributions,
                round_commitments=commitments,
                live_big_blind=False,
                acting_player=None,
                showdown=True,
            )

        return replace(
            self,
            street=Street(int(self.street) + 1),
            round_histories=(*histories, ()),
            contributions=contributions,
            round_commitments=(0, 0),
            betting_level=0,
            live_big_blind=False,
            acting_player=None,
        )

    def _passive_action_closes_round(self, amount_to_call: int) -> bool:
        """Return whether a check or call completes the current betting round."""
        if amount_to_call > 0:
            return not self.live_big_blind
        if self.live_big_blind:
            return True
        history = self.round_histories[-1]
        return bool(history and history[-1] is Action.CHECK_CALL)

    @property
    def _starting_player_for_street(self) -> PlayerId:
        """Return the first player under heads-up positional rules."""
        if self.street is Street.PREFLOP:
            return self.configuration.button_player
        return 1 - self.configuration.button_player

    @property
    def _bet_size(self) -> int:
        """Return the fixed-limit wager size for the current street."""
        if self.street in (Street.PREFLOP, Street.FLOP):
            return self.configuration.small_bet_units
        return self.configuration.big_bet_units


class HoldemGame(ExtensiveFormGame):
    """Factory for configurable heads-up fixed-limit Hold'em states."""

    @property
    def game_id(self) -> GameId:
        """Return the shared Hold'em engine identifier."""
        return GameId.HOLD_EM

    def initial_state(self, configuration: GameConfiguration) -> HoldemState:
        """Create the physical-card chance root for an explicit configuration."""
        if not isinstance(configuration, HoldemConfig):
            raise TypeError("configuration must be a HoldemConfig")

        if configuration.synthetic_flop_start:
            starting_contribution = configuration.small_bet_units
            contributions = (starting_contribution, starting_contribution)
            commitments = (0, 0)
            betting_level = 0
            live_big_blind = False
        else:
            button = configuration.button_player
            big_blind = 1 - button
            contributions = _values_by_player(
                button,
                configuration.small_blind_units,
                big_blind,
                configuration.big_blind_units,
            )
            commitments = contributions
            betting_level = 1
            live_big_blind = True

        return HoldemState(
            configuration=configuration,
            street=configuration.start_street,
            contributions=contributions,
            round_commitments=commitments,
            betting_level=betting_level,
            live_big_blind=live_big_blind,
        )


def _replace_player_cards(
    hole_cards: tuple[tuple[int, ...], tuple[int, ...]],
    player: PlayerId,
    cards: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return hole-card tuples with one player's cards replaced."""
    if player == 0:
        return cards, hole_cards[1]
    return hole_cards[0], cards


def _add_to_player(values: tuple[int, int], player: PlayerId, amount: int) -> tuple[int, int]:
    if player == 0:
        return values[0] + amount, values[1]
    return values[0], values[1] + amount


def _values_by_player(
    first_player: PlayerId,
    first_value: int,
    second_player: PlayerId,
    second_value: int,
) -> tuple[int, int]:
    """Place two values into canonical Player-0/Player-1 order."""
    values = [0, 0]
    values[first_player] = first_value
    values[second_player] = second_value
    return values[0], values[1]


def _normalise_chip_amount(name: str, value: ChipAmount) -> Fraction:
    """Convert a finite numeric chip amount to an exact fraction."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return Fraction(str(value)) if isinstance(value, float) else Fraction(value)
