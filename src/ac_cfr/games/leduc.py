"""Canonical two-player Leduc poker engine and physical-card encoding."""

from dataclasses import dataclass, field, replace
from enum import IntEnum

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


class LeducRank(IntEnum):
    """Leduc ranks ordered from weakest to strongest."""

    JACK = 0
    QUEEN = 1
    KING = 2


class LeducSuit(IntEnum):
    """Physical suits retained in the canonical Leduc game."""

    FIRST = 0
    SECOND = 1


LEDUC_SUIT_COUNT = len(LeducSuit)


def encode_card(rank: LeducRank, suit: LeducSuit) -> int:
    """Encode one physical Leduc card as a compact rank-major integer."""
    if not isinstance(rank, LeducRank):
        raise TypeError("rank must be a LeducRank")
    if not isinstance(suit, LeducSuit):
        raise TypeError("suit must be a LeducSuit")
    return int(rank) * LEDUC_SUIT_COUNT + int(suit)


def card_rank(card: int) -> LeducRank:
    """Return the rank stored in a valid compact Leduc card."""
    _validate_card(card)
    return LeducRank(card // LEDUC_SUIT_COUNT)


def card_suit(card: int) -> LeducSuit:
    """Return the physical suit stored in a valid compact Leduc card."""
    _validate_card(card)
    return LeducSuit(card % LEDUC_SUIT_COUNT)


LEDUC_DECK = tuple(encode_card(rank, suit) for rank in LeducRank for suit in LeducSuit)
LEDUC_PRIVATE_DEALS = tuple(
    (first, second) for first in LEDUC_DECK for second in LEDUC_DECK if first != second
)

_PRIVATE_DEAL_OUTCOMES = tuple(
    ChanceOutcome(outcome=deal_id, probability=1.0 / len(LEDUC_PRIVATE_DEALS))
    for deal_id in range(len(LEDUC_PRIVATE_DEALS))
)
_PUBLIC_OUTCOMES = tuple(
    tuple(
        ChanceOutcome(outcome=card, probability=1.0 / (len(LEDUC_DECK) - 2))
        for card in LEDUC_DECK
        if card not in private_cards
    )
    for private_cards in LEDUC_PRIVATE_DEALS
)
_PUBLIC_CARDS = tuple(
    tuple(outcome.outcome for outcome in outcomes) for outcomes in _PUBLIC_OUTCOMES
)
_OPEN_ACTIONS = (Action.CHECK_CALL, Action.BET_RAISE)
_FACING_BET_ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE)
_CAPPED_ACTIONS = (Action.FOLD, Action.CHECK_CALL)
_HISTORY_SEPARATOR = len(Action)


@dataclass(frozen=True, slots=True)
class LeducConfig:
    """Fixed OpenSpiel-compatible canonical Leduc rules."""

    game_id: GameId = field(default=GameId.LEDUC, init=False)
    utility_unit: UtilityUnit = field(default=UtilityUnit.CHIP, init=False)
    player_count: int = field(default=2, init=False)
    ante: int = field(default=1, init=False)
    private_cards_per_player: int = field(default=1, init=False)
    public_cards: int = field(default=1, init=False)
    betting_rounds: int = field(default=2, init=False)
    bet_sizes: tuple[int, int] = field(default=(2, 4), init=False)
    max_bets_per_round: int = field(default=2, init=False)
    starting_players: tuple[int, int] = field(default=(0, 0), init=False)
    suit_isomorphism: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class LeducState(ExtensiveFormState):
    """Immutable state for one canonical physical-card Leduc hand."""

    configuration: LeducConfig
    private_deal_id: int = -1
    private_cards: tuple[int, int] | None = None
    public_card: int | None = None
    round_index: int = 0
    round_histories: tuple[tuple[Action, ...], ...] = ((),)
    contributions: tuple[int, int] = (1, 1)
    round_commitments: tuple[int, int] = (0, 0)
    aggressive_actions: int = 0
    acting_player: PlayerId | None = None
    folded_player: PlayerId | None = None
    showdown: bool = False

    @property
    def node_type(self) -> NodeType:
        """Return the kind of transition available from this state."""
        if self.folded_player is not None or self.showdown:
            return NodeType.TERMINAL
        if self.private_cards is None or (self.round_index == 1 and self.public_card is None):
            return NodeType.CHANCE
        return NodeType.PLAYER

    @property
    def current_player(self) -> PlayerId | None:
        """Return the acting player at a decision node."""
        return self.acting_player if self.node_type is NodeType.PLAYER else None

    def legal_actions(self) -> tuple[Action, ...]:
        """Return strict legal actions in canonical storage order."""
        if self.node_type is not NodeType.PLAYER or self.acting_player is None:
            return ()
        amount_to_call = max(self.round_commitments) - self.round_commitments[self.acting_player]
        if amount_to_call == 0:
            return _OPEN_ACTIONS
        if self.aggressive_actions < self.configuration.max_bets_per_round:
            return _FACING_BET_ACTIONS
        return _CAPPED_ACTIONS

    def chance_outcomes(self) -> tuple[ChanceOutcome, ...]:
        """Return ordered physical-card outcomes for the current chance node."""
        if self.node_type is not NodeType.CHANCE:
            return ()
        if self.private_cards is None:
            return _PRIVATE_DEAL_OUTCOMES
        return _PUBLIC_OUTCOMES[self.private_deal_id]

    def apply_action(self, action: int) -> "LeducState":
        """Return the state after a strict chance or player transition."""
        if self.node_type is NodeType.TERMINAL:
            raise ValueError("cannot act in a terminal state")
        if self.node_type is NodeType.CHANCE:
            return self._apply_chance(action)

        parsed_action = require_legal_action(action, self.legal_actions())
        return self._apply_player_action(parsed_action)

    def utility(self, player: PlayerId) -> float:
        """Return terminal net-chip utility for the requested player."""
        validate_player(player)
        if self.node_type is not NodeType.TERMINAL or self.private_cards is None:
            raise ValueError("utility is available only at terminal states")

        pot = self.contributions[0] + self.contributions[1]
        if self.folded_player is not None:
            winner = 1 - self.folded_player
            player_zero_utility = (
                float(pot - self.contributions[0]) if winner == 0 else float(-self.contributions[0])
            )
        else:
            assert self.public_card is not None
            comparison = _compare_hands(self.private_cards, self.public_card)
            if comparison == 0:
                player_zero_utility = pot / 2.0 - self.contributions[0]
            elif comparison > 0:
                player_zero_utility = float(pot - self.contributions[0])
            else:
                player_zero_utility = float(-self.contributions[0])
        return player_zero_utility if player == 0 else -player_zero_utility

    def information_state(self) -> InformationState:
        """Return only cards and action history visible to the acting player."""
        player = self.current_player
        if player is None or self.private_cards is None:
            raise ValueError("information state is available only at player nodes")

        history_encoding = tuple(
            value
            for history in self.round_histories
            for value in (*(int(action) for action in history), _HISTORY_SEPARATOR)
        )
        return InformationState(
            game_id=GameId.LEDUC,
            player=player,
            encoding=(
                self.private_cards[player],
                self.public_card if self.public_card is not None else -1,
                self.round_index,
                *history_encoding,
            ),
            legal_actions=self.legal_actions(),
        )

    def _apply_chance(self, outcome: int) -> "LeducState":
        if self.private_cards is None:
            deal_id = _require_private_deal(outcome)
            return LeducState(
                configuration=self.configuration,
                private_deal_id=deal_id,
                private_cards=LEDUC_PRIVATE_DEALS[deal_id],
                acting_player=self.configuration.starting_players[0],
            )

        if isinstance(outcome, bool) or not isinstance(outcome, int):
            raise TypeError("chance outcome must be an integer")
        if outcome not in _PUBLIC_CARDS[self.private_deal_id]:
            raise ValueError(f"unavailable public card: {outcome}")
        return LeducState(
            configuration=self.configuration,
            private_deal_id=self.private_deal_id,
            private_cards=self.private_cards,
            public_card=outcome,
            round_index=1,
            round_histories=self.round_histories,
            contributions=self.contributions,
            acting_player=self.configuration.starting_players[1],
        )

    def _apply_player_action(self, action: Action) -> "LeducState":
        assert self.acting_player is not None
        player = self.acting_player
        opponent = 1 - player
        history = (*self.round_histories[-1], action)
        histories = (*self.round_histories[:-1], history)

        if action is Action.FOLD:
            return self._replace(
                round_histories=histories, acting_player=None, folded_player=player
            )

        amount_to_call = max(self.round_commitments) - self.round_commitments[player]
        if action is Action.BET_RAISE:
            payment = amount_to_call + self.configuration.bet_sizes[self.round_index]
            return self._replace(
                round_histories=histories,
                contributions=_add_to_player(self.contributions, player, payment),
                round_commitments=_add_to_player(self.round_commitments, player, payment),
                aggressive_actions=self.aggressive_actions + 1,
                acting_player=opponent,
            )

        contributions = _add_to_player(self.contributions, player, amount_to_call)
        commitments = _add_to_player(self.round_commitments, player, amount_to_call)
        closes_round = amount_to_call > 0 or (
            len(history) >= 2 and history[-2] is Action.CHECK_CALL
        )
        if not closes_round:
            return self._replace(
                round_histories=histories,
                contributions=contributions,
                round_commitments=commitments,
                acting_player=opponent,
            )
        if self.round_index == 1:
            return self._replace(
                round_histories=histories,
                contributions=contributions,
                round_commitments=commitments,
                acting_player=None,
                showdown=True,
            )
        return LeducState(
            configuration=self.configuration,
            private_deal_id=self.private_deal_id,
            private_cards=self.private_cards,
            round_index=1,
            round_histories=(*histories, ()),
            contributions=contributions,
        )

    def _replace(self, **changes: object) -> "LeducState":
        return replace(self, **changes)


class LeducGame(ExtensiveFormGame):
    """Factory for canonical physical-card Leduc states."""

    @property
    def game_id(self) -> GameId:
        """Return the canonical Leduc identifier."""
        return GameId.LEDUC

    def initial_state(self, configuration: GameConfiguration) -> LeducState:
        """Create a chance root from the canonical Leduc configuration."""
        if not isinstance(configuration, LeducConfig):
            raise TypeError("configuration must be a LeducConfig")
        return LeducState(configuration)


def compare_leduc_hands(private_cards: tuple[int, int], public_card: int) -> int:
    """Compare two Leduc hands, returning positive, zero, or negative for Player 0."""
    if len(private_cards) != 2:
        raise ValueError("private_cards must contain exactly two cards")
    for card in (*private_cards, public_card):
        _validate_card(card)
    if len({*private_cards, public_card}) != 3:
        raise ValueError("Leduc showdown cards must be physically distinct")
    return _compare_hands(private_cards, public_card)


def _compare_hands(private_cards: tuple[int, int], public_card: int) -> int:
    public_rank = public_card // LEDUC_SUIT_COUNT
    first_rank = private_cards[0] // LEDUC_SUIT_COUNT
    second_rank = private_cards[1] // LEDUC_SUIT_COUNT
    first_pair = first_rank == public_rank
    second_pair = second_rank == public_rank
    if first_pair != second_pair:
        return 1 if first_pair else -1
    return (first_rank > second_rank) - (first_rank < second_rank)


def _add_to_player(values: tuple[int, int], player: PlayerId, amount: int) -> tuple[int, int]:
    if player == 0:
        return values[0] + amount, values[1]
    return values[0], values[1] + amount


def _require_private_deal(outcome: int) -> int:
    if isinstance(outcome, bool) or not isinstance(outcome, int):
        raise TypeError("chance outcome must be an integer")
    if not 0 <= outcome < len(LEDUC_PRIVATE_DEALS):
        raise ValueError(f"unknown private deal: {outcome}")
    return outcome


def _validate_card(card: int) -> None:
    if isinstance(card, bool) or not isinstance(card, int):
        raise TypeError("card must be an integer")
    if not 0 <= card < len(LEDUC_DECK):
        raise ValueError(f"card must be between 0 and {len(LEDUC_DECK) - 1}")
