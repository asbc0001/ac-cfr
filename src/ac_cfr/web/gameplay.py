"""Server-authoritative ephemeral poker-hand management."""

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from math import fsum
from random import Random, SystemRandom

from ac_cfr.agents.base import Strategy, validate_strategy
from ac_cfr.games import leduc
from ac_cfr.games.base import Action, ExtensiveFormState, GameId, NodeType
from ac_cfr.games.holdem.cards import card_to_string as holdem_card_to_string
from ac_cfr.games.holdem.engine import HoldemConfig, HoldemGame, HoldemState, Street
from ac_cfr.games.holdem.evaluator.reference import HandCategory, score_five_cards
from ac_cfr.games.kuhn import KuhnCard, KuhnConfig, KuhnGame, KuhnState
from ac_cfr.persistence.registry import ResolvedStrategy, StrategyRegistry

DEFAULT_HAND_TTL_SECONDS = 30 * 60

_KUHN_CARD_NAMES = {
    KuhnCard.JACK: "J",
    KuhnCard.QUEEN: "Q",
    KuhnCard.KING: "K",
}
_LEDUC_RANK_NAMES = "JQK"
_LEDUC_SUIT_NAMES = "cd"
_HOLDEM_RANK_NAMES = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

_GameState = KuhnState | leduc.LeducState | HoldemState


class HandNotFoundError(LookupError):
    """Raised when an opaque hand identifier is unknown or expired."""


class StaleHandVersionError(ValueError):
    """Raised when a mutation does not target the current hand version."""


class InvalidHandActionError(ValueError):
    """Raised when a requested action cannot be applied to the current hand."""


@dataclass(frozen=True, slots=True)
class ActionProbability:
    """One legal action and its disclosed sampling probability."""

    action: Action
    label: str
    probability: float


@dataclass(frozen=True, slots=True)
class AIDecision:
    """An AI strategy disclosed only after its action has been applied."""

    probabilities: tuple[ActionProbability, ...]
    chosen_action: Action


@dataclass(frozen=True, slots=True)
class ActionHistoryEntry:
    """One attributed player action within a named betting round."""

    street: str
    actor: str
    action: str


@dataclass(frozen=True, slots=True)
class TerminalSummary:
    """Terminal reason, hand classes, and cards relevant to the result."""

    reason: str
    headline: str
    human_hand: str | None
    opponent_hand: str | None
    highlighted_cards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicHandView:
    """Only state that the human player is permitted to observe."""

    hand_id: str
    state_version: int
    strategy_id: str
    game: str
    game_version: str
    human_player: int
    human_position: str
    current_hand: str
    private_cards: tuple[str, ...]
    opponent_cards: tuple[str, ...]
    board: tuple[str, ...]
    pot: float
    action_history: tuple[ActionHistoryEntry, ...]
    legal_actions: tuple[tuple[Action, str, float | None], ...]
    ai_decision: AIDecision | None
    terminal: bool
    human_utility: float | None
    result: str | None
    terminal_summary: TerminalSummary | None


@dataclass(slots=True)
class _HandSession:
    """Mutable server-owned state for one short-lived hand."""

    hand_id: str
    resolved_strategy: ResolvedStrategy
    state: _GameState
    human_player: int
    rng: Random
    state_version: int
    last_access: float
    last_ai_decision: AIDecision | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class HandStore:
    """Own ephemeral hands, cached frozen agents, and mutation locks."""

    __slots__ = (
        "_clock",
        "_hand_ttl_seconds",
        "_hands",
        "_lock",
        "_master_rng",
        "_registry",
        "_resolved_strategies",
        "_token_factory",
    )

    def __init__(
        self,
        registry: StrategyRegistry,
        *,
        hand_ttl_seconds: float = DEFAULT_HAND_TTL_SECONDS,
        master_rng: Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        if not isinstance(registry, StrategyRegistry):
            raise TypeError("registry must be a StrategyRegistry")
        if hand_ttl_seconds <= 0:
            raise ValueError("hand_ttl_seconds must be positive")
        self._registry = registry
        self._hand_ttl_seconds = float(hand_ttl_seconds)
        self._master_rng = master_rng if master_rng is not None else SystemRandom()
        self._clock = clock
        self._token_factory = token_factory
        self._hands: dict[str, _HandSession] = {}
        self._resolved_strategies: dict[str, ResolvedStrategy] = {}
        self._lock = threading.Lock()

    def create(self, strategy_id: str) -> PublicHandView:
        """Create, deal, and advance one hand until the human can act."""
        resolved = self._resolve_strategy(strategy_id)
        with self._lock:
            self._discard_expired(self._clock())
            hand_id = self._new_hand_id()
            human_player = self._master_rng.randrange(2)
            hand_rng = Random(self._master_rng.getrandbits(128))
            session = _HandSession(
                hand_id=hand_id,
                resolved_strategy=resolved,
                state=_initial_state(resolved),
                human_player=human_player,
                rng=hand_rng,
                state_version=0,
                last_access=self._clock(),
            )
            self._hands[hand_id] = session

        try:
            with session.lock:
                _advance_to_human_or_terminal(session)
                session.last_access = self._clock()
                return _public_view(session)
        except Exception:
            with self._lock:
                self._hands.pop(hand_id, None)
            raise

    def act(self, hand_id: str, *, expected_version: int, action: int) -> PublicHandView:
        """Apply one current, legal human action exactly once."""
        session = self._active_session(hand_id)
        with session.lock:
            if expected_version != session.state_version:
                raise StaleHandVersionError("hand state version is stale")
            _validate_session_context(session)
            if session.state.is_terminal:
                raise InvalidHandActionError("the hand is already terminal")
            if session.state.current_player != session.human_player:
                raise InvalidHandActionError("it is not the human player's turn")
            if isinstance(action, bool) or not isinstance(action, int):
                raise InvalidHandActionError("action must be an integer")
            try:
                parsed_action = Action(action)
            except ValueError as error:
                raise InvalidHandActionError("action is unknown") from error
            if parsed_action not in session.state.legal_actions():
                raise InvalidHandActionError("action is illegal in the current state")

            session.last_ai_decision = None
            session.state = session.state.apply_action(parsed_action)
            session.state_version += 1
            _advance_to_human_or_terminal(session)
            session.last_access = self._clock()
            return _public_view(session)

    def discard(self, hand_id: str, *, expected_version: int) -> None:
        """Discard exactly one current version of an ephemeral hand."""
        session = self._active_session(hand_id)
        with session.lock:
            if expected_version != session.state_version:
                raise StaleHandVersionError("hand state version is stale")
            # Invalidate an action that was queued before this discard acquired the lock.
            session.state_version += 1
            with self._lock:
                if self._hands.get(hand_id) is session:
                    del self._hands[hand_id]

    def _resolve_strategy(self, strategy_id: str) -> ResolvedStrategy:
        if not isinstance(strategy_id, str) or not strategy_id:
            raise ValueError("strategy_id must be a non-empty string")
        with self._lock:
            resolved = self._resolved_strategies.get(strategy_id)
            if resolved is None:
                resolved = self._registry.resolve(strategy_id)
                self._resolved_strategies[strategy_id] = resolved
            return resolved

    def _active_session(self, hand_id: str) -> _HandSession:
        if not isinstance(hand_id, str) or not hand_id:
            raise HandNotFoundError("hand does not exist")
        now = self._clock()
        with self._lock:
            session = self._hands.get(hand_id)
            if session is None or now - session.last_access >= self._hand_ttl_seconds:
                self._hands.pop(hand_id, None)
                raise HandNotFoundError("hand does not exist or has expired")
            session.last_access = now
            return session

    def _new_hand_id(self) -> str:
        for _ in range(10):
            hand_id = self._token_factory()
            if isinstance(hand_id, str) and hand_id and hand_id not in self._hands:
                return hand_id
        raise RuntimeError("could not create a unique hand identifier")

    def _discard_expired(self, now: float) -> None:
        expired = tuple(
            hand_id
            for hand_id, session in self._hands.items()
            if now - session.last_access >= self._hand_ttl_seconds
        )
        for hand_id in expired:
            del self._hands[hand_id]


def _initial_state(resolved: ResolvedStrategy) -> _GameState:
    game_id = GameId(resolved.entry.game)
    if game_id is GameId.KUHN:
        return KuhnGame().initial_state(KuhnConfig())
    if game_id is GameId.LEDUC:
        return leduc.LeducGame().initial_state(leduc.LeducConfig())
    if not isinstance(resolved.game, HoldemConfig):
        raise ValueError("modified HULHE strategy has an incompatible game configuration")
    return HoldemGame().initial_state(resolved.game)


def _advance_to_human_or_terminal(session: _HandSession) -> None:
    while not session.state.is_terminal:
        if session.state.node_type is NodeType.CHANCE:
            outcomes = session.state.chance_outcomes()
            outcome = _sample_weighted(
                tuple(item.outcome for item in outcomes),
                tuple(item.probability for item in outcomes),
                session.rng,
            )
            session.state = session.state.apply_action(outcome)
            session.state_version += 1
            continue
        if session.state.current_player == session.human_player:
            return

        information_state = session.state.information_state()
        legal_actions = information_state.legal_actions
        strategy = validate_strategy(
            session.resolved_strategy.agent.get_strategy(information_state, legal_actions),
            legal_actions,
        )
        action = Action(_sample_weighted(legal_actions, strategy, session.rng))
        session.last_ai_decision = AIDecision(
            probabilities=tuple(
                ActionProbability(candidate, _action_label(candidate, legal_actions), probability)
                for candidate, probability in zip(legal_actions, strategy, strict=True)
            ),
            chosen_action=action,
        )
        session.state = session.state.apply_action(action)
        session.state_version += 1


def _sample_weighted[T](values: tuple[T, ...], probabilities: Strategy, rng: Random) -> T:
    if not values or len(values) != len(probabilities):
        raise ValueError("sampling values and probabilities are inconsistent")
    draw = rng.random() * fsum(probabilities)
    cumulative = 0.0
    for value, probability in zip(values, probabilities, strict=True):
        cumulative += probability
        if draw < cumulative:
            return value
    return values[-1]


def _validate_session_context(session: _HandSession) -> None:
    entry = session.resolved_strategy.entry
    snapshot_metadata = session.resolved_strategy.snapshot_metadata
    configuration = session.state.configuration
    configuration_id = configuration.configuration_id
    if configuration_id is None or configuration_id.value != entry.game_version:
        raise RuntimeError("hand game configuration no longer matches its strategy")
    if configuration.game_id.value != entry.game:
        raise RuntimeError("hand game no longer matches its strategy")
    expected_encoding = (
        snapshot_metadata.state_encoding
        if snapshot_metadata is not None
        else configuration.state_encoding_id.value
    )
    if expected_encoding != entry.state_encoding:
        raise RuntimeError("hand encoding no longer matches its strategy")


def _public_view(session: _HandSession) -> PublicHandView:
    state = session.state
    private_cards, opponent_cards, board, pot = _visible_state(
        state,
        session.human_player,
    )
    legal_actions = (
        tuple(
            (
                action,
                _action_label(action, state.legal_actions()),
                _action_cost(state, action),
            )
            for action in state.legal_actions()
        )
        if state.current_player == session.human_player
        else ()
    )
    utility = state.utility(session.human_player) if state.is_terminal else None
    return PublicHandView(
        hand_id=session.hand_id,
        state_version=session.state_version,
        strategy_id=session.resolved_strategy.entry.strategy_id,
        game=session.resolved_strategy.entry.game,
        game_version=session.resolved_strategy.entry.game_version,
        human_player=session.human_player,
        human_position=_human_position(state, session.human_player),
        current_hand=_current_hand_label(state, session.human_player),
        private_cards=private_cards,
        # Finished hands have no strategically hidden information left.
        opponent_cards=opponent_cards if state.is_terminal else (),
        board=board,
        pot=pot,
        action_history=_action_history(state, session.human_player),
        legal_actions=legal_actions,
        ai_decision=session.last_ai_decision,
        terminal=state.is_terminal,
        human_utility=utility,
        result=_result_text(utility),
        terminal_summary=(
            _terminal_summary(state, session.human_player) if state.is_terminal else None
        ),
    )


def _visible_state(
    state: ExtensiveFormState,
    human_player: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    float,
]:
    opponent_player = 1 - human_player
    if isinstance(state, KuhnState):
        private_cards = (
            (_KUHN_CARD_NAMES[state.private_cards[human_player]],)
            if state.private_cards is not None
            else ()
        )
        opponent_cards = (
            (_KUHN_CARD_NAMES[state.private_cards[opponent_player]],)
            if state.private_cards is not None
            else ()
        )
        return (
            private_cards,
            opponent_cards,
            (),
            _kuhn_pot(state),
        )
    if isinstance(state, leduc.LeducState):
        private_cards = (
            (_leduc_card_to_string(state.private_cards[human_player]),)
            if state.private_cards is not None
            else ()
        )
        opponent_cards = (
            (_leduc_card_to_string(state.private_cards[opponent_player]),)
            if state.private_cards is not None
            else ()
        )
        board = (_leduc_card_to_string(state.public_card),) if state.public_card is not None else ()
        return (
            private_cards,
            opponent_cards,
            board,
            float(sum(state.contributions)),
        )
    if isinstance(state, HoldemState):
        private_cards = tuple(
            holdem_card_to_string(card) for card in state.hole_cards[human_player]
        )
        opponent_cards = tuple(
            holdem_card_to_string(card) for card in state.hole_cards[opponent_player]
        )
        board = tuple(holdem_card_to_string(card) for card in state.board_cards)
        pot = float(state.pot * state.configuration.base_unit)
        return (
            private_cards,
            opponent_cards,
            board,
            pot,
        )
    raise TypeError("unsupported web game state")


def _action_history(state: _GameState, human_player: int) -> tuple[ActionHistoryEntry, ...]:
    if isinstance(state, KuhnState):
        return _round_action_history(
            (state.action_history,),
            (state.configuration.starting_player,),
            ("Hand",),
            human_player,
        )
    if isinstance(state, leduc.LeducState):
        round_count = len(state.round_histories)
        return _round_action_history(
            state.round_histories,
            state.configuration.starting_players[:round_count],
            ("Round 1", "Round 2")[:round_count],
            human_player,
        )
    if isinstance(state, HoldemState):
        street_indexes = tuple(
            range(
                int(state.configuration.start_street),
                int(state.configuration.start_street) + len(state.round_histories),
            )
        )
        starting_players = tuple(
            state.configuration.button_player
            if street_index == int(state.configuration.start_street) == 0
            else 1 - state.configuration.button_player
            for street_index in street_indexes
        )
        street_names = tuple(Street(index).name.title() for index in street_indexes)
        return _round_action_history(
            state.round_histories,
            starting_players,
            street_names,
            human_player,
        )
    raise TypeError("unsupported web game state")


def _round_action_history(
    round_histories: tuple[tuple[Action, ...], ...],
    starting_players: tuple[int, ...],
    street_names: tuple[str, ...],
    human_player: int,
) -> tuple[ActionHistoryEntry, ...]:
    entries = []
    for history, starting_player, street in zip(
        round_histories,
        starting_players,
        street_names,
        strict=True,
    ):
        previous_action: Action | None = None
        for index, action in enumerate(history):
            actor = (starting_player + index) % 2
            entries.append(
                ActionHistoryEntry(
                    street=street,
                    actor="You" if actor == human_player else "AI",
                    action=_historical_action_label(action, previous_action),
                )
            )
            previous_action = action
    return tuple(entries)


def _historical_action_label(action: Action, previous_action: Action | None) -> str:
    facing_bet = previous_action is Action.BET_RAISE
    if action is Action.CHECK_CALL:
        return "Call" if facing_bet else "Check"
    if action is Action.BET_RAISE:
        return "Raise" if facing_bet else "Bet"
    return "Fold"


def _terminal_summary(state: _GameState, human_player: int) -> TerminalSummary:
    if isinstance(state, KuhnState):
        return _kuhn_terminal_summary(state, human_player)
    if isinstance(state, leduc.LeducState):
        return _leduc_terminal_summary(state, human_player)
    if isinstance(state, HoldemState):
        return _holdem_terminal_summary(state, human_player)
    raise TypeError("unsupported web game state")


def _current_hand_label(state: _GameState, human_player: int) -> str:
    if isinstance(state, KuhnState):
        assert state.private_cards is not None
        return f"High card: {_KUHN_CARD_NAMES[state.private_cards[human_player]]}"
    if isinstance(state, leduc.LeducState):
        assert state.private_cards is not None
        card = state.private_cards[human_player]
        pair = state.public_card is not None and leduc.card_rank(card) == leduc.card_rank(
            state.public_card
        )
        return _leduc_hand_label(card, pair)
    if isinstance(state, HoldemState):
        visible_cards = (*state.hole_cards[human_player], *state.board_cards)
        strength = max(score_five_cards(cards) for cards in combinations(visible_cards, 5))
        return _holdem_hand_label(strength)
    raise TypeError("unsupported web game state")


def _kuhn_terminal_summary(state: KuhnState, human_player: int) -> TerminalSummary:
    assert state.private_cards is not None
    utility = state.utility(human_player)
    if state.action_history[-1] is Action.FOLD:
        return _fold_summary(utility)
    human_card = _KUHN_CARD_NAMES[state.private_cards[human_player]]
    opponent_card = _KUHN_CARD_NAMES[state.private_cards[1 - human_player]]
    winner_card = human_card if utility > 0 else opponent_card
    return TerminalSummary(
        reason="showdown",
        headline=f"{'You win' if utility > 0 else 'AI wins'} with {winner_card} high",
        human_hand=f"{human_card} high",
        opponent_hand=f"{opponent_card} high",
        highlighted_cards=(winner_card,),
    )


def _leduc_terminal_summary(
    state: leduc.LeducState,
    human_player: int,
) -> TerminalSummary:
    assert state.private_cards is not None
    utility = state.utility(human_player)
    if state.folded_player is not None:
        return _fold_summary(utility)
    assert state.public_card is not None
    public_rank = leduc.card_rank(state.public_card)
    human_card = state.private_cards[human_player]
    opponent_card = state.private_cards[1 - human_player]
    human_pair = leduc.card_rank(human_card) == public_rank
    opponent_pair = leduc.card_rank(opponent_card) == public_rank
    human_label = _leduc_hand_label(human_card, human_pair)
    opponent_label = _leduc_hand_label(opponent_card, opponent_pair)
    if utility == 0:
        headline = f"Tie with {_lower_initial(human_label)}"
        highlighted = (
            _leduc_card_to_string(human_card),
            _leduc_card_to_string(opponent_card),
            _leduc_card_to_string(state.public_card),
        )
    else:
        human_won = utility > 0
        winner_card = human_card if human_won else opponent_card
        winner_pair = human_pair if human_won else opponent_pair
        winner_label = human_label if human_won else opponent_label
        headline = f"{'You win' if human_won else 'AI wins'} with {_lower_initial(winner_label)}"
        highlighted = (_leduc_card_to_string(winner_card),)
        if winner_pair:
            highlighted += (_leduc_card_to_string(state.public_card),)
    return TerminalSummary(
        reason="showdown",
        headline=headline,
        human_hand=human_label,
        opponent_hand=opponent_label,
        highlighted_cards=highlighted,
    )


def _holdem_terminal_summary(state: HoldemState, human_player: int) -> TerminalSummary:
    utility = state.utility(human_player)
    if state.folded_player is not None:
        return _fold_summary(utility)
    human_cards, human_strength = _best_holdem_hand(
        state.hole_cards[human_player],
        state.board_cards,
    )
    opponent_cards, opponent_strength = _best_holdem_hand(
        state.hole_cards[1 - human_player],
        state.board_cards,
    )
    human_label = _holdem_hand_label(human_strength)
    opponent_label = _holdem_hand_label(opponent_strength)
    if utility == 0:
        headline = f"Tie with {_lower_initial(human_label)}"
        highlighted_cards = (*human_cards, *opponent_cards)
    elif utility > 0:
        headline = f"You win with {_lower_initial(human_label)}"
        highlighted_cards = human_cards
    else:
        headline = f"AI wins with {_lower_initial(opponent_label)}"
        highlighted_cards = opponent_cards
    return TerminalSummary(
        reason="showdown",
        headline=headline,
        human_hand=human_label,
        opponent_hand=opponent_label,
        highlighted_cards=tuple(dict.fromkeys(highlighted_cards)),
    )


def _fold_summary(utility: float) -> TerminalSummary:
    return TerminalSummary(
        reason="fold",
        headline="You win: AI folded" if utility > 0 else "AI wins: you folded",
        human_hand=None,
        opponent_hand=None,
        highlighted_cards=(),
    )


def _lower_initial(text: str) -> str:
    return f"{text[:1].lower()}{text[1:]}"


def _leduc_hand_label(card: int, pair: bool) -> str:
    rank = _LEDUC_RANK_NAMES[int(leduc.card_rank(card))]
    return f"Pair: {rank}" if pair else f"High card: {rank}"


def _best_holdem_hand(
    hole_cards: tuple[int, ...],
    board_cards: tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    visible_cards = (*hole_cards, *board_cards)
    best_cards = max(
        combinations(visible_cards, 5),
        key=lambda cards: (score_five_cards(cards), tuple(sorted(cards))),
    )
    return (
        tuple(holdem_card_to_string(card) for card in best_cards),
        score_five_cards(best_cards),
    )


def _holdem_hand_label(strength: tuple[int, ...]) -> str:
    category = HandCategory(strength[0])
    ranks = tuple(_HOLDEM_RANK_NAMES[int(rank)] for rank in strength[1:])
    if category is HandCategory.HIGH_CARD:
        return f"High card: {ranks[0]}"
    if category is HandCategory.PAIR:
        return f"Pair: {ranks[0]}"
    if category is HandCategory.TWO_PAIR:
        return f"Two pair: {ranks[0]}, {ranks[1]}"
    if category is HandCategory.THREE_OF_A_KIND:
        return f"Three of a kind: {ranks[0]}"
    if category is HandCategory.STRAIGHT:
        return f"Straight: {ranks[0]} high"
    if category is HandCategory.FLUSH:
        return f"Flush: {ranks[0]} high"
    if category is HandCategory.FULL_HOUSE:
        return f"Full house: {ranks[0]} over {ranks[1]}"
    if category is HandCategory.FOUR_OF_A_KIND:
        return f"Four of a kind: {ranks[0]}"
    return f"Straight flush: {ranks[0]} high"


def _kuhn_pot(state: KuhnState) -> float:
    pot = 2
    previous_action: Action | None = None
    for action in state.action_history:
        if action is Action.BET_RAISE or (
            action is Action.CHECK_CALL and previous_action is Action.BET_RAISE
        ):
            pot += 1
        previous_action = action
    return float(pot)


def _leduc_card_to_string(card: int) -> str:
    rank = _LEDUC_RANK_NAMES[int(leduc.card_rank(card))]
    suit = _LEDUC_SUIT_NAMES[int(leduc.card_suit(card))]
    return f"{rank}{suit}"


def _action_label(action: Action, legal_actions: tuple[Action, ...]) -> str:
    facing_bet = Action.FOLD in legal_actions
    if action is Action.CHECK_CALL:
        return "Call" if facing_bet else "Check"
    if action is Action.BET_RAISE:
        return "Raise" if facing_bet else "Bet"
    return "Fold"


def _action_cost(state: _GameState, action: Action) -> float | None:
    if action is Action.FOLD:
        return None
    if isinstance(state, KuhnState):
        return 1.0 if action is Action.BET_RAISE or Action.FOLD in state.legal_actions() else None
    if isinstance(state, leduc.LeducState):
        assert state.current_player is not None
        amount_to_call = (
            max(state.round_commitments) - state.round_commitments[state.current_player]
        )
        amount = amount_to_call
        if action is Action.BET_RAISE:
            amount += state.configuration.bet_sizes[state.round_index]
        return float(amount) if amount else None
    if isinstance(state, HoldemState):
        amount = state.amount_to_call
        if action is Action.BET_RAISE:
            bet_size = (
                state.configuration.small_bet_units
                if state.street in (Street.PREFLOP, Street.FLOP)
                else state.configuration.big_bet_units
            )
            amount += bet_size
        chip_amount = float(amount * state.configuration.base_unit)
        return chip_amount if chip_amount else None
    raise TypeError("unsupported web game state")


def _human_position(state: ExtensiveFormState, human_player: int) -> str:
    if isinstance(state, HoldemState):
        return "Button" if human_player == state.configuration.button_player else "Out of position"
    return f"Player {human_player + 1}"


def _result_text(utility: float | None) -> str | None:
    if utility is None:
        return None
    if utility > 0:
        return f"You win {utility:g} chips"
    if utility < 0:
        return f"You lose {-utility:g} chips"
    return "Draw"
