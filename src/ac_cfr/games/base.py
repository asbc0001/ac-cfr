"""Common contracts and compact encodings for extensive-form games."""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import fsum, isclose, isfinite
from numbers import Real
from typing import Protocol, Self

PLAYER_COUNT = 2

PlayerId = int
NodeId = int
InformationSetId = int
Encoding = tuple[int, ...]


class GameId(StrEnum):
    """Stable identifiers for canonical game rules."""

    KUHN = "kuhn"
    LEDUC = "leduc"
    HOLD_EM = "holdem"


class UtilityUnit(StrEnum):
    """Units used to report terminal utility."""

    CHIP = "chip"


class Action(IntEnum):
    """Context-dependent poker actions in stable storage order."""

    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2


ACTION_ORDER = tuple(Action)
_ACTION_POSITION = {action: position for position, action in enumerate(ACTION_ORDER)}

ActionHistory = tuple[Action, ...]


class NodeType(IntEnum):
    """Kinds of nodes in an extensive-form game tree."""

    CHANCE = 0
    PLAYER = 1
    TERMINAL = 2


@dataclass(frozen=True, slots=True)
class ChanceOutcome:
    """One compact chance outcome and its true probability."""

    outcome: int
    probability: float
    multiplicity: int = 1

    def __post_init__(self) -> None:
        _validate_non_negative_integer("outcome", self.outcome)
        if isinstance(self.probability, bool) or not isinstance(self.probability, Real):
            raise TypeError("probability must be a real number")

        probability = float(self.probability)
        if not isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("probability must be finite and in the interval (0, 1]")
        object.__setattr__(self, "probability", probability)

        if isinstance(self.multiplicity, bool) or not isinstance(self.multiplicity, int):
            raise TypeError("multiplicity must be an integer")
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be positive")


@dataclass(frozen=True, slots=True)
class InformationState:
    """Player-visible information used to make one strategy decision."""

    game_id: GameId
    player: PlayerId
    encoding: Encoding
    legal_actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.game_id, GameId):
            raise TypeError("game_id must be a GameId")
        validate_player(self.player)
        if not isinstance(self.encoding, tuple):
            raise TypeError("encoding must be a tuple")
        for value in self.encoding:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("information-state encoding values must be integers")
        validate_legal_actions(self.legal_actions)


class GameConfiguration(Protocol):
    """Minimum configuration contract accepted by a game implementation."""

    @property
    def game_id(self) -> GameId:
        """Return the identifier for the configured game rules."""
        ...


class ExtensiveFormState(ABC):
    """Underlying state contract shared by games, trainers, and evaluators."""

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Return whether this is a chance, player, or terminal node."""
        raise NotImplementedError

    @property
    def is_chance_node(self) -> bool:
        """Return whether chance must select the next transition."""
        return self.node_type is NodeType.CHANCE

    @property
    def is_terminal(self) -> bool:
        """Return whether the hand has ended."""
        return self.node_type is NodeType.TERMINAL

    @property
    @abstractmethod
    def current_player(self) -> PlayerId | None:
        """Return the acting player, or None at chance and terminal nodes."""
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self) -> tuple[Action, ...]:
        """Return strict legal actions in stable order at a player node."""
        raise NotImplementedError

    @abstractmethod
    def chance_outcomes(self) -> tuple[ChanceOutcome, ...]:
        """Return the exact distribution at a chance node."""
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: int) -> Self:
        """Return the state reached by one legal player action or chance outcome."""
        raise NotImplementedError

    @abstractmethod
    def utility(self, player: PlayerId) -> float:
        """Return terminal net-chip utility from the requested player's perspective."""
        raise NotImplementedError

    @abstractmethod
    def information_state(self) -> InformationState:
        """Return only information visible to the acting player."""
        raise NotImplementedError


class ExtensiveFormGame(ABC):
    """Factory contract for creating game states from explicit rules."""

    @property
    @abstractmethod
    def game_id(self) -> GameId:
        """Return the identifier for this game's exact rules."""
        raise NotImplementedError

    @abstractmethod
    def initial_state(self, configuration: GameConfiguration) -> ExtensiveFormState:
        """Create the root state for an explicit compatible configuration."""
        raise NotImplementedError


class DeterministicIdRegistry[KeyType: Hashable]:
    """Assign zero-based IDs in an explicitly controlled encounter order.

    Tree builders call ``assign`` during deterministic depth-first traversal. The
    registry never derives IDs from hashes or iterates over its internal mapping.
    """

    def __init__(self) -> None:
        self._ids: dict[KeyType, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def assign(self, key: KeyType) -> int:
        """Return the existing ID for a key or assign the next sequential ID."""
        identifier = self._ids.get(key)
        if identifier is not None:
            return identifier

        identifier = len(self._ids)
        self._ids[key] = identifier
        return identifier

    def identifier_for(self, key: KeyType) -> int:
        """Return an already assigned ID."""
        return self._ids[key]


def validate_player(player: PlayerId) -> None:
    """Validate a two-player game identifier."""
    if isinstance(player, bool) or not isinstance(player, int):
        raise TypeError("player must be an integer")
    if not 0 <= player < PLAYER_COUNT:
        raise ValueError(f"player must be between 0 and {PLAYER_COUNT - 1}")


def validate_legal_actions(actions: tuple[Action, ...]) -> None:
    """Validate a non-empty, unique action tuple in canonical order."""
    if not isinstance(actions, tuple):
        raise TypeError("legal actions must be a tuple")
    if not actions:
        raise ValueError("legal actions must not be empty")
    if any(not isinstance(action, Action) for action in actions):
        raise TypeError("every legal action must be an Action")
    if len(set(actions)) != len(actions):
        raise ValueError("legal actions must not contain duplicates")

    positions = tuple(_ACTION_POSITION[action] for action in actions)
    if positions != tuple(sorted(positions)):
        raise ValueError("legal actions must use canonical action order")


def validate_chance_outcomes(outcomes: Iterable[ChanceOutcome]) -> tuple[ChanceOutcome, ...]:
    """Validate and return a finite chance distribution in supplied order."""
    outcome_tuple = tuple(outcomes)
    if not outcome_tuple:
        raise ValueError("chance outcomes must not be empty")
    if any(not isinstance(outcome, ChanceOutcome) for outcome in outcome_tuple):
        raise TypeError("every chance outcome must be a ChanceOutcome")

    outcome_ids = tuple(outcome.outcome for outcome in outcome_tuple)
    if len(set(outcome_ids)) != len(outcome_ids):
        raise ValueError("chance outcome identifiers must be unique")

    probability_sum = fsum(outcome.probability for outcome in outcome_tuple)
    if not isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("chance outcome probabilities must sum to 1")
    return outcome_tuple


def require_legal_action(action: int, legal_actions: tuple[Action, ...]) -> Action:
    """Return a legal action enum or reject an invalid transition."""
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError("action must be an integer")
    try:
        parsed_action = Action(action)
    except ValueError as error:
        raise ValueError(f"unknown action: {action}") from error
    if parsed_action not in legal_actions:
        raise ValueError(f"illegal action: {parsed_action.name}")
    return parsed_action


def _validate_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
