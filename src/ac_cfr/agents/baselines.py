"""Simple frozen policies used as untrained comparison agents."""

from collections import Counter
from enum import IntEnum
from itertools import combinations

from ac_cfr.agents.base import PlayableAgent, Strategy
from ac_cfr.games.base import Action, GameId, InformationState
from ac_cfr.games.holdem.cards import SUIT_COUNT, Rank
from ac_cfr.games.holdem.engine import Street
from ac_cfr.games.holdem.evaluator.reference import HandCategory, HandStrength, score_five_cards
from ac_cfr.games.holdem.information_state import parse_holdem_visible_state

RULE_BASED_AGENT_ID = "rule_based_v1"
_WHEEL_MASK = (1 << int(Rank.ACE)) | 0b1111
_STRAIGHT_MASKS = (_WHEEL_MASK, *(0b11111 << start for start in range(9)))


class _StrengthBucket(IntEnum):
    """Coarse fixed-policy strength buckets."""

    VERY_WEAK = 0
    WEAK_MADE = 1
    MARGINAL_OR_DRAW = 2
    STRONG = 3
    VERY_STRONG = 4


class BaselineAgent(PlayableAgent):
    """Uniform-random policy over the currently legal actions."""

    __slots__ = ()

    def get_strategy(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
    ) -> Strategy:
        """Return a uniform distribution using no hidden game information."""
        if not isinstance(information_state, InformationState):
            raise TypeError("information_state must be an InformationState")
        if legal_actions != information_state.legal_actions:
            raise ValueError("legal_actions must match the information state")
        probability = 1.0 / len(legal_actions)
        return tuple(probability for _ in legal_actions)


class RuleBasedAgent(PlayableAgent):
    """Deterministic visible-hand-strength policy for modified HULHE."""

    __slots__ = ()

    def get_strategy(
        self,
        information_state: InformationState,
        legal_actions: tuple[Action, ...],
    ) -> Strategy:
        """Return a one-hot legal strategy from the frozen version-one rules."""
        if not isinstance(information_state, InformationState):
            raise TypeError("information_state must be an InformationState")
        if legal_actions != information_state.legal_actions:
            raise ValueError("legal_actions must match the information state")
        if information_state.game_id is not GameId.HOLD_EM:
            raise ValueError("RuleBasedAgent supports Hold'em only")

        visible_state = parse_holdem_visible_state(information_state)
        if visible_state.street < int(Street.FLOP):
            raise ValueError("RuleBasedAgent requires a flop, turn, or river decision")
        bucket = _strength_bucket(
            visible_state.hole_cards,
            visible_state.board_cards,
            street=visible_state.street,
        )
        action = _select_action(bucket, visible_state.street, legal_actions)
        return tuple(1.0 if legal_action is action else 0.0 for legal_action in legal_actions)


def _strength_bucket(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
    *,
    street: int,
) -> _StrengthBucket:
    """Classify visible cards using current made strength and simple draws."""
    cards = (*hole_cards, *board_cards)
    strength = _best_visible_strength(cards)
    category = HandCategory(strength[0])
    private_made_hand = _has_private_made_hand(
        hole_cards,
        board_cards,
        strength,
        street=street,
    )
    if private_made_hand and category >= HandCategory.FULL_HOUSE:
        return _StrengthBucket.VERY_STRONG
    if private_made_hand and category >= HandCategory.TWO_PAIR:
        return _StrengthBucket.STRONG

    private_draw = street != int(Street.RIVER) and (
        _has_private_four_flush(hole_cards, board_cards)
        or _has_private_open_ended_draw(hole_cards, board_cards)
    )
    if private_draw or (
        category is HandCategory.PAIR
        and private_made_hand
        and _is_top_pair_or_overpair(hole_cards, board_cards)
    ):
        return _StrengthBucket.MARGINAL_OR_DRAW
    if category is HandCategory.PAIR and private_made_hand:
        return _StrengthBucket.WEAK_MADE
    return _StrengthBucket.VERY_WEAK


def _best_visible_strength(cards: tuple[int, ...]) -> HandStrength:
    """Return the strongest inspectable five-card strength in visible cards."""
    if len(cards) not in (5, 6, 7):
        raise ValueError("made-hand classification requires five to seven visible cards")
    return max(score_five_cards(subset) for subset in combinations(cards, 5))


def _hole_cards_improve_board(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
) -> bool:
    """Return whether either private card improves the river's five-card hand."""
    if len(board_cards) != 5:
        raise ValueError("board comparison requires exactly five river cards")
    board_strength = score_five_cards(board_cards)
    visible_cards = (*hole_cards, *board_cards)
    best_strength = max(score_five_cards(cards) for cards in combinations(visible_cards, 5))
    return best_strength > board_strength


def _has_private_made_hand(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
    strength: HandStrength,
    *,
    street: int,
) -> bool:
    """Return whether a hole card contributes to the current made hand."""
    if street == int(Street.RIVER):
        return _hole_cards_improve_board(hole_cards, board_cards)

    category = HandCategory(strength[0])
    hole_ranks = {card // SUIT_COUNT for card in hole_cards}
    made_rank_positions = {
        HandCategory.PAIR: (1,),
        HandCategory.TWO_PAIR: (1, 2),
        HandCategory.THREE_OF_A_KIND: (1,),
        HandCategory.FULL_HOUSE: (1, 2),
        HandCategory.FOUR_OF_A_KIND: (1,),
    }
    if positions := made_rank_positions.get(category):
        return any(strength[position] in hole_ranks for position in positions)
    if category is HandCategory.STRAIGHT:
        straight_mask = _straight_mask(strength[1])
        return _private_rank_is_required(hole_cards, board_cards, straight_mask)
    if category is HandCategory.FLUSH:
        return _private_card_forms_flush(hole_cards, board_cards)
    if category is HandCategory.STRAIGHT_FLUSH:
        straight_mask = _straight_mask(strength[1])
        return _private_card_forms_straight_flush(hole_cards, board_cards, straight_mask)
    return False


def _is_top_pair_or_overpair(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
) -> bool:
    """Return whether the private pair is top pair or a pocket overpair."""
    hole_ranks = tuple(card // SUIT_COUNT for card in hole_cards)
    highest_board_rank = max(card // SUIT_COUNT for card in board_cards)
    if hole_ranks[0] == hole_ranks[1]:
        return hole_ranks[0] > highest_board_rank
    board_ranks = {card // SUIT_COUNT for card in board_cards}
    return highest_board_rank in hole_ranks and highest_board_rank in board_ranks


def _private_rank_is_required(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
    rank_mask: int,
) -> bool:
    """Return whether a hole rank supplies a missing rank in a made pattern."""
    board_rank_mask = 0
    for card in board_cards:
        board_rank_mask |= 1 << (card // SUIT_COUNT)
    missing_rank_mask = rank_mask & ~board_rank_mask
    return any(missing_rank_mask & (1 << (card // SUIT_COUNT)) for card in hole_cards)


def _private_card_forms_flush(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
) -> bool:
    """Return whether a hole card contributes to a made flush."""
    visible_suits = Counter(card % SUIT_COUNT for card in (*hole_cards, *board_cards))
    board_suits = Counter(card % SUIT_COUNT for card in board_cards)
    return any(count >= 5 and board_suits[suit] < 5 for suit, count in visible_suits.items())


def _private_card_forms_straight_flush(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
    straight_mask: int,
) -> bool:
    """Return whether a hole card contributes to a made straight flush."""
    for suit in range(SUIT_COUNT):
        hole_mask = 0
        board_mask = 0
        for card in hole_cards:
            if card % SUIT_COUNT == suit:
                hole_mask |= 1 << (card // SUIT_COUNT)
        for card in board_cards:
            if card % SUIT_COUNT == suit:
                board_mask |= 1 << (card // SUIT_COUNT)
        visible_mask = hole_mask | board_mask
        if (
            visible_mask & straight_mask == straight_mask
            and board_mask & straight_mask != straight_mask
        ):
            return True
    return False


def _straight_mask(high_rank: int) -> int:
    """Return the five ranks ending at one straight's high card."""
    return _WHEEL_MASK if high_rank == int(Rank.FIVE) else 0b11111 << (high_rank - 4)


def _has_private_four_flush(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
) -> bool:
    """Return whether a private card is needed to form a four-card flush draw."""
    visible_suits = Counter(card % SUIT_COUNT for card in (*hole_cards, *board_cards))
    board_suits = Counter(card % SUIT_COUNT for card in board_cards)
    return any(count == 4 and board_suits[suit] < 4 for suit, count in visible_suits.items())


def _has_private_open_ended_draw(
    hole_cards: tuple[int, int],
    board_cards: tuple[int, ...],
) -> bool:
    """Return whether a private rank is needed for an open-ended straight draw."""
    visible_rank_mask = 0
    board_rank_mask = 0
    for card in (*hole_cards, *board_cards):
        visible_rank_mask |= 1 << (card // SUIT_COUNT)
    for card in board_cards:
        board_rank_mask |= 1 << (card // SUIT_COUNT)
    for start in range(9):
        draw_mask = 0b1111 << start
        if visible_rank_mask & draw_mask == draw_mask and board_rank_mask & draw_mask != draw_mask:
            return True
    return False


def _select_action(
    bucket: _StrengthBucket,
    street: int,
    legal_actions: tuple[Action, ...],
) -> Action:
    """Apply the frozen action table while respecting the betting cap."""
    facing_bet = Action.FOLD in legal_actions
    can_bet_or_raise = Action.BET_RAISE in legal_actions

    if bucket is _StrengthBucket.VERY_STRONG:
        return Action.BET_RAISE if can_bet_or_raise else Action.CHECK_CALL
    if bucket is _StrengthBucket.STRONG:
        if facing_bet:
            if street == int(Street.RIVER) and can_bet_or_raise:
                return Action.BET_RAISE
            return Action.CHECK_CALL
        return Action.BET_RAISE if can_bet_or_raise else Action.CHECK_CALL
    if bucket is _StrengthBucket.MARGINAL_OR_DRAW:
        return Action.CHECK_CALL
    if bucket is _StrengthBucket.WEAK_MADE:
        return Action.FOLD if facing_bet else Action.CHECK_CALL
    return Action.FOLD if facing_bet else Action.CHECK_CALL
