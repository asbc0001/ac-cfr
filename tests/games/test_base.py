from dataclasses import FrozenInstanceError

import pytest

from ac_cfr.games.base import (
    ACTION_ORDER,
    Action,
    ChanceOutcome,
    DeterministicIdRegistry,
    GameId,
    InformationState,
    NodeType,
    validate_chance_outcomes,
    validate_legal_actions,
    validate_player,
)


def test_game_identifiers_are_simple_and_stable() -> None:
    assert GameId.KUHN == "kuhn"
    assert GameId.LEDUC == "leduc"


def test_action_and_node_encodings_are_compact_and_stable() -> None:
    assert ACTION_ORDER == (Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE)
    assert tuple(int(action) for action in ACTION_ORDER) == (0, 1, 2)
    assert tuple(int(node_type) for node_type in NodeType) == (0, 1, 2)


def test_legal_actions_require_unique_canonical_action_tuples() -> None:
    validate_legal_actions((Action.FOLD, Action.CHECK_CALL))
    validate_legal_actions((Action.CHECK_CALL, Action.BET_RAISE))
    validate_legal_actions(ACTION_ORDER)

    for invalid_actions in (
        (),
        (Action.CHECK_CALL, Action.FOLD),
        (Action.FOLD, Action.FOLD),
    ):
        with pytest.raises(ValueError):
            validate_legal_actions(invalid_actions)
    with pytest.raises(TypeError):
        validate_legal_actions((0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_legal_actions([Action.FOLD])  # type: ignore[arg-type]


def test_chance_distribution_preserves_order_probability_and_multiplicity() -> None:
    outcomes = validate_chance_outcomes(
        (
            ChanceOutcome(outcome=10, probability=1 / 3, multiplicity=2),
            ChanceOutcome(outcome=11, probability=2 / 3, multiplicity=4),
        )
    )

    assert tuple(outcome.outcome for outcome in outcomes) == (10, 11)
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    assert tuple(outcome.multiplicity for outcome in outcomes) == (2, 4)


def test_chance_distribution_rejects_duplicate_or_incomplete_outcomes() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_chance_outcomes((ChanceOutcome(0, 0.5), ChanceOutcome(0, 0.5)))
    with pytest.raises(ValueError, match="sum to 1"):
        validate_chance_outcomes((ChanceOutcome(0, 0.4), ChanceOutcome(1, 0.5)))
    for invalid_probability in (0.0, 1.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ChanceOutcome(0, invalid_probability)
    with pytest.raises(ValueError):
        ChanceOutcome(-1, 1.0)
    with pytest.raises(TypeError):
        ChanceOutcome(True, 1.0)
    with pytest.raises(ValueError):
        ChanceOutcome(0, 1.0, 0)
    with pytest.raises(TypeError):
        ChanceOutcome(0, 1.0, True)


def test_player_validation_is_strict() -> None:
    validate_player(0)
    validate_player(1)

    for invalid_player in (-1, 2):
        with pytest.raises(ValueError):
            validate_player(invalid_player)
    for invalid_player in (True, 0.0):
        with pytest.raises(TypeError):
            validate_player(invalid_player)  # type: ignore[arg-type]


def test_information_state_is_immutable_player_visible_data() -> None:
    information_state = InformationState(
        game_id=GameId.KUHN,
        player=0,
        encoding=(2, 1, 0),
        legal_actions=(Action.CHECK_CALL, Action.BET_RAISE),
    )

    assert information_state.encoding == (2, 1, 0)
    with pytest.raises(FrozenInstanceError):
        information_state.player = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="encoding values"):
        InformationState(
            game_id=GameId.KUHN,
            player=0,
            encoding=(1, "hidden"),  # type: ignore[arg-type]
            legal_actions=(Action.CHECK_CALL,),
        )


def test_deterministic_ids_follow_explicit_encounter_order() -> None:
    first_registry: DeterministicIdRegistry[tuple[int, ...]] = DeterministicIdRegistry()
    second_registry: DeterministicIdRegistry[tuple[int, ...]] = DeterministicIdRegistry()
    keys = ((2, 0), (1, 4), (2, 0), (0, 3))

    first_ids = tuple(first_registry.assign(key) for key in keys)
    second_ids = tuple(second_registry.assign(key) for key in keys)

    assert first_ids == second_ids == (0, 1, 0, 2)
    assert first_registry.identifier_for((1, 4)) == 1
    assert len(first_registry) == 3
