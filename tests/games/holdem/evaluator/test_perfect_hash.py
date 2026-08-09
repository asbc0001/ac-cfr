import os
from random import Random

import numpy as np
import pytest
from phevaluator import evaluate_cards

from ac_cfr.games.holdem.cards import DECK, card_to_string
from ac_cfr.games.holdem.evaluator.generation import (
    build_lookup_tables,
    seven_card_rank_vectors,
)
from ac_cfr.games.holdem.evaluator.perfect_hash import (
    FLUSH_MASK_COUNT,
    INVALID_RANK,
    NON_FLUSH_VECTOR_COUNT,
    _load_tables,
    evaluate_holdem,
    quinary_hash,
)
from ac_cfr.games.holdem.evaluator.reference import evaluate_seven_cards_reference


def test_packaged_tables_and_quinary_hash_are_complete() -> None:
    non_flush_table, flush_table = _load_tables()
    assert non_flush_table.shape == (NON_FLUSH_VECTOR_COUNT,)
    assert flush_table.shape == (FLUSH_MASK_COUNT,)
    assert non_flush_table.dtype == flush_table.dtype == np.dtype("<u2")
    assert np.all(non_flush_table != INVALID_RANK)
    assert np.count_nonzero(flush_table) == sum(
        1 for mask in range(FLUSH_MASK_COUNT) if 5 <= mask.bit_count() <= 7
    )
    assert all(1 <= rank <= 7_462 for rank in non_flush_table)
    assert {quinary_hash(vector) for vector in seven_card_rank_vectors()} == set(
        range(NON_FLUSH_VECTOR_COUNT)
    )


def test_production_evaluator_validates_input_and_matches_both_oracles() -> None:
    assert evaluate_holdem((48, 44), (40, 36, 32, 0, 5)) == 1
    _assert_random_hands_agree(hand_count=5_000, seed=19_837)

    with pytest.raises(ValueError, match="exactly 2"):
        evaluate_holdem((0,), (1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="distinct"):
        evaluate_holdem((0, 1), (1, 2, 3, 4, 5))


@pytest.mark.slow
def test_million_hand_oracle_comparison() -> None:
    hand_count = int(os.environ.get("EVALUATOR_RANDOM_HANDS", "1000000"))
    _assert_random_hands_agree(hand_count=hand_count, seed=91_271)


@pytest.mark.slow
def test_lookup_generation_is_reproducible() -> None:
    generated_non_flush, generated_flush = build_lookup_tables()
    packaged_non_flush, packaged_flush = _load_tables()
    assert np.array_equal(generated_non_flush, packaged_non_flush)
    assert np.array_equal(generated_flush, packaged_flush)


def _assert_random_hands_agree(hand_count: int, seed: int) -> None:
    random_number_generator = Random(seed)
    for _ in range(hand_count):
        cards = tuple(random_number_generator.sample(DECK, 7))
        hole_cards = cards[:2]
        board_cards = cards[2:]
        expected = evaluate_seven_cards_reference(hole_cards, board_cards)
        independent = evaluate_cards(*(card_to_string(card) for card in cards))
        assert evaluate_holdem(hole_cards, board_cards) == expected == independent
