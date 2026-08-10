"""Direct seven-card evaluator backed by compact perfect-hash tables."""

import json
from collections.abc import Sequence
from functools import cache
from hashlib import sha256
from importlib import resources

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.holdem.cards import RANK_COUNT, SUIT_COUNT, validate_holdem_cards

TABLE_SCHEMA = "ac_cfr_holdem_evaluator_v1"
TABLE_DTYPE = np.dtype("<u2")
NON_FLUSH_VECTOR_COUNT = 49_205
FLUSH_MASK_COUNT = 1 << RANK_COUNT
INVALID_RANK = 0


def _build_ways() -> tuple[tuple[int, ...], ...]:
    """Precompute bounded rank-count suffix combinations for perfect hashing."""
    ways = [[0] * 8 for _ in range(RANK_COUNT + 1)]
    ways[0][0] = 1
    for length in range(1, RANK_COUNT + 1):
        for total in range(8):
            ways[length][total] = sum(
                ways[length - 1][total - count] for count in range(min(4, total) + 1)
            )
    return tuple(tuple(row) for row in ways)


QUINARY_WAYS = _build_ways()


def evaluate_holdem(hole_cards: Sequence[int], board_cards: Sequence[int]) -> int:
    """Validate and rank exactly two hole cards and five board cards."""
    cards = validate_holdem_cards(hole_cards, board_cards)
    return _evaluate_seven_cards_unchecked(cards)


def quinary_hash(rank_counts: Sequence[int]) -> int:
    """Map one valid seven-card rank-count vector bijectively into 0..49,204."""
    if len(rank_counts) != RANK_COUNT:
        raise ValueError(f"rank_counts must contain {RANK_COUNT} entries")
    if any(isinstance(count, bool) or not isinstance(count, int) for count in rank_counts):
        raise TypeError("rank counts must be integers")
    if any(not 0 <= count <= 4 for count in rank_counts) or sum(rank_counts) != 7:
        raise ValueError("rank counts must be in 0..4 and sum to seven")
    return _quinary_hash_unchecked(rank_counts)


def _quinary_hash_unchecked(rank_counts: Sequence[int]) -> int:
    """Rank a validated quinary vector in deterministic lexicographic order."""
    index = 0
    remaining = 7
    for position, count in enumerate(rank_counts):
        suffix_length = RANK_COUNT - position - 1
        for candidate in range(count):
            suffix_total = remaining - candidate
            if 0 <= suffix_total < len(QUINARY_WAYS[suffix_length]):
                index += QUINARY_WAYS[suffix_length][suffix_total]
        remaining -= count
    return index


def _evaluate_seven_cards_unchecked(
    cards: tuple[int, int, int, int, int, int, int],
) -> int:
    """Evaluate seven validated cards by one direct table lookup."""
    rank_counts = [0] * RANK_COUNT
    suit_counts = [0] * SUIT_COUNT
    suit_masks = [0] * SUIT_COUNT
    for card in cards:
        rank = card // SUIT_COUNT
        suit = card % SUIT_COUNT
        rank_counts[rank] += 1
        suit_counts[suit] += 1
        suit_masks[suit] |= 1 << rank

    non_flush_table, flush_table = _load_tables()
    for suit, count in enumerate(suit_counts):
        if count >= 5:
            rank = int(flush_table[suit_masks[suit]])
            break
    else:
        rank = int(non_flush_table[_quinary_hash_unchecked(rank_counts)])
    if rank == INVALID_RANK:
        raise RuntimeError("valid hand reached an unpopulated evaluator-table entry")
    return rank


@cache
def _load_tables() -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    """Load and integrity-check packaged evaluator tables once per process."""
    data_root = resources.files("ac_cfr.games.holdem.evaluator.data")
    metadata = json.loads(data_root.joinpath("metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema") != TABLE_SCHEMA
        or metadata.get("dtype") != TABLE_DTYPE.str
        or metadata.get("strength_class_count") != 7_462
    ):
        raise RuntimeError("incompatible Hold'em evaluator table metadata")

    tables: list[NDArray[np.uint16]] = []
    combined_payload = bytearray()
    for table_name, expected_shape in (
        ("non_flush", (NON_FLUSH_VECTOR_COUNT,)),
        ("flush", (FLUSH_MASK_COUNT,)),
    ):
        table_metadata = metadata["tables"][table_name]
        if tuple(table_metadata["shape"]) != expected_shape:
            raise RuntimeError(f"invalid {table_name} evaluator-table shape")
        payload = data_root.joinpath(table_metadata["file"]).read_bytes()
        if sha256(payload).hexdigest() != table_metadata["sha256"]:
            raise RuntimeError(f"invalid {table_name} evaluator-table checksum")
        table = np.frombuffer(payload, dtype=TABLE_DTYPE)
        if table.shape != expected_shape:
            raise RuntimeError(f"invalid {table_name} evaluator-table byte length")
        tables.append(table)
        combined_payload.extend(payload)
    if sha256(combined_payload).hexdigest() != metadata["combined_sha256"]:
        raise RuntimeError("invalid combined evaluator-table checksum")
    return tables[0], tables[1]
