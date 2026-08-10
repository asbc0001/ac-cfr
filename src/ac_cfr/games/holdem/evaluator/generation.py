"""Deterministic offline construction of production evaluator tables."""

import json
from collections.abc import Iterator, Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ac_cfr.games.holdem.cards import RANK_COUNT, SUIT_COUNT
from ac_cfr.games.holdem.evaluator.perfect_hash import (
    FLUSH_MASK_COUNT,
    INVALID_RANK,
    NON_FLUSH_VECTOR_COUNT,
    TABLE_DTYPE,
    TABLE_SCHEMA,
    quinary_hash,
)
from ac_cfr.games.holdem.evaluator.reference import _evaluate_seven_cards_unchecked


def build_lookup_tables() -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    """Build every non-flush hash and valid flush-mask entry exactly once."""
    non_flush_table = np.full(NON_FLUSH_VECTOR_COUNT, INVALID_RANK, dtype=TABLE_DTYPE)
    populated_indices: set[int] = set()
    for rank_counts in seven_card_rank_vectors():
        index = quinary_hash(rank_counts)
        if index in populated_indices:
            raise RuntimeError("quinary hash collision during table generation")
        cards = _non_flush_representative(rank_counts)
        non_flush_table[index] = _evaluate_seven_cards_unchecked(cards)
        populated_indices.add(index)
    if len(populated_indices) != NON_FLUSH_VECTOR_COUNT or np.any(non_flush_table == INVALID_RANK):
        raise RuntimeError("non-flush table was not populated exactly once")

    flush_table = np.full(FLUSH_MASK_COUNT, INVALID_RANK, dtype=TABLE_DTYPE)
    for rank_mask in range(FLUSH_MASK_COUNT):
        if 5 <= rank_mask.bit_count() <= 7:
            cards = _flush_representative(rank_mask)
            flush_table[rank_mask] = _evaluate_seven_cards_unchecked(cards)
    return non_flush_table, flush_table


def seven_card_rank_vectors() -> Iterator[tuple[int, ...]]:
    """Yield valid quinary vectors in deterministic lexicographic order."""
    counts = [0] * RANK_COUNT

    def visit(position: int, remaining: int) -> Iterator[tuple[int, ...]]:
        if position == RANK_COUNT:
            if remaining == 0:
                yield tuple(counts)
            return
        for count in range(min(4, remaining) + 1):
            counts[position] = count
            yield from visit(position + 1, remaining - count)

    yield from visit(0, 7)


def write_lookup_data(output_directory: Path) -> dict[str, object]:
    """Atomically write deterministic binary tables and canonical metadata."""
    non_flush_table, flush_table = build_lookup_tables()
    output_directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "non_flush": non_flush_table.tobytes(order="C"),
        "flush": flush_table.tobytes(order="C"),
    }
    files = {"non_flush": "non_flush.bin", "flush": "flush.bin"}
    metadata: dict[str, object] = {
        "schema": TABLE_SCHEMA,
        "dtype": TABLE_DTYPE.str,
        "strength_class_count": 7_462,
        "tables": {
            name: {
                "file": files[name],
                "shape": [len(payload) // TABLE_DTYPE.itemsize],
                "sha256": sha256(payload).hexdigest(),
            }
            for name, payload in payloads.items()
        },
        "combined_sha256": sha256(payloads["non_flush"] + payloads["flush"]).hexdigest(),
    }
    for name, payload in payloads.items():
        _atomic_write(output_directory / files[name], payload)
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_directory / "metadata.json", metadata_bytes)
    return metadata


def _non_flush_representative(
    rank_counts: Sequence[int],
) -> tuple[int, int, int, int, int, int, int]:
    """Construct a deterministic seven-card hand without a five-card flush."""
    suit_counts = [0] * SUIT_COUNT
    cards: list[int] = []
    for rank, count in enumerate(rank_counts):
        available_suits = sorted(range(SUIT_COUNT), key=lambda suit: (suit_counts[suit], suit))
        for suit in available_suits[:count]:
            cards.append(rank * SUIT_COUNT + suit)
            suit_counts[suit] += 1
    if len(cards) != 7 or max(suit_counts) >= 5 or len(set(cards)) != 7:
        raise RuntimeError("invalid deterministic non-flush representative")
    return _seven_card_tuple(cards)


def _flush_representative(rank_mask: int) -> tuple[int, int, int, int, int, int, int]:
    """Construct a deterministic seven-card hand for one flush rank mask."""
    cards = [rank * SUIT_COUNT for rank in range(RANK_COUNT) if rank_mask & (1 << rank)]
    if len(cards) == 7:
        return _seven_card_tuple(cards)
    for rank in range(RANK_COUNT):
        for suit in range(1, SUIT_COUNT):
            card = rank * SUIT_COUNT + suit
            if card not in cards:
                cards.append(card)
            if len(cards) == 7:
                return _seven_card_tuple(cards)
    raise RuntimeError("could not construct deterministic flush representative")


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace one generated table file through a same-directory temporary file."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def _seven_card_tuple(cards: Sequence[int]) -> tuple[int, int, int, int, int, int, int]:
    return cards[0], cards[1], cards[2], cards[3], cards[4], cards[5], cards[6]
