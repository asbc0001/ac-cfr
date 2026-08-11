"""Stable compatibility identifiers for tabular poker artefacts."""

from dataclasses import fields
from fractions import Fraction
from hashlib import sha256

import numpy as np

from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.games.tree import IndexedGameTree

ACTION_SPACE_ID = "poker"
TABULAR_MODEL_CONFIG_ID = "tabular_average_strategy"


def tree_compatibility_digest(tree: IndexedGameTree) -> str:
    """Hash the complete indexed-tree layout used to interpret tabular arrays."""
    if not isinstance(tree, IndexedGameTree):
        raise TypeError("tree must be an IndexedGameTree")

    digest = sha256()
    digest.update(tree.game_id.value.encode("utf-8"))
    for field in fields(tree):
        if field.name == "game_id":
            continue
        array = getattr(tree, field.name)
        if not isinstance(array, np.ndarray):
            raise TypeError(f"tree field {field.name} must be a NumPy array")
        little_endian_dtype = array.dtype.newbyteorder("<")
        canonical_array = np.asarray(array, dtype=little_endian_dtype)
        digest.update(field.name.encode("ascii"))
        digest.update(canonical_array.dtype.str.encode("ascii"))
        digest.update(str(canonical_array.shape).encode("ascii"))
        digest.update(canonical_array.tobytes(order="C"))
    return digest.hexdigest()


def holdem_compatibility_digest(config: HoldemConfig) -> str:
    """Hash every rule needed to interpret an on-demand Hold'em strategy."""
    if not isinstance(config, HoldemConfig):
        raise TypeError("config must be a HoldemConfig")
    digest = sha256()
    values = (
        int(config.start_street),
        config.max_bets_per_round,
        Fraction(config.small_blind),
        Fraction(config.big_blind),
        Fraction(config.small_bet),
        Fraction(config.big_bet),
        config.button_player,
        config.synthetic_flop_start,
    )
    digest.update(repr(values).encode("ascii"))
    return digest.hexdigest()
