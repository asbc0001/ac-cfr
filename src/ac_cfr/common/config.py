"""Stable configuration and state-encoding compatibility identifiers."""

from enum import StrEnum


class GameConfigurationId(StrEnum):
    """Identifiers for exact canonical game configurations."""

    KUHN = "kuhn"
    LEDUC = "leduc"
    HULHE = "hulhe"
    MODIFIED_HULHE = "modified_hulhe"


class StateEncodingId(StrEnum):
    """Identifiers for current player-visible state encodings.

    An incompatible successor receives a ``_v2`` suffix while these original
    identifiers remain unchanged.
    """

    KUHN = "kuhn"
    LEDUC = "leduc"
    HOLD_EM = "holdem"
