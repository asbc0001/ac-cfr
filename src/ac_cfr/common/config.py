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
    LEDUC_NEURAL = "leduc_neural"
    HOLD_EM = "holdem"


class ModelConfigId(StrEnum):
    """Identifiers for reconstructable neural-network configurations."""

    LEDUC_DEEP_CFR_BASELINE = "leduc_deep_cfr"
    LEDUC_DEEP_CFR_SMALL = "leduc_deep_cfr_small"

    # Retain the original name for existing checkpoints and snapshots.
    LEDUC_DEEP_CFR = LEDUC_DEEP_CFR_BASELINE


class OptimizerId(StrEnum):
    """Identifiers for configured neural-network optimisers."""

    ADAM = "adam"
