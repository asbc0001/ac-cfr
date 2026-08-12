"""Deterministic seed derivation for independent random-number streams."""

from dataclasses import dataclass
from enum import StrEnum
from hashlib import blake2b
from random import Random

_MAX_ROOT_SEED = (1 << 128) - 1
_MAX_STREAM_INDEX = (1 << 64) - 1
# Derived seeds fit libraries that accept signed 64-bit seed values.
_DERIVED_SEED_MASK = (1 << 63) - 1


class RngStream(StrEnum):
    """Names for logically independent sources of randomness."""

    CHANCE = "chance"
    POLICY = "policy"
    BOOTSTRAP = "bootstrap"
    RESERVOIR = "reservoir"
    NETWORK = "network"
    NETWORK_TRAINING = "network_training"
    DATA_LOADER = "data_loader"
    WORKER = "worker"


@dataclass(frozen=True, slots=True)
class SeedDeriver:
    """Derive stable child seeds from one reproducible root seed."""

    root_seed: int

    def __post_init__(self) -> None:
        _validate_bounded_integer("root_seed", self.root_seed, _MAX_ROOT_SEED)

    def derive(self, stream: RngStream, index: int = 0) -> int:
        """Return a deterministic seed for a named stream and optional worker index.

        BLAKE2b is fast, available in the standard library, and supports a
        personalization label. Unlike Python's built-in hash, its result is stable
        across processes. It is used here for reproducibility, not security.
        """
        if not isinstance(stream, RngStream):
            raise TypeError("stream must be an RngStream")
        _validate_bounded_integer("index", index, _MAX_STREAM_INDEX)

        stream_name = stream.value.encode("ascii")
        payload = b"".join(
            (
                self.root_seed.to_bytes(16, byteorder="big"),
                len(stream_name).to_bytes(1, byteorder="big"),
                stream_name,
                index.to_bytes(8, byteorder="big"),
            )
        )
        digest = blake2b(payload, digest_size=8, person=b"ac_cfr_rng").digest()
        return int.from_bytes(digest, byteorder="big") & _DERIVED_SEED_MASK

    def python_rng(self, stream: RngStream, index: int = 0) -> Random:
        """Create an independent Python RNG whose state can be saved and restored."""
        return Random(self.derive(stream, index))


def _validate_bounded_integer(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
