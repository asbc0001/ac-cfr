"""Strict TOML configuration loading for Hold'em games."""

import tomllib
from pathlib import Path

from ac_cfr.common.config import GameConfigurationId, StateEncodingId
from ac_cfr.games.holdem.engine import HoldemConfig, Street

_FORMAT_VERSION = 1
_TOP_LEVEL_FIELDS = {
    "format_version",
    "configuration_id",
    "state_encoding_id",
    "rules",
}
_RULE_FIELDS = {
    "start_street",
    "max_bets_per_round",
    "small_blind",
    "big_blind",
    "small_bet",
    "big_bet",
    "button_player",
    "synthetic_flop_start",
}
_STREETS = {
    "preflop": Street.PREFLOP,
    "flop": Street.FLOP,
}


def load_holdem_config(path: Path) -> HoldemConfig:
    """Load and validate one versioned Hold'em game preset."""
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Hold'em configuration is unreadable: {path}") from error

    if set(values) != _TOP_LEVEL_FIELDS:
        raise ValueError("Hold'em configuration fields are incompatible")
    if values["format_version"] != _FORMAT_VERSION:
        raise ValueError("Hold'em configuration format_version is incompatible")

    rules = values["rules"]
    if not isinstance(rules, dict) or set(rules) != _RULE_FIELDS:
        raise ValueError("Hold'em rules fields are incompatible")

    try:
        start_street = _STREETS[rules["start_street"]]
        configuration_id = GameConfigurationId(values["configuration_id"])
        state_encoding_id = StateEncodingId(values["state_encoding_id"])
        resolved_rules = {**rules, "start_street": start_street}
        config = HoldemConfig(**resolved_rules)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Hold'em configuration values are invalid") from error

    if config.configuration_id is not configuration_id:
        raise ValueError("Hold'em configuration_id does not match its rules")
    if config.state_encoding_id is not state_encoding_id:
        raise ValueError("Hold'em state_encoding_id does not match its rules")
    return config
