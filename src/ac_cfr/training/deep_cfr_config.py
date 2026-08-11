"""Strict TOML configuration loading for Deep CFR runs."""

import tomllib
from collections.abc import Mapping
from pathlib import Path

from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig

_FORMAT_VERSION = 1
_RUN_FIELDS = {"checkpoint_interval"}
_TRAINING_FIELDS = {
    "iterations",
    "traversals_per_player",
    "advantage_reservoir_capacity",
    "strategy_reservoir_capacity",
    "advantage_training_steps",
    "strategy_training_steps",
    "training_batch_size",
    "learning_rate",
    "validation_fraction",
    "max_gradient_norm",
    "dropout_probability",
    "seed",
    "snapshot_iterations",
    "model_config_id",
    "state_encoding_id",
    "optimizer_id",
}
_RUNTIME_FIELDS = {"inference_batch_size", "cpu_threads", "device"}


def load_deep_cfr_run_config(
    path: Path,
    *,
    run_id: str,
    overrides: Mapping[str, object] | None = None,
) -> DeepCFRRunConfig:
    """Load a preset, apply supplied overrides, and return one resolved configuration."""
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Deep CFR configuration is unreadable: {path}") from error
    if set(values) != {"format_version", "run", "training", "runtime"}:
        raise ValueError("Deep CFR configuration sections are incompatible")
    if values["format_version"] != _FORMAT_VERSION:
        raise ValueError("Deep CFR configuration format_version is incompatible")

    run = _strict_table(values["run"], _RUN_FIELDS, "run")
    training = _strict_table(values["training"], _TRAINING_FIELDS, "training")
    runtime = _strict_table(values["runtime"], _RUNTIME_FIELDS, "runtime")
    _apply_overrides(run, training, runtime, overrides or {})
    try:
        checkpoint_interval = run["checkpoint_interval"]
        if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int):
            raise TypeError("checkpoint_interval must be an integer")
        return DeepCFRRunConfig(
            run_id=run_id,
            checkpoint_interval=checkpoint_interval,
            training=DeepCFRTrainingConfig.from_dict(training),
            runtime=DeepCFRRuntimeConfig.from_dict(runtime),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Deep CFR configuration values are invalid") from error


def _strict_table(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Deep CFR {name} configuration fields are incompatible")
    return value.copy()


def _apply_overrides(
    run: dict[str, object],
    training: dict[str, object],
    runtime: dict[str, object],
    overrides: Mapping[str, object],
) -> None:
    known_fields = _RUN_FIELDS | _TRAINING_FIELDS | _RUNTIME_FIELDS
    unknown_fields = set(overrides) - known_fields
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"unknown Deep CFR configuration overrides: {names}")
    for name, value in overrides.items():
        if name in run:
            run[name] = value
        elif name in training:
            training[name] = (
                list(value) if name == "snapshot_iterations" and isinstance(value, tuple) else value
            )
        else:
            runtime[name] = value
