"""Strict TOML configuration loading for Deep CFR runs."""

import tomllib
from collections.abc import Mapping
from pathlib import Path

from ac_cfr.common.config import DeepCFRImplementationId
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig

_FORMAT_VERSION = 2
_RUN_FIELDS = {"implementation", "checkpoint_interval"}
_TRAINING_FIELDS = {
    "iterations",
    "traversals_per_player",
    "advantage_reservoir_capacity",
    "strategy_reservoir_capacity",
    "advantage_training_steps",
    "strategy_training_steps",
    "advantage_batch_size",
    "strategy_batch_size",
    "learning_rate",
    "validation_fraction",
    "max_gradient_norm",
    "dropout_probability",
    "seed",
    "snapshot_iterations",
    "game_configuration_id",
    "model_config_id",
    "state_encoding_id",
    "optimizer_id",
}
_LEGACY_TRAINING_FIELDS = _TRAINING_FIELDS - {
    "advantage_batch_size",
    "strategy_batch_size",
    "game_configuration_id",
} | {"training_batch_size"}
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
    format_version = values["format_version"]
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise ValueError("Deep CFR configuration format_version is incompatible")
    if format_version not in (1, _FORMAT_VERSION):
        raise ValueError("Deep CFR configuration format_version is incompatible")

    run = _strict_table(values["run"], _RUN_FIELDS, "run")
    training_fields = _LEGACY_TRAINING_FIELDS if format_version == 1 else _TRAINING_FIELDS
    training = _strict_table(values["training"], training_fields, "training")
    if format_version == 1:
        batch_size = training.pop("training_batch_size")
        training["advantage_batch_size"] = batch_size
        training["strategy_batch_size"] = batch_size
        training["game_configuration_id"] = "leduc"
    runtime = _strict_table(values["runtime"], _RUNTIME_FIELDS, "runtime")
    _apply_overrides(run, training, runtime, overrides or {})
    try:
        checkpoint_interval = run["checkpoint_interval"]
        if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int):
            raise TypeError("checkpoint_interval must be an integer")
        return DeepCFRRunConfig(
            run_id=run_id,
            implementation=DeepCFRImplementationId(run["implementation"]),
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
