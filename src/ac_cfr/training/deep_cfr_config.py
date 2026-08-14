"""Strict TOML configuration loading for Deep CFR runs."""

import math
import tomllib
from collections.abc import Mapping
from pathlib import Path

from ac_cfr.common.config import DeepCFRImplementationId
from ac_cfr.common.environment import effective_cpu_count
from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig

_FORMAT_VERSION = 5
_RUN_FIELDS = {"implementation", "checkpoint_interval", "checkpoint_retention"}
_VERSION_TWO_RUN_FIELDS = _RUN_FIELDS - {"checkpoint_retention"}
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
    "validation_split_id",
    "max_gradient_norm",
    "dropout_probability",
    "seed",
    "snapshot_iterations",
    "game_configuration_id",
    "model_config_id",
    "state_encoding_id",
    "optimizer_id",
}
_VERSION_FOUR_TRAINING_FIELDS = _TRAINING_FIELDS - {"validation_split_id"}
_LEGACY_TRAINING_FIELDS = _VERSION_FOUR_TRAINING_FIELDS - {
    "advantage_batch_size",
    "strategy_batch_size",
    "game_configuration_id",
} | {"training_batch_size"}
_RUNTIME_FIELDS = {
    "inference_batch_size",
    "cpu_threads",
    "device",
    "traversal_workers",
    "storage_budget_bytes",
}
_VERSION_THREE_RUNTIME_FIELDS = _RUNTIME_FIELDS - {"storage_budget_bytes"}
_VERSION_TWO_RUNTIME_FIELDS = _VERSION_THREE_RUNTIME_FIELDS - {"traversal_workers"}


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
    if format_version not in (1, 2, 3, 4, _FORMAT_VERSION):
        raise ValueError("Deep CFR configuration format_version is incompatible")

    run = _strict_table(
        values["run"], _RUN_FIELDS if format_version >= 3 else _VERSION_TWO_RUN_FIELDS, "run"
    )
    run.setdefault("checkpoint_retention", 2)
    training_fields = (
        _LEGACY_TRAINING_FIELDS
        if format_version == 1
        else _VERSION_FOUR_TRAINING_FIELDS
        if format_version <= 4
        else _TRAINING_FIELDS
    )
    training = _strict_table(values["training"], training_fields, "training")
    if format_version == 1:
        batch_size = training.pop("training_batch_size")
        training["advantage_batch_size"] = batch_size
        training["strategy_batch_size"] = batch_size
        training["game_configuration_id"] = "leduc"
    training.setdefault("validation_split_id", "sample")
    runtime_fields = (
        _RUNTIME_FIELDS
        if format_version >= 4
        else _VERSION_THREE_RUNTIME_FIELDS
        if format_version == 3
        else _VERSION_TWO_RUNTIME_FIELDS
    )
    runtime = _strict_table(values["runtime"], runtime_fields, "runtime")
    runtime.setdefault("traversal_workers", 1)
    runtime.setdefault("storage_budget_bytes", None)
    _apply_overrides(run, training, runtime, overrides or {})
    _resolve_traversal_workers(runtime)
    try:
        checkpoint_interval = run["checkpoint_interval"]
        if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int):
            raise TypeError("checkpoint_interval must be an integer")
        checkpoint_retention = run["checkpoint_retention"]
        if isinstance(checkpoint_retention, bool) or not isinstance(checkpoint_retention, int):
            raise TypeError("checkpoint_retention must be an integer")
        return DeepCFRRunConfig(
            run_id=run_id,
            implementation=DeepCFRImplementationId(run["implementation"]),
            checkpoint_interval=checkpoint_interval,
            checkpoint_retention=checkpoint_retention,
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


def _resolve_traversal_workers(runtime: dict[str, object]) -> None:
    """Resolve zero once to a bounded machine-aware worker count."""
    workers = runtime["traversal_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ValueError("traversal_workers must be an integer")
    if workers != 0:
        return
    # Preserve capacity for the coordinator, reservoir admission, and GPU feeding.
    available_cpus = effective_cpu_count()
    runtime["traversal_workers"] = min(14, max(1, math.floor(available_cpus * 0.8)))
