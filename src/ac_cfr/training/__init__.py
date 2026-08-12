"""Reusable poker-solver training orchestration."""

from importlib import import_module
from typing import TYPE_CHECKING

from ac_cfr.training.config import DeepCFRRuntimeConfig, DeepCFRTrainingConfig
from ac_cfr.training.reservoirs import AdvantageSample, StrategySample, UniformReservoir

if TYPE_CHECKING:
    from ac_cfr.training.runner import (
        TabularTrainingConfig,
        TrainingOutcome,
        resume_tabular_training,
        start_tabular_training,
    )

_RUNNER_EXPORTS = frozenset(
    {
        "TabularTrainingConfig",
        "TrainingOutcome",
        "resume_tabular_training",
        "start_tabular_training",
    }
)

__all__ = (
    "AdvantageSample",
    "DeepCFRTrainingConfig",
    "DeepCFRRuntimeConfig",
    "StrategySample",
    "TabularTrainingConfig",
    "TrainingOutcome",
    "UniformReservoir",
    "resume_tabular_training",
    "start_tabular_training",
)


def __getattr__(name: str) -> object:
    """Load tabular orchestration only when its public export is requested."""
    if name not in _RUNNER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("ac_cfr.training.runner"), name)
    globals()[name] = value
    return value
