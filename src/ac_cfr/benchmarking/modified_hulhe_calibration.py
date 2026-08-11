"""Collect-once neural-fit and GPU batch calibration for modified HULHE."""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median
from threading import Event, Thread
from time import perf_counter
from typing import Final

import numpy as np
import psutil
import torch
from numpy.typing import NDArray

from ac_cfr.benchmarking.harness import environment_record, report_progress
from ac_cfr.common.config import DeepCFRImplementationId, GameConfigurationId
from ac_cfr.common.provenance import code_revision
from ac_cfr.common.rng import RngStream, SeedDeriver
from ac_cfr.games.holdem.engine import HoldemConfig
from ac_cfr.models import DeepCFRNetwork, build_deep_cfr_network
from ac_cfr.persistence.files import write_csv, write_json
from ac_cfr.solvers import DeepCFR
from ac_cfr.solvers.naive_deep_cfr import (
    NetworkFitPoint,
    train_network_tensor_milestones,
)
from ac_cfr.training.deep_cfr_config import load_deep_cfr_run_config
from ac_cfr.training.deep_cfr_runner import DeepCFRRunConfig

CALIBRATION_ID = "modified_hulhe_deep_cfr"
DEFAULT_PRESET = Path("configs/deep_cfr/modified_hulhe_calibration.toml")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/modified-hulhe-calibration")
DEFAULT_FIT_MILESTONES: Final = (4_000, 8_000, 16_000, 32_000)
DEFAULT_BATCH_WARMUP_STEPS = 50
DEFAULT_BATCH_TIMED_STEPS = 200
DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS = 0.2
_FIRST_ITERATION_NETWORK_SEED_BASE = 3

_FIT_FIELDS: Final = (
    "update_steps",
    "segment_training_seconds",
    "cumulative_training_seconds",
    "segment_updates_per_second",
    "training_samples",
    "validation_samples",
    "training_loss",
    "validation_loss",
)
_BATCH_FIELDS: Final = (
    "batch_size",
    "warmup_steps",
    "timed_steps",
    "timed_seconds",
    "updates_per_second",
    "median_gpu_utilisation_percent",
    "gpu_utilisation_source",
    "peak_vram_allocated_bytes",
    "peak_vram_reserved_bytes",
    "reservoir_digest",
)


@dataclass(frozen=True, slots=True)
class ModifiedHulheCalibrationSchedule:
    """Bounded diagnostic work applied to one frozen advantage reservoir."""

    player: int = 0
    fit_milestones: tuple[int, ...] = DEFAULT_FIT_MILESTONES
    batch_sizes: tuple[int, ...] = (10_000, 20_000)
    batch_warmup_steps: int = DEFAULT_BATCH_WARMUP_STEPS
    batch_timed_steps: int = DEFAULT_BATCH_TIMED_STEPS
    gpu_sample_interval_seconds: float = DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        _validate_increasing_positive_integers("fit_milestones", self.fit_milestones)
        _validate_increasing_positive_integers("batch_sizes", self.batch_sizes)
        _validate_positive_integer("batch_warmup_steps", self.batch_warmup_steps)
        _validate_positive_integer("batch_timed_steps", self.batch_timed_steps)
        if (
            isinstance(self.gpu_sample_interval_seconds, bool)
            or not isinstance(self.gpu_sample_interval_seconds, (int, float))
            or self.gpu_sample_interval_seconds <= 0.0
        ):
            raise ValueError("gpu_sample_interval_seconds must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible schedule values."""
        return {
            "player": self.player,
            "fit_milestones": list(self.fit_milestones),
            "batch_sizes": list(self.batch_sizes),
            "batch_warmup_steps": self.batch_warmup_steps,
            "batch_timed_steps": self.batch_timed_steps,
            "gpu_sample_interval_seconds": self.gpu_sample_interval_seconds,
        }


def run_modified_hulhe_calibration(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    preset_path: Path = DEFAULT_PRESET,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run the production collect-once calibration declared by a TOML preset."""
    config = load_deep_cfr_run_config(
        preset_path,
        run_id="modified-hulhe-calibration",
    )
    schedule = ModifiedHulheCalibrationSchedule(
        batch_sizes=(
            config.training.advantage_batch_size,
            config.training.advantage_batch_size * 2,
        )
    )
    return calibrate_modified_hulhe(
        config,
        schedule,
        output_directory,
        preset_path=preset_path,
        progress_callback=progress_callback,
    )


def calibrate_modified_hulhe(
    config: DeepCFRRunConfig,
    schedule: ModifiedHulheCalibrationSchedule,
    output_directory: Path,
    *,
    preset_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Collect one reservoir, fit one network, and compare batches on fixed data."""
    _validate_calibration_config(config)
    if not isinstance(schedule, ModifiedHulheCalibrationSchedule):
        raise TypeError("schedule must be a ModifiedHulheCalibrationSchedule")
    if output_directory.exists() and (
        not output_directory.is_dir() or any(output_directory.iterdir())
    ):
        raise FileExistsError(
            f"calibration output path is not an empty directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.runtime.cpu_threads)
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")

    solver = DeepCFR(HoldemConfig.modified(), config.training, config.runtime)
    report_progress(progress_callback, "collecting one fixed advantage reservoir")
    collection_started = perf_counter()
    solver.collect_calibration_traversals(schedule.player)
    collection_seconds = perf_counter() - collection_started
    reservoir = solver.advantage_reservoirs[schedule.player]
    states, masks, sample_iterations, advantages = reservoir.arrays
    reservoir_digest = _array_digest(states, masks, sample_iterations, advantages)
    network_seed_index = _FIRST_ITERATION_NETWORK_SEED_BASE + schedule.player

    tensors = (
        torch.from_numpy(states),
        torch.from_numpy(masks),
        torch.from_numpy(sample_iterations.astype(np.float32)),
        torch.from_numpy(advantages),
    )
    report_progress(progress_callback, "training one continuous milestone diagnostic fit")
    fit_network = _new_network(config, network_seed_index)
    _reset_cuda_peak_memory(config.runtime.device)
    fit_sampler = _GpuUtilisationSampler(
        config.runtime.device,
        schedule.gpu_sample_interval_seconds,
    )
    fit_sampler.start()
    try:
        fit_points = _fit_points(
            fit_network,
            tensors,
            config,
            schedule.fit_milestones,
            config.training.advantage_batch_size,
            network_seed_index,
        )
    finally:
        fit_sampler.stop()
    _require_gpu_utilisation(config.runtime.device, fit_sampler)
    fit_records = _fit_records(fit_points)
    fit_peak_allocated = _cuda_peak_memory(config.runtime.device, reserved=False)
    fit_peak_reserved = _cuda_peak_memory(config.runtime.device, reserved=True)
    del fit_network
    _release_cuda_cache(config.runtime.device)

    batch_records: list[dict[str, object]] = []
    for batch_size in schedule.batch_sizes:
        report_progress(progress_callback, f"benchmarking warmed batch size {batch_size:,}")
        network = _new_network(config, network_seed_index)
        _reset_cuda_peak_memory(config.runtime.device)
        sampler = _GpuUtilisationSampler(
            config.runtime.device,
            schedule.gpu_sample_interval_seconds,
        )

        try:
            points = _fit_points(
                network,
                tensors,
                config,
                (
                    schedule.batch_warmup_steps,
                    schedule.batch_warmup_steps + schedule.batch_timed_steps,
                ),
                batch_size,
                network_seed_index,
                milestone_callback=_batch_sampler_callback(
                    sampler,
                    schedule.batch_warmup_steps,
                    schedule.batch_timed_steps,
                ),
            )
        finally:
            sampler.stop()
        _require_gpu_utilisation(config.runtime.device, sampler)
        timed_seconds = points[1].training_seconds - points[0].training_seconds
        batch_records.append(
            {
                "batch_size": batch_size,
                "warmup_steps": schedule.batch_warmup_steps,
                "timed_steps": schedule.batch_timed_steps,
                "timed_seconds": timed_seconds,
                "updates_per_second": schedule.batch_timed_steps / timed_seconds,
                "median_gpu_utilisation_percent": sampler.median_utilisation,
                "gpu_utilisation_source": sampler.source,
                "peak_vram_allocated_bytes": _cuda_peak_memory(
                    config.runtime.device, reserved=False
                ),
                "peak_vram_reserved_bytes": _cuda_peak_memory(config.runtime.device, reserved=True),
                "reservoir_digest": reservoir_digest,
            }
        )
        del network
        _release_cuda_cache(config.runtime.device)

    if _array_digest(states, masks, sample_iterations, advantages) != reservoir_digest:
        raise RuntimeError("calibration reservoir changed after collection")
    fit_path = output_directory / "network_fit.csv"
    batch_path = output_directory / "batch_throughput.csv"
    write_csv(fit_path, _FIT_FIELDS, fit_records)
    write_csv(batch_path, _BATCH_FIELDS, batch_records)
    metadata_path = output_directory / "calibration.json"
    write_json(
        metadata_path,
        {
            "about": (
                "Collect-once modified-HULHE advantage-network fit and warmed GPU batch "
                "calibration. This is not a complete Deep CFR training run."
            ),
            "calibration_id": CALIBRATION_ID,
            "code_revision": code_revision(),
            "preset": None if preset_path is None else str(preset_path),
            "resolved_run_config": config.to_dict(),
            "schedule": schedule.to_dict(),
            "collection": {
                "traversing_player": schedule.player,
                "traversals": config.training.traversals_per_player,
                "seconds": collection_seconds,
                "traversals_per_second": (
                    config.training.traversals_per_player / collection_seconds
                ),
                "retained_advantage_samples": len(reservoir),
                "advantage_samples_seen": reservoir.samples_seen,
                "bytes_per_sample": reservoir.bytes_per_sample,
                "allocated_reservoir_bytes": reservoir.resident_bytes,
                "allocated_all_reservoir_bytes": sum(
                    memory.resident_bytes
                    for memory in (*solver.advantage_reservoirs, solver.strategy_reservoir)
                ),
                "retained_strategy_samples": len(solver.strategy_reservoir),
                "strategy_samples_seen": solver.strategy_reservoir.samples_seen,
                "process_rss_bytes_after_collection": psutil.Process().memory_info().rss,
                "reservoir_digest": reservoir_digest,
            },
            "fit": {
                "fresh_network": True,
                "single_continuous_optimizer": True,
                "fixed_train_validation_split": True,
                "batch_size": config.training.advantage_batch_size,
                "median_gpu_utilisation_percent": fit_sampler.median_utilisation,
                "gpu_utilisation_source": fit_sampler.source,
                "peak_vram_allocated_bytes": fit_peak_allocated,
                "peak_vram_reserved_bytes": fit_peak_reserved,
            },
            "batch_comparison": {
                "fresh_identically_initialised_network_per_batch": True,
                "same_reservoir_and_split_seed": True,
                "training_time_excludes_warmup_and_loss_evaluation": True,
            },
            "environment": environment_record(
                "torch", "numpy", "numba", "psutil", device=config.runtime.device
            ),
            "files": {
                "network_fit": fit_path.name,
                "batch_throughput": batch_path.name,
            },
        },
    )
    return metadata_path


def _fit_points(
    network: DeepCFRNetwork,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    config: DeepCFRRunConfig,
    milestones: tuple[int, ...],
    batch_size: int,
    seed_index: int,
    milestone_callback: Callable[[NetworkFitPoint], None] | None = None,
) -> tuple[NetworkFitPoint, ...]:
    """Apply the production advantage-loss semantics to one fixed tensor dataset."""
    states, masks, sample_iterations, advantages = tensors
    seed_deriver = SeedDeriver(config.training.seed)
    return train_network_tensor_milestones(
        network=network,
        states=states,
        action_masks=masks,
        targets=advantages,
        sample_iterations=sample_iterations,
        current_iteration=1,
        update_milestones=milestones,
        batch_size=batch_size,
        learning_rate=config.training.learning_rate,
        data_seed=seed_deriver.derive(RngStream.DATA_LOADER, seed_index),
        training_seed=seed_deriver.derive(RngStream.NETWORK_TRAINING, seed_index),
        strategy_targets=False,
        validation_fraction=config.training.validation_fraction,
        max_gradient_norm=config.training.max_gradient_norm,
        milestone_callback=milestone_callback,
    )


def _new_network(config: DeepCFRRunConfig, seed_index: int) -> DeepCFRNetwork:
    """Construct one reproducibly initialised configured network on the target device."""
    seed = SeedDeriver(config.training.seed).derive(RngStream.NETWORK, seed_index)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build_deep_cfr_network(
            config.training.model_config_id,
            dropout_probability=config.training.dropout_probability,
        ).to(config.runtime.device)


def _fit_records(points: tuple[NetworkFitPoint, ...]) -> list[dict[str, object]]:
    """Convert cumulative fit observations into compact milestone rows."""
    records: list[dict[str, object]] = []
    prior_steps = 0
    prior_seconds = 0.0
    for point in points:
        segment_steps = point.update_steps - prior_steps
        segment_seconds = point.training_seconds - prior_seconds
        if segment_seconds <= 0.0:
            raise RuntimeError("network-fit timing must be positive")
        records.append(
            {
                "update_steps": point.update_steps,
                "segment_training_seconds": segment_seconds,
                "cumulative_training_seconds": point.training_seconds,
                "segment_updates_per_second": segment_steps / segment_seconds,
                "training_samples": point.training_samples,
                "validation_samples": point.validation_samples,
                "training_loss": point.training_loss,
                "validation_loss": point.validation_loss,
            }
        )
        prior_steps = point.update_steps
        prior_seconds = point.training_seconds
    return records


def _array_digest(*arrays: NDArray[np.generic]) -> str:
    """Return one deterministic digest proving every fit used unchanged samples."""
    digest = sha256()
    for array in arrays:
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _batch_sampler_callback(
    sampler: "_GpuUtilisationSampler",
    warmup_steps: int,
    timed_steps: int,
) -> Callable[[NetworkFitPoint], None]:
    """Start GPU sampling after warmup and stop after the timed update segment."""
    final_steps = warmup_steps + timed_steps

    def observe(point: NetworkFitPoint) -> None:
        if point.update_steps == warmup_steps:
            sampler.start()
        elif point.update_steps == final_steps:
            sampler.stop()

    return observe


class _GpuUtilisationSampler:
    """Sample CUDA utilisation without introducing another runtime dependency."""

    def __init__(self, device: str, interval_seconds: float) -> None:
        self._enabled = device == "cuda"
        self._device = torch.device(device)
        self._interval_seconds = interval_seconds
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._values: list[float] = []
        self._source: str | None = None

    @property
    def median_utilisation(self) -> float | None:
        """Return median sampled GPU utilisation, or None when unavailable."""
        return median(self._values) if self._values else None

    @property
    def source(self) -> str | None:
        """Return the successful GPU-utilisation measurement backend."""
        return self._source

    def start(self) -> None:
        """Start background sampling only for CUDA calibration."""
        if not self._enabled:
            return
        self._thread = Thread(target=self._sample, name="gpu-utilisation-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop and join the sampler before its values are read."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _sample(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            value = self._sample_once()
            if value is None:
                return
            self._values.append(value)

    def _sample_once(self) -> float | None:
        try:
            value = float(torch.cuda.utilization(self._device))
            self._source = "torch.cuda.utilization"
            return value
        except (ImportError, RuntimeError):
            pass
        device_index = (
            self._device.index if self._device.index is not None else torch.cuda.current_device()
        )
        try:
            completed = subprocess.run(
                (
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(device_index),
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=max(1.0, self._interval_seconds * 4.0),
            )
            value = float(completed.stdout.strip().splitlines()[0])
            self._source = "nvidia-smi"
            return value
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None


def _validate_calibration_config(config: DeepCFRRunConfig) -> None:
    if not isinstance(config, DeepCFRRunConfig):
        raise TypeError("config must be a DeepCFRRunConfig")
    if config.implementation is not DeepCFRImplementationId.OPTIMISED:
        raise ValueError("modified-HULHE calibration requires the optimised implementation")
    if config.training.game_configuration_id is not GameConfigurationId.MODIFIED_HULHE:
        raise ValueError("calibration requires the modified-HULHE game configuration")


def _validate_increasing_positive_integers(name: str, values: tuple[int, ...]) -> None:
    if (
        not isinstance(values, tuple)
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values
        )
        or tuple(sorted(set(values))) != values
    ):
        raise ValueError(f"{name} must be unique, positive, and increasing")


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _reset_cuda_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _cuda_peak_memory(device: str, *, reserved: bool) -> int | None:
    if device != "cuda":
        return None
    cuda_device = torch.device(device)
    measure = torch.cuda.max_memory_reserved if reserved else torch.cuda.max_memory_allocated
    return int(measure(cuda_device))


def _release_cuda_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def _require_gpu_utilisation(device: str, sampler: _GpuUtilisationSampler) -> None:
    if device == "cuda" and sampler.median_utilisation is None:
        raise RuntimeError("GPU utilisation could not be measured through PyTorch or nvidia-smi")
