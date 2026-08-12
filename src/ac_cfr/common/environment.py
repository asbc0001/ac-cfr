"""Provider-independent runtime environment and hardware metadata."""

import os
import platform
import socket
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil
import torch

from ac_cfr.common.provenance import code_revision

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def cpu_affinity_count() -> int:
    """Return the number of logical CPUs on which this process may run."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def cpu_quota_count(cgroup_root: Path = _CGROUP_ROOT) -> float | None:
    """Return the cgroup CPU quota as an equivalent CPU count, if limited."""
    quota = _read_cpu_quota(cgroup_root / "cpu.max")
    if quota is not None:
        return quota
    return _read_cpu_quota(
        cgroup_root / "cpu" / "cpu.cfs_quota_us",
        cgroup_root / "cpu" / "cpu.cfs_period_us",
    )


def effective_cpu_count() -> float:
    """Return usable CPU capacity after affinity and cgroup limits."""
    capacities = [float(os.cpu_count() or 1), float(cpu_affinity_count())]
    quota = cpu_quota_count()
    if quota is not None:
        capacities.append(quota)
    return min(capacities)


def environment_record(
    *packages: str,
    device: str | None = None,
) -> dict[str, object]:
    """Return reproducible source, software, CPU, memory, and optional GPU metadata."""
    is_wsl2 = "microsoft" in platform.release().lower()
    record: dict[str, object] = {
        "code_revision": code_revision(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "available_cpu_count": cpu_affinity_count(),
        "cpu_quota_count": cpu_quota_count(),
        "effective_cpu_count": effective_cpu_count(),
        "available_memory_bytes": psutil.virtual_memory().available,
        "total_memory_bytes": psutil.virtual_memory().total,
        "wsl2": is_wsl2,
        "wsl_config_paths": (
            sorted(str(path) for path in Path("/mnt/c/Users").glob("*/.wslconfig"))
            if is_wsl2
            else []
        ),
        "executable": sys.executable,
    }
    for package in packages:
        try:
            record[package] = version(package)
        except PackageNotFoundError:
            record[package] = "unavailable"
    if device is not None:
        record["device"] = device
    if device == "cuda" and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        record.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_index": device_index,
                "cuda_device_name": properties.name,
                "cuda_device_total_memory_bytes": properties.total_memory,
                "cuda_compute_capability": [properties.major, properties.minor],
            }
        )
    return record


def _read_cpu_quota(quota_path: Path, period_path: Path | None = None) -> float | None:
    """Read either a cgroup v2 cpu.max file or the equivalent v1 files."""
    try:
        if period_path is None:
            quota_value, period_value = quota_path.read_text(encoding="utf-8").split()
            if quota_value == "max":
                return None
        else:
            quota_value = quota_path.read_text(encoding="utf-8").strip()
            period_value = period_path.read_text(encoding="utf-8").strip()
        quota = int(quota_value)
        period = int(period_value)
    except (OSError, UnicodeError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period
