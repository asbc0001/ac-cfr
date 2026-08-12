"""Provider-independent runtime environment and hardware metadata."""

import os
import platform
import shutil
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


def memory_limit_bytes(cgroup_root: Path = _CGROUP_ROOT) -> int | None:
    """Return the cgroup memory limit, if one is configured."""
    limit = _read_cgroup_integer(cgroup_root / "memory.max")
    if limit is not None:
        return limit
    return _read_cgroup_integer(cgroup_root / "memory" / "memory.limit_in_bytes")


def memory_usage_bytes(cgroup_root: Path = _CGROUP_ROOT) -> int | None:
    """Return current cgroup memory use, if it is available."""
    usage = _read_cgroup_integer(cgroup_root / "memory.current")
    if usage is not None:
        return usage
    return _read_cgroup_integer(cgroup_root / "memory" / "memory.usage_in_bytes")


def effective_memory_bytes(cgroup_root: Path = _CGROUP_ROOT) -> tuple[int, int]:
    """Return available and total memory constrained by the current cgroup."""
    host_memory = psutil.virtual_memory()
    limit = memory_limit_bytes(cgroup_root)
    if limit is None:
        return int(host_memory.available), int(host_memory.total)
    usage = memory_usage_bytes(cgroup_root)
    cgroup_available = limit if usage is None else max(0, limit - usage)
    return min(int(host_memory.available), cgroup_available), min(int(host_memory.total), limit)


def effective_storage_remaining_bytes(path: Path, storage_budget_bytes: int | None) -> int:
    """Return backing free space capped by the remaining configured run budget."""
    filesystem_free = shutil.disk_usage(path).free
    if storage_budget_bytes is None:
        return filesystem_free
    used_bytes = sum(
        entry.stat(follow_symlinks=False).st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )
    return min(filesystem_free, max(0, storage_budget_bytes - used_bytes))


def environment_record(
    *packages: str,
    device: str | None = None,
) -> dict[str, object]:
    """Return reproducible source, software, CPU, memory, and optional GPU metadata."""
    is_wsl2 = "microsoft" in platform.release().lower()
    host_memory = psutil.virtual_memory()
    available_memory, total_memory = effective_memory_bytes()
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
        "available_memory_bytes": available_memory,
        "total_memory_bytes": total_memory,
        "host_available_memory_bytes": host_memory.available,
        "host_total_memory_bytes": host_memory.total,
        "memory_limit_bytes": memory_limit_bytes(),
        "memory_usage_bytes": memory_usage_bytes(),
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


def _read_cgroup_integer(path: Path) -> int | None:
    """Read one positive cgroup integer, treating missing and unlimited values as absent."""
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value == "max":
            return None
        parsed = int(value)
    except (OSError, UnicodeError, ValueError):
        return None
    return parsed if parsed > 0 else None
