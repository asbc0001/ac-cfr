"""Provider-independent runtime environment and hardware metadata."""

import platform
import socket
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil
import torch

from ac_cfr.common.provenance import code_revision


def environment_record(
    *packages: str,
    device: str | None = None,
) -> dict[str, object]:
    """Return reproducible source, software, CPU, memory, and optional GPU metadata."""
    process = psutil.Process()
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
        "available_cpu_count": (
            len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else None
        ),
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
