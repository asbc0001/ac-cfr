from pathlib import Path

import pytest

from ac_cfr.common.environment import (
    cpu_quota_count,
    effective_memory_bytes,
    effective_storage_remaining_bytes,
)


def test_cpu_quota_count_reads_cgroup_limits(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("1360000 100000\n", encoding="utf-8")

    assert cpu_quota_count(tmp_path) == pytest.approx(13.6)

    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    assert cpu_quota_count(tmp_path) is None


def test_effective_memory_bytes_respects_cgroup_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "memory.max").write_text("250000000000\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("10000000000\n", encoding="utf-8")
    host_memory = type(
        "Memory",
        (),
        {"available": 2_000_000_000_000, "total": 2_100_000_000_000},
    )
    monkeypatch.setattr("ac_cfr.common.environment.psutil.virtual_memory", lambda: host_memory)

    assert effective_memory_bytes(tmp_path) == (240_000_000_000, 250_000_000_000)


def test_effective_storage_remaining_bytes_subtracts_run_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "checkpoint.pt").write_bytes(b"x" * 30)
    monkeypatch.setattr(
        "ac_cfr.common.environment.shutil.disk_usage",
        lambda _: type("DiskUsage", (), {"free": 1_000}),
    )

    assert effective_storage_remaining_bytes(tmp_path, 100) == 70
