from pathlib import Path

import pytest

from ac_cfr.common.environment import cpu_quota_count


def test_cpu_quota_count_reads_cgroup_limits(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("1360000 100000\n", encoding="utf-8")

    assert cpu_quota_count(tmp_path) == pytest.approx(13.6)

    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    assert cpu_quota_count(tmp_path) is None
