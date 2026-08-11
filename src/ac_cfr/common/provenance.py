"""Small helpers for recording source-code provenance."""

import subprocess


def code_revision() -> str:
    """Return the current Git revision, marking uncommitted work explicitly."""
    try:
        revision_result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = revision_result.stdout.strip()
    if not revision:
        return "unknown"
    return f"{revision}-dirty" if status_result.stdout else revision
