"""Install registry-declared playable strategy artefacts with integrity checks."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from ac_cfr.persistence.registry import (
    StrategyRegistryEntry,
    load_strategy_registry,
    strategy_artifact_path,
)

DEFAULT_REGISTRY_PATH = Path("configs/strategy_registry.json")
DEFAULT_RELEASE_BASE_URL = "https://github.com/asbc0001/ac-cfr/releases/download"


class _Readable(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class _Writable(Protocol):
    def write(self, data: bytes, /) -> object: ...


def install_strategy_artifacts(
    *,
    registry_path: Path,
    project_root: Path,
    strategy_ids: tuple[str, ...] = (),
    source_directory: Path | None = None,
    release_base_url: str = DEFAULT_RELEASE_BASE_URL,
) -> tuple[Path, ...]:
    """Install selected file-backed entries from staging or published releases."""
    registry = load_strategy_registry(registry_path, project_root=project_root)
    requested = set(strategy_ids)
    known_ids = {entry.strategy_id for entry in registry.entries}
    unknown = requested - known_ids
    if unknown:
        raise ValueError(f"unknown strategy_id values: {', '.join(sorted(unknown))}")

    entries = tuple(
        entry
        for entry in registry.entries
        if entry.local_path is not None and (not requested or entry.strategy_id in requested)
    )
    if requested and len(entries) != len(requested):
        raise ValueError("selected baseline strategies do not have downloadable artefacts")

    installed = []
    for entry in entries:
        destination = strategy_artifact_path(entry, project_root=project_root)
        if _matches_registry(destination, entry):
            installed.append(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        _install_entry(
            entry,
            destination,
            source_directory=source_directory,
            release_base_url=release_base_url,
        )
        installed.append(destination)
    return tuple(installed)


def main(argv: Sequence[str] | None = None) -> int:
    """Download or copy playable artefacts declared by the strategy registry."""
    parser = argparse.ArgumentParser(
        description="Install and verify playable strategy artefacts.",
    )
    parser.add_argument("--strategy-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--strategy-id", action="append", default=[])
    parser.add_argument(
        "--source-directory",
        type=Path,
        help="Copy release assets from this directory instead of downloading them.",
    )
    parser.add_argument("--release-base-url", default=DEFAULT_RELEASE_BASE_URL)
    parsed = parser.parse_args(argv)
    installed = install_strategy_artifacts(
        registry_path=parsed.strategy_registry,
        project_root=parsed.project_root,
        strategy_ids=tuple(parsed.strategy_id),
        source_directory=parsed.source_directory,
        release_base_url=parsed.release_base_url,
    )
    for path in installed:
        print(path)
    return 0


def _install_entry(
    entry: StrategyRegistryEntry,
    destination: Path,
    *,
    source_directory: Path | None,
    release_base_url: str,
) -> None:
    if entry.file_size is None or entry.sha256 is None or entry.release_id is None:
        raise ValueError("file-backed registry entry is incomplete")
    temporary_path: Path | None = None
    try:
        # The destination directory keeps final replacement on one filesystem and atomic.
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            if source_directory is not None:
                source = source_directory / destination.name
                if not source.is_file():
                    raise ValueError(f"release asset is missing: {source.name}")
                with source.open("rb") as input_file:
                    _copy_limited(input_file, temporary, entry.file_size)
            else:
                release = quote(entry.release_id, safe="")
                filename = quote(destination.name, safe="")
                url = f"{release_base_url.rstrip('/')}/{release}/{filename}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    _copy_limited(response, temporary, entry.file_size)
        if not _matches_registry(temporary_path, entry):
            raise ValueError(f"release asset failed integrity validation: {destination.name}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _matches_registry(path: Path, entry: StrategyRegistryEntry) -> bool:
    if entry.file_size is None or entry.sha256 is None or not path.is_file():
        return False
    if path.stat().st_size != entry.file_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry.sha256


def _copy_limited(source: _Readable, destination: _Writable, expected_size: int) -> None:
    """Copy no more than the registry-declared number of bytes."""
    copied = 0
    while chunk := source.read(min(1024 * 1024, expected_size - copied + 1)):
        copied += len(chunk)
        if copied > expected_size:
            raise ValueError("release asset exceeds its declared size")
        destination.write(chunk)


if __name__ == "__main__":
    raise SystemExit(main())
