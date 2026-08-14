"""Small durable-file helpers shared by project persistence formats."""

import csv
import errno
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import BinaryIO, TextIO, cast

_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_CHECKPOINT_RETRY_DELAYS_SECONDS = (1.0, 5.0, 15.0, 30.0)
_RETRYABLE_STORAGE_ERRNOS = frozenset(
    {
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.EIO,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ESTALE,
        errno.ETIMEDOUT,
    }
)


@contextmanager
def atomic_binary_writer(path: Path) -> Iterator[BinaryIO]:
    """Replace a file only after its temporary replacement is fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            yield cast(BinaryIO, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def staged_atomic_binary_writer(
    path: Path,
    *,
    staging_directory: Path | None = None,
) -> Iterator[BinaryIO]:
    """Stage a large file locally before verified, retried atomic publication."""
    resolved_staging_directory = staging_directory or checkpoint_staging_directory()
    resolved_staging_directory.mkdir(parents=True, exist_ok=True)
    staged_path: Path | None = None
    staging_complete = False
    published = False
    try:
        with NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".staged",
            dir=resolved_staging_directory,
            delete=False,
        ) as staged_file:
            staged_path = Path(staged_file.name)
            yield cast(BinaryIO, staged_file)
            staged_file.flush()
            os.fsync(staged_file.fileno())
        staging_complete = True
        assert staged_path is not None
        _publish_staged_file(staged_path, path)
        published = True
    except BaseException as error:
        if staged_path is not None:
            if staging_complete and staged_path.exists():
                error.add_note(f"complete local staging file retained at {staged_path}")
            elif not staging_complete:
                staged_path.unlink(missing_ok=True)
        raise
    finally:
        if published and staged_path is not None:
            staged_path.unlink(missing_ok=True)


def checkpoint_staging_directory() -> Path:
    """Return the local directory used for recoverable checkpoint staging files."""
    configured = os.environ.get("AC_CFR_CHECKPOINT_STAGING_DIRECTORY")
    return (
        Path(configured).expanduser() if configured else Path(gettempdir()) / "ac-cfr-checkpoints"
    )


def _publish_staged_file(staged_path: Path, path: Path) -> None:
    """Publish one complete staged file, retrying transient storage failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged_digest = _sha256_file(staged_path)
    total_attempts = len(_CHECKPOINT_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(total_attempts):
        try:
            _publish_staged_file_once(staged_path, staged_digest, path)
            return
        except OSError as error:
            retryable = error.errno in _RETRYABLE_STORAGE_ERRNOS
            if not retryable or attempt == total_attempts - 1:
                raise
            delay = _CHECKPOINT_RETRY_DELAYS_SECONDS[attempt]
            print(
                f"checkpoint publication attempt {attempt + 1}/{total_attempts} failed "
                f"with {error}; retrying in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _publish_staged_file_once(
    staged_path: Path,
    staged_digest: bytes,
    path: Path,
) -> None:
    """Copy, flush, verify and atomically install one staged file."""
    temporary_path: Path | None = None
    try:
        copied_digest = hashlib.sha256()
        with NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with staged_path.open("rb") as staged_file:
                while chunk := staged_file.read(_COPY_CHUNK_BYTES):
                    temporary_file.write(chunk)
                    copied_digest.update(chunk)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if temporary_path.stat().st_size != staged_path.stat().st_size:
            raise OSError(errno.EIO, "checkpoint publication size mismatch")
        if copied_digest.digest() != staged_digest:
            raise OSError(errno.EIO, "checkpoint publication checksum mismatch")
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> bytes:
    """Return a bounded-memory SHA-256 digest for one local staged file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.digest()


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Write UTF-8 text through an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            yield cast(TextIO, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_csv(
    path: Path,
    fields: Sequence[str],
    records: Iterable[Mapping[str, object]],
) -> None:
    """Atomically write CSV records using a fixed column order."""
    with atomic_text_writer(path) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value: object) -> None:
    """Atomically write deterministic, human-readable JSON."""
    with atomic_text_writer(path) as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _sync_directory(directory: Path) -> None:
    """Flush directory metadata after atomically replacing a file."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
