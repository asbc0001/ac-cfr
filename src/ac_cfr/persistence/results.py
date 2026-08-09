"""Compact, resume-safe CSV metric and evaluation records."""

import csv
from pathlib import Path
from typing import Final

from ac_cfr.persistence.files import atomic_text_writer

RESULT_FIELDS: Final = (
    "game",
    "game_version",
    "utility_unit",
    "solver",
    "run_id",
    "strategy_snapshot_id",
    "source_checkpoint_id",
    "iteration",
    "seed",
    "elapsed_training_seconds",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
    "traversals",
    "traversals_per_second",
    "memory_metric",
    "memory_mb",
    "gpu_memory_mb",
    "advantage_loss",
    "opponent_id",
    "hands",
    "paired_deals",
    "mbb_per_game",
    "confidence_level",
    "confidence_interval_method",
    "confidence_interval_low",
    "confidence_interval_high",
)
RESULT_KEY_FIELDS: Final = (
    "run_id",
    "strategy_snapshot_id",
    "source_checkpoint_id",
    "iteration",
    "seed",
    "opponent_id",
)

ResultRecord = dict[str, str]


class CsvResultStore:
    """Upsert compact records by a stable composite key and write atomically."""

    __slots__ = ("_path", "_records")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records = self._read_existing()

    @property
    def records(self) -> tuple[ResultRecord, ...]:
        """Return independent copies of the current records."""
        return tuple(record.copy() for record in self._records.values())

    def upsert(self, values: dict[str, object]) -> None:
        """Insert or replace one complete logical measurement."""
        unknown_fields = set(values) - set(RESULT_FIELDS)
        if unknown_fields:
            raise ValueError(f"unknown result fields: {sorted(unknown_fields)}")
        record = {field: _stringify(values.get(field)) for field in RESULT_FIELDS}
        _validate_required_fields(record)
        self._records[_record_key(record)] = record
        self._write()

    def replace(self, records: list[dict[str, object]]) -> None:
        """Replace all records after validating their keys and fields."""
        replacement: dict[tuple[str, ...], ResultRecord] = {}
        for values in records:
            unknown_fields = set(values) - set(RESULT_FIELDS)
            if unknown_fields:
                raise ValueError(f"unknown result fields: {sorted(unknown_fields)}")
            record = {field: _stringify(values.get(field)) for field in RESULT_FIELDS}
            _validate_required_fields(record)
            key = _record_key(record)
            if key in replacement:
                raise ValueError("result records contain a duplicate composite key")
            replacement[key] = record
        self._records = replacement
        self._write()

    def _read_existing(self) -> dict[tuple[str, ...], ResultRecord]:
        if not self._path.exists():
            return {}
        with self._path.open(encoding="utf-8", newline="") as results_file:
            reader = csv.DictReader(results_file)
            if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
                raise ValueError("results file has an incompatible header")
            records: dict[tuple[str, ...], ResultRecord] = {}
            for row in reader:
                record = {field: row[field] for field in RESULT_FIELDS}
                _validate_required_fields(record)
                key = _record_key(record)
                if key in records:
                    raise ValueError("results file contains a duplicate composite key")
                records[key] = record
            return records

    def _write(self) -> None:
        with atomic_text_writer(self._path) as results_file:
            writer: csv.DictWriter[str] = csv.DictWriter(
                results_file,
                fieldnames=list(RESULT_FIELDS),
                lineterminator="\n",
            )
            writer.writeheader()
            for record in sorted(self._records.values(), key=_record_sort_key):
                writer.writerow(record)


def _record_key(record: ResultRecord) -> tuple[str, ...]:
    return tuple(record[field] for field in RESULT_KEY_FIELDS)


def _validate_required_fields(record: ResultRecord) -> None:
    required_fields = ("game", "game_version", "solver", "run_id", "iteration", "seed")
    if any(not record[field] for field in required_fields):
        raise ValueError("result record is missing a required field")
    try:
        iteration = int(record["iteration"])
        int(record["seed"])
    except ValueError as error:
        raise ValueError("result iteration and seed must be integers") from error
    if iteration < 0:
        raise ValueError("result iteration must not be negative")


def _record_sort_key(record: ResultRecord) -> tuple[str | int, ...]:
    return (
        record["game"],
        record["solver"],
        record["run_id"],
        int(record["iteration"]),
        record["strategy_snapshot_id"],
        record["opponent_id"],
        int(record["seed"]),
    )


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise TypeError("result values must not be booleans")
    return str(value)
