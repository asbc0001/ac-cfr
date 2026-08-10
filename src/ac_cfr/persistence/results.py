"""Compact, resume-safe CSV records for training and exact evaluation."""

import csv
from pathlib import Path
from typing import Final

from ac_cfr.persistence.files import atomic_text_writer

TRAINING_METRIC_FIELDS: Final = (
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
)
TRAINING_METRIC_KEY_FIELDS: Final = ("run_id", "iteration", "seed")

EVALUATION_RESULT_FIELDS: Final = (
    "game",
    "game_version",
    "utility_unit",
    "solver",
    "run_id",
    "strategy_snapshot_id",
    "source_checkpoint_id",
    "iteration",
    "seed",
    "expected_value_player_zero",
    "exploitability",
    "nash_conv",
)
EVALUATION_RESULT_KEY_FIELDS: Final = (
    "run_id",
    "strategy_snapshot_id",
    "source_checkpoint_id",
    "iteration",
    "seed",
)

# Wider files from the first reporting implementation are projected onto the relevant schema.
LEGACY_RESULT_FIELDS: Final = (
    *TRAINING_METRIC_FIELDS,
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

ResultRecord = dict[str, str]


class TrainingMetricStore:
    """Store one compact row per periodic measurement in a training run."""

    __slots__ = ("_store",)

    def __init__(self, path: Path) -> None:
        self._store = _CsvRecordStore(
            path,
            fields=TRAINING_METRIC_FIELDS,
            key_fields=TRAINING_METRIC_KEY_FIELDS,
        )

    @property
    def records(self) -> tuple[ResultRecord, ...]:
        """Return independent copies of the current records."""
        return self._store.records

    def upsert(self, values: dict[str, object]) -> None:
        """Insert or replace one training measurement."""
        self._store.upsert(values)

    def replace(self, records: list[dict[str, object]]) -> None:
        """Replace all measurements, including compatible legacy records."""
        self._store.replace(records)


class EvaluationResultStore:
    """Store one compact exact-evaluation result per frozen strategy."""

    __slots__ = ("_store",)

    def __init__(self, path: Path) -> None:
        self._store = _CsvRecordStore(
            path,
            fields=EVALUATION_RESULT_FIELDS,
            key_fields=EVALUATION_RESULT_KEY_FIELDS,
        )

    @property
    def records(self) -> tuple[ResultRecord, ...]:
        """Return independent copies of the current records."""
        return self._store.records

    def upsert(self, values: dict[str, object]) -> None:
        """Insert or replace one exact-evaluation result."""
        self._store.upsert(values)


class _CsvRecordStore:
    """Upsert records by a stable key and replace the CSV atomically."""

    __slots__ = ("_fields", "_key_fields", "_path", "_records")

    def __init__(
        self,
        path: Path,
        *,
        fields: tuple[str, ...],
        key_fields: tuple[str, ...],
    ) -> None:
        self._path = path
        self._fields = fields
        self._key_fields = key_fields
        self._records = self._read_existing()

    @property
    def records(self) -> tuple[ResultRecord, ...]:
        """Return independent copies of records in their current stored order."""
        return tuple(record.copy() for record in self._records.values())

    def upsert(self, values: dict[str, object]) -> None:
        """Normalise and atomically insert or replace one keyed record."""
        record = self._normalise(values)
        self._records[self._record_key(record)] = record
        self._write()

    def replace(self, records: list[dict[str, object]]) -> None:
        """Validate and atomically replace the complete record collection."""
        replacement: dict[tuple[str, ...], ResultRecord] = {}
        for values in records:
            record = self._normalise(values, allow_legacy=True)
            key = self._record_key(record)
            if key in replacement:
                raise ValueError("result records contain a duplicate composite key")
            replacement[key] = record
        self._records = replacement
        self._write()

    def _read_existing(self) -> dict[tuple[str, ...], ResultRecord]:
        """Load a current or compatible legacy CSV without duplicate keys."""
        if not self._path.exists():
            return {}
        with self._path.open(encoding="utf-8", newline="") as results_file:
            reader = csv.DictReader(results_file)
            header = tuple(reader.fieldnames or ())
            if header not in (self._fields, LEGACY_RESULT_FIELDS):
                raise ValueError("results file has an incompatible header")
            records: dict[tuple[str, ...], ResultRecord] = {}
            for row in reader:
                values: dict[str, object] = {field: row.get(field) for field in header}
                record = self._normalise(values, allow_legacy=True)
                key = self._record_key(record)
                if key in records:
                    raise ValueError("results file contains a duplicate composite key")
                records[key] = record
            return records

    def _normalise(
        self,
        values: dict[str, object],
        *,
        allow_legacy: bool = False,
    ) -> ResultRecord:
        """Project supported values onto this store's exact string schema."""
        provided_fields = set(values)
        if not provided_fields <= set(self._fields) and not (
            allow_legacy and provided_fields == set(LEGACY_RESULT_FIELDS)
        ):
            unknown_fields = provided_fields - set(self._fields)
            raise ValueError(f"unknown result fields: {sorted(unknown_fields)}")
        record = {field: _stringify(values.get(field)) for field in self._fields}
        _validate_required_fields(record)
        return record

    def _record_key(self, record: ResultRecord) -> tuple[str, ...]:
        return tuple(record[field] for field in self._key_fields)

    def _write(self) -> None:
        """Atomically write records in deterministic order."""
        with atomic_text_writer(self._path) as results_file:
            writer: csv.DictWriter[str] = csv.DictWriter(
                results_file,
                fieldnames=list(self._fields),
                lineterminator="\n",
            )
            writer.writeheader()
            for record in sorted(self._records.values(), key=_record_sort_key):
                writer.writerow(record)


def _validate_required_fields(record: ResultRecord) -> None:
    """Validate required identifiers and numeric key fields."""
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
        int(record["seed"]),
    )


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise TypeError("result values must not be booleans")
    return str(value)
