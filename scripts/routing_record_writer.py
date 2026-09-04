"""E3 RoutingRecord v1 JSONL persistence layer (E3 P0 continuation).

Responsibility split with ``scripts.routing_record``:
- ``routing_record`` owns the v1 data model, validation, and the canonical
  per-record JSONL line (``RoutingRecord.to_json`` / ``from_json``: compact,
  sorted keys, deterministic byte output).
- This module owns file persistence only: creating/opening a JSONL stream,
  append semantics, one record per physical line, and explicit errors for
  non-RoutingRecord or non-JSON-serializable input. It never re-serializes
  records (the byte content is exactly ``record.to_json() + "\\n"``), never
  invents fields, and never writes raw upstream snapshots: only
  ``RoutingRecord`` instances are accepted, so the canonical
  ``source_snapshot_keys`` provenance survives verbatim.

Guarantees:
- One ``RoutingRecord`` == one physical JSONL line, LF-terminated.
- Deterministic output: ``json.dumps(..., sort_keys=True)`` is already applied
  inside ``RoutingRecord.to_json``, so repeated writes of equal records
  produce byte-identical files.
- Append is the default; ``append=False`` creates or truncates the target.
- Illegal input raises; data is never silently dropped or partially written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from scripts.routing_record import RoutingRecord


class RoutingRecordWriterError(RuntimeError):
    """Raised when a RoutingRecord cannot be serialized into one JSONL line."""


class RoutingRecordWriter:
    """Stream one validated v1 RoutingRecord per JSONL line to a file."""

    def __init__(self, path: Path | str, *, append: bool = True) -> None:
        self.path = Path(path)
        self._handle = self.path.open(
            mode="a" if append else "w",
            encoding="utf-8",
            newline="\n",
        )

    @property
    def closed(self) -> bool:
        """Return whether the underlying file stream has been closed."""
        return self._handle is None or self._handle.closed

    def write(self, record: Any) -> None:
        """Serialize and append one record; raises on any illegal input."""
        if not isinstance(record, RoutingRecord):
            raise TypeError(
                f"expected a RoutingRecord, got {type(record).__name__}; "
                "raw upstream snapshots are never written to JSONL"
            )
        if self._handle is None:
            raise ValueError("RoutingRecordWriter is closed")
        try:
            line = record.to_json()
        except (TypeError, ValueError) as exc:
            raise RoutingRecordWriterError(
                f"record run_id={record.run_id!r} (schema {record.schema_version}) "
                f"is not JSON-serializable: {exc}"
            ) from exc
        self._handle.write(line)
        self._handle.write("\n")

    def write_all(self, records: Iterable[Any]) -> int:
        """Append every record and return the number of lines written."""
        count = 0
        for record in records:
            self.write(record)
            count += 1
        return count

    def close(self) -> None:
        """Close the underlying file stream (idempotent)."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RoutingRecordWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def iter_records(path: Path | str) -> Iterator[RoutingRecord]:
    """Yield RoutingRecords from a writer-produced JSONL file.

    Blank lines are tolerated; a malformed JSON or schema-invalid line raises
    a ``ValueError`` annotated with the file path and line number instead of
    being silently skipped.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as stream:
        for line_no, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = RoutingRecord.from_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid RoutingRecord JSONL line {file_path}:{line_no}: {exc}"
                ) from exc
            yield record


__all__ = ["RoutingRecordWriter", "RoutingRecordWriterError", "iter_records"]