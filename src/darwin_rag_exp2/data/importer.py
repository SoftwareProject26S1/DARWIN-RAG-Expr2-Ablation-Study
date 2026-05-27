"""Read the raw notice JSONL without modifying the baseline input."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import orjson
from pydantic import ValidationError

from .schema import NoticeRecord


IssueType = Literal["invalid_json", "schema_error"]


@dataclass(frozen=True)
class ImportIssue:
    """One row that cannot be admitted to the validated raw-record stream."""

    line_number: int
    issue_type: IssueType
    message: str


@dataclass(frozen=True)
class ImportResult:
    """Validated rows and input-integrity metadata for one source file."""

    source_path: Path
    source_sha256: str
    line_count: int
    records: tuple[NoticeRecord, ...]
    issues: tuple[ImportIssue, ...]

    @property
    def invalid_json_count(self) -> int:
        return sum(issue.issue_type == "invalid_json" for issue in self.issues)

    @property
    def schema_error_count(self) -> int:
        return sum(issue.issue_type == "schema_error" for issue in self.issues)


def _file_sha256(source_path: Path) -> str:
    digest = sha256()
    with source_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_notice_export(source_path: Path) -> ImportResult:
    """Parse and validate all source rows while preserving import failures."""

    records: list[NoticeRecord] = []
    issues: list[ImportIssue] = []
    line_count = 0

    with source_path.open("rb") as source:
        for line_count, line in enumerate(source, start=1):
            try:
                raw_record = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                issues.append(ImportIssue(line_count, "invalid_json", str(exc)))
                continue

            try:
                records.append(NoticeRecord.model_validate(raw_record))
            except ValidationError as exc:
                issues.append(ImportIssue(line_count, "schema_error", str(exc)))

    return ImportResult(
        source_path=source_path,
        source_sha256=_file_sha256(source_path),
        line_count=line_count,
        records=tuple(records),
        issues=tuple(issues),
    )
