"""Quality audit and report serialization for the read-only source corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from statistics import median

import orjson

from .importer import ImportIssue, load_notice_export


@dataclass(frozen=True)
class TextLengthStatistics:
    """Character-length summary over schema-valid source notice text."""

    median: float | int
    p95: int
    maximum: int


@dataclass(frozen=True)
class AuditReport:
    """Stable raw-corpus measurements used by later Exp2 phases."""

    source_path: str
    source_sha256: str
    record_count: int
    valid_record_count: int
    invalid_json_count: int
    schema_error_count: int
    duplicate_id_count: int
    duplicate_url_count: int
    text_length_mismatch_count: int
    title_only_count: int
    text_length_characters: TextLengthStatistics | None
    category_counts: dict[str, int]
    issues: tuple[ImportIssue, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


def _duplicate_occurrence_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _text_length_statistics(lengths: list[int]) -> TextLengthStatistics | None:
    if not lengths:
        return None
    sorted_lengths = sorted(lengths)
    p95_index = math.ceil(len(sorted_lengths) * 0.95) - 1
    return TextLengthStatistics(
        median=median(sorted_lengths),
        p95=sorted_lengths[p95_index],
        maximum=sorted_lengths[-1],
    )


def audit_notice_export(source_path: Path) -> AuditReport:
    """Compute non-destructive baseline measurements for one raw export."""

    imported = load_notice_export(source_path)
    records = list(imported.records)
    category_counts = Counter(record.category for record in records)

    return AuditReport(
        source_path=str(source_path),
        source_sha256=imported.source_sha256,
        record_count=imported.line_count,
        valid_record_count=len(records),
        invalid_json_count=imported.invalid_json_count,
        schema_error_count=imported.schema_error_count,
        duplicate_id_count=_duplicate_occurrence_count([record.id for record in records]),
        duplicate_url_count=_duplicate_occurrence_count([record.url for record in records]),
        text_length_mismatch_count=sum(
            len(record.text) != record.text_length for record in records
        ),
        title_only_count=sum(
            record.text.strip() == record.title.strip() for record in records
        ),
        text_length_characters=_text_length_statistics(
            [len(record.text) for record in records]
        ),
        category_counts=dict(sorted(category_counts.items())),
        issues=imported.issues,
    )


def _markdown_cell(value: str) -> str:
    return " ".join(value.replace("|", "/").split())


def _render_markdown(report: AuditReport) -> str:
    statistics = report.text_length_characters
    lines = [
        "# Raw Notice Dataset Audit",
        "",
        f"- Source: `{report.source_path}`",
        f"- SHA-256: `{report.source_sha256}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Records | {report.record_count:,} |",
        f"| Valid records | {report.valid_record_count:,} |",
        f"| Invalid JSON | {report.invalid_json_count:,} |",
        f"| Schema errors | {report.schema_error_count:,} |",
        f"| Duplicate IDs | {report.duplicate_id_count:,} |",
        f"| Duplicate URLs | {report.duplicate_url_count:,} |",
        f"| Text length mismatches | {report.text_length_mismatch_count:,} |",
        f"| Title-only records | {report.title_only_count:,} |",
        "",
        "## Text Length Characters",
        "",
    ]
    if statistics is None:
        lines.append("No schema-valid text rows.")
    else:
        lines.extend(
            [
                "| Statistic | Characters |",
                "|---|---:|",
                f"| Median | {statistics.median:,} |",
                f"| P95 | {statistics.p95:,} |",
                f"| Maximum | {statistics.maximum:,} |",
            ]
        )
    lines.extend(["", "## Category Counts", "", "| Category | Records |", "|---|---:|"])
    lines.extend(
        f"| {category} | {count:,} |"
        for category, count in report.category_counts.items()
    )
    if report.issues:
        lines.extend(["", "## Import Issues", "", "| Line | Type | Message |", "|---:|---|---|"])
        lines.extend(
            f"| {issue.line_number} | {issue.issue_type} | {_markdown_cell(issue.message)} |"
            for issue in report.issues
        )
    return "\n".join(lines) + "\n"


def write_audit_artifacts(report: AuditReport, output_dir: Path) -> None:
    """Write machine-readable and reviewable audit outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("audit.json").write_bytes(
        orjson.dumps(report.as_dict(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )
    output_dir.joinpath("audit.md").write_text(_render_markdown(report), encoding="utf-8")
