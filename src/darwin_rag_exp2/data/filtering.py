"""Category admission and quality filtering for the Phase 3 source corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal

import orjson
import yaml

from .importer import load_notice_export
from .schema import NoticeRecord


FilterReason = Literal[
    "excluded_category",
    "unsupported_category",
    "title_only",
    "body_too_short",
    "category_below_minimum",
]

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass(frozen=True)
class CorpusFilterConfig:
    """Frozen Phase 3 filtering thresholds and category policy."""

    primary_categories: tuple[str, ...]
    excluded_category_reasons: Mapping[str, str]
    minimum_body_tokens: int
    min_primary_source_documents: int
    category_mapping: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExcludedRecord:
    """A source notice removed from the primary study population."""

    source_record: NoticeRecord
    reason: FilterReason
    reason_detail: str
    body_token_count: int | None = None
    primary_category: str | None = None


@dataclass(frozen=True)
class CorpusPreparationResult:
    """The admitted corpus plus exclusion accounting and lineage metadata."""

    source_path: str
    source_sha256: str
    record_count: int
    valid_record_count: int
    config: CorpusFilterConfig
    admitted_records: tuple[NoticeRecord, ...]
    excluded_records: tuple[ExcludedRecord, ...]
    admitted_category_counts: dict[str, int]
    filter_reason_counts: dict[str, int]
    rejected_primary_categories: dict[str, int]


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        msg = f"YAML root must be a mapping: {path}"
        raise ValueError(msg)
    return payload


def load_corpus_filter_config(
    experiment_config_path: Path,
    category_mapping_path: Path,
) -> CorpusFilterConfig:
    """Load Phase 3 settings from the experiment and category mapping files."""

    experiment_config = _load_yaml(experiment_config_path)
    category_config = _load_yaml(category_mapping_path)

    experiment_section = experiment_config.get("experiment", {})
    data_section = experiment_config.get("data", {})
    chunking_section = experiment_config.get("chunking", {})
    if not isinstance(experiment_section, dict):
        raise ValueError("experiment config section must be a mapping")
    if not isinstance(data_section, dict):
        raise ValueError("data config section must be a mapping")
    if not isinstance(chunking_section, dict):
        raise ValueError("chunking config section must be a mapping")

    primary_categories = experiment_section.get("primary_categories")
    min_primary_source_documents = data_section.get("min_primary_source_documents")
    minimum_body_tokens = chunking_section.get("minimum_information_tokens")
    if not isinstance(primary_categories, list) or not all(
        isinstance(category, str) for category in primary_categories
    ):
        raise ValueError("experiment.primary_categories must be a list of strings")
    if not isinstance(min_primary_source_documents, int):
        raise ValueError("data.min_primary_source_documents must be an integer")
    if not isinstance(minimum_body_tokens, int):
        raise ValueError("chunking.minimum_information_tokens must be an integer")

    excluded_category_reasons = category_config.get("excluded_categories", {})
    category_mapping = category_config.get("primary_category_mapping", {})
    if not isinstance(excluded_category_reasons, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in excluded_category_reasons.items()
    ):
        raise ValueError("excluded_categories must be a mapping of strings")
    if not isinstance(category_mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in category_mapping.items()
    ):
        raise ValueError("primary_category_mapping must be a mapping of strings")

    return CorpusFilterConfig(
        primary_categories=tuple(primary_categories),
        excluded_category_reasons=dict(excluded_category_reasons),
        minimum_body_tokens=minimum_body_tokens,
        min_primary_source_documents=min_primary_source_documents,
        category_mapping=dict(category_mapping),
    )


def _normalized_space(value: str) -> str:
    return " ".join(value.split())


def _body_text(record: NoticeRecord) -> str:
    text = _normalized_space(record.text)
    title = _normalized_space(record.title)
    if not title:
        return text
    if text == title:
        return ""
    title_prefix = f"{title} "
    if text.startswith(title_prefix):
        return text[len(title_prefix) :].strip()
    return text


def _token_count(value: str) -> int:
    return len(_TOKEN_PATTERN.findall(value))


def _sorted_counts(values: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(values.items()))


def prepare_corpus(
    source_path: Path,
    config: CorpusFilterConfig,
) -> CorpusPreparationResult:
    """Apply Phase 3 category and quality filtering to a raw notice export."""

    imported = load_notice_export(source_path)
    if imported.issues:
        msg = "Phase 3 corpus preparation requires a schema-valid Phase 2 export"
        raise ValueError(msg)

    primary_categories = set(config.primary_categories)
    tentative_records: list[tuple[NoticeRecord, str, int]] = []
    excluded_records: list[ExcludedRecord] = []

    for record in imported.records:
        mapped_category = config.category_mapping.get(record.category, record.category)
        if record.category in config.excluded_category_reasons:
            excluded_records.append(
                ExcludedRecord(
                    source_record=record,
                    reason="excluded_category",
                    reason_detail=config.excluded_category_reasons[record.category],
                    primary_category=mapped_category,
                )
            )
            continue
        if mapped_category not in primary_categories:
            excluded_records.append(
                ExcludedRecord(
                    source_record=record,
                    reason="unsupported_category",
                    reason_detail="source category is not part of the primary study population",
                    primary_category=mapped_category,
                )
            )
            continue

        body_text = _body_text(record)
        if not body_text:
            excluded_records.append(
                ExcludedRecord(
                    source_record=record,
                    reason="title_only",
                    reason_detail="text equals title after whitespace normalization",
                    body_token_count=0,
                    primary_category=mapped_category,
                )
            )
            continue

        body_token_count = _token_count(body_text)
        if body_token_count < config.minimum_body_tokens:
            excluded_records.append(
                ExcludedRecord(
                    source_record=record,
                    reason="body_too_short",
                    reason_detail=(
                        f"body_token_count={body_token_count} "
                        f"< minimum_body_tokens={config.minimum_body_tokens}"
                    ),
                    body_token_count=body_token_count,
                    primary_category=mapped_category,
                )
            )
            continue
        tentative_records.append((record, mapped_category, body_token_count))

    tentative_counts = Counter(category for _, category, _ in tentative_records)
    rejected_primary_categories = {
        category: count
        for category, count in sorted(tentative_counts.items())
        if count < config.min_primary_source_documents
    }

    admitted_records: list[NoticeRecord] = []
    for record, mapped_category, body_token_count in tentative_records:
        if mapped_category in rejected_primary_categories:
            excluded_records.append(
                ExcludedRecord(
                    source_record=record,
                    reason="category_below_minimum",
                    reason_detail=(
                        f"category_count={rejected_primary_categories[mapped_category]} "
                        f"< min_primary_source_documents="
                        f"{config.min_primary_source_documents}"
                    ),
                    body_token_count=body_token_count,
                    primary_category=mapped_category,
                )
            )
            continue
        admitted_records.append(record)

    admitted_counts = Counter(
        config.category_mapping.get(record.category, record.category)
        for record in admitted_records
    )
    reason_counts = Counter(record.reason for record in excluded_records)

    return CorpusPreparationResult(
        source_path=str(source_path),
        source_sha256=imported.source_sha256,
        record_count=imported.line_count,
        valid_record_count=len(imported.records),
        config=config,
        admitted_records=tuple(admitted_records),
        excluded_records=tuple(excluded_records),
        admitted_category_counts=_sorted_counts(admitted_counts),
        filter_reason_counts=_sorted_counts(reason_counts),
        rejected_primary_categories=rejected_primary_categories,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            output.write(b"\n")


def _excluded_payload(record: ExcludedRecord) -> dict[str, object]:
    return {
        "id": record.source_record.id,
        "category": record.source_record.category,
        "reason": record.reason,
        "reason_detail": record.reason_detail,
    }


def write_corpus_artifacts(
    result: CorpusPreparationResult,
    output_dir: Path,
) -> None:
    """Write Phase 3 corpus artifacts and lineage manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_dir / "admitted.jsonl",
        [record.model_dump() for record in result.admitted_records],
    )
    _write_jsonl(
        output_dir / "excluded.jsonl",
        [_excluded_payload(record) for record in result.excluded_records],
    )

    output_dir.joinpath("category_counts.json").write_bytes(
        orjson.dumps(
            result.admitted_category_counts,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
    output_dir.joinpath("filter_reason_counts.json").write_bytes(
        orjson.dumps(
            result.filter_reason_counts,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
    output_dir.joinpath("manifest.json").write_bytes(
        orjson.dumps(
            {
                "phase": 3,
                "source_path": result.source_path,
                "source_sha256": result.source_sha256,
                "record_count": result.record_count,
                "valid_record_count": result.valid_record_count,
                "admitted_record_count": len(result.admitted_records),
                "excluded_record_count": len(result.excluded_records),
                "admitted_category_counts": result.admitted_category_counts,
                "filter_reason_counts": result.filter_reason_counts,
                "rejected_primary_categories": result.rejected_primary_categories,
                "primary_categories": list(result.config.primary_categories),
                "excluded_category_reasons": dict(
                    result.config.excluded_category_reasons
                ),
                "minimum_body_tokens": result.config.minimum_body_tokens,
                "min_primary_source_documents": (
                    result.config.min_primary_source_documents
                ),
                "artifact_files": [
                    "admitted.jsonl",
                    "excluded.jsonl",
                    "category_counts.json",
                    "filter_reason_counts.json",
                    "manifest.json",
                ],
            },
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
