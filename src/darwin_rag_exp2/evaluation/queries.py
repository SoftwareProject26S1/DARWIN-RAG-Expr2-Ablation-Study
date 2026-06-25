"""Phase 8 query annotation validation."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import orjson
import pyarrow.parquet as pq
import yaml


LEGACY_QUERY_FIELDS = {
    "query_id",
    "query",
    "gold_chunks",
    "reference_answer",
    "gold_categories",
    "query_type",
}
V2_SCHEMA_VERSION = "eval_v3_overlap_aware_rechunked"
V2_QUERY_FIELDS = {
    "query_id",
    "query",
    "reference_answer",
    "query_type",
    "expected_categories",
    "source_id",
    "gold_chunk_ids",
    "neighbor_chunk_ids",
    "graded_relevance",
    "requires_multi_category",
    "gold_category_set",
    "gold_category_pure",
    "evidence_unit",
    "schema_version",
}
V2_OPTIONAL_QUERY_FIELDS = {"manual_patch_note", "source_ids"}
GOLD_CHUNK_FIELDS = {"chunk_id", "source_id", "chunk_index", "relevance", "role"}
QUERY_TYPES = ("single_category", "multi_category", "ambiguous")
GOLD_ROLES = ("direct", "direct_overlap", "support")


@dataclass(frozen=True)
class QueryValidationConfig:
    """Config values used by the strict Phase 8 query validator."""

    primary_categories: tuple[str, ...]
    expected_dev_count: int
    expected_test_count: int
    non_single_fraction: float
    non_single_tolerance: float = 0.05


def load_query_validation_config(
    config_path: Path,
    *,
    non_single_tolerance: float = 0.05,
) -> QueryValidationConfig:
    """Load Phase 8 validation defaults from the experiment YAML."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = payload.get("experiment") or {}
    data = payload.get("data") or {}
    evaluation = payload.get("evaluation") or {}
    if not isinstance(experiment, Mapping):
        raise ValueError("config experiment section must be a mapping")
    if not isinstance(data, Mapping):
        raise ValueError("config data section must be a mapping")
    if not isinstance(evaluation, Mapping):
        raise ValueError("config evaluation section must be a mapping")
    categories = experiment.get("primary_categories")
    if not isinstance(categories, list) or not all(
        isinstance(category, str) for category in categories
    ):
        raise ValueError("config must define experiment.primary_categories")
    return QueryValidationConfig(
        primary_categories=tuple(categories),
        expected_dev_count=int(evaluation.get("dev_queries", 80)),
        expected_test_count=int(evaluation.get("test_queries", 240)),
        non_single_fraction=float(data.get("query_type_non_single_fraction", 0.30)),
        non_single_tolerance=non_single_tolerance,
    )


def validate_query_splits(
    *,
    dev_path: Path,
    test_path: Path,
    chunks_path: Path,
    config: QueryValidationConfig,
) -> dict[str, object]:
    """Validate dev/test annotation JSONL files against the Phase 8 contract."""

    chunk_ids = _load_chunk_ids(chunks_path)
    dev_rows = _load_jsonl(dev_path, split="dev")
    test_rows = _load_jsonl(test_path, split="test")
    _validate_expected_count("dev", dev_rows, config.expected_dev_count)
    _validate_expected_count("test", test_rows, config.expected_test_count)
    _validate_id_uniqueness(dev_rows, test_rows)
    validated_dev = _validate_split_rows(
        "dev",
        dev_rows,
        chunk_ids=chunk_ids,
        config=config,
    )
    validated_test = _validate_split_rows(
        "test",
        test_rows,
        chunk_ids=chunk_ids,
        config=config,
    )
    query_hashes = _query_hashes(
        dev_path=dev_path,
        test_path=test_path,
        dev_rows=validated_dev,
        test_rows=validated_test,
    )
    return {
        "phase": 8,
        "artifact_type": "query_validation",
        "valid": True,
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "config": {
            "primary_categories": list(config.primary_categories),
            "expected_dev_count": config.expected_dev_count,
            "expected_test_count": config.expected_test_count,
            "non_single_fraction": config.non_single_fraction,
            "non_single_tolerance": config.non_single_tolerance,
        },
        "splits": {
            "dev": _split_summary(validated_dev),
            "test": _split_summary(validated_test),
        },
        "query_hashes": query_hashes,
    }


def write_query_validation_artifacts(
    output_dir: Path,
    report: Mapping[str, object],
) -> None:
    """Write validation report, distribution table, hashes, and manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = [
        "validation.json",
        "distribution.csv",
        "query_hashes.json",
        "manifest.json",
    ]
    _write_json(output_dir / "validation.json", report)
    _write_distribution_csv(output_dir / "distribution.csv", report)
    _write_json(output_dir / "query_hashes.json", report["query_hashes"])
    _write_json(
        output_dir / "manifest.json",
        {
            "phase": 8,
            "artifact_type": "query_validation",
            "valid": bool(report.get("valid")),
            "splits": report.get("splits", {}),
            "artifact_files": artifact_files,
        },
    )


def _load_jsonl(path: Path, *, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = orjson.loads(stripped)
            except orjson.JSONDecodeError as error:
                raise ValueError(f"{split} line {line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{split} line {line_number}: row must be an object")
            rows.append({"_line_number": line_number, **dict(row)})
    if not rows:
        raise ValueError(f"{split}: no query rows found in {path}")
    return rows


def _load_chunk_ids(chunks_path: Path) -> set[str]:
    rows = pq.read_table(chunks_path, columns=["chunk_id"]).to_pylist()
    chunk_ids = {str(row["chunk_id"]) for row in rows}
    if not chunk_ids:
        raise ValueError(f"no chunk IDs found in {chunks_path}")
    return chunk_ids


def _validate_expected_count(
    split: str,
    rows: Sequence[Mapping[str, object]],
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"{split}: expected {expected_count} rows, found {len(rows)}"
        )


def _validate_id_uniqueness(
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
) -> None:
    for split, rows in (("dev", dev_rows), ("test", test_rows)):
        ids = [str(row.get("query_id", "")) for row in rows]
        duplicates = sorted(query_id for query_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"{split}: duplicate query_id values: {duplicates}")
    dev_ids = {str(row.get("query_id", "")) for row in dev_rows}
    test_ids = {str(row.get("query_id", "")) for row in test_rows}
    overlap = sorted(dev_ids.intersection(test_ids))
    if overlap:
        raise ValueError(f"dev/test query_id overlap: {overlap}")


def _validate_split_rows(
    split: str,
    rows: Sequence[Mapping[str, object]],
    *,
    chunk_ids: set[str],
    config: QueryValidationConfig,
) -> list[dict[str, object]]:
    validated: list[dict[str, object]] = []
    for row in rows:
        line_number = int(row["_line_number"])
        payload = {key: value for key, value in row.items() if key != "_line_number"}
        validated.append(
            _validate_row(
                split,
                line_number,
                payload,
                chunk_ids=chunk_ids,
                config=config,
            )
        )
    if not all(row.get("schema_version") == V2_SCHEMA_VERSION for row in validated):
        _validate_non_single_fraction(split, validated, config)
    return validated


def _validate_row(
    split: str,
    line_number: int,
    row: Mapping[str, object],
    *,
    chunk_ids: set[str],
    config: QueryValidationConfig,
) -> dict[str, object]:
    if row.get("schema_version") == V2_SCHEMA_VERSION:
        return _normalize_v2_row(
            split,
            line_number,
            row,
            chunk_ids=chunk_ids,
            config=config,
        )
    if "schema_version" in row:
        raise ValueError(
            f"{split} line {line_number}: unknown schema_version {row['schema_version']!r}"
        )

    fields = set(row)
    missing = sorted(LEGACY_QUERY_FIELDS.difference(fields))
    if missing:
        raise ValueError(f"{split} line {line_number}: missing fields {missing}")
    extra = sorted(fields.difference(LEGACY_QUERY_FIELDS))
    if extra:
        raise ValueError(f"{split} line {line_number}: unexpected fields {extra}")

    query_id = str(row["query_id"]).strip()
    prefix = f"{split}_q"
    if not query_id.startswith(prefix):
        raise ValueError(
            f"{split} line {line_number}: query_id {query_id!r} must start with {prefix}"
        )
    query = str(row["query"]).strip()
    if not query:
        raise ValueError(f"{split} line {line_number} {query_id}: empty query")
    if not str(row["reference_answer"]).strip():
        raise ValueError(
            f"{split} line {line_number} {query_id}: empty reference_answer"
        )

    gold_chunks = _string_list(row["gold_chunks"])
    if not gold_chunks:
        raise ValueError(
            f"{split} line {line_number} {query_id}: gold_chunks must not be empty"
        )
    duplicate_chunks = _duplicates(gold_chunks)
    if duplicate_chunks:
        raise ValueError(
            f"{split} line {line_number} {query_id}: duplicate gold_chunks {duplicate_chunks}"
        )
    unknown_chunks = sorted(chunk for chunk in gold_chunks if chunk not in chunk_ids)
    if unknown_chunks:
        raise ValueError(
            f"{split} line {line_number} {query_id}: unknown gold_chunks {unknown_chunks}"
        )

    gold_categories = _string_list(row["gold_categories"])
    if not gold_categories:
        raise ValueError(
            f"{split} line {line_number} {query_id}: gold_categories must not be empty"
        )
    duplicate_categories = _duplicates(gold_categories)
    if duplicate_categories:
        raise ValueError(
            f"{split} line {line_number} {query_id}: duplicate gold_categories {duplicate_categories}"
        )
    allowed_categories = set(config.primary_categories)
    invalid_categories = sorted(
        category for category in gold_categories if category not in allowed_categories
    )
    if invalid_categories:
        raise ValueError(
            f"{split} line {line_number} {query_id}: invalid gold_categories {invalid_categories}"
        )

    query_type = str(row["query_type"])
    if query_type not in QUERY_TYPES:
        raise ValueError(
            f"{split} line {line_number} {query_id}: invalid query_type {query_type!r}"
        )
    if query_type == "single_category" and len(gold_categories) != 1:
        raise ValueError(
            f"{split} line {line_number} {query_id}: single_category requires exactly one gold_category"
        )
    if query_type == "multi_category" and len(gold_categories) < 2:
        raise ValueError(
            f"{split} line {line_number} {query_id}: multi_category requires at least two gold_categories"
        )

    return {
        "query_id": query_id,
        "query": query,
        "gold_chunks": gold_chunks,
        "reference_answer": str(row["reference_answer"]).strip(),
        "gold_categories": gold_categories,
        "query_type": query_type,
    }


def _normalize_v2_row(
    split: str,
    line_number: int,
    row: Mapping[str, object],
    *,
    chunk_ids: set[str],
    config: QueryValidationConfig,
) -> dict[str, object]:
    fields = set(row)
    missing = sorted(V2_QUERY_FIELDS.difference(fields))
    if missing:
        raise ValueError(f"{split} line {line_number}: missing v2 fields {missing}")
    extra = sorted(fields.difference(V2_QUERY_FIELDS | V2_OPTIONAL_QUERY_FIELDS))
    query_id = str(row["query_id"]).strip()
    if extra:
        raise ValueError(
            f"{split} line {line_number} {query_id}: unexpected v2 fields {extra}"
        )
    prefix = f"{split}_q"
    if not query_id.startswith(prefix):
        raise ValueError(
            f"{split} line {line_number}: query_id {query_id!r} must start with {prefix}"
        )
    query = str(row["query"]).strip()
    if not query:
        raise ValueError(f"{split} line {line_number} {query_id}: empty query")
    reference_answer = str(row["reference_answer"]).strip()
    if not reference_answer:
        raise ValueError(
            f"{split} line {line_number} {query_id}: empty reference_answer"
        )
    if not isinstance(row["source_id"], str) or not row["source_id"].strip():
        raise ValueError(f"{split} line {line_number} {query_id}: invalid source_id")
    if "source_ids" in row:
        source_ids = row["source_ids"]
        if (
            not isinstance(source_ids, (list, tuple))
            or not source_ids
            or not all(isinstance(source_id, str) and source_id.strip() for source_id in source_ids)
        ):
            raise ValueError(f"{split} line {line_number} {query_id}: invalid source_ids")
    if "manual_patch_note" in row and (
        not isinstance(row["manual_patch_note"], str) or not row["manual_patch_note"].strip()
    ):
        raise ValueError(f"{split} line {line_number} {query_id}: invalid manual_patch_note")
    if not isinstance(row["evidence_unit"], str) or not row["evidence_unit"].strip():
        raise ValueError(f"{split} line {line_number} {query_id}: invalid evidence_unit")

    raw_gold_chunks = row["gold_chunk_ids"]
    if not isinstance(raw_gold_chunks, list) or not raw_gold_chunks:
        raise ValueError(
            f"{split} line {line_number} {query_id}: gold_chunk_ids must be a non-empty list"
        )
    gold_chunk_rows: list[dict[str, object]] = []
    gold_chunks: list[str] = []
    for item in raw_gold_chunks:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{split} line {line_number} {query_id}: gold_chunk_ids entries must be objects"
            )
        item_fields = set(item)
        if item_fields != GOLD_CHUNK_FIELDS:
            raise ValueError(
                f"{split} line {line_number} {query_id}: invalid gold_chunk_ids object shape"
            )
        chunk_id = item["chunk_id"]
        source_id = item["source_id"]
        chunk_index = item["chunk_index"]
        relevance = item["relevance"]
        role = item["role"]
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(
                f"{split} line {line_number} {query_id}: invalid gold_chunk_ids chunk_id"
            )
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                f"{split} line {line_number} {query_id}: invalid gold_chunk_ids source_id"
            )
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError(
                f"{split} line {line_number} {query_id}: invalid gold_chunk_ids chunk_index"
            )
        if isinstance(relevance, bool) or relevance not in (1, 2):
            raise ValueError(
                f"{split} line {line_number} {query_id}: invalid gold_chunk_ids relevance"
            )
        if role not in GOLD_ROLES:
            raise ValueError(
                f"{split} line {line_number} {query_id}: invalid gold_chunk_ids role"
            )
        gold_chunks.append(chunk_id)
        gold_chunk_rows.append(dict(item))

    duplicate_chunks = _duplicates(gold_chunks)
    if duplicate_chunks:
        raise ValueError(
            f"{split} line {line_number} {query_id}: duplicate gold_chunk_ids {duplicate_chunks}"
        )
    unknown_chunks = sorted(chunk for chunk in gold_chunks if chunk not in chunk_ids)
    if unknown_chunks:
        raise ValueError(
            f"{split} line {line_number} {query_id}: unknown gold_chunk_ids {unknown_chunks}"
        )

    neighbor_chunk_ids = _strict_string_list(row["neighbor_chunk_ids"])
    duplicate_neighbors = _duplicates(neighbor_chunk_ids)
    if duplicate_neighbors:
        raise ValueError(
            f"{split} line {line_number} {query_id}: duplicate neighbor_chunk_ids {duplicate_neighbors}"
        )
    unknown_neighbors = sorted(chunk for chunk in neighbor_chunk_ids if chunk not in chunk_ids)
    if unknown_neighbors:
        raise ValueError(
            f"{split} line {line_number} {query_id}: unknown neighbor_chunk_ids {unknown_neighbors}"
        )
    overlapping_neighbors = sorted(set(neighbor_chunk_ids).intersection(gold_chunks))
    if overlapping_neighbors:
        raise ValueError(
            f"{split} line {line_number} {query_id}: neighbor_chunk_ids overlap gold chunks {overlapping_neighbors}"
        )

    graded_relevance = row["graded_relevance"]
    if not isinstance(graded_relevance, Mapping):
        raise ValueError(
            f"{split} line {line_number} {query_id}: graded_relevance must be an object"
        )
    graded_keys = {str(chunk_id) for chunk_id in graded_relevance}
    gold_chunk_set = set(gold_chunks)
    missing_graded_gold = sorted(gold_chunk_set.difference(graded_keys))
    if missing_graded_gold:
        raise ValueError(
            f"{split} line {line_number} {query_id}: graded_relevance must cover gold_chunk_ids"
        )
    unknown_graded_chunks = sorted(
        graded_keys.difference(gold_chunk_set.union(neighbor_chunk_ids))
    )
    if unknown_graded_chunks:
        raise ValueError(
            f"{split} line {line_number} {query_id}: unknown graded_relevance chunks {unknown_graded_chunks}"
        )
    normalized_relevance: dict[str, float] = {}
    for chunk_id, value in graded_relevance.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"{split} line {line_number} {query_id}: graded_relevance values must be numeric"
            )
        score = float(value)
        if score < 0.0 or score > 1.0:
            raise ValueError(
                f"{split} line {line_number} {query_id}: graded_relevance values must be in [0, 1]"
            )
        normalized_relevance[str(chunk_id)] = score

    allowed_categories = set(config.primary_categories)
    expected_categories = _valid_categories(
        split,
        line_number,
        query_id,
        "expected_categories",
        row["expected_categories"],
        allowed_categories,
    )
    gold_categories = _valid_categories(
        split,
        line_number,
        query_id,
        "gold_category_set",
        row["gold_category_set"],
        allowed_categories,
    )
    if set(expected_categories) != set(gold_categories):
        raise ValueError(
            f"{split} line {line_number} {query_id}: expected_categories must match gold_category_set"
        )
    requires_multi_category = row["requires_multi_category"]
    if not isinstance(requires_multi_category, bool):
        raise ValueError(
            f"{split} line {line_number} {query_id}: requires_multi_category must be boolean"
        )
    if requires_multi_category != (len(gold_categories) > 1):
        raise ValueError(
            f"{split} line {line_number} {query_id}: requires_multi_category inconsistent with gold_category_set"
        )
    if not isinstance(row["gold_category_pure"], bool):
        raise ValueError(
            f"{split} line {line_number} {query_id}: gold_category_pure must be boolean"
        )
    query_type = str(row["query_type"]).strip()
    if not query_type:
        raise ValueError(f"{split} line {line_number} {query_id}: empty query_type")

    return {
        **dict(row),
        "query_id": query_id,
        "query": query,
        "reference_answer": reference_answer,
        "query_type": query_type,
        "gold_chunk_ids": gold_chunk_rows,
        "neighbor_chunk_ids": neighbor_chunk_ids,
        "graded_relevance": normalized_relevance,
        "gold_chunks": gold_chunks,
        "gold_categories": gold_categories,
    }


def _validate_non_single_fraction(
    split: str,
    rows: Sequence[Mapping[str, object]],
    config: QueryValidationConfig,
) -> None:
    non_single_count = sum(
        1 for row in rows if str(row["query_type"]) != "single_category"
    )
    fraction = non_single_count / len(rows)
    lower = config.non_single_fraction - config.non_single_tolerance
    upper = config.non_single_fraction + config.non_single_tolerance
    if fraction < lower or fraction > upper:
        raise ValueError(
            f"{split}: non-single query fraction {fraction:.3f} outside "
            f"[{lower:.3f}, {upper:.3f}]"
        )


def _split_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    query_type_counts = {
        query_type: count
        for query_type, count in sorted(
            Counter(str(row["query_type"]) for row in rows).items()
        )
    }
    category_counts = {
        category: count
        for category, count in sorted(
            Counter(
                category
                for row in rows
                for category in _string_list(row["gold_categories"])
            ).items()
        )
    }
    non_single_count = sum(
        count
        for query_type, count in query_type_counts.items()
        if query_type != "single_category"
    )
    summary = {
        "row_count": len(rows),
        "query_type_counts": query_type_counts,
        "category_counts": category_counts,
        "non_single_count": non_single_count,
        "non_single_fraction": _metric(non_single_count / len(rows)),
    }
    schema_versions = [
        str(row["schema_version"])
        for row in rows
        if row.get("schema_version") is not None
    ]
    if schema_versions:
        summary["schema_version_counts"] = {
            schema_version: count
            for schema_version, count in sorted(Counter(schema_versions).items())
        }
    return summary


def _query_hashes(
    *,
    dev_path: Path,
    test_path: Path,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = [("dev", row) for row in dev_rows] + [("test", row) for row in test_rows]
    queries = [
        {
            "split": split,
            "query_id": str(row["query_id"]),
            "sha256": _canonical_row_sha256(row),
        }
        for split, row in rows
    ]
    return {
        "files": {
            "dev": {
                "path": str(dev_path),
                "sha256": _file_sha256(dev_path),
                "row_count": len(dev_rows),
            },
            "test": {
                "path": str(test_path),
                "sha256": _file_sha256(test_path),
                "row_count": len(test_rows),
            },
        },
        "queries": queries,
    }


def _canonical_row_sha256(row: Mapping[str, object]) -> str:
    payload = {
        key: row[key]
        for key in sorted(row)
    }
    return sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _write_distribution_csv(path: Path, report: Mapping[str, object]) -> None:
    rows: list[dict[str, object]] = []
    splits = report.get("splits")
    if isinstance(splits, Mapping):
        for split, summary in splits.items():
            if not isinstance(summary, Mapping):
                continue
            for dimension, key in (
                ("query_type", "query_type_counts"),
                ("gold_category", "category_counts"),
            ):
                counts = summary.get(key)
                if not isinstance(counts, Mapping):
                    continue
                total = sum(int(count) for count in counts.values())
                for value, count in sorted(counts.items()):
                    rows.append(
                        {
                            "split": split,
                            "dimension": dimension,
                            "value": value,
                            "count": int(count),
                            "fraction": _metric(int(count) / total) if total else 0.0,
                        }
                    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["split", "dimension", "value", "count", "fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("expected a JSON array")
    return [str(item) for item in value]


def _strict_string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected a JSON string array")
    return list(value)


def _valid_categories(
    split: str,
    line_number: int,
    query_id: str,
    field: str,
    value: object,
    allowed_categories: set[str],
) -> list[str]:
    categories = _strict_string_list(value)
    if not categories:
        raise ValueError(
            f"{split} line {line_number} {query_id}: {field} must not be empty"
        )
    duplicates = _duplicates(categories)
    if duplicates:
        raise ValueError(
            f"{split} line {line_number} {query_id}: duplicate {field} {duplicates}"
        )
    invalid = sorted(category for category in categories if category not in allowed_categories)
    if invalid:
        raise ValueError(
            f"{split} line {line_number} {query_id}: invalid {field} {invalid}"
        )
    return categories


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(value: float) -> float:
    return round(float(value), 6)
