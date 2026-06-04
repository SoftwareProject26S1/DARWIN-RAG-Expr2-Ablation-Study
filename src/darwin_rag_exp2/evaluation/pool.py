"""Phase 8 query annotation candidate-pool export."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import orjson
import pyarrow.parquet as pq


@dataclass(frozen=True)
class QueryPoolResult:
    """Rows and manifest for a deterministic query annotation pool."""

    rows: list[dict[str, object]]
    manifest: dict[str, object]


def build_query_pool(
    *,
    chunks_path: Path,
    primary_categories: tuple[str, ...],
    per_category: int,
    seed: int,
    preview_chars: int = 240,
) -> QueryPoolResult:
    """Sample chunk candidates per category for human query annotation."""

    if per_category <= 0:
        raise ValueError("per_category must be positive")
    if preview_chars <= 0:
        raise ValueError("preview_chars must be positive")
    chunk_rows = pq.read_table(chunks_path).to_pylist()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    allowed = set(primary_categories)
    for row in chunk_rows:
        category = str(row.get("category", ""))
        if category in allowed:
            grouped[category].append(dict(row))

    rng = random.Random(seed)
    output_rows: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    for category in primary_categories:
        rows = sorted(grouped.get(category, []), key=lambda row: str(row.get("chunk_id", "")))
        shuffled = list(rows)
        rng.shuffle(shuffled)
        selected = sorted(
            shuffled[:per_category],
            key=lambda row: str(row.get("chunk_id", "")),
        )
        category_counts[category] = len(selected)
        output_rows.extend(
            _pool_row(row, preview_chars=preview_chars)
            for row in selected
        )

    manifest = {
        "phase": 8,
        "artifact_type": "query_pool",
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "seed": seed,
        "per_category": per_category,
        "preview_chars": preview_chars,
        "primary_categories": list(primary_categories),
        "row_count": len(output_rows),
        "category_counts": {
            category: category_counts[category]
            for category in sorted(category_counts)
        },
        "artifact_files": [
            "query_pool.jsonl",
            "query_pool.csv",
            "manifest.json",
        ],
    }
    return QueryPoolResult(rows=output_rows, manifest=manifest)


def write_query_pool_artifacts(output_dir: Path, result: QueryPoolResult) -> None:
    """Write query-pool JSONL, CSV, and manifest artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "query_pool.jsonl", result.rows)
    _write_csv(output_dir / "query_pool.csv", result.rows)
    _write_json(output_dir / "manifest.json", result.manifest)


def _pool_row(row: dict[str, object], *, preview_chars: int) -> dict[str, object]:
    return {
        "chunk_id": str(row.get("chunk_id", "")),
        "source_id": str(row.get("source_id", "")),
        "category": str(row.get("category", "")),
        "title": str(row.get("title", "")),
        "body_preview": _preview(str(row.get("body_text", "")), preview_chars),
        "date": str(row.get("date", "")),
        "url": str(row.get("url", "")),
    }


def _preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 3:
        return compact[:limit]
    return compact[: limit - 3] + "..."


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            output.write(b"\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "chunk_id",
        "source_id",
        "category",
        "title",
        "body_preview",
        "date",
        "url",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
