from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2.evaluation.pool import (
    build_query_pool,
    write_query_pool_artifacts,
)


def test_build_query_pool_samples_deterministically_by_category(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    _write_chunks(chunks_path)

    first = build_query_pool(
        chunks_path=chunks_path,
        primary_categories=("학사", "장학"),
        per_category=2,
        seed=7,
        preview_chars=24,
    )
    second = build_query_pool(
        chunks_path=chunks_path,
        primary_categories=("학사", "장학"),
        per_category=2,
        seed=7,
        preview_chars=24,
    )

    assert first.rows == second.rows
    assert len(first.rows) == 4
    assert {row["category"] for row in first.rows} == {"학사", "장학"}
    assert all(len(str(row["body_preview"])) <= 24 for row in first.rows)
    assert first.manifest["category_counts"] == {"장학": 2, "학사": 2}


def test_write_query_pool_artifacts_writes_jsonl_csv_and_manifest(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "pool"
    _write_chunks(chunks_path)
    result = build_query_pool(
        chunks_path=chunks_path,
        primary_categories=("학사", "장학"),
        per_category=1,
        seed=42,
        preview_chars=60,
    )

    write_query_pool_artifacts(output_path, result)

    assert {
        "query_pool.jsonl",
        "query_pool.csv",
        "manifest.json",
    }.issubset({path.name for path in output_path.iterdir()})
    rows = [
        json.loads(line)
        for line in (output_path / "query_pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert set(rows[0]) == {
        "chunk_id",
        "source_id",
        "category",
        "title",
        "body_preview",
        "date",
        "url",
    }
    assert manifest["artifact_type"] == "query_pool"
    assert manifest["seed"] == 42


def _write_chunks(path) -> None:
    rows = []
    for category in ["학사", "장학"]:
        for index in range(3):
            rows.append(
                {
                    "chunk_id": f"{category}-{index}",
                    "source_id": f"source-{category}-{index}",
                    "category": category,
                    "title": f"{category} 공지 {index}",
                    "body_text": f"{category} 본문 {index} " * 20,
                    "date": f"2026-06-0{index + 1}",
                    "url": f"https://example.test/{category}/{index}",
                }
            )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
