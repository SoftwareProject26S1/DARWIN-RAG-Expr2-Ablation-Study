"""Artifact writers for Phase 4 chunk outputs."""

from __future__ import annotations

from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .chunking import ChunkingResult, NoticeChunk


def write_chunk_artifacts(result: ChunkingResult, output_dir: Path) -> None:
    """Write chunk JSONL, Parquet, histogram, and manifest artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_chunk_payload(chunk) for chunk in result.chunks]
    _write_jsonl(output_dir / "chunks.jsonl", rows)
    _write_parquet(output_dir / "chunks.parquet", rows)
    output_dir.joinpath("length_histogram.json").write_bytes(
        orjson.dumps(
            result.length_histogram,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
    manifest = {
        **result.manifest,
        "artifact_files": [
            "chunks.jsonl",
            "chunks.parquet",
            "length_histogram.json",
            "manifest.json",
        ],
    }
    output_dir.joinpath("manifest.json").write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            output.write(b"\n")


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows, schema=_chunk_schema())
    pq.write_table(table, path)


def _chunk_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("category", pa.string()),
            pa.field("title", pa.string()),
            pa.field("title_prefix", pa.string()),
            pa.field("body_text", pa.string()),
            pa.field("classifier_text", pa.string()),
            pa.field("body_token_count", pa.int64()),
            pa.field("title_token_count", pa.int64()),
            pa.field("classifier_token_count", pa.int64()),
            pa.field("url", pa.string()),
            pa.field("slug", pa.string()),
            pa.field("date", pa.string()),
            pa.field("source", pa.string()),
            pa.field("collected_at", pa.string()),
        ]
    )


def _chunk_payload(chunk: NoticeChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "chunk_index": chunk.chunk_index,
        "category": chunk.category,
        "title": chunk.title,
        "title_prefix": chunk.title_prefix,
        "body_text": chunk.body_text,
        "classifier_text": chunk.classifier_text,
        "body_token_count": chunk.body_token_count,
        "title_token_count": chunk.title_token_count,
        "classifier_token_count": chunk.classifier_token_count,
        "url": chunk.url,
        "slug": chunk.slug,
        "date": chunk.date,
        "source": chunk.source,
        "collected_at": chunk.collected_at,
    }
