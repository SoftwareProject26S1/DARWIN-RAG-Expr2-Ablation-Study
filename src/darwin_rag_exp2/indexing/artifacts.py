"""Build Phase 7 embedding and index artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .embeddings import EmbeddingModel, l2_normalize
from .partitions import build_partition_assignments


@dataclass(frozen=True)
class IndexingConfig:
    """Config values used by Phase 7 index construction."""

    embedding_model: str
    normalize_embeddings: bool
    similarity_metric: str


@dataclass(frozen=True)
class IndexBuildResult:
    """Metadata returned after writing index artifacts."""

    manifest: dict[str, object]


class IndexWriter(Protocol):
    """Minimal vector index writer used by the artifact builder."""

    def write(self, path: Path, vectors: list[list[float]]) -> None:
        """Write vectors to an index path."""


def load_indexing_config(config_path: Path) -> IndexingConfig:
    """Load Phase 7 defaults from the experiment config."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    retrieval = payload.get("retrieval", {})
    return IndexingConfig(
        embedding_model=str(models.get("embedder", "BAAI/bge-m3")),
        normalize_embeddings=bool(retrieval.get("normalize_embeddings", True)),
        similarity_metric=str(
            retrieval.get("similarity_metric", "cosine_via_inner_product")
        ),
    )


def build_index_artifacts(
    *,
    chunks_path: Path,
    predictions_path: Path,
    output_dir: Path,
    embedding_model: EmbeddingModel,
    index_writer: IndexWriter,
    ingest_threshold: float,
    embedding_model_name: str,
    normalize_embeddings: bool = True,
    similarity_metric: str = "cosine_via_inner_product",
) -> IndexBuildResult:
    """Build unified and category-partition vector index artifacts."""

    chunk_rows = _read_chunks(chunks_path)
    prediction_rows = _read_prediction_rows(predictions_path)
    chunk_by_id = {str(row["chunk_id"]): row for row in chunk_rows}
    missing_prediction_chunks = sorted(
        chunk_id
        for chunk_id in chunk_by_id
        if chunk_id not in {str(row.get("chunk_id")) for row in prediction_rows}
    )
    if missing_prediction_chunks:
        raise ValueError(
            f"missing predictions for {len(missing_prediction_chunks)} chunks"
        )

    texts = [str(row["body_text"]) for row in chunk_rows]
    vectors = embedding_model.encode(texts)
    if len(vectors) != len(chunk_rows):
        raise ValueError("embedding model returned a different row count")
    vectors = l2_normalize(vectors) if normalize_embeddings else [
        [float(value) for value in vector]
        for vector in vectors
    ]
    dimension = len(vectors[0]) if vectors else 0
    if dimension == 0:
        raise ValueError("cannot build indexes without vectors")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_writer.write(output_dir / "unified.faiss", vectors)
    unified_id_rows = [
        _id_map_row(index, row, partition_category=None)
        for index, row in enumerate(chunk_rows)
    ]
    _write_parquet(output_dir / "unified_id_map.parquet", unified_id_rows)

    assignments = build_partition_assignments(
        prediction_rows,
        ingest_threshold=ingest_threshold,
    )
    assignment_rows = _enrich_assignments(assignments, chunk_by_id)
    _write_parquet(output_dir / "partition_assignments.parquet", assignment_rows)

    category_indexes = _write_category_indexes(
        output_dir,
        category_rows=assignment_rows,
        chunk_by_id=chunk_by_id,
        vector_by_chunk_id={
            str(row["chunk_id"]): vector
            for row, vector in zip(chunk_rows, vectors, strict=True)
        },
        index_writer=index_writer,
    )
    manifest: dict[str, object] = {
        "phase": 7,
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _file_sha256(predictions_path),
        "embedding_model": embedding_model_name,
        "embedding_dimension": dimension,
        "normalize_embeddings": normalize_embeddings,
        "similarity_metric": similarity_metric,
        "index_backend": getattr(index_writer, "index_backend", type(index_writer).__name__),
        "ingest_threshold": ingest_threshold,
        "chunk_count": len(chunk_rows),
        "partition_assignment_count": len(assignment_rows),
        "category_indexes": category_indexes,
        "artifact_files": [
            "unified.faiss",
            "unified_id_map.parquet",
            "partition_assignments.parquet",
            "manifest.json",
        ]
        + [
            file_name
            for category_index in category_indexes
            for file_name in [
                str(category_index["index_file"]),
                str(category_index["id_map_file"]),
            ]
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return IndexBuildResult(manifest=manifest)


def _read_chunks(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"no chunks found in {path}")
    required = {"chunk_id", "source_id", "category", "body_text"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"chunk row missing required columns: {sorted(missing)}")
    return sorted(rows, key=lambda row: str(row["chunk_id"]))


def _read_prediction_rows(path: Path) -> list[dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"no predictions found in {path}")
    return sorted(rows, key=lambda row: str(row["chunk_id"]))


def _enrich_assignments(
    assignments: Sequence[Mapping[str, object]],
    chunk_by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for assignment in assignments:
        chunk_id = str(assignment["chunk_id"])
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"assignment references unknown chunk_id {chunk_id!r}")
        rows.append(
            {
                "chunk_id": chunk_id,
                "source_id": str(chunk["source_id"]),
                "source_category": str(chunk["category"]),
                "partition_category": str(assignment["category"]),
                "category": str(assignment["category"]),
                "probability": float(assignment["probability"]),
                "assignment_reason": str(assignment["assignment_reason"]),
            }
        )
    return rows


def _write_category_indexes(
    output_dir: Path,
    *,
    category_rows: Sequence[Mapping[str, object]],
    chunk_by_id: Mapping[str, Mapping[str, object]],
    vector_by_chunk_id: Mapping[str, list[float]],
    index_writer: IndexWriter,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in category_rows:
        grouped[str(row["partition_category"])].append(row)

    category_indexes: list[dict[str, object]] = []
    for category_index, category in enumerate(sorted(grouped)):
        rows = sorted(grouped[category], key=lambda row: str(row["chunk_id"]))
        prefix = f"category_{category_index:03d}"
        index_path = output_dir / f"{prefix}.faiss"
        id_map_path = output_dir / f"{prefix}_id_map.parquet"
        vectors = [
            vector_by_chunk_id[str(row["chunk_id"])]
            for row in rows
        ]
        index_writer.write(index_path, vectors)
        id_rows = [
            {
                **_id_map_row(
                    vector_index,
                    chunk_by_id[str(row["chunk_id"])],
                    partition_category=category,
                ),
                "probability": float(row["probability"]),
                "assignment_reason": str(row["assignment_reason"]),
            }
            for vector_index, row in enumerate(rows)
        ]
        _write_parquet(id_map_path, id_rows)
        category_indexes.append(
            {
                "category": category,
                "chunk_count": len(rows),
                "index_file": index_path.name,
                "id_map_file": id_map_path.name,
            }
        )
    return category_indexes


def _id_map_row(
    vector_index: int,
    chunk: Mapping[str, object],
    *,
    partition_category: str | None,
) -> dict[str, object]:
    row = {
        "vector_index": vector_index,
        "chunk_id": str(chunk["chunk_id"]),
        "source_id": str(chunk["source_id"]),
        "source_category": str(chunk["category"]),
    }
    if partition_category is not None:
        row["partition_category"] = partition_category
    return row


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows)), path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
