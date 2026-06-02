"""Build and validate reusable chunk embedding artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .embeddings import EmbeddingModel, l2_normalize


@dataclass(frozen=True)
class EmbeddingArtifactResult:
    """Metadata returned after writing chunk embedding artifacts."""

    manifest: dict[str, object]


@dataclass(frozen=True)
class LoadedEmbeddingArtifacts:
    """Validated precomputed embedding matrix and metadata."""

    vectors: np.ndarray
    manifest: dict[str, object]


def build_embedding_artifacts(
    *,
    chunks_path: Path,
    output_dir: Path,
    embedding_model: EmbeddingModel,
    embedding_model_name: str,
    normalize_embeddings: bool,
    similarity_metric: str,
) -> EmbeddingArtifactResult:
    """Embed canonical chunk rows and write reusable vectors plus id map."""

    chunk_rows = _read_chunk_rows(chunks_path)
    vectors = embedding_model.encode([str(row["body_text"]) for row in chunk_rows])
    if len(vectors) != len(chunk_rows):
        raise ValueError("embedding model returned a different row count")
    normalized_vectors = l2_normalize(vectors) if normalize_embeddings else [
        [float(value) for value in vector]
        for vector in vectors
    ]
    vector_array = np.asarray(normalized_vectors, dtype=np.float32)
    if vector_array.ndim != 2 or vector_array.shape[0] == 0 or vector_array.shape[1] == 0:
        raise ValueError("embedding matrix must be non-empty and two-dimensional")

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "vectors.npy"
    id_map_path = output_dir / "id_map.parquet"
    np.save(vectors_path, vector_array)
    _write_parquet(
        id_map_path,
        [
            _id_map_row(vector_index, row)
            for vector_index, row in enumerate(chunk_rows)
        ],
    )

    manifest: dict[str, object] = {
        "phase": 7,
        "artifact_type": "chunk_embeddings",
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "embedding_model": embedding_model_name,
        "embedding_dimension": int(vector_array.shape[1]),
        "normalize_embeddings": normalize_embeddings,
        "similarity_metric": similarity_metric,
        "chunk_count": int(vector_array.shape[0]),
        "vectors_sha256": _file_sha256(vectors_path),
        "id_map_sha256": _file_sha256(id_map_path),
        "artifact_files": [
            "vectors.npy",
            "id_map.parquet",
            "manifest.json",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return EmbeddingArtifactResult(manifest=manifest)


def load_embedding_artifacts(
    embeddings_dir: Path,
    *,
    chunks_path: Path,
    expected_embedding_model: str,
    expected_normalize_embeddings: bool,
    expected_similarity_metric: str | None = None,
) -> LoadedEmbeddingArtifacts:
    """Load and validate precomputed embedding artifacts for a chunk file."""

    manifest_path = embeddings_dir / "manifest.json"
    vectors_path = embeddings_dir / "vectors.npy"
    id_map_path = embeddings_dir / "id_map.parquet"
    manifest = orjson.loads(manifest_path.read_bytes())

    _require_equal(
        manifest.get("chunks_sha256"),
        _file_sha256(chunks_path),
        "chunks sha256",
    )
    _require_equal(
        manifest.get("embedding_model"),
        expected_embedding_model,
        "embedding model",
    )
    _require_equal(
        manifest.get("normalize_embeddings"),
        expected_normalize_embeddings,
        "normalize_embeddings",
    )
    if expected_similarity_metric is not None:
        _require_equal(
            manifest.get("similarity_metric"),
            expected_similarity_metric,
            "similarity_metric",
        )

    vectors = np.load(vectors_path)
    if vectors.dtype != np.float32:
        raise ValueError("embedding vectors must use float32 dtype")
    if vectors.ndim != 2:
        raise ValueError("embedding vectors must be a two-dimensional matrix")

    chunk_rows = _read_chunk_rows(chunks_path)
    id_map_rows = pq.read_table(id_map_path).to_pylist()
    if vectors.shape[0] != len(chunk_rows):
        raise ValueError("embedding vector row count does not match chunks")
    if len(id_map_rows) != len(chunk_rows):
        raise ValueError("embedding id_map row count does not match chunks")
    if int(manifest.get("chunk_count", -1)) != len(chunk_rows):
        raise ValueError("embedding manifest chunk_count does not match chunks")
    if int(manifest.get("embedding_dimension", -1)) != int(vectors.shape[1]):
        raise ValueError("embedding manifest dimension does not match vectors")

    expected_chunk_ids = [str(row["chunk_id"]) for row in chunk_rows]
    actual_chunk_ids = [str(row.get("chunk_id")) for row in id_map_rows]
    if actual_chunk_ids != expected_chunk_ids:
        raise ValueError("embedding id_map chunk_id order does not match chunks")
    actual_indexes = [int(row.get("vector_index")) for row in id_map_rows]
    if actual_indexes != list(range(len(id_map_rows))):
        raise ValueError("embedding id_map vector_index must be contiguous")

    _require_equal(
        manifest.get("vectors_sha256"),
        _file_sha256(vectors_path),
        "vectors sha256",
    )
    _require_equal(
        manifest.get("id_map_sha256"),
        _file_sha256(id_map_path),
        "id_map sha256",
    )
    return LoadedEmbeddingArtifacts(
        vectors=vectors,
        manifest=dict(manifest),
    )


def _read_chunk_rows(path: Path) -> list[dict[str, object]]:
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


def _id_map_row(vector_index: int, chunk: Mapping[str, object]) -> dict[str, object]:
    return {
        "vector_index": vector_index,
        "chunk_id": str(chunk["chunk_id"]),
        "source_id": str(chunk["source_id"]),
        "source_category": str(chunk["category"]),
    }


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"embedding artifact {label} mismatch")


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
