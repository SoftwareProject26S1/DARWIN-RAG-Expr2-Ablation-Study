import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from darwin_rag_exp2.cli import main
from darwin_rag_exp2.indexing.artifacts import build_index_artifacts
from darwin_rag_exp2.indexing.embedding_artifacts import build_embedding_artifacts
from darwin_rag_exp2.indexing.embeddings import HashEmbeddingModel


class RecordingIndexWriter:
    def __init__(self) -> None:
        self.writes: list[tuple[Path, int, int]] = []

    def write(self, path: Path, vectors: list[list[float]]) -> None:
        dimension = len(vectors[0]) if vectors else 0
        self.writes.append((path, len(vectors), dimension))
        path.write_text(
            json.dumps({"count": len(vectors), "dimension": dimension}),
            encoding="utf-8",
        )


def test_build_index_artifacts_writes_manifest_id_maps_and_partition_provenance(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    writer = RecordingIndexWriter()
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "학사 일정 안내"),
                chunk_row("c2", "장학 신청 안내"),
                chunk_row("c3", "교환학생 모집 안내"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                prediction_row("c1", {"학사": 0.8, "장학": 0.2}),
                prediction_row("c2", {"학사": 0.7, "장학": 0.75}),
                prediction_row("c3", {"국제교류": 0.55, "학사": 0.45}),
            ]
        ),
        predictions_path,
    )

    result = build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        index_writer=writer,
        ingest_threshold=0.6,
        embedding_model_name="deterministic-hash",
    )

    manifest = json.loads((output_path / "manifest.json").read_text())
    assignments = pq.read_table(output_path / "partition_assignments.parquet").to_pylist()

    assert result.manifest["phase"] == 7
    assert manifest["chunk_count"] == 3
    assert manifest["embedding_model"] == "deterministic-hash"
    assert manifest["normalize_embeddings"] is True
    assert manifest["similarity_metric"] == "cosine_via_inner_product"
    assert manifest["ingest_threshold"] == 0.6
    assert (output_path / "unified.faiss").exists()
    assert (output_path / "unified_id_map.parquet").exists()
    assert len([write for write in writer.writes if write[0].name.endswith(".faiss")]) == 4
    assert [
        (row["chunk_id"], row["category"], row["assignment_reason"])
        for row in assignments
    ] == [
        ("c1", "학사", "threshold"),
        ("c2", "장학", "threshold"),
        ("c2", "학사", "threshold"),
        ("c3", "국제교류", "top1_fallback"),
    ]


def test_build_index_artifacts_reuses_precomputed_embeddings_without_encoding(
    tmp_path,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    embeddings_path = tmp_path / "embeddings"
    output_path = tmp_path / "indexes"
    writer = RecordingIndexWriter()
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "학사 일정 안내"),
                chunk_row("c2", "장학 신청 안내"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                prediction_row("c1", {"학사": 0.8}),
                prediction_row("c2", {"장학": 0.9}),
            ]
        ),
        predictions_path,
    )
    build_embedding_artifacts(
        chunks_path=chunks_path,
        output_dir=embeddings_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        embedding_model_name="deterministic-hash",
        normalize_embeddings=True,
        similarity_metric="cosine_via_inner_product",
    )

    result = build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=ExplodingEmbeddingModel(),
        embedding_artifacts_dir=embeddings_path,
        index_writer=writer,
        ingest_threshold=0.6,
        embedding_model_name="deterministic-hash",
    )

    assert result.manifest["embedding_artifacts_path"] == str(embeddings_path)
    assert result.manifest["embedding_vectors_sha256"]
    assert writer.writes[0] == (output_path / "unified.faiss", 2, 6)


def test_build_index_artifacts_rejects_incompatible_embedding_artifacts(
    tmp_path,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    embeddings_path = tmp_path / "embeddings"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist([chunk_row("c1", "학사 일정 안내")]),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist([prediction_row("c1", {"학사": 0.8})]),
        predictions_path,
    )
    build_embedding_artifacts(
        chunks_path=chunks_path,
        output_dir=embeddings_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        embedding_model_name="deterministic-hash",
        normalize_embeddings=True,
        similarity_metric="cosine_via_inner_product",
    )

    with pytest.raises(ValueError, match="embedding model"):
        build_index_artifacts(
            chunks_path=chunks_path,
            predictions_path=predictions_path,
            output_dir=output_path,
            embedding_model=HashEmbeddingModel(dimension=6),
            embedding_artifacts_dir=embeddings_path,
            index_writer=RecordingIndexWriter(),
            ingest_threshold=0.6,
            embedding_model_name="different-model",
        )


def test_build_embeddings_and_indexes_cli_reuses_precomputed_embeddings(
    tmp_path,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    config_path = tmp_path / "experiment.yaml"
    embeddings_path = tmp_path / "embeddings"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "학사 일정 안내"),
                chunk_row("c2", "장학 신청 안내"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                prediction_row("c1", {"학사": 0.8}),
                prediction_row("c2", {"장학": 0.9}),
            ]
        ),
        predictions_path,
    )
    config_path.write_text(
        """
models:
  embedder: deterministic-hash
retrieval:
  normalize_embeddings: true
  similarity_metric: cosine_via_inner_product
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "darwin_rag_exp2.cli.FaissIndexWriter",
        lambda: RecordingIndexWriter(),
    )

    assert main(
        [
            "build-embeddings",
            "--chunks",
            str(chunks_path),
            "--config",
            str(config_path),
            "--embedding-backend",
            "hash",
            "--output",
            str(embeddings_path),
        ]
    ) == 0
    assert main(
        [
            "build-indexes",
            "--chunks",
            str(chunks_path),
            "--predictions",
            str(predictions_path),
            "--config",
            str(config_path),
            "--embeddings",
            str(embeddings_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    assert (embeddings_path / "vectors.npy").exists()
    assert (output_path / "unified.faiss").exists()
    assert manifest["embedding_artifacts_path"] == str(embeddings_path)


class ExplodingEmbeddingModel:
    def encode(self, texts):
        raise AssertionError("precomputed embedding path must not encode texts")


def chunk_row(chunk_id: str, body_text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": f"{chunk_id}-source",
        "category": "학사",
        "body_text": body_text,
    }


def prediction_row(chunk_id: str, probabilities: dict[str, float]) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "probabilities_json": json.dumps(probabilities, ensure_ascii=False),
    }
