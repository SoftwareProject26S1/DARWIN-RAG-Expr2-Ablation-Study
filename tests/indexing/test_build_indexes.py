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
    assert manifest["partition_top_k"] == 1
    assert manifest["partition_assignment_level"] == "source"
    assert manifest["prediction_artifact_level"] == "chunk"
    assert manifest["source_probability_aggregation"] == "max"
    assert manifest["source_count"] == 3
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


def test_build_index_artifacts_records_top_k_partition_assignments(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
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
                prediction_row("c1", {"학사": 0.8, "장학": 0.5}),
                prediction_row("c2", {"학사": 0.2, "장학": 0.9}),
            ]
        ),
        predictions_path,
    )

    build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        index_writer=writer,
        ingest_threshold=0.6,
        partition_top_k=2,
        embedding_model_name="deterministic-hash",
    )

    manifest = json.loads((output_path / "manifest.json").read_text())
    assignments = pq.read_table(output_path / "partition_assignments.parquet").to_pylist()

    assert manifest["partition_top_k"] == 2
    assert manifest["partition_assignment_count"] == 4
    assert [
        (row["chunk_id"], row["category"], row["assignment_reason"])
        for row in assignments
    ] == [
        ("c1", "학사", "threshold"),
        ("c1", "장학", "top_k_fallback"),
        ("c2", "장학", "threshold"),
        ("c2", "학사", "top_k_fallback"),
    ]
    assert len([write for write in writer.writes if write[0].name.endswith(".faiss")]) == 3


def test_build_index_artifacts_assigns_partitions_at_source_level(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    writer = RecordingIndexWriter()
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "학사 일정 안내", source_id="source-1"),
                chunk_row("c2", "장학 신청 안내", source_id="source-1"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                prediction_row("c1", {"학사": 0.9, "장학": 0.1, "채용": 0.4}),
                prediction_row("c2", {"학사": 0.2, "장학": 0.85, "채용": 0.3}),
            ]
        ),
        predictions_path,
    )

    build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        index_writer=writer,
        ingest_threshold=0.8,
        partition_top_k=2,
        embedding_model_name="deterministic-hash",
    )

    manifest = json.loads((output_path / "manifest.json").read_text())
    assignments = pq.read_table(output_path / "partition_assignments.parquet").to_pylist()

    assert manifest["partition_assignment_level"] == "source"
    assert manifest["prediction_artifact_level"] == "chunk"
    assert manifest["source_probability_aggregation"] == "max"
    assert manifest["source_count"] == 1
    assert manifest["partition_assignment_count"] == 4
    assert [
        (row["chunk_id"], row["category"], row["probability"], row["assignment_reason"])
        for row in assignments
    ] == [
        ("c1", "학사", 0.9, "threshold"),
        ("c1", "장학", 0.85, "threshold"),
        ("c2", "학사", 0.9, "threshold"),
        ("c2", "장학", 0.85, "threshold"),
    ]


def test_build_index_artifacts_uses_source_level_prediction_artifact(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "학사 일정 안내", source_id="source-1"),
                chunk_row("c2", "장학 신청 안내", source_id="source-1"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                source_prediction_row(
                    "source-1",
                    {"학사": 0.9, "장학": 0.85, "채용": 0.1},
                ),
            ]
        ),
        predictions_path,
    )

    build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        index_writer=RecordingIndexWriter(),
        ingest_threshold=0.8,
        embedding_model_name="deterministic-hash",
    )

    manifest = json.loads((output_path / "manifest.json").read_text())
    assignments = pq.read_table(output_path / "partition_assignments.parquet").to_pylist()

    assert manifest["prediction_artifact_level"] == "source"
    assert manifest["source_probability_aggregation"] == "provided"
    assert manifest["partition_assignment_count"] == 4
    assert [
        (row["chunk_id"], row["category"], row["probability"], row["assignment_reason"])
        for row in assignments
    ] == [
        ("c1", "학사", 0.9, "threshold"),
        ("c1", "장학", 0.85, "threshold"),
        ("c2", "학사", 0.9, "threshold"),
        ("c2", "장학", 0.85, "threshold"),
    ]


def test_build_index_artifacts_applies_top_k_after_source_aggregation(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "학사 일정 안내", source_id="source-1"),
                chunk_row("c2", "장학 신청 안내", source_id="source-1"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                prediction_row("c1", {"학사": 0.9, "장학": 0.55, "채용": 0.1}),
                prediction_row("c2", {"학사": 0.3, "장학": 0.5, "채용": 0.4}),
            ]
        ),
        predictions_path,
    )

    build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        index_writer=RecordingIndexWriter(),
        ingest_threshold=0.8,
        partition_top_k=2,
        embedding_model_name="deterministic-hash",
    )

    assignments = pq.read_table(output_path / "partition_assignments.parquet").to_pylist()

    assert [
        (row["chunk_id"], row["category"], row["probability"], row["assignment_reason"])
        for row in assignments
    ] == [
        ("c1", "학사", 0.9, "threshold"),
        ("c1", "장학", 0.55, "top_k_fallback"),
        ("c2", "학사", 0.9, "threshold"),
        ("c2", "장학", 0.55, "top_k_fallback"),
    ]


def test_build_index_artifacts_rejects_unknown_prediction_chunk(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist([chunk_row("c1", "학사 일정 안내")]),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                prediction_row("c1", {"학사": 0.8}),
                prediction_row("unknown", {"학사": 0.8}),
            ]
        ),
        predictions_path,
    )

    with pytest.raises(ValueError, match="unknown prediction chunk_id"):
        build_index_artifacts(
            chunks_path=chunks_path,
            predictions_path=predictions_path,
            output_dir=output_path,
            embedding_model=HashEmbeddingModel(dimension=6),
            index_writer=RecordingIndexWriter(),
            ingest_threshold=0.6,
            embedding_model_name="deterministic-hash",
        )


def test_build_index_artifacts_rejects_missing_prediction_source(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "학사 일정 안내", source_id="source-1"),
                chunk_row("c2", "장학 신청 안내", source_id="source-2"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [source_prediction_row("source-1", {"학사": 0.8})]
        ),
        predictions_path,
    )

    with pytest.raises(ValueError, match="missing predictions for 1 sources"):
        build_index_artifacts(
            chunks_path=chunks_path,
            predictions_path=predictions_path,
            output_dir=output_path,
            embedding_model=HashEmbeddingModel(dimension=6),
            index_writer=RecordingIndexWriter(),
            ingest_threshold=0.6,
            embedding_model_name="deterministic-hash",
        )


def test_build_index_artifacts_rejects_unknown_prediction_source(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist([chunk_row("c1", "학사 일정 안내", source_id="source-1")]),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                source_prediction_row("source-1", {"학사": 0.8}),
                source_prediction_row("unknown", {"학사": 0.8}),
            ]
        ),
        predictions_path,
    )

    with pytest.raises(ValueError, match="unknown prediction source_id"):
        build_index_artifacts(
            chunks_path=chunks_path,
            predictions_path=predictions_path,
            output_dir=output_path,
            embedding_model=HashEmbeddingModel(dimension=6),
            index_writer=RecordingIndexWriter(),
            ingest_threshold=0.6,
            embedding_model_name="deterministic-hash",
        )


def test_build_index_artifacts_rejects_duplicate_prediction_source(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist([chunk_row("c1", "학사 일정 안내", source_id="source-1")]),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                source_prediction_row("source-1", {"학사": 0.8}),
                source_prediction_row("source-1", {"장학": 0.7}),
            ]
        ),
        predictions_path,
    )

    with pytest.raises(ValueError, match="duplicate prediction source_id"):
        build_index_artifacts(
            chunks_path=chunks_path,
            predictions_path=predictions_path,
            output_dir=output_path,
            embedding_model=HashEmbeddingModel(dimension=6),
            index_writer=RecordingIndexWriter(),
            ingest_threshold=0.6,
            embedding_model_name="deterministic-hash",
        )


def test_build_index_artifacts_rejects_mixed_prediction_levels(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    pq.write_table(
        pa.Table.from_pylist([chunk_row("c1", "학사 일정 안내", source_id="source-1")]),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                prediction_row("c1", {"학사": 0.8}),
                source_prediction_row("source-1", {"학사": 0.8}),
            ]
        ),
        predictions_path,
    )

    with pytest.raises(ValueError, match="mixed prediction levels"):
        build_index_artifacts(
            chunks_path=chunks_path,
            predictions_path=predictions_path,
            output_dir=output_path,
            embedding_model=HashEmbeddingModel(dimension=6),
            index_writer=RecordingIndexWriter(),
            ingest_threshold=0.6,
            embedding_model_name="deterministic-hash",
        )


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
                chunk_row("c1", "학사 일정 안내", source_id="source-1"),
                chunk_row("c2", "장학 신청 안내", source_id="source-1"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                source_prediction_row("source-1", {"학사": 0.8, "장학": 0.9}),
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


def test_build_index_artifacts_encodes_embedding_text(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "indexes"
    embedding_model = RecordingEmbeddingModel()
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row(
                    "c1",
                    "본문만",
                    embedding_text="제목 포함\n\n본문만",
                ),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist([prediction_row("c1", {"학사": 0.8})]),
        predictions_path,
    )

    build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=embedding_model,
        index_writer=RecordingIndexWriter(),
        ingest_threshold=0.6,
        embedding_model_name="recording",
    )

    assert embedding_model.encoded_texts == ["제목 포함\n\n본문만"]


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
                chunk_row("c1", "학사 일정 안내", source_id="source-1"),
                chunk_row("c2", "장학 신청 안내", source_id="source-1"),
            ]
        ),
        chunks_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                source_prediction_row("source-1", {"학사": 0.8, "장학": 0.6}),
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
indexing:
  ingest_threshold: 0.7
  partition_top_k: 2
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
    assert manifest["ingest_threshold"] == 0.7
    assert manifest["partition_top_k"] == 2
    assert manifest["partition_assignment_level"] == "source"
    assert manifest["prediction_artifact_level"] == "source"
    assert manifest["source_probability_aggregation"] == "provided"
    assert manifest["source_count"] == 1
    assignments = pq.read_table(output_path / "partition_assignments.parquet").to_pylist()
    assert [
        (row["chunk_id"], row["category"], row["assignment_reason"])
        for row in assignments
    ] == [
        ("c1", "학사", "threshold"),
        ("c1", "장학", "top_k_fallback"),
        ("c2", "학사", "threshold"),
        ("c2", "장학", "top_k_fallback"),
    ]


class ExplodingEmbeddingModel:
    def encode(self, texts):
        raise AssertionError("precomputed embedding path must not encode texts")


class RecordingEmbeddingModel:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, texts) -> list[list[float]]:
        self.encoded_texts = list(texts)
        return [[1.0, 0.0] for _ in self.encoded_texts]


def chunk_row(
    chunk_id: str,
    body_text: str,
    *,
    embedding_text: str | None = None,
    source_id: str | None = None,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id or f"{chunk_id}-source",
        "category": "학사",
        "body_text": body_text,
        "embedding_text": embedding_text or body_text,
    }


def prediction_row(chunk_id: str, probabilities: dict[str, float]) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "probabilities_json": json.dumps(probabilities, ensure_ascii=False),
    }


def source_prediction_row(
    source_id: str,
    probabilities: dict[str, float],
    *,
    category: str = "학사",
) -> dict[str, object]:
    predicted_category, confidence = max(
        probabilities.items(),
        key=lambda item: (item[1], item[0]),
    )
    return {
        "source_id": source_id,
        "category": category,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "probabilities_json": json.dumps(probabilities, ensure_ascii=False),
    }
