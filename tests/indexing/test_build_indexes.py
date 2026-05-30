import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2.indexing.artifacts import build_index_artifacts
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
