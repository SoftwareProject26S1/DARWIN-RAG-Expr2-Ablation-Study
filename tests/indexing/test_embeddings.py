import math
import json
from types import SimpleNamespace
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from darwin_rag_exp2.indexing.embedding_artifacts import (
    build_embedding_artifacts,
    load_embedding_artifacts,
)
from darwin_rag_exp2.indexing.embeddings import (
    HashEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    l2_normalize,
)


def test_l2_normalize_makes_each_vector_unit_length() -> None:
    vectors = l2_normalize([[3.0, 4.0], [5.0, 12.0]])

    assert vectors == [[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]]
    assert [round(math.sqrt(sum(value * value for value in row)), 12) for row in vectors] == [
        1.0,
        1.0,
    ]


def test_l2_normalize_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        l2_normalize([[0.0, 0.0]])


def test_hash_embedding_model_is_deterministic_and_normalized() -> None:
    model = HashEmbeddingModel(dimension=8)

    first = model.encode(["학사 일정 안내", "장학 신청 안내"])
    second = model.encode(["학사 일정 안내", "장학 신청 안내"])

    assert first == second
    assert first[0] != first[1]
    assert all(
        round(math.sqrt(sum(value * value for value in row)), 12) == 1.0
        for row in first
    )


def test_sentence_transformer_embedding_model_passes_device(monkeypatch) -> None:
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, *, device=None):
            calls.append({"model_name": model_name, "device": device})

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    SentenceTransformerEmbeddingModel("BAAI/bge-m3", device="cuda:1")

    assert calls == [{"model_name": "BAAI/bge-m3", "device": "cuda:1"}]


def test_build_embedding_artifacts_writes_vectors_id_map_and_manifest(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "embeddings"
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c2", "s2", "장학", "장학 신청 안내"),
                chunk_row("c1", "s1", "학사", "학사 일정 안내"),
            ]
        ),
        chunks_path,
    )

    result = build_embedding_artifacts(
        chunks_path=chunks_path,
        output_dir=output_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        embedding_model_name="deterministic-hash",
        normalize_embeddings=True,
        similarity_metric="cosine_via_inner_product",
    )

    vectors = np.load(output_path / "vectors.npy")
    id_rows = pq.read_table(output_path / "id_map.parquet").to_pylist()
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))

    assert result.manifest["phase"] == 7
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 6)
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.ones(2))
    assert [
        (row["vector_index"], row["chunk_id"], row["source_id"], row["source_category"])
        for row in id_rows
    ] == [
        (0, "c1", "s1", "학사"),
        (1, "c2", "s2", "장학"),
    ]
    assert manifest["artifact_files"] == [
        "vectors.npy",
        "id_map.parquet",
        "manifest.json",
    ]
    assert manifest["chunk_count"] == 2
    assert manifest["embedding_dimension"] == 6
    assert manifest["embedding_model"] == "deterministic-hash"
    assert manifest["normalize_embeddings"] is True
    assert manifest["similarity_metric"] == "cosine_via_inner_product"
    assert manifest["vectors_sha256"] == result.manifest["vectors_sha256"]
    assert manifest["id_map_sha256"] == result.manifest["id_map_sha256"]


def test_load_embedding_artifacts_rejects_chunk_id_order_mismatch(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "embeddings"
    pq.write_table(
        pa.Table.from_pylist(
            [
                chunk_row("c1", "s1", "학사", "학사 일정 안내"),
                chunk_row("c2", "s2", "장학", "장학 신청 안내"),
            ]
        ),
        chunks_path,
    )
    build_embedding_artifacts(
        chunks_path=chunks_path,
        output_dir=output_path,
        embedding_model=HashEmbeddingModel(dimension=6),
        embedding_model_name="deterministic-hash",
        normalize_embeddings=True,
        similarity_metric="cosine_via_inner_product",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "vector_index": 0,
                    "chunk_id": "c2",
                    "source_id": "s2",
                    "source_category": "장학",
                },
                {
                    "vector_index": 1,
                    "chunk_id": "c1",
                    "source_id": "s1",
                    "source_category": "학사",
                },
            ]
        ),
        output_path / "id_map.parquet",
    )

    with pytest.raises(ValueError, match="chunk_id order"):
        load_embedding_artifacts(
            output_path,
            chunks_path=chunks_path,
            expected_embedding_model="deterministic-hash",
            expected_normalize_embeddings=True,
        )


def chunk_row(
    chunk_id: str,
    source_id: str,
    category: str,
    body_text: str,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "category": category,
        "body_text": body_text,
    }
