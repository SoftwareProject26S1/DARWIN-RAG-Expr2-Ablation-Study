import sys
from types import SimpleNamespace

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from darwin_rag_exp2.retrieval.faiss_backend import FaissSearchBackend


class FakeIndex:
    def __init__(self, *, distances: list[float], ids: list[int]) -> None:
        self.distances = distances
        self.ids = ids

    def search(self, query_vectors: np.ndarray, top_k: int):
        assert query_vectors.dtype == np.float32
        assert query_vectors.shape == (1, 2)
        return (
            np.asarray([self.distances[:top_k]], dtype=np.float32),
            np.asarray([self.ids[:top_k]], dtype=np.int64),
        )


def test_faiss_search_backend_maps_unified_and_category_hits(tmp_path, monkeypatch) -> None:
    indexes_path = tmp_path / "indexes"
    indexes_path.mkdir()
    (indexes_path / "unified.faiss").write_bytes(b"fake")
    (indexes_path / "category_000.faiss").write_bytes(b"fake")
    (indexes_path / "manifest.json").write_bytes(
        orjson.dumps(
            {
                "category_indexes": [
                    {
                        "category": "학사",
                        "index_file": "category_000.faiss",
                        "id_map_file": "category_000_id_map.parquet",
                    }
                ]
            }
        )
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "vector_index": 0,
                    "chunk_id": "c0",
                    "source_id": "s0",
                    "source_category": "학사",
                },
                {
                    "vector_index": 1,
                    "chunk_id": "c1",
                    "source_id": "s1",
                    "source_category": "장학",
                },
            ]
        ),
        indexes_path / "unified_id_map.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "vector_index": 0,
                    "chunk_id": "c2",
                    "source_id": "s2",
                    "source_category": "학사",
                    "partition_category": "학사",
                }
            ]
        ),
        indexes_path / "category_000_id_map.parquet",
    )

    fake_faiss = SimpleNamespace(
        read_index=lambda path: (
            FakeIndex(distances=[0.7, 0.2], ids=[1, 0])
            if path.endswith("unified.faiss")
            else FakeIndex(distances=[0.8], ids=[0])
        )
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    backend = FaissSearchBackend(indexes_path)

    unified_hits = backend.search_unified([1.0, 0.0], top_k=2)
    category_hits = backend.search_category("학사", [1.0, 0.0], top_k=1)

    assert [(hit.chunk_id, hit.rank) for hit in unified_hits] == [
        ("c1", 1),
        ("c0", 2),
    ]
    assert [hit.similarity for hit in unified_hits] == pytest.approx([0.7, 0.2])
    assert category_hits[0].chunk_id == "c2"
    assert category_hits[0].source_category == "학사"
    assert category_hits[0].similarity == pytest.approx(0.8)
