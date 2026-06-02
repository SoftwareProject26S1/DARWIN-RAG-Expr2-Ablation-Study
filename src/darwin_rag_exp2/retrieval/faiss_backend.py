"""FAISS-backed search surface for Phase 9 retrieval variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pyarrow.parquet as pq

from .types import SearchHit


class FaissSearchBackend:
    """Load frozen Phase 7 FAISS indexes and expose variant search methods."""

    def __init__(self, indexes_dir: Path) -> None:
        self.indexes_dir = indexes_dir
        self.manifest = _load_manifest(indexes_dir / "manifest.json")
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "faiss-cpu is required to read Phase 7 indexes; "
                "install the indexing dependency group first"
            ) from error

        self._faiss = faiss
        self._unified_index = faiss.read_index(str(indexes_dir / "unified.faiss"))
        self._unified_id_rows = _load_id_map(indexes_dir / "unified_id_map.parquet")
        self._category_indexes: dict[str, Any] = {}
        self._category_id_rows: dict[str, dict[int, dict[str, object]]] = {}
        for row in self._category_manifest_rows():
            category = str(row["category"])
            self._category_indexes[category] = faiss.read_index(
                str(indexes_dir / str(row["index_file"]))
            )
            self._category_id_rows[category] = _load_id_map(
                indexes_dir / str(row["id_map_file"])
            )

    def search_unified(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        """Return top hits from the unified index."""

        return _search_index(
            self._unified_index,
            self._unified_id_rows,
            query_embedding,
            top_k=top_k,
        )

    def search_category(
        self,
        category: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        """Return top hits from one category partition index."""

        if category not in self._category_indexes:
            raise ValueError(f"unknown category index {category!r}")
        return _search_index(
            self._category_indexes[category],
            self._category_id_rows[category],
            query_embedding,
            top_k=top_k,
        )

    def _category_manifest_rows(self) -> Sequence[Mapping[str, object]]:
        rows = self.manifest.get("category_indexes")
        if not isinstance(rows, list):
            raise ValueError("index manifest must contain category_indexes")
        return [dict(row) for row in rows]


def _search_index(
    index: Any,
    id_rows: Mapping[int, Mapping[str, object]],
    query_embedding: Sequence[float],
    *,
    top_k: int,
) -> list[SearchHit]:
    if top_k <= 0:
        return []
    query_array = np.asarray([query_embedding], dtype="float32")
    distances, indices = index.search(query_array, top_k)
    hits: list[SearchHit] = []
    for offset, vector_index in enumerate(indices[0].tolist(), start=1):
        if int(vector_index) < 0:
            continue
        row = id_rows.get(int(vector_index))
        if row is None:
            raise ValueError(f"FAISS returned unmapped vector index {vector_index}")
        hits.append(
            SearchHit(
                chunk_id=str(row["chunk_id"]),
                source_id=str(row["source_id"]),
                source_category=str(row["source_category"]),
                similarity=float(distances[0][offset - 1]),
                rank=offset,
            )
        )
    return hits


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"missing index manifest: {path}")
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("index manifest must be an object")
    return dict(payload)


def _load_id_map(path: Path) -> dict[int, dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    id_rows: dict[int, dict[str, object]] = {}
    for row in rows:
        vector_index = int(row["vector_index"])
        id_rows[vector_index] = dict(row)
    return id_rows
