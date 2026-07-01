"""Embedding helpers for Phase 7 index construction."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import math
from typing import Protocol


class EmbeddingModel(Protocol):
    """Minimal embedding model interface used by the index builder."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed input texts in their original order."""


class HashEmbeddingModel:
    """Deterministic local embedding model for tests and smoke plumbing."""

    def __init__(self, *, dimension: int = 32) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            values: list[float] = []
            counter = 0
            while len(values) < self.dimension:
                digest = sha256(f"{counter}:{text}".encode("utf-8")).digest()
                for byte in digest:
                    values.append((float(byte) / 127.5) - 1.0)
                    if len(values) == self.dimension:
                        break
                counter += 1
            vectors.append(values)
        return l2_normalize(vectors)


class SentenceTransformerEmbeddingModel:
    """SentenceTransformers wrapper for the documented BAAI/bge-m3 embedder."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for the production embedder; "
                "install the indexing dependency group first"
            ) from error
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


def l2_normalize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return vectors scaled to unit L2 length."""

    normalized: list[list[float]] = []
    for vector in vectors:
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ValueError("cannot L2-normalize a zero vector")
        normalized.append([value / norm for value in values])
    return normalized
