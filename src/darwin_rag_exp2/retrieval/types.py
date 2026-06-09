"""Shared types for Phase 9 retrieval variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class QueryFeatures:
    """Query annotation plus model-derived features used by retrieval variants."""

    query_id: str
    query: str
    embedding: list[float]
    probabilities: dict[str, float]
    gold_chunks: tuple[str, ...] = ()
    gold_categories: tuple[str, ...] = ()
    query_type: str = ""


@dataclass(frozen=True)
class SearchHit:
    """One hit returned from a unified or category vector index."""

    chunk_id: str
    source_id: str
    source_category: str
    similarity: float
    rank: int


@dataclass(frozen=True)
class PartitionHit:
    """One category-partition occurrence of a retrieved chunk."""

    chunk_id: str
    source_id: str
    source_category: str
    partition_category: str
    similarity: float
    rank: int


@dataclass(frozen=True)
class RankedChunk:
    """A scored retrieval result with enough provenance for later reporting."""

    chunk_id: str
    source_id: str
    source_category: str
    partition_category: str | None
    rank: int
    score: float
    similarity: float
    similarity_norm: float
    query_category_probability: float | None = None
    lambda_value: float | None = None
    scoring_method: str = ""


@dataclass(frozen=True)
class VariantResult:
    """Ranked output for one primary retrieval variant."""

    variant: str
    query_id: str
    top10: tuple[RankedChunk, ...]
    top5_contexts: tuple[RankedChunk, ...]


@dataclass(frozen=True)
class PrimaryRunSettings:
    """Frozen settings shared by primary retrieval variants."""

    candidate_k_per_partition: int
    report_top_k: int
    generation_context_top_n: int
    theta_route: float
    lambda_fixed: float
    lambda_by_category: dict[str, float]


class SearchBackend(Protocol):
    """Vector search surface used by Phase 9 variant runners."""

    def search_unified(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        """Return top hits from the unified index."""

    def search_category(
        self,
        category: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        """Return top hits from one category partition index."""
