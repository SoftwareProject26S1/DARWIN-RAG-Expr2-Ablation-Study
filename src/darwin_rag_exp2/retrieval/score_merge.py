"""Score-merge functions for Phase 9 primary retrieval variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .types import PartitionHit, RankedChunk


def cosine_to_unit_interval(similarity: float) -> float:
    """Map cosine/inner-product similarity from [-1, 1] to [0, 1]."""

    return (float(similarity) + 1.0) / 2.0


def score_merge_candidates(
    candidates: Sequence[PartitionHit],
    *,
    query_probabilities: Mapping[str, float],
    lambda_by_category: Mapping[str, float],
    limit: int,
    scoring_method: str = "score_merge",
) -> tuple[RankedChunk, ...]:
    """Score partition hits and keep each chunk's best-scoring occurrence."""

    best_by_chunk_id: dict[str, RankedChunk] = {}
    for candidate in candidates:
        category = candidate.partition_category
        if category not in lambda_by_category:
            raise ValueError(f"missing lambda for category {category!r}")
        if category not in query_probabilities:
            raise ValueError(f"missing query probability for category {category!r}")
        lambda_value = float(lambda_by_category[category])
        query_probability = float(query_probabilities[category])
        similarity_norm = cosine_to_unit_interval(candidate.similarity)
        score = (
            lambda_value * similarity_norm
            + (1.0 - lambda_value) * query_probability
        )
        ranked = RankedChunk(
            chunk_id=candidate.chunk_id,
            source_id=candidate.source_id,
            source_category=candidate.source_category,
            partition_category=category,
            rank=0,
            score=_metric(score),
            similarity=_metric(candidate.similarity),
            similarity_norm=_metric(similarity_norm),
            query_category_probability=_metric(query_probability),
            lambda_value=_metric(lambda_value),
            scoring_method=scoring_method,
        )
        previous = best_by_chunk_id.get(candidate.chunk_id)
        if previous is None or _is_better_occurrence(ranked, previous):
            best_by_chunk_id[candidate.chunk_id] = ranked

    ordered = sorted(
        best_by_chunk_id.values(),
        key=lambda row: (
            -row.score,
            -row.similarity,
            row.partition_category or "",
            row.chunk_id,
        ),
    )
    return _rerank(ordered[:limit])


def _is_better_occurrence(candidate: RankedChunk, previous: RankedChunk) -> bool:
    return (
        candidate.score,
        candidate.similarity,
        -(ord(candidate.partition_category[0]) if candidate.partition_category else 0),
    ) > (
        previous.score,
        previous.similarity,
        -(ord(previous.partition_category[0]) if previous.partition_category else 0),
    )


def _rerank(rows: Sequence[RankedChunk]) -> tuple[RankedChunk, ...]:
    return tuple(
        RankedChunk(
            chunk_id=row.chunk_id,
            source_id=row.source_id,
            source_category=row.source_category,
            partition_category=row.partition_category,
            rank=index,
            score=row.score,
            similarity=row.similarity,
            similarity_norm=row.similarity_norm,
            query_category_probability=row.query_category_probability,
            lambda_value=row.lambda_value,
            scoring_method=row.scoring_method,
        )
        for index, row in enumerate(rows, start=1)
    )


def _metric(value: float) -> float:
    return round(float(value), 12)
