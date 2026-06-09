"""Primary Phase 9 retrieval variant runners."""

from __future__ import annotations

from collections.abc import Mapping

from .routing import soft_route_categories, top1_category
from .score_merge import cosine_to_unit_interval, score_merge_candidates
from .types import (
    PartitionHit,
    PrimaryRunSettings,
    QueryFeatures,
    RankedChunk,
    SearchBackend,
    SearchHit,
    VariantResult,
)


PRIMARY_VARIANTS = ("B0", "B1", "B2-score", "P-score")
SEARCH_MODE_CATEGORY_SCORE_MERGE = "category-score-merge"
SEARCH_MODE_UNIFIED_PRIOR_RERANK = "unified-prior-rerank"
SEARCH_MODES = (
    SEARCH_MODE_CATEGORY_SCORE_MERGE,
    SEARCH_MODE_UNIFIED_PRIOR_RERANK,
)


def run_primary_variants(
    query: QueryFeatures,
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
    search_mode: str = SEARCH_MODE_CATEGORY_SCORE_MERGE,
    unified_candidate_k: int = 100,
) -> dict[str, VariantResult]:
    """Run all four primary retrieval variants for one query."""

    _validate_search_mode(search_mode)
    return {
        "B0": run_b0(query, search_backend=search_backend, settings=settings),
        "B1": run_b1(query, search_backend=search_backend, settings=settings),
        "B2-score": run_b2_score(
            query,
            search_backend=search_backend,
            settings=settings,
            search_mode=search_mode,
            unified_candidate_k=unified_candidate_k,
        ),
        "P-score": run_p_score(
            query,
            search_backend=search_backend,
            settings=settings,
            search_mode=search_mode,
            unified_candidate_k=unified_candidate_k,
        ),
    }


def run_b0(
    query: QueryFeatures,
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
) -> VariantResult:
    """Run the unified-index similarity baseline."""

    hits = search_backend.search_unified(
        query.embedding,
        top_k=settings.report_top_k,
    )
    ranked = _rank_similarity_hits(hits, partition_category=None)
    return _variant_result("B0", query.query_id, ranked, settings)


def run_b1(
    query: QueryFeatures,
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
) -> VariantResult:
    """Run the hard-routing top-1 category baseline."""

    category = top1_category(query.probabilities)
    hits = search_backend.search_category(
        category,
        query.embedding,
        top_k=settings.report_top_k,
    )
    ranked = _rank_similarity_hits(hits, partition_category=category)
    return _variant_result("B1", query.query_id, ranked, settings)


def run_b2_score(
    query: QueryFeatures,
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
    search_mode: str = SEARCH_MODE_CATEGORY_SCORE_MERGE,
    unified_candidate_k: int = 100,
) -> VariantResult:
    """Run soft routing with a single fixed lambda score-merge coefficient."""

    _validate_search_mode(search_mode)
    if search_mode == SEARCH_MODE_UNIFIED_PRIOR_RERANK:
        ranked = _run_unified_prior_rerank(
            query,
            search_backend=search_backend,
            settings=settings,
            lambda_by_category={
                category: settings.lambda_fixed
                for category in settings.lambda_by_category
            },
            unified_candidate_k=unified_candidate_k,
        )
        return _variant_result("B2-score", query.query_id, ranked, settings)

    categories = soft_route_categories(
        query.probabilities,
        theta_route=settings.theta_route,
    )
    lambda_by_category = {
        category: settings.lambda_fixed
        for category in categories
    }
    ranked = _run_score_merge(
        query,
        search_backend=search_backend,
        settings=settings,
        categories=categories,
        lambda_by_category=lambda_by_category,
    )
    return _variant_result("B2-score", query.query_id, ranked, settings)


def run_p_score(
    query: QueryFeatures,
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
    search_mode: str = SEARCH_MODE_CATEGORY_SCORE_MERGE,
    unified_candidate_k: int = 100,
) -> VariantResult:
    """Run soft routing with category-specific adaptive lambda values."""

    _validate_search_mode(search_mode)
    if search_mode == SEARCH_MODE_UNIFIED_PRIOR_RERANK:
        ranked = _run_unified_prior_rerank(
            query,
            search_backend=search_backend,
            settings=settings,
            lambda_by_category=settings.lambda_by_category,
            unified_candidate_k=unified_candidate_k,
        )
        return _variant_result("P-score", query.query_id, ranked, settings)

    categories = soft_route_categories(
        query.probabilities,
        theta_route=settings.theta_route,
    )
    lambda_by_category = {
        category: settings.lambda_by_category[category]
        for category in categories
    }
    ranked = _run_score_merge(
        query,
        search_backend=search_backend,
        settings=settings,
        categories=categories,
        lambda_by_category=lambda_by_category,
    )
    return _variant_result("P-score", query.query_id, ranked, settings)


def _run_unified_prior_rerank(
    query: QueryFeatures,
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
    lambda_by_category: Mapping[str, float],
    unified_candidate_k: int,
) -> tuple[RankedChunk, ...]:
    if unified_candidate_k <= 0:
        raise ValueError("unified_candidate_k must be positive")
    hits = search_backend.search_unified(
        query.embedding,
        top_k=unified_candidate_k,
    )
    candidates = [
        PartitionHit(
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
            source_category=hit.source_category,
            partition_category=hit.source_category,
            similarity=hit.similarity,
            rank=hit.rank,
        )
        for hit in hits
    ]
    return score_merge_candidates(
        candidates,
        query_probabilities=query.probabilities,
        lambda_by_category=lambda_by_category,
        limit=settings.report_top_k,
        scoring_method=SEARCH_MODE_UNIFIED_PRIOR_RERANK,
    )


def _run_score_merge(
    query: QueryFeatures,
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
    categories: tuple[str, ...],
    lambda_by_category: Mapping[str, float],
) -> tuple[RankedChunk, ...]:
    candidates: list[PartitionHit] = []
    for category in categories:
        hits = search_backend.search_category(
            category,
            query.embedding,
            top_k=settings.candidate_k_per_partition,
        )
        candidates.extend(_partition_hits(hits, partition_category=category))
    return score_merge_candidates(
        candidates,
        query_probabilities=query.probabilities,
        lambda_by_category=lambda_by_category,
        limit=settings.report_top_k,
    )


def _rank_similarity_hits(
    hits: list[SearchHit],
    *,
    partition_category: str | None,
) -> tuple[RankedChunk, ...]:
    ordered = sorted(
        hits,
        key=lambda hit: (-float(hit.similarity), hit.chunk_id),
    )
    return tuple(
        RankedChunk(
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
            source_category=hit.source_category,
            partition_category=partition_category,
            rank=index,
            score=_metric(cosine_to_unit_interval(hit.similarity)),
            similarity=_metric(hit.similarity),
            similarity_norm=_metric(cosine_to_unit_interval(hit.similarity)),
            scoring_method="similarity",
        )
        for index, hit in enumerate(ordered, start=1)
    )


def _partition_hits(
    hits: list[SearchHit],
    *,
    partition_category: str,
) -> list[PartitionHit]:
    return [
        PartitionHit(
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
            source_category=hit.source_category,
            partition_category=partition_category,
            similarity=hit.similarity,
            rank=hit.rank,
        )
        for hit in hits
    ]


def _variant_result(
    variant: str,
    query_id: str,
    ranked: tuple[RankedChunk, ...],
    settings: PrimaryRunSettings,
) -> VariantResult:
    top10 = ranked[: settings.report_top_k]
    return VariantResult(
        variant=variant,
        query_id=query_id,
        top10=top10,
        top5_contexts=top10[: settings.generation_context_top_n],
    )


def _metric(value: float) -> float:
    return round(float(value), 12)


def _validate_search_mode(search_mode: str) -> None:
    if search_mode not in SEARCH_MODES:
        raise ValueError(
            f"unknown search mode {search_mode!r}; expected one of {SEARCH_MODES}"
        )
