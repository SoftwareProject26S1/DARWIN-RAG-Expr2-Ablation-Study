"""Retrieval metric helpers for Phase 9 reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def retrieval_metrics_at_k(
    *,
    ranked_chunk_ids: Sequence[str],
    gold_chunk_ids: Iterable[str],
    k: int,
) -> dict[str, float]:
    """Compute basic binary-relevance retrieval metrics at k."""

    if k <= 0:
        raise ValueError("k must be positive")
    gold = {str(chunk_id) for chunk_id in gold_chunk_ids}
    if not gold:
        raise ValueError("gold_chunk_ids must not be empty")
    top_k = [str(chunk_id) for chunk_id in ranked_chunk_ids[:k]]
    relevant_positions = [
        index
        for index, chunk_id in enumerate(top_k, start=1)
        if chunk_id in gold
    ]
    hit = 1.0 if relevant_positions else 0.0
    recall = len(set(top_k).intersection(gold)) / len(gold)
    mrr = 1.0 / relevant_positions[0] if relevant_positions else 0.0
    ndcg = _ndcg_at_k(top_k, gold)
    return {
        f"hit@{k}": _metric(hit),
        f"recall@{k}": _metric(recall),
        f"mrr@{k}": _metric(mrr),
        f"ndcg@{k}": _metric(ndcg),
    }


def _ndcg_at_k(ranked_chunk_ids: Sequence[str], gold: set[str]) -> float:
    dcg = 0.0
    for position, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in gold:
            dcg += 1.0 / math.log2(position + 1)
    ideal_relevant_count = min(len(gold), len(ranked_chunk_ids))
    if ideal_relevant_count == 0:
        return 0.0
    ideal_dcg = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, ideal_relevant_count + 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _metric(value: float) -> float:
    return round(float(value), 6)
