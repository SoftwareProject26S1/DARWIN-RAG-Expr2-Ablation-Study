"""Rank-fusion fixtures used to document optional RRF behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def vanilla_rrf_scores(
    partition_rankings: Mapping[str, Sequence[str]],
    *,
    rank_constant: int = 60,
) -> dict[str, float]:
    """Compute vanilla RRF scores from ranks only, with no lambda/probability terms."""

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = {}
    for category in sorted(partition_rankings):
        for rank, chunk_id in enumerate(partition_rankings[category], start=1):
            scores[str(chunk_id)] = scores.get(str(chunk_id), 0.0) + (
                1.0 / (rank_constant + rank)
            )
    return {
        chunk_id: round(score, 12)
        for chunk_id, score in sorted(scores.items())
    }
