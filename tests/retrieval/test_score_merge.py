from darwin_rag_exp2.retrieval.score_merge import score_merge_candidates
from darwin_rag_exp2.retrieval.types import PartitionHit


def test_score_merge_uses_unit_interval_similarity_and_keeps_best_chunk_occurrence() -> None:
    candidates = [
        PartitionHit(
            chunk_id="c1",
            source_id="s1",
            source_category="학사",
            partition_category="학사",
            similarity=0.2,
            rank=1,
        ),
        PartitionHit(
            chunk_id="c1",
            source_id="s1",
            source_category="학사",
            partition_category="장학",
            similarity=0.8,
            rank=1,
        ),
        PartitionHit(
            chunk_id="c2",
            source_id="s2",
            source_category="장학",
            partition_category="장학",
            similarity=0.1,
            rank=2,
        ),
    ]

    ranked = score_merge_candidates(
        candidates,
        query_probabilities={"학사": 0.9, "장학": 0.1},
        lambda_by_category={"학사": 0.2, "장학": 0.9},
        limit=10,
    )

    assert [result.chunk_id for result in ranked] == ["c1", "c2"]
    assert ranked[0].partition_category == "학사"
    assert ranked[0].score == 0.84
    assert ranked[0].similarity_norm == 0.6
