import pytest

from darwin_rag_exp2.evaluation import retrieval_metrics as metrics_module


def test_retrieval_metrics_report_hit_recall_mrr_and_ndcg_at_k() -> None:
    metrics = metrics_module.retrieval_metrics_at_k(
        ranked_chunk_ids=["c3", "c2", "c1", "c4"],
        gold_chunk_ids={"c1", "c2"},
        k=3,
    )

    assert metrics == {
        "hit@3": 1.0,
        "recall@3": 1.0,
        "mrr@3": 0.5,
        "ndcg@3": 0.693426,
    }


def test_graded_retrieval_metrics_report_ndcg_at_k() -> None:
    metrics = metrics_module.graded_retrieval_metrics_at_k(
        ranked_chunk_ids=["c2", "c1", "c3"],
        relevance_by_chunk_id={"c1": 1.0, "c2": 0.5, "c3": 0.0},
        k=2,
    )

    assert metrics == {"graded_ndcg@2": 0.859719}


def test_graded_retrieval_metrics_reject_empty_relevance() -> None:
    with pytest.raises(ValueError, match="relevance_by_chunk_id must not be empty"):
        metrics_module.graded_retrieval_metrics_at_k(
            ranked_chunk_ids=["c1"],
            relevance_by_chunk_id={},
            k=1,
        )
