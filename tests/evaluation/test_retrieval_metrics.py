from darwin_rag_exp2.evaluation.retrieval_metrics import retrieval_metrics_at_k


def test_retrieval_metrics_report_hit_recall_mrr_and_ndcg_at_k() -> None:
    metrics = retrieval_metrics_at_k(
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
