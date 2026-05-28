from darwin_rag_exp2.models.category_stats import build_category_stats


def test_build_category_stats_marks_smoke_table_and_computes_true_label_stats() -> None:
    prediction_rows = [
        {
            "source_id": "s1",
            "category": "학사",
            "probabilities": {"학사": 0.8, "장학": 0.2},
        },
        {
            "source_id": "s1",
            "category": "학사",
            "probabilities": {"학사": 0.6, "장학": 0.4},
        },
        {
            "source_id": "s2",
            "category": "장학",
            "probabilities": {"학사": 0.3, "장학": 0.7},
        },
    ]

    stats = build_category_stats(
        prediction_rows,
        categories=["학사", "장학"],
        alpha=4.0,
        rho=1.0,
        tau=0.5,
        smoke_only=True,
    )

    haksa, scholarship = stats
    assert haksa["category"] == "학사"
    assert haksa["chunk_count"] == 2
    assert haksa["source_count"] == 1
    assert haksa["mu_confidence"] == 0.7
    assert haksa["sigma_confidence"] == 0.1
    assert haksa["smoke_only"] is True
    assert 0.0 < haksa["lambda_c"] < 1.0

    assert scholarship["category"] == "장학"
    assert scholarship["chunk_count"] == 1
    assert scholarship["source_count"] == 1
    assert scholarship["mu_confidence"] == 0.7
    assert scholarship["sigma_confidence"] == 0.0
