from darwin_rag_exp2.retrieval.rrf import vanilla_rrf_scores


def test_vanilla_rrf_is_invariant_to_fixed_or_adaptive_lambda_interpretation() -> None:
    partition_rankings = {
        "학사": ["c1", "c2"],
        "장학": ["c2", "c1"],
    }

    fixed_lambda_scores = vanilla_rrf_scores(partition_rankings)
    adaptive_lambda_scores = vanilla_rrf_scores(partition_rankings)

    assert fixed_lambda_scores == adaptive_lambda_scores
    assert fixed_lambda_scores["c1"] == fixed_lambda_scores["c2"]
