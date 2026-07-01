from pathlib import Path

import pytest

from darwin_rag_exp2.retrieval import tuning
from darwin_rag_exp2.retrieval.tuning import (
    tune_adaptive_lambda_parameters,
    tune_primary_settings,
)
from darwin_rag_exp2.retrieval.types import QueryFeatures, SearchHit


class TuneSearchBackend:
    def search_unified(self, query_embedding: list[float], *, top_k: int) -> list[SearchHit]:
        _ = (query_embedding, top_k)
        return []

    def search_category(
        self,
        category: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        _ = (query_embedding, top_k)
        if category == "학사":
            return [SearchHit("c1", "s1", "학사", 1.0, 1)]
        if category == "장학":
            return [SearchHit("c2", "s2", "장학", 0.0, 1)]
        return []


def test_tune_primary_settings_selects_b2_score_dev_ndcg() -> None:
    query = QueryFeatures(
        query_id="dev_q0001",
        query="장학 일정 알려줘",
        embedding=[1.0, 0.0],
        probabilities={"학사": 0.6, "장학": 0.9},
        gold_chunks=("c2",),
        gold_categories=("장학",),
        query_type="single_category",
    )

    settings, diagnostics = tune_primary_settings(
        [query],
        search_backend=TuneSearchBackend(),
        candidate_k_per_partition=1,
        report_top_k=1,
        generation_context_top_n=1,
        theta_candidates=[0.6],
        lambda_fixed_candidates=[1.0, 0.0],
        lambda_by_category={"학사": 0.8, "장학": 0.7},
        metric_key="ndcg@1",
    )

    assert settings.theta_route == 0.6
    assert settings.lambda_fixed == 0.0
    assert diagnostics["best_metric"] == 1.0
    assert diagnostics["best_variant"] == "B2-score"
    assert diagnostics["trials"] == [
        {
            "lambda_fixed": 1.0,
            "metric": 0.0,
            "query_type_metrics": {
                "single_category": {
                    "hit@1": 0.0,
                    "mrr@1": 0.0,
                    "ndcg@1": 0.0,
                    "query_count": 1,
                    "recall@1": 0.0,
                },
            },
            "theta_route": 0.6,
        },
        {
            "lambda_fixed": 0.0,
            "metric": 1.0,
            "query_type_metrics": {
                "single_category": {
                    "hit@1": 1.0,
                    "mrr@1": 1.0,
                    "ndcg@1": 1.0,
                    "query_count": 1,
                    "recall@1": 1.0,
                },
            },
            "theta_route": 0.6,
        },
    ]


def test_tune_primary_settings_prefers_larger_lambda_when_dev_metric_ties() -> None:
    query = QueryFeatures(
        query_id="dev_q0001",
        query="학사 일정 알려줘",
        embedding=[1.0, 0.0],
        probabilities={"학사": 0.9},
        gold_chunks=("c1",),
        gold_categories=("학사",),
        query_type="core_single_category",
    )

    settings, diagnostics = tune_primary_settings(
        [query],
        search_backend=TuneSearchBackend(),
        candidate_k_per_partition=1,
        report_top_k=1,
        generation_context_top_n=1,
        theta_candidates=[0.5],
        lambda_fixed_candidates=[0.9, 1.0],
        lambda_by_category={"학사": 0.8},
        metric_key="ndcg@1",
    )

    assert settings.lambda_fixed == 1.0
    assert diagnostics["best_metric"] == 1.0


def test_tune_primary_settings_records_query_type_metrics() -> None:
    queries = [
        QueryFeatures(
            query_id="dev_q0001",
            query="학사 일정 알려줘",
            embedding=[1.0, 0.0],
            probabilities={"학사": 0.9, "장학": 0.1},
            gold_chunks=("c1",),
            gold_categories=("학사",),
            query_type="core_single_category",
        ),
        QueryFeatures(
            query_id="dev_q0002",
            query="학사와 장학을 같이 알려줘",
            embedding=[1.0, 0.0],
            probabilities={"학사": 0.9, "장학": 0.1},
            gold_chunks=("c2",),
            gold_categories=("장학",),
            query_type="multi_category",
        ),
    ]

    _, diagnostics = tune_primary_settings(
        queries,
        search_backend=TuneSearchBackend(),
        candidate_k_per_partition=1,
        report_top_k=1,
        generation_context_top_n=1,
        theta_candidates=[0.5],
        lambda_fixed_candidates=[1.0],
        lambda_by_category={"학사": 0.8, "장학": 0.7},
        metric_key="ndcg@1",
    )

    expected_query_type_metrics = {
        "core_single_category": {
            "hit@1": 1.0,
            "mrr@1": 1.0,
            "ndcg@1": 1.0,
            "query_count": 1,
            "recall@1": 1.0,
        },
        "multi_category": {
            "hit@1": 0.0,
            "mrr@1": 0.0,
            "ndcg@1": 0.0,
            "query_count": 1,
            "recall@1": 0.0,
        },
    }
    assert diagnostics["trials"] == [
        {
            "lambda_fixed": 1.0,
            "metric": 0.5,
            "query_type_metrics": expected_query_type_metrics,
            "theta_route": 0.5,
        },
    ]
    assert diagnostics["best_query_type_metrics"] == expected_query_type_metrics


def test_tune_adaptive_lambda_parameters_selects_p_score_dev_ndcg() -> None:
    query = QueryFeatures(
        query_id="dev_q0001",
        query="장학 일정 알려줘",
        embedding=[1.0, 0.0],
        probabilities={"학사": 0.6, "장학": 0.95},
        gold_chunks=("c2",),
        gold_categories=("장학",),
        query_type="single_category",
    )
    base_settings, _ = tune_primary_settings(
        [query],
        search_backend=TuneSearchBackend(),
        candidate_k_per_partition=1,
        report_top_k=1,
        generation_context_top_n=1,
        theta_candidates=[0.6],
        lambda_fixed_candidates=[0.5],
        lambda_by_category={"학사": 0.5, "장학": 0.5},
        metric_key="ndcg@1",
    )
    stats_rows = [
        {"category": "학사", "mu_confidence": 0.9, "sigma_confidence": 0.5},
        {"category": "장학", "mu_confidence": 0.5, "sigma_confidence": 0.5},
    ]

    tuned_settings, diagnostics = tune_adaptive_lambda_parameters(
        [query],
        search_backend=TuneSearchBackend(),
        base_settings=base_settings,
        category_stats_rows=stats_rows,
        alpha_candidates=[8.0],
        rho_candidates=[0.0, 10.0],
        tau_candidates=[0.5],
        metric_key="ndcg@1",
    )

    assert diagnostics["best_variant"] == "P-score"
    assert diagnostics["best_parameters"] == {
        "alpha": 8.0,
        "rho": 10.0,
        "tau": 0.5,
    }
    assert tuned_settings.lambda_by_category["장학"] < 0.01


def test_tuning_query_metadata_records_dev_v2_lineage_and_metric() -> None:
    metadata = tuning.tuning_query_metadata(
        Path("data/annotations/queries_dev_v2.jsonl"),
        metric_key="ndcg@10",
    )

    assert metadata == {
        "queries_path": "data/annotations/queries_dev_v2.jsonl",
        "query_file": "queries_dev_v2.jsonl",
        "query_split": "dev",
        "schema_version": "eval_v3_overlap_aware_rechunked",
        "metric_key": "ndcg@10",
    }


def test_tuning_query_metadata_rejects_test_v2_queries() -> None:
    with pytest.raises(ValueError, match="tuning requires dev queries"):
        _ = tuning.tuning_query_metadata(
            Path("data/annotations/queries_test_v2.jsonl"),
            metric_key="ndcg@10",
        )


def test_tuning_query_metadata_rejects_notdev_filename() -> None:
    with pytest.raises(ValueError, match="tuning requires dev queries"):
        _ = tuning.tuning_query_metadata(
            Path("queries_notdev_v2.jsonl"),
            metric_key="ndcg@10",
        )
