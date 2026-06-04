from __future__ import annotations

from darwin_rag_exp2.evaluation.retrieval_analysis import analyze_primary_results


def test_analyze_primary_results_reports_variant_metrics_and_paired_delta() -> None:
    rows = _sample_result_rows()

    analysis = analyze_primary_results(rows, metric_key="ndcg@10", top_failures=5)

    metrics_by_variant = {
        row["variant"]: row
        for row in analysis["metrics_by_variant"]
    }
    assert metrics_by_variant["B2-score"]["ndcg@10"] == 0.5
    assert metrics_by_variant["P-score"]["ndcg@10"] == 0.75

    comparison = analysis["paired_comparison"]
    assert comparison["metric"] == "ndcg@10"
    assert comparison["mean_delta"] == 0.25
    assert comparison["median_delta"] == 0.25
    assert comparison["wins"] == 1
    assert comparison["ties"] == 1
    assert comparison["losses"] == 0


def test_analyze_primary_results_reports_breakdowns_and_variant_equivalence() -> None:
    rows = _sample_result_rows()

    analysis = analyze_primary_results(rows, metric_key="ndcg@10", top_failures=5)

    query_type_breakdown = {
        (row["query_type"], row["variant"]): row
        for row in analysis["breakdown_by_query_type"]
    }
    assert query_type_breakdown[("ambiguous", "P-score")]["ndcg@10"] == 0.5

    category_breakdown = {
        (row["gold_category"], row["variant"]): row
        for row in analysis["breakdown_by_gold_category"]
    }
    assert category_breakdown[("장학", "P-score")]["query_count"] == 1

    equivalence = {
        row["query_id"]: row
        for row in analysis["variant_equivalence"]
    }
    assert equivalence["q1"]["b1_b2_p_top10_equal"] is True
    assert equivalence["q2"]["b1_b2_p_top10_equal"] is False


def test_analyze_primary_results_reports_routing_and_strict_gold_failures() -> None:
    rows = _sample_result_rows()

    analysis = analyze_primary_results(
        rows,
        metric_key="ndcg@10",
        top_failures=5,
        chunk_lookup={
            "c3": {
                "chunk_id": "c3",
                "source_id": "s3",
                "category": "장학",
                "title": "정답 장학 공지",
                "body_text": "정답 본문입니다.",
            }
        },
    )

    routing_by_variant = {
        row["variant"]: row
        for row in analysis["routing_diagnostics"]
    }
    assert routing_by_variant["B2-score"]["route_width_mean"] == 1.5
    assert routing_by_variant["B2-score"]["route_width_1_rate"] == 0.5
    assert routing_by_variant["B2-score"]["route_width_ge2_rate"] == 0.5
    assert routing_by_variant["B2-score"]["top1_fallback_rate"] == 0.5

    failure_by_query = {
        row["query_id"]: row
        for row in analysis["failure_cases"]
    }
    b2_failure = failure_by_query["q2"]["variants"]["B2-score"]
    assert b2_failure["chunk_hit"] is False
    assert b2_failure["source_hit"] is True
    assert b2_failure["top10"][0]["chunk_id"] == "c4"


def test_analyze_primary_results_supports_legacy_rows_without_routing() -> None:
    legacy_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"query_probabilities", "routing"}
        }
        for row in _sample_result_rows()
    ]

    analysis = analyze_primary_results(
        legacy_rows,
        metric_key="ndcg@10",
        top_failures=5,
    )

    assert analysis["summary"]["legacy_row_count"] == len(legacy_rows)
    routing_by_variant = {
        row["variant"]: row
        for row in analysis["routing_diagnostics"]
    }
    assert routing_by_variant["B2-score"]["legacy_rows"] == 2
    assert routing_by_variant["B2-score"]["route_width_mean"] == 1.0


def _sample_result_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(
        _query_rows(
            query_id="q1",
            query="수강신청 변경 기간은?",
            query_type="single_category",
            gold_chunks=["c1"],
            gold_categories=["학사"],
            metrics_by_variant={
                "B0": 1.0,
                "B1": 1.0,
                "B2-score": 1.0,
                "P-score": 1.0,
            },
            top10_by_variant={
                "B0": [_hit("c1", "s1", "학사", None)],
                "B1": [_hit("c1", "s1", "학사", "학사")],
                "B2-score": [_hit("c1", "s1", "학사", "학사")],
                "P-score": [_hit("c1", "s1", "학사", "학사")],
            },
            routing_by_variant={
                "B0": _routing("unified", "학사", [], 0),
                "B1": _routing("top1", "학사", ["학사"], 1),
                "B2-score": _routing("top1_fallback", "학사", ["학사"], 1),
                "P-score": _routing("top1_fallback", "학사", ["학사"], 1),
            },
        )
    )
    rows.extend(
        _query_rows(
            query_id="q2",
            query="장학 안내가 맞는지 애매한데 신청 기간을 알려줘",
            query_type="ambiguous",
            gold_chunks=["c3"],
            gold_categories=["장학"],
            metrics_by_variant={
                "B0": 0.0,
                "B1": 0.0,
                "B2-score": 0.0,
                "P-score": 0.5,
            },
            top10_by_variant={
                "B0": [_hit("c2", "s2", "학사", None)],
                "B1": [_hit("c4", "s3", "장학", "장학")],
                "B2-score": [_hit("c4", "s3", "장학", "장학")],
                "P-score": [
                    _hit("c5", "s5", "학사", "학사"),
                    _hit("c3", "s3", "장학", "장학", rank=2),
                ],
            },
            routing_by_variant={
                "B0": _routing("unified", "장학", [], 0),
                "B1": _routing("top1", "장학", ["장학"], 1),
                "B2-score": _routing("soft_threshold", "장학", ["장학", "학사"], 2),
                "P-score": _routing("soft_threshold", "장학", ["장학", "학사"], 2),
            },
        )
    )
    return rows


def _query_rows(
    *,
    query_id: str,
    query: str,
    query_type: str,
    gold_chunks: list[str],
    gold_categories: list[str],
    metrics_by_variant: dict[str, float],
    top10_by_variant: dict[str, list[dict[str, object]]],
    routing_by_variant: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, ndcg in metrics_by_variant.items():
        top10 = top10_by_variant[variant]
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "variant": variant,
                "query_type": query_type,
                "gold_chunks": gold_chunks,
                "gold_categories": gold_categories,
                "query_probabilities": {"학사": 0.45, "장학": 0.55},
                "routing": routing_by_variant[variant],
                "metrics": {
                    "hit@10": 1.0 if ndcg > 0.0 else 0.0,
                    "mrr@10": ndcg,
                    "ndcg@10": ndcg,
                    "recall@10": 1.0 if ndcg > 0.0 else 0.0,
                },
                "top10": top10,
                "top5_contexts": top10[:1],
            }
        )
    return rows


def _hit(
    chunk_id: str,
    source_id: str,
    source_category: str,
    partition_category: str | None,
    *,
    rank: int = 1,
) -> dict[str, object]:
    row: dict[str, object] = {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_category": source_category,
        "rank": rank,
        "score": 1.0 / rank,
        "similarity": 0.9 / rank,
        "similarity_norm": 0.95 / rank,
        "scoring_method": "similarity",
    }
    if partition_category is not None:
        row["partition_category"] = partition_category
    return row


def _routing(
    mode: str,
    top1_category: str,
    routed_categories: list[str],
    route_width: int,
) -> dict[str, object]:
    return {
        "mode": mode,
        "top1_category": top1_category,
        "routed_categories": routed_categories,
        "route_width": route_width,
    }
