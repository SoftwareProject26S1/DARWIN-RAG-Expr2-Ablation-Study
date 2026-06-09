from __future__ import annotations

from darwin_rag_exp2.evaluation.retrieval_report_html import render_primary_report_html


def test_render_primary_report_html_contains_required_visual_sections() -> None:
    analysis = _analysis_payload()

    html = render_primary_report_html(analysis)

    assert "Variant별 metric bar chart" in html
    assert "P-score - B2-score delta 분포 histogram" in html
    assert "query_type/category별 heatmap" in html
    assert "Route width 분포" in html
    assert "실패 query Top-N" in html
    assert html.count("<svg") >= 4
    assert "<table" in html


def test_render_primary_report_html_escapes_query_and_preview_text() -> None:
    analysis = _analysis_payload()
    analysis["failure_cases"][0]["query"] = "<script>alert('x')</script>"
    analysis["failure_cases"][0]["variants"]["P-score"]["top10"][0][
        "title"
    ] = "<b>공지</b>"

    html = render_primary_report_html(analysis)

    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "<b>공지</b>" not in html
    assert "&lt;b&gt;공지&lt;/b&gt;" in html


def _analysis_payload() -> dict[str, object]:
    return {
        "summary": {
            "query_count": 2,
            "variant_count": 4,
            "metric": "ndcg@10",
        },
        "metrics_by_variant": [
            {"variant": "B0", "query_count": 2, "hit@10": 0.5, "mrr@10": 0.4, "ndcg@10": 0.45, "recall@10": 0.5},
            {"variant": "B1", "query_count": 2, "hit@10": 0.5, "mrr@10": 0.4, "ndcg@10": 0.45, "recall@10": 0.5},
            {"variant": "B2-score", "query_count": 2, "hit@10": 0.5, "mrr@10": 0.5, "ndcg@10": 0.5, "recall@10": 0.5},
            {"variant": "P-score", "query_count": 2, "hit@10": 1.0, "mrr@10": 0.75, "ndcg@10": 0.75, "recall@10": 1.0},
        ],
        "paired_comparison": {
            "metric": "ndcg@10",
            "mean_delta": 0.25,
            "median_delta": 0.25,
            "wins": 1,
            "ties": 1,
            "losses": 0,
            "bootstrap_ci95_low": 0.0,
            "bootstrap_ci95_high": 0.5,
            "wilcoxon_p_value": 1.0,
        },
        "paired_deltas": [
            {"query_id": "q1", "delta": 0.0, "b2_score": 1.0, "p_score": 1.0},
            {"query_id": "q2", "delta": 0.5, "b2_score": 0.0, "p_score": 0.5},
        ],
        "breakdown_by_query_type": [
            {"query_type": "single_category", "variant": "P-score", "query_count": 1, "ndcg@10": 1.0},
            {"query_type": "ambiguous", "variant": "P-score", "query_count": 1, "ndcg@10": 0.5},
        ],
        "breakdown_by_gold_category": [
            {"gold_category": "학사", "variant": "P-score", "query_count": 1, "ndcg@10": 1.0},
            {"gold_category": "장학", "variant": "P-score", "query_count": 1, "ndcg@10": 0.5},
        ],
        "routing_diagnostics": [
            {
                "variant": "B2-score",
                "query_count": 2,
                "route_width_mean": 1.5,
                "route_width_1_rate": 0.5,
                "route_width_ge2_rate": 0.5,
                "top1_fallback_rate": 0.5,
            },
            {
                "variant": "P-score",
                "query_count": 2,
                "route_width_mean": 1.5,
                "route_width_1_rate": 0.5,
                "route_width_ge2_rate": 0.5,
                "top1_fallback_rate": 0.5,
            },
        ],
        "failure_cases": [
            {
                "query_id": "q2",
                "query": "장학 신청 기간은?",
                "query_type": "ambiguous",
                "gold_categories": ["장학"],
                "gold_chunks": ["c3"],
                "variants": {
                    "B2-score": {
                        "metric": 0.0,
                        "chunk_hit": False,
                        "source_hit": True,
                        "top10": [{"chunk_id": "c4", "source_id": "s3", "title": "비슷한 공지"}],
                    },
                    "P-score": {
                        "metric": 0.5,
                        "chunk_hit": True,
                        "source_hit": True,
                        "top10": [{"chunk_id": "c3", "source_id": "s3", "title": "정답 공지"}],
                    },
                },
            }
        ],
    }
