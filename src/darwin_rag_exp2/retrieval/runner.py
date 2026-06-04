"""Phase 9 primary retrieval run orchestration and artifact writing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import orjson

from darwin_rag_exp2.evaluation.retrieval_metrics import retrieval_metrics_at_k

from .routing import soft_route_categories, top1_category
from .types import (
    PrimaryRunSettings,
    QueryFeatures,
    RankedChunk,
    SearchBackend,
    VariantResult,
)
from .variants import run_primary_variants


def run_primary_queries(
    queries: Sequence[QueryFeatures],
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
) -> list[dict[str, object]]:
    """Run all primary variants for each query and return report rows."""

    rows: list[dict[str, object]] = []
    for query in queries:
        variant_results = run_primary_variants(
            query,
            search_backend=search_backend,
            settings=settings,
        )
        for variant_result in variant_results.values():
            rows.append(_result_row(query, variant_result, settings))
    return rows


def write_primary_run(
    *,
    output_dir: Path,
    result_rows: Sequence[Mapping[str, object]],
    settings: PrimaryRunSettings,
    run_metadata: Mapping[str, object] | None = None,
) -> None:
    """Write Phase 9 result rows and a small manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "results.jsonl", result_rows)
    query_ids = {str(row["query_id"]) for row in result_rows}
    variants = {str(row["variant"]) for row in result_rows}
    manifest: dict[str, object] = {
        "phase": 9,
        "artifact_type": "primary_retrieval_run",
        "query_count": len(query_ids),
        "variant_count": len(variants),
        "row_count": len(result_rows),
        "variants": sorted(variants),
        "settings": _settings_payload(settings),
        "artifact_files": ["results.jsonl", "manifest.json"],
    }
    if run_metadata:
        manifest["run_metadata"] = dict(run_metadata)
    _write_json(output_dir / "manifest.json", manifest)


def _result_row(
    query: QueryFeatures,
    variant_result: VariantResult,
    settings: PrimaryRunSettings,
) -> dict[str, object]:
    metric_values = retrieval_metrics_at_k(
        ranked_chunk_ids=[row.chunk_id for row in variant_result.top10],
        gold_chunk_ids=query.gold_chunks,
        k=settings.report_top_k,
    )
    return {
        "query_id": query.query_id,
        "query": query.query,
        "variant": variant_result.variant,
        "query_type": query.query_type,
        "gold_chunks": list(query.gold_chunks),
        "gold_categories": list(query.gold_categories),
        "query_probabilities": dict(query.probabilities),
        "routing": _routing_payload(query, variant_result.variant, settings),
        "metrics": metric_values,
        "top10": [_ranked_payload(row) for row in variant_result.top10],
        "top5_contexts": [
            _ranked_payload(row)
            for row in variant_result.top5_contexts
        ],
    }


def _routing_payload(
    query: QueryFeatures,
    variant: str,
    settings: PrimaryRunSettings,
) -> dict[str, object]:
    top1 = top1_category(query.probabilities)
    if variant == "B0":
        return {
            "mode": "unified",
            "top1_category": top1,
            "routed_categories": [],
            "route_width": 0,
        }
    if variant == "B1":
        return {
            "mode": "top1",
            "top1_category": top1,
            "routed_categories": [top1],
            "route_width": 1,
        }

    routed = soft_route_categories(
        query.probabilities,
        theta_route=settings.theta_route,
    )
    threshold_categories = [
        category
        for category, probability in query.probabilities.items()
        if float(probability) >= settings.theta_route
    ]
    return {
        "mode": "soft_threshold" if threshold_categories else "top1_fallback",
        "top1_category": top1,
        "routed_categories": list(routed),
        "route_width": len(routed),
    }


def _ranked_payload(row: RankedChunk) -> dict[str, object]:
    payload = asdict(row)
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _settings_payload(settings: PrimaryRunSettings) -> dict[str, object]:
    return {
        "candidate_k_per_partition": settings.candidate_k_per_partition,
        "report_top_k": settings.report_top_k,
        "generation_context_top_n": settings.generation_context_top_n,
        "theta_route": settings.theta_route,
        "lambda_fixed": settings.lambda_fixed,
        "lambda_by_category": dict(settings.lambda_by_category),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            output.write(b"\n")
