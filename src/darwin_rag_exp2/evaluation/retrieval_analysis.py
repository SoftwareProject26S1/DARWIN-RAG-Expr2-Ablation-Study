"""Phase 9 primary retrieval result analysis artifacts."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import orjson
import pyarrow.parquet as pq


CANONICAL_VARIANTS = ("B0", "B1", "B2-score", "P-score")
PRIMARY_PAIR = ("B2-score", "P-score")
DEFAULT_METRIC_KEYS = ("hit@10", "mrr@10", "ndcg@10", "recall@10")


def load_primary_result_rows(run_dir: Path) -> list[dict[str, object]]:
    """Load Phase 9 ``results.jsonl`` rows from a primary run directory."""

    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"missing primary results: {results_path}")
    rows: list[dict[str, object]] = []
    with results_path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = orjson.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"result row {line_number} must be an object")
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"no result rows found in {results_path}")
    return rows


def load_chunk_lookup(chunks_path: Path | None) -> dict[str, dict[str, object]]:
    """Load optional chunk metadata keyed by ``chunk_id`` for failure previews."""

    if chunks_path is None:
        return {}
    rows = pq.read_table(chunks_path).to_pylist()
    lookup: dict[str, dict[str, object]] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id:
            lookup[chunk_id] = dict(row)
    return lookup


def analyze_primary_results(
    result_rows: Sequence[Mapping[str, object]],
    *,
    metric_key: str = "ndcg@10",
    top_failures: int = 20,
    chunk_lookup: Mapping[str, Mapping[str, object]] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Build aggregate and diagnostic tables for Phase 9 retrieval rows."""

    if top_failures < 0:
        raise ValueError("top_failures must not be negative")
    rows = [dict(row) for row in result_rows]
    if not rows:
        raise ValueError("result_rows must not be empty")
    lookup = chunk_lookup or {}
    grouped = _rows_by_query_and_variant(rows)
    variants = _ordered_variants(rows)
    metric_keys = _metric_keys(rows)
    if metric_key not in metric_keys:
        raise ValueError(f"metric {metric_key!r} not found in result rows")

    paired_deltas = _paired_deltas(grouped, metric_key=metric_key)
    analysis: dict[str, object] = {
        "summary": {
            "query_count": len(grouped),
            "variant_count": len(variants),
            "row_count": len(rows),
            "variants": variants,
            "metric": metric_key,
            "legacy_row_count": sum(1 for row in rows if "routing" not in row),
        },
        "metrics_by_variant": _metrics_by_variant(rows, variants, metric_keys),
        "breakdown_by_query_type": _breakdown_by_query_type(rows, variants, metric_keys),
        "breakdown_by_gold_category": _breakdown_by_gold_category(
            rows,
            variants,
            metric_keys,
        ),
        "paired_comparison": _paired_comparison(
            paired_deltas,
            metric_key=metric_key,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "paired_deltas": paired_deltas,
        "routing_diagnostics": _routing_diagnostics(rows, variants),
        "variant_equivalence": _variant_equivalence(grouped),
        "failure_cases": _failure_cases(
            grouped,
            metric_key=metric_key,
            top_failures=top_failures,
            chunk_lookup=lookup,
        ),
    }
    return analysis


def write_primary_analysis(
    *,
    output_dir: Path,
    analysis: Mapping[str, object],
    run_dir: Path,
    chunks_path: Path | None = None,
) -> None:
    """Write machine-readable tables and a static HTML retrieval report."""

    from .retrieval_report_html import render_primary_report_html

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = [
        "summary.json",
        "metrics_by_variant.csv",
        "breakdown_by_query_type.csv",
        "breakdown_by_gold_category.csv",
        "paired_comparison.json",
        "paired_deltas.csv",
        "routing_diagnostics.csv",
        "variant_equivalence.csv",
        "failure_cases.jsonl",
        "failure_cases.md",
        "report.html",
        "manifest.json",
    ]
    _write_json(output_dir / "summary.json", analysis["summary"])
    _write_csv(output_dir / "metrics_by_variant.csv", analysis["metrics_by_variant"])
    _write_csv(
        output_dir / "breakdown_by_query_type.csv",
        analysis["breakdown_by_query_type"],
    )
    _write_csv(
        output_dir / "breakdown_by_gold_category.csv",
        analysis["breakdown_by_gold_category"],
    )
    _write_json(output_dir / "paired_comparison.json", analysis["paired_comparison"])
    _write_csv(output_dir / "paired_deltas.csv", analysis["paired_deltas"])
    _write_csv(output_dir / "routing_diagnostics.csv", analysis["routing_diagnostics"])
    _write_csv(output_dir / "variant_equivalence.csv", analysis["variant_equivalence"])
    _write_jsonl(output_dir / "failure_cases.jsonl", analysis["failure_cases"])
    _write_failure_markdown(output_dir / "failure_cases.md", analysis["failure_cases"])
    (output_dir / "report.html").write_text(
        render_primary_report_html(analysis),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "phase": 9,
            "artifact_type": "primary_retrieval_analysis",
            "run_dir": str(run_dir),
            "chunks_path": str(chunks_path) if chunks_path is not None else None,
            "summary": analysis["summary"],
            "artifact_files": artifact_files,
        },
    )


def _rows_by_query_and_variant(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        query_id = str(row["query_id"])
        variant = str(row["variant"])
        if variant in grouped[query_id]:
            raise ValueError(
                f"duplicate row for query_id={query_id!r}, variant={variant!r}"
            )
        grouped[query_id][variant] = dict(row)
    return dict(grouped)


def _ordered_variants(rows: Sequence[Mapping[str, object]]) -> list[str]:
    found = {str(row["variant"]) for row in rows}
    ordered = [variant for variant in CANONICAL_VARIANTS if variant in found]
    ordered.extend(sorted(found.difference(ordered)))
    return ordered


def _metric_keys(rows: Sequence[Mapping[str, object]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("each result row must contain a metrics object")
        found.update(str(key) for key in metrics)
    ordered = [key for key in DEFAULT_METRIC_KEYS if key in found]
    ordered.extend(sorted(found.difference(ordered)))
    return ordered


def _metrics_by_variant(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant in variants:
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        result: dict[str, object] = {
            "variant": variant,
            "query_count": len(variant_rows),
        }
        result.update(_mean_metrics(variant_rows, metric_keys))
        output.append(result)
    return output


def _breakdown_by_query_type(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, object]]:
    query_types = sorted({str(row.get("query_type", "")) for row in rows})
    output: list[dict[str, object]] = []
    for query_type in query_types:
        for variant in variants:
            group_rows = [
                row
                for row in rows
                if str(row.get("query_type", "")) == query_type
                and str(row["variant"]) == variant
            ]
            if not group_rows:
                continue
            result: dict[str, object] = {
                "query_type": query_type,
                "variant": variant,
                "query_count": len(group_rows),
            }
            result.update(_mean_metrics(group_rows, metric_keys))
            output.append(result)
    return output


def _breakdown_by_gold_category(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, object]]:
    exploded: list[tuple[str, Mapping[str, object]]] = []
    for row in rows:
        for category in _string_list(row.get("gold_categories")):
            exploded.append((category, row))
    categories = sorted({category for category, _ in exploded})
    output: list[dict[str, object]] = []
    for category in categories:
        for variant in variants:
            group_rows = [
                row
                for row_category, row in exploded
                if row_category == category and str(row["variant"]) == variant
            ]
            if not group_rows:
                continue
            result: dict[str, object] = {
                "gold_category": category,
                "variant": variant,
                "query_count": len(group_rows),
            }
            result.update(_mean_metrics(group_rows, metric_keys))
            output.append(result)
    return output


def _mean_metrics(
    rows: Sequence[Mapping[str, object]],
    metric_keys: Sequence[str],
) -> dict[str, float]:
    means: dict[str, float] = {}
    for key in metric_keys:
        values = [
            float(_metrics(row)[key])
            for row in rows
            if key in _metrics(row)
        ]
        means[key] = _metric(sum(values) / len(values)) if values else 0.0
    return means


def _paired_deltas(
    grouped: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    metric_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    left_variant, right_variant = PRIMARY_PAIR
    for query_id in sorted(grouped):
        by_variant = grouped[query_id]
        if left_variant not in by_variant or right_variant not in by_variant:
            continue
        left = by_variant[left_variant]
        right = by_variant[right_variant]
        left_value = float(_metrics(left)[metric_key])
        right_value = float(_metrics(right)[metric_key])
        rows.append(
            {
                "query_id": query_id,
                "query": str(right.get("query", left.get("query", ""))),
                "query_type": str(right.get("query_type", left.get("query_type", ""))),
                "gold_categories": ",".join(
                    _string_list(right.get("gold_categories", left.get("gold_categories")))
                ),
                "b2_score": _metric(left_value),
                "p_score": _metric(right_value),
                "delta": _metric(right_value - left_value),
            }
        )
    return rows


def _paired_comparison(
    paired_deltas: Sequence[Mapping[str, object]],
    *,
    metric_key: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    deltas = [float(row["delta"]) for row in paired_deltas]
    wins = sum(1 for value in deltas if value > 0.0)
    losses = sum(1 for value in deltas if value < 0.0)
    ties = sum(1 for value in deltas if value == 0.0)
    ci_low, ci_high = _bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed)
    return {
        "metric": metric_key,
        "query_count": len(deltas),
        "mean_delta": _metric(sum(deltas) / len(deltas)) if deltas else 0.0,
        "median_delta": _metric(median(deltas)) if deltas else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "wilcoxon_p_value": _wilcoxon_signed_rank_p_value(deltas),
    }


def _routing_diagnostics(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant in variants:
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        route_rows = [_routing_payload(row) for row in variant_rows]
        widths = [int(row["route_width"]) for row in route_rows]
        modes = [str(row["mode"]) for row in route_rows]
        count = len(route_rows)
        output.append(
            {
                "variant": variant,
                "query_count": count,
                "legacy_rows": sum(1 for row in route_rows if bool(row["legacy"])),
                "route_width_mean": _metric(sum(widths) / count) if count else 0.0,
                "route_width_1_rate": _rate(widths, lambda value: value == 1),
                "route_width_ge2_rate": _rate(widths, lambda value: value >= 2),
                "top1_fallback_rate": _rate(modes, lambda value: value == "top1_fallback"),
                "soft_threshold_rate": _rate(modes, lambda value: value == "soft_threshold"),
                "top1_rate": _rate(modes, lambda value: value == "top1"),
                "unified_rate": _rate(modes, lambda value: value == "unified"),
            }
        )
    return output


def _variant_equivalence(
    grouped: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for query_id in sorted(grouped):
        by_variant = grouped[query_id]
        if not all(variant in by_variant for variant in ("B1", "B2-score", "P-score")):
            continue
        b1_ids = _top_chunk_ids(by_variant["B1"])
        b2_ids = _top_chunk_ids(by_variant["B2-score"])
        p_ids = _top_chunk_ids(by_variant["P-score"])
        sample = by_variant["P-score"]
        output.append(
            {
                "query_id": query_id,
                "query": str(sample.get("query", "")),
                "query_type": str(sample.get("query_type", "")),
                "gold_categories": ",".join(_string_list(sample.get("gold_categories"))),
                "b1_b2_top10_equal": b1_ids == b2_ids,
                "b2_p_top10_equal": b2_ids == p_ids,
                "b1_b2_p_top10_equal": b1_ids == b2_ids == p_ids,
            }
        )
    return output


def _failure_cases(
    grouped: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    metric_key: str,
    top_failures: int,
    chunk_lookup: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for query_id in sorted(grouped):
        by_variant = grouped[query_id]
        sample = next(iter(by_variant.values()))
        variants: dict[str, object] = {}
        has_failure = False
        for variant in _ordered_variants(by_variant.values()):
            row = by_variant[variant]
            variant_payload = _failure_variant_payload(
                row,
                metric_key=metric_key,
                chunk_lookup=chunk_lookup,
            )
            variants[variant] = variant_payload
            if (
                float(variant_payload["metric"]) < 1.0
                or variant_payload["chunk_hit"] is False
            ):
                has_failure = True
        if not has_failure:
            continue
        candidates.append(
            {
                "query_id": query_id,
                "query": str(sample.get("query", "")),
                "query_type": str(sample.get("query_type", "")),
                "gold_chunks": _string_list(sample.get("gold_chunks")),
                "gold_categories": _string_list(sample.get("gold_categories")),
                "variants": variants,
            }
        )
    candidates.sort(key=lambda row: (_failure_sort_metric(row), str(row["query_id"])))
    return candidates[:top_failures] if top_failures else []


def _failure_variant_payload(
    row: Mapping[str, object],
    *,
    metric_key: str,
    chunk_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    gold_chunks = set(_string_list(row.get("gold_chunks")))
    gold_sources = {
        str(chunk_lookup[chunk_id].get("source_id"))
        for chunk_id in gold_chunks
        if chunk_id in chunk_lookup and chunk_lookup[chunk_id].get("source_id") is not None
    }
    top10 = [_enrich_hit(hit, chunk_lookup) for hit in _top_hits(row)]
    top_chunk_ids = {str(hit.get("chunk_id")) for hit in top10}
    top_source_ids = {str(hit.get("source_id")) for hit in top10 if hit.get("source_id")}
    return {
        "metric": _metric(float(_metrics(row).get(metric_key, 0.0))),
        "chunk_hit": bool(gold_chunks.intersection(top_chunk_ids)),
        "source_hit": bool(gold_sources.intersection(top_source_ids)) if gold_sources else None,
        "routing": _routing_payload(row),
        "top10": top10,
    }


def _enrich_hit(
    hit: Mapping[str, object],
    chunk_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    payload = dict(hit)
    chunk_id = str(payload.get("chunk_id", ""))
    chunk = chunk_lookup.get(chunk_id)
    if chunk:
        payload.setdefault("title", str(chunk.get("title", "")))
        payload.setdefault("body_preview", _preview(str(chunk.get("body_text", ""))))
        payload.setdefault("chunk_category", str(chunk.get("category", "")))
    return payload


def _failure_sort_metric(row: Mapping[str, object]) -> tuple[float, float]:
    variants = row.get("variants")
    if not isinstance(variants, Mapping):
        return (1.0, 1.0)
    p_payload = variants.get("P-score")
    b2_payload = variants.get("B2-score")
    p_value = (
        float(p_payload.get("metric", 1.0))
        if isinstance(p_payload, Mapping)
        else 1.0
    )
    b2_value = (
        float(b2_payload.get("metric", 1.0))
        if isinstance(b2_payload, Mapping)
        else 1.0
    )
    return (p_value, b2_value)


def _routing_payload(row: Mapping[str, object]) -> dict[str, object]:
    routing = row.get("routing")
    if isinstance(routing, Mapping):
        routed_categories = _string_list(routing.get("routed_categories"))
        route_width = routing.get("route_width", len(routed_categories))
        return {
            "mode": str(routing.get("mode", "")),
            "top1_category": str(routing.get("top1_category", "")),
            "routed_categories": routed_categories,
            "route_width": int(route_width),
            "legacy": False,
        }
    return _legacy_routing_payload(row)


def _legacy_routing_payload(row: Mapping[str, object]) -> dict[str, object]:
    variant = str(row.get("variant", ""))
    if variant == "B0":
        return {
            "mode": "unified",
            "top1_category": "",
            "routed_categories": [],
            "route_width": 0,
            "legacy": True,
        }
    categories = sorted(
        {
            str(hit.get("partition_category"))
            for hit in _top_hits(row)
            if hit.get("partition_category") is not None
        }
    )
    return {
        "mode": "legacy_observed" if categories else "legacy_unknown",
        "top1_category": categories[0] if categories else "",
        "routed_categories": categories,
        "route_width": len(categories),
        "legacy": True,
    }


def _metrics(row: Mapping[str, object]) -> Mapping[str, object]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("result row metrics must be a mapping")
    return metrics


def _top_hits(row: Mapping[str, object]) -> list[dict[str, object]]:
    top10 = row.get("top10")
    if not isinstance(top10, list):
        return []
    return [
        dict(hit)
        for hit in top10
        if isinstance(hit, Mapping)
    ]


def _top_chunk_ids(row: Mapping[str, object]) -> list[str]:
    return [str(hit.get("chunk_id")) for hit in _top_hits(row)]


def _string_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _rate(values: Sequence[Any], predicate) -> float:
    if not values:
        return 0.0
    return _metric(sum(1 for value in values if predicate(value)) / len(values))


def _bootstrap_ci(
    deltas: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    if samples <= 0:
        mean_delta = _metric(sum(deltas) / len(deltas))
        return (mean_delta, mean_delta)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [rng.choice(deltas) for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return (_metric(means[low_index]), _metric(means[high_index]))


def _wilcoxon_signed_rank_p_value(deltas: Sequence[float]) -> float:
    nonzero = [float(value) for value in deltas if float(value) != 0.0]
    if not nonzero:
        return 1.0
    ranks = _absolute_ranks(nonzero)
    observed = sum(rank for value, rank in zip(nonzero, ranks, strict=True) if value > 0)
    total = sum(ranks)
    if len(ranks) <= 20:
        possible = [0.0]
        for rank in ranks:
            possible = [value for score in possible for value in (score, score + rank)]
        lower = sum(1 for value in possible if value <= observed) / len(possible)
        upper = sum(1 for value in possible if value >= observed) / len(possible)
        return _metric(min(1.0, 2.0 * min(lower, upper)))
    mean = total / 2.0
    variance = sum(rank * rank for rank in ranks) / 4.0
    if variance == 0.0:
        return 1.0
    z = abs(observed - mean) / math.sqrt(variance)
    return _metric(math.erfc(z / math.sqrt(2.0)))


def _absolute_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted((abs(value), index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for _, original_index in ordered[position:end]:
            ranks[original_index] = average_rank
        position = end
    return ranks


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _write_jsonl(path: Path, rows: object) -> None:
    if not isinstance(rows, Sequence):
        raise ValueError("jsonl rows must be a sequence")
    with path.open("wb") as output:
        for row in rows:
            output.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            output.write(b"\n")


def _write_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ValueError("csv rows must be a sequence")
    dict_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    fieldnames: list[str] = []
    for row in dict_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def _write_failure_markdown(path: Path, rows: object) -> None:
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ValueError("failure rows must be a sequence")
    lines = [
        "# Phase 9 Retrieval Failure Cases",
        "",
        "| query_id | query_type | gold_categories | query |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {query_id} | {query_type} | {gold_categories} | {query} |".format(
                query_id=str(row.get("query_id", "")),
                query_type=str(row.get("query_type", "")),
                gold_categories=", ".join(_string_list(row.get("gold_categories"))),
                query=str(row.get("query", "")).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}..."


def _metric(value: float) -> float:
    return round(float(value), 6)
