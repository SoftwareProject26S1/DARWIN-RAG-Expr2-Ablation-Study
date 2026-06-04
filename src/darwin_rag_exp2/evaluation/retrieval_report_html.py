"""Static HTML report rendering for Phase 9 retrieval analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any


METRIC_KEYS = ("hit@10", "mrr@10", "ndcg@10", "recall@10")
COLORS = {
    "B0": "#4c78a8",
    "B1": "#f58518",
    "B2-score": "#54a24b",
    "P-score": "#b279a2",
}


def render_primary_report_html(analysis: Mapping[str, object]) -> str:
    """Render a dependency-free static HTML report for primary retrieval analysis."""

    summary = _mapping(analysis.get("summary"))
    title = "Phase 9 Retrieval Analysis"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="ko">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{escape(title)}</h1>",
            _summary_cards(summary),
            _variant_metric_section(_rows(analysis.get("metrics_by_variant"))),
            _delta_histogram_section(
                _rows(analysis.get("paired_deltas")),
                _mapping(analysis.get("paired_comparison")),
            ),
            _heatmap_section(
                _rows(analysis.get("breakdown_by_query_type")),
                _rows(analysis.get("breakdown_by_gold_category")),
                str(summary.get("metric", "ndcg@10")),
            ),
            _route_width_section(_rows(analysis.get("routing_diagnostics"))),
            _failure_table_section(_rows(analysis.get("failure_cases"))),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _summary_cards(summary: Mapping[str, object]) -> str:
    return (
        '<section class="summary-grid">'
        + _card("Queries", summary.get("query_count", ""))
        + _card("Variants", summary.get("variant_count", ""))
        + _card("Metric", summary.get("metric", ""))
        + _card("Rows", summary.get("row_count", ""))
        + "</section>"
    )


def _card(label: str, value: object) -> str:
    return (
        '<div class="summary-card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(str(value))}</strong>"
        "</div>"
    )


def _variant_metric_section(rows: Sequence[Mapping[str, object]]) -> str:
    return (
        '<section class="panel">'
        "<h2>Variant별 metric bar chart</h2>"
        "<p>B0, B1, B2-score, P-score의 Top-10 retrieval metric 평균입니다.</p>"
        + _variant_metric_svg(rows)
        + _table(rows, preferred_columns=["variant", "query_count", *METRIC_KEYS])
        + "</section>"
    )


def _variant_metric_svg(rows: Sequence[Mapping[str, object]]) -> str:
    variants = [str(row.get("variant", "")) for row in rows]
    width = 900
    height = 320
    margin_left = 70
    margin_bottom = 54
    plot_width = width - margin_left - 24
    plot_height = height - 48 - margin_bottom
    metric_gap = plot_width / max(len(METRIC_KEYS), 1)
    bar_width = min(28.0, metric_gap / max(len(variants), 1) * 0.72)
    parts = [_svg_frame(width, height, margin_left, 24, plot_width, plot_height)]
    for metric_index, metric in enumerate(METRIC_KEYS):
        group_x = margin_left + metric_index * metric_gap + metric_gap * 0.16
        for variant_index, variant in enumerate(variants):
            row = rows[variant_index]
            value = _float(row.get(metric))
            x = group_x + variant_index * (bar_width + 4)
            bar_height = plot_height * max(0.0, min(1.0, value))
            y = 24 + plot_height - bar_height
            parts.append(
                '<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                'fill="{fill}"><title>{title}</title></rect>'.format(
                    x=x,
                    y=y,
                    w=bar_width,
                    h=bar_height,
                    fill=COLORS.get(variant, "#777777"),
                    title=escape(f"{variant} {metric}: {value:.3f}"),
                )
            )
        label_x = margin_left + metric_index * metric_gap + metric_gap * 0.5
        parts.append(
            f'<text x="{label_x:.1f}" y="{height - 18}" text-anchor="middle">{escape(metric)}</text>'
        )
    parts.extend(_legend(variants, width - 280, 26))
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _delta_histogram_section(
    deltas: Sequence[Mapping[str, object]],
    comparison: Mapping[str, object],
) -> str:
    stats = (
        f"wins/ties/losses: {comparison.get('wins', 0)}/"
        f"{comparison.get('ties', 0)}/{comparison.get('losses', 0)}, "
        f"mean delta: {comparison.get('mean_delta', 0)}"
    )
    return (
        '<section class="panel">'
        "<h2>P-score - B2-score delta 분포 histogram</h2>"
        f"<p>{escape(stats)}</p>"
        + _delta_histogram_svg(deltas)
        + _table(
            [comparison],
            preferred_columns=[
                "metric",
                "query_count",
                "mean_delta",
                "median_delta",
                "wins",
                "ties",
                "losses",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "wilcoxon_p_value",
            ],
        )
        + "</section>"
    )


def _delta_histogram_svg(deltas: Sequence[Mapping[str, object]]) -> str:
    values = [_float(row.get("delta")) for row in deltas]
    width = 900
    height = 260
    margin_left = 70
    plot_width = width - margin_left - 28
    plot_height = height - 74
    if not values:
        return f'<svg viewBox="0 0 {width} {height}"><text x="40" y="60">No deltas</text></svg>'
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    bin_count = min(12, max(3, int(len(values) ** 0.5) + 1))
    counts = [0 for _ in range(bin_count)]
    for value in values:
        index = int((value - min_value) / (max_value - min_value) * bin_count)
        counts[min(index, bin_count - 1)] += 1
    max_count = max(counts) or 1
    bar_gap = 4
    bar_width = (plot_width - bar_gap * (bin_count - 1)) / bin_count
    parts = [_svg_frame(width, height, margin_left, 24, plot_width, plot_height)]
    for index, count in enumerate(counts):
        bar_height = plot_height * count / max_count
        x = margin_left + index * (bar_width + bar_gap)
        y = 24 + plot_height - bar_height
        low = min_value + (max_value - min_value) * index / bin_count
        high = min_value + (max_value - min_value) * (index + 1) / bin_count
        parts.append(
            '<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            'fill="#4c78a8"><title>{title}</title></rect>'.format(
                x=x,
                y=y,
                w=bar_width,
                h=bar_height,
                title=escape(f"{low:.3f} to {high:.3f}: {count}"),
            )
        )
    parts.append(
        f'<text x="{margin_left}" y="{height - 20}">{min_value:.3f}</text>'
    )
    parts.append(
        f'<text x="{width - 60}" y="{height - 20}" text-anchor="end">{max_value:.3f}</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _heatmap_section(
    query_type_rows: Sequence[Mapping[str, object]],
    category_rows: Sequence[Mapping[str, object]],
    metric: str,
) -> str:
    return (
        '<section class="panel">'
        "<h2>query_type/category별 heatmap</h2>"
        f"<p>{escape(metric)} 평균입니다. multi-category query는 category별로 펼쳐 집계됩니다.</p>"
        "<h3>query_type</h3>"
        + _heatmap_svg(query_type_rows, label_key="query_type", metric=metric)
        + _heatmap_table(query_type_rows, label_key="query_type", metric=metric)
        + "<h3>gold_category</h3>"
        + _heatmap_svg(category_rows, label_key="gold_category", metric=metric)
        + _heatmap_table(category_rows, label_key="gold_category", metric=metric)
        + "</section>"
    )


def _heatmap_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    label_key: str,
    metric: str,
) -> str:
    labels = sorted({str(row.get(label_key, "")) for row in rows})
    variants = [
        variant
        for variant in ("B0", "B1", "B2-score", "P-score")
        if any(str(row.get("variant")) == variant for row in rows)
    ]
    if not labels or not variants:
        return '<svg viewBox="0 0 900 90"><text x="24" y="44">No heatmap rows</text></svg>'
    cell_width = 120
    cell_height = 34
    label_width = 180
    top = 36
    width = label_width + cell_width * len(variants) + 28
    height = top + cell_height * len(labels) + 24
    by_cell = {
        (str(row.get(label_key, "")), str(row.get("variant", ""))): _float(row.get(metric))
        for row in rows
    }
    parts: list[str] = []
    for column_index, variant in enumerate(variants):
        x = label_width + column_index * cell_width
        parts.append(
            f'<text x="{x + cell_width / 2:.1f}" y="22" text-anchor="middle">{escape(variant)}</text>'
        )
    for row_index, label in enumerate(labels):
        y = top + row_index * cell_height
        parts.append(
            f'<text x="{label_width - 10}" y="{y + 22}" text-anchor="end">{escape(label)}</text>'
        )
        for column_index, variant in enumerate(variants):
            x = label_width + column_index * cell_width
            value = by_cell.get((label, variant), 0.0)
            parts.append(
                '<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" '
                'fill="{fill}" stroke="#d9e2ec"><title>{title}</title></rect>'.format(
                    x=x,
                    y=y,
                    w=cell_width,
                    h=cell_height,
                    fill=_heat_color(value),
                    title=escape(f"{label} {variant} {metric}: {value:.3f}"),
                )
            )
            parts.append(
                f'<text x="{x + cell_width / 2:.1f}" y="{y + 22}" text-anchor="middle">{value:.3f}</text>'
            )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _heatmap_table(
    rows: Sequence[Mapping[str, object]],
    *,
    label_key: str,
    metric: str,
) -> str:
    labels = sorted({str(row.get(label_key, "")) for row in rows})
    variants = [variant for variant in ("B0", "B1", "B2-score", "P-score") if any(str(row.get("variant")) == variant for row in rows)]
    by_cell = {
        (str(row.get(label_key, "")), str(row.get("variant", ""))): _float(row.get(metric))
        for row in rows
    }
    header = "<tr><th>{}</th>{}</tr>".format(
        escape(label_key),
        "".join(f"<th>{escape(variant)}</th>" for variant in variants),
    )
    body = []
    for label in labels:
        cells = [f"<th>{escape(label)}</th>"]
        for variant in variants:
            value = by_cell.get((label, variant), 0.0)
            cells.append(
                '<td style="background:{color}">{value:.3f}</td>'.format(
                    color=_heat_color(value),
                    value=value,
                )
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="heatmap">{header}{"".join(body)}</table>'


def _route_width_section(rows: Sequence[Mapping[str, object]]) -> str:
    return (
        '<section class="panel">'
        "<h2>Route width 분포</h2>"
        "<p>soft routing이 실제로 여러 category를 열었는지 확인합니다.</p>"
        + _route_width_svg(rows)
        + _table(
            rows,
            preferred_columns=[
                "variant",
                "query_count",
                "route_width_mean",
                "route_width_1_rate",
                "route_width_ge2_rate",
                "top1_fallback_rate",
                "legacy_rows",
            ],
        )
        + "</section>"
    )


def _route_width_svg(rows: Sequence[Mapping[str, object]]) -> str:
    variants = [str(row.get("variant", "")) for row in rows]
    metrics = ("route_width_1_rate", "route_width_ge2_rate", "top1_fallback_rate")
    colors = {
        "route_width_1_rate": "#f58518",
        "route_width_ge2_rate": "#54a24b",
        "top1_fallback_rate": "#e45756",
    }
    width = 900
    height = 300
    margin_left = 74
    plot_width = width - margin_left - 24
    plot_height = height - 78
    group_gap = plot_width / max(len(variants), 1)
    bar_width = min(42.0, group_gap / 4)
    parts = [_svg_frame(width, height, margin_left, 24, plot_width, plot_height)]
    for variant_index, variant in enumerate(variants):
        row = rows[variant_index]
        group_x = margin_left + variant_index * group_gap + group_gap * 0.25
        for metric_index, metric in enumerate(metrics):
            value = _float(row.get(metric))
            bar_height = plot_height * max(0.0, min(1.0, value))
            x = group_x + metric_index * (bar_width + 5)
            y = 24 + plot_height - bar_height
            parts.append(
                '<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                'fill="{fill}"><title>{title}</title></rect>'.format(
                    x=x,
                    y=y,
                    w=bar_width,
                    h=bar_height,
                    fill=colors[metric],
                    title=escape(f"{variant} {metric}: {value:.3f}"),
                )
            )
        label_x = margin_left + variant_index * group_gap + group_gap * 0.5
        parts.append(
            f'<text x="{label_x:.1f}" y="{height - 22}" text-anchor="middle">{escape(variant)}</text>'
        )
    parts.extend(_legend(metrics, width - 360, 26, colors=colors))
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _failure_table_section(rows: Sequence[Mapping[str, object]]) -> str:
    table_rows = []
    for row in rows:
        variants = _mapping(row.get("variants"))
        b2 = _mapping(variants.get("B2-score"))
        p_score = _mapping(variants.get("P-score"))
        table_rows.append(
            {
                "query_id": row.get("query_id", ""),
                "query_type": row.get("query_type", ""),
                "gold_categories": ", ".join(_string_list(row.get("gold_categories"))),
                "B2 metric": b2.get("metric", ""),
                "P metric": p_score.get("metric", ""),
                "B2 top": _top_preview(b2),
                "P top": _top_preview(p_score),
                "query": row.get("query", ""),
            }
        )
    return (
        '<section class="panel">'
        "<h2>실패 query Top-N</h2>"
        "<p>선택 metric 기준 실패 또는 낮은 성능 query입니다.</p>"
        + _table(
            table_rows,
            preferred_columns=[
                "query_id",
                "query_type",
                "gold_categories",
                "B2 metric",
                "P metric",
                "B2 top",
                "P top",
                "query",
            ],
        )
        + "</section>"
    )


def _top_preview(variant_payload: Mapping[str, object]) -> str:
    top10 = variant_payload.get("top10")
    if not isinstance(top10, list) or not top10:
        return ""
    first = _mapping(top10[0])
    parts = [
        str(first.get("chunk_id", "")),
        str(first.get("source_category", first.get("chunk_category", ""))),
        str(first.get("title", "")),
    ]
    return " / ".join(part for part in parts if part)


def _table(
    rows: Sequence[Mapping[str, object]],
    *,
    preferred_columns: Sequence[str],
) -> str:
    if not rows:
        return '<p class="empty">No rows</p>'
    columns = [column for column in preferred_columns if any(column in row for row in rows)]
    for row in rows:
        for key in row:
            key_string = str(key)
            if key_string not in columns:
                columns.append(key_string)
    header = "<tr>{}</tr>".format(
        "".join(f"<th>{escape(column)}</th>" for column in columns)
    )
    body = []
    for row in rows:
        body.append(
            "<tr>{}</tr>".format(
                "".join(
                    f"<td>{escape(_display(row.get(column, '')))}</td>"
                    for column in columns
                )
            )
        )
    return f"<table>{header}{''.join(body)}</table>"


def _svg_frame(
    width: int,
    height: int,
    margin_left: int,
    margin_top: int,
    plot_width: float,
    plot_height: float,
) -> str:
    x0 = margin_left
    y0 = margin_top
    y1 = margin_top + plot_height
    parts = [
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#444"/>',
        f'<line x1="{x0}" y1="{y1}" x2="{x0 + plot_width}" y2="{y1}" stroke="#444"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = y1 - plot_height * value
        parts.append(
            f'<line x1="{x0 - 4}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="#444"/>'
        )
        parts.append(
            f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>'
        )
    return "".join(parts)


def _legend(
    labels: Sequence[str],
    x: int,
    y: int,
    *,
    colors: Mapping[str, str] | None = None,
) -> list[str]:
    palette = colors or COLORS
    parts: list[str] = []
    for index, label in enumerate(labels):
        item_y = y + index * 20
        parts.append(
            f'<rect x="{x}" y="{item_y}" width="12" height="12" fill="{palette.get(label, "#777777")}"/>'
        )
        parts.append(
            f'<text x="{x + 18}" y="{item_y + 11}">{escape(str(label))}</text>'
        )
    return parts


def _heat_color(value: float) -> str:
    bounded = max(0.0, min(1.0, value))
    red = int(245 - 70 * bounded)
    green = int(247 - 10 * bounded)
    blue = int(250 - 125 * bounded)
    return f"rgb({red},{green},{blue})"


def _rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _display(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _css() -> str:
    return """
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f7f8fa; color: #1f2933; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 24px 56px; }
h1 { margin: 0 0 22px; font-size: 30px; font-weight: 720; }
h2 { margin: 0 0 8px; font-size: 21px; }
h3 { margin: 24px 0 8px; font-size: 16px; }
p { margin: 0 0 16px; color: #52606d; line-height: 1.55; }
.summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
.summary-card { background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px 16px; }
.summary-card span { display: block; color: #627d98; font-size: 13px; margin-bottom: 4px; }
.summary-card strong { font-size: 22px; }
.panel { background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 18px; margin-top: 16px; overflow-x: auto; }
svg { width: 100%; height: auto; margin: 8px 0 16px; }
svg text { fill: #334e68; font-size: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
th, td { border: 1px solid #d9e2ec; padding: 7px 8px; text-align: left; vertical-align: top; }
th { background: #f0f4f8; color: #334e68; font-weight: 650; }
.heatmap td { text-align: right; font-variant-numeric: tabular-nums; }
.empty { color: #829ab1; }
@media (max-width: 760px) {
  main { padding: 20px 12px 36px; }
  .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""
