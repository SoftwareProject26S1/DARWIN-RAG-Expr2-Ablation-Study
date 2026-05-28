"""Category-level calibrated confidence statistics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def build_category_stats(
    prediction_rows: Sequence[Mapping[str, object]],
    categories: Sequence[str],
    *,
    alpha: float = 8.0,
    rho: float = 4.0,
    tau: float = 0.5,
    smoke_only: bool = False,
) -> list[dict[str, object]]:
    """Summarize true-label calibrated probabilities by category."""

    stats: list[dict[str, object]] = []
    for category in categories:
        matching = [
            row
            for row in prediction_rows
            if str(row.get("category")) == category
        ]
        confidences = [
            _probability_for(row, category)
            for row in matching
        ]
        mu_confidence = _mean(confidences)
        sigma_confidence = _population_std(confidences, mu_confidence)
        stats.append(
            {
                "category": category,
                "chunk_count": len(matching),
                "source_count": len(
                    {str(row.get("source_id")) for row in matching}
                ),
                "mu_confidence": _metric(mu_confidence),
                "sigma_confidence": _metric(sigma_confidence),
                "lambda_c": _metric(
                    _sigmoid(alpha * (mu_confidence - tau) - rho * sigma_confidence)
                ),
                "smoke_only": smoke_only,
            }
        )
    return stats


def _probability_for(row: Mapping[str, object], category: str) -> float:
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return 0.0
    value = probabilities.get(category, 0.0)
    return float(value)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _population_std(values: Sequence[float], mean: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _sigmoid(value: float) -> float:
    if value >= 0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    numerator = math.exp(value)
    return numerator / (1.0 + numerator)


def _metric(value: float) -> float:
    return round(float(value), 12)
