"""Category partition assignment rules for Phase 7 indexes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json


def build_partition_assignments(
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    ingest_threshold: float,
) -> list[dict[str, object]]:
    """Assign each chunk to all threshold-passing categories or top-1 fallback."""

    if ingest_threshold < 0.0 or ingest_threshold > 1.0:
        raise ValueError("ingest_threshold must be between 0 and 1")

    assignments: list[dict[str, object]] = []
    for row in prediction_rows:
        chunk_id = str(row.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("prediction rows must contain chunk_id")
        probabilities = _extract_probabilities(row)
        if not probabilities:
            raise ValueError(f"prediction row {chunk_id!r} has no probabilities")

        selected = [
            (category, probability, "threshold")
            for category, probability in probabilities.items()
            if probability >= ingest_threshold
        ]
        if not selected:
            category, probability = max(
                probabilities.items(),
                key=lambda item: (item[1], item[0]),
            )
            selected = [(category, probability, "top1_fallback")]

        for category, probability, reason in sorted(
            selected,
            key=lambda item: (-item[1], item[0]),
        ):
            assignments.append(
                {
                    "chunk_id": chunk_id,
                    "category": category,
                    "probability": round(float(probability), 12),
                    "assignment_reason": reason,
                }
            )
    return assignments


def _extract_probabilities(row: Mapping[str, object]) -> dict[str, float]:
    probabilities_json = row.get("probabilities_json")
    if isinstance(probabilities_json, str):
        decoded = json.loads(probabilities_json)
        if not isinstance(decoded, dict):
            raise ValueError("probabilities_json must decode to an object")
        return {
            str(category): float(probability)
            for category, probability in decoded.items()
        }

    probabilities = row.get("probabilities")
    if isinstance(probabilities, Mapping):
        return {
            str(category): float(probability)
            for category, probability in probabilities.items()
        }
    return {}
