"""Load Phase 8 query rows and attach Phase 9 model features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import orjson

from darwin_rag_exp2.indexing.embeddings import EmbeddingModel, l2_normalize

from .types import QueryFeatures


def load_query_rows(path: Path) -> list[dict[str, object]]:
    """Read query annotation rows from JSONL."""

    rows: list[dict[str, object]] = []
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = orjson.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"query row {line_number} must be an object")
            _validate_query_row(row, line_number=line_number)
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"no query rows found in {path}")
    return rows


def embed_query_rows(
    query_rows: Sequence[Mapping[str, object]],
    *,
    embedding_model: EmbeddingModel,
    normalize_embeddings: bool,
) -> dict[str, list[float]]:
    """Embed query text rows keyed by query_id."""

    texts = [str(row["query"]) for row in query_rows]
    vectors = embedding_model.encode(texts)
    if len(vectors) != len(query_rows):
        raise ValueError(
            "embedding model returned a different number of vectors than queries"
        )
    if normalize_embeddings:
        vectors = l2_normalize(vectors)
    return {
        str(row["query_id"]): [float(value) for value in vector]
        for row, vector in zip(query_rows, vectors, strict=True)
    }


def probabilities_from_query_rows(
    query_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    """Extract precomputed calibrated category probabilities from query rows."""

    probabilities_by_query_id: dict[str, dict[str, float]] = {}
    for row in query_rows:
        query_id = str(row["query_id"])
        probabilities = _extract_probabilities(row)
        if not probabilities:
            raise ValueError(f"query_id {query_id!r} has no category probabilities")
        probabilities_by_query_id[query_id] = probabilities
    return probabilities_by_query_id


def oracle_probabilities_from_query_rows(
    query_rows: Sequence[Mapping[str, object]],
    *,
    categories: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Build diagnostic oracle probabilities from query gold categories."""

    ordered_categories = tuple(dict.fromkeys(str(category) for category in categories))
    if not ordered_categories:
        raise ValueError("oracle probabilities require at least one known category")
    known_categories = set(ordered_categories)
    probabilities_by_query_id: dict[str, dict[str, float]] = {}
    for row in query_rows:
        query_id = str(row["query_id"])
        gold_categories = tuple(
            dict.fromkeys(str(category) for category in row["gold_categories"])
        )
        if not gold_categories:
            raise ValueError(f"query_id {query_id!r} has no gold categories")
        unknown = sorted(set(gold_categories).difference(known_categories))
        if unknown:
            raise ValueError(
                f"query_id {query_id!r} has unknown gold categories: {unknown}"
            )
        oracle_probability = 1.0 / len(gold_categories)
        probabilities_by_query_id[query_id] = {
            category: (oracle_probability if category in gold_categories else 0.0)
            for category in ordered_categories
        }
    return probabilities_by_query_id


def build_query_features(
    query_rows: Sequence[Mapping[str, object]],
    *,
    embeddings_by_query_id: Mapping[str, Sequence[float]],
    probabilities_by_query_id: Mapping[str, Mapping[str, float]],
) -> list[QueryFeatures]:
    """Combine query annotations with embeddings and category probabilities."""

    features: list[QueryFeatures] = []
    for row in query_rows:
        query_id = str(row["query_id"])
        if query_id not in embeddings_by_query_id:
            raise ValueError(f"missing embedding for query_id {query_id!r}")
        if query_id not in probabilities_by_query_id:
            raise ValueError(f"missing probabilities for query_id {query_id!r}")
        features.append(
            QueryFeatures(
                query_id=query_id,
                query=str(row["query"]),
                embedding=[
                    float(value)
                    for value in embeddings_by_query_id[query_id]
                ],
                probabilities={
                    str(category): float(probability)
                    for category, probability in probabilities_by_query_id[query_id].items()
                },
                gold_chunks=tuple(str(value) for value in row["gold_chunks"]),
                gold_categories=tuple(str(value) for value in row["gold_categories"]),
                query_type=str(row["query_type"]),
            )
        )
    return features


def _validate_query_row(row: Mapping[str, object], *, line_number: int) -> None:
    required = {
        "query_id",
        "query",
        "gold_chunks",
        "reference_answer",
        "gold_categories",
        "query_type",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"query row {line_number} missing fields: {sorted(missing)}")
    if not str(row["query_id"]).strip():
        raise ValueError(f"query row {line_number} has empty query_id")
    if not str(row["query"]).strip():
        raise ValueError(f"query row {line_number} has empty query")
    if not isinstance(row["gold_chunks"], list) or not row["gold_chunks"]:
        raise ValueError(f"query row {line_number} must contain gold_chunks")
    if not isinstance(row["gold_categories"], list) or not row["gold_categories"]:
        raise ValueError(f"query row {line_number} must contain gold_categories")


def _extract_probabilities(row: Mapping[str, object]) -> dict[str, float]:
    probabilities = row.get("probabilities")
    if isinstance(probabilities, Mapping):
        return {
            str(category): float(probability)
            for category, probability in probabilities.items()
        }

    probabilities_json = row.get("probabilities_json")
    if isinstance(probabilities_json, str) and probabilities_json.strip():
        decoded = orjson.loads(probabilities_json)
        if not isinstance(decoded, Mapping):
            raise ValueError("probabilities_json must decode to an object")
        return {
            str(category): float(probability)
            for category, probability in decoded.items()
        }

    return {}
