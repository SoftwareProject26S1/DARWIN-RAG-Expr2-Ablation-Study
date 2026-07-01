"""Load Phase 8 query rows and attach Phase 9 model features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import orjson

from darwin_rag_exp2.evaluation import queries as query_validation
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
            rows.append(_normalize_query_row(row, line_number=line_number))
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
                graded_relevance=_graded_relevance(row),
                neighbor_chunk_ids=_string_tuple(row.get("neighbor_chunk_ids", ())),
                source_id=str(row.get("source_id") or ""),
                evidence_unit=str(row.get("evidence_unit") or ""),
                schema_version=str(row.get("schema_version") or ""),
                requires_multi_category=_optional_bool(row, "requires_multi_category"),
                gold_category_pure=_optional_bool(row, "gold_category_pure"),
            )
        )
    return features


def _normalize_query_row(
    row: Mapping[str, object],
    *,
    line_number: int,
) -> dict[str, object]:
    if row.get("schema_version") == query_validation.V2_SCHEMA_VERSION:
        return _normalize_v2_query_row(row, line_number=line_number)

    _validate_legacy_query_row(row, line_number=line_number)
    return dict(row)


def _validate_legacy_query_row(row: Mapping[str, object], *, line_number: int) -> None:
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


def _normalize_v2_query_row(
    row: Mapping[str, object],
    *,
    line_number: int,
) -> dict[str, object]:
    helper_row = dict(row)
    graded_relevance = row.get("graded_relevance")
    gold_chunks = _gold_chunk_ids(row)
    if isinstance(graded_relevance, Mapping) and gold_chunks:
        helper_row["graded_relevance"] = {
            chunk_id: graded_relevance[chunk_id]
            for chunk_id in gold_chunks
            if chunk_id in graded_relevance
        }
    normalized = query_validation._normalize_v2_row(
        _split_from_query_id(row),
        line_number,
        helper_row,
        chunk_ids=_known_chunk_ids(row),
        config=query_validation.QueryValidationConfig(
            primary_categories=_known_categories(row),
            expected_dev_count=0,
            expected_test_count=0,
            non_single_fraction=0.0,
        ),
    )
    normalized["graded_relevance"] = _graded_relevance(row)
    return normalized


def _split_from_query_id(row: Mapping[str, object]) -> str:
    query_id = str(row.get("query_id", ""))
    if query_id.startswith("test_q"):
        return "test"
    return "dev"


def _known_chunk_ids(row: Mapping[str, object]) -> set[str]:
    chunk_ids = set(_string_tuple(row.get("neighbor_chunk_ids", ())))
    chunk_ids.update(_gold_chunk_ids(row))
    return chunk_ids


def _gold_chunk_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    gold_chunk_ids = row.get("gold_chunk_ids")
    if not isinstance(gold_chunk_ids, list):
        return ()
    chunks: list[str] = []
    for item in gold_chunk_ids:
        if isinstance(item, Mapping):
            chunk_id = item.get("chunk_id")
            if isinstance(chunk_id, str):
                chunks.append(chunk_id)
    return tuple(chunks)


def _known_categories(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *_string_tuple(row.get("expected_categories", ())),
                *_string_tuple(row.get("gold_category_set", ())),
            ]
        )
    )


def _graded_relevance(row: Mapping[str, object]) -> dict[str, float]:
    graded_relevance = row.get("graded_relevance", {})
    if not isinstance(graded_relevance, Mapping):
        return {}
    normalized: dict[str, float] = {}
    for chunk_id, score in graded_relevance.items():
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError("graded_relevance values must be numeric")
        relevance = float(score)
        if relevance < 0.0 or relevance > 1.0:
            raise ValueError("graded_relevance values must be in [0, 1]")
        normalized[str(chunk_id)] = relevance
    return normalized


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value)


def _optional_bool(row: Mapping[str, object], key: str) -> bool | None:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return None


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
