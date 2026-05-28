"""Source-level fold construction for crossfit classifier artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFold:
    """One crossfit split with source IDs held out as a unit."""

    fold_index: int
    training_source_ids: tuple[str, ...]
    validation_source_ids: tuple[str, ...]


def build_source_folds(
    rows: Sequence[Mapping[str, object]],
    *,
    fold_count: int,
) -> list[SourceFold]:
    """Build deterministic source-level folds from chunk-like rows."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")

    source_to_category: dict[str, str] = {}
    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        category = str(row.get("category", "")).strip()
        if not source_id:
            raise ValueError("rows must contain non-empty source_id values")
        if not category:
            raise ValueError("rows must contain non-empty category values")
        previous = source_to_category.get(source_id)
        if previous is not None and previous != category:
            raise ValueError(f"source_id {source_id!r} has multiple categories")
        source_to_category[source_id] = category

    if not source_to_category:
        raise ValueError("crossfit folds require at least one source")
    if fold_count > len(source_to_category):
        raise ValueError("fold_count must not exceed source count")

    sources_by_category: dict[str, list[str]] = defaultdict(list)
    for source_id, category in source_to_category.items():
        sources_by_category[category].append(source_id)

    validation_buckets: list[list[str]] = [[] for _ in range(fold_count)]
    for category_offset, category in enumerate(sorted(sources_by_category)):
        for source_offset, source_id in enumerate(sorted(sources_by_category[category])):
            fold_index = (source_offset + category_offset) % fold_count
            validation_buckets[fold_index].append(source_id)

    all_sources = tuple(sorted(source_to_category))
    folds: list[SourceFold] = []
    for fold_index, validation_sources in enumerate(validation_buckets):
        validation_source_ids = tuple(sorted(validation_sources))
        if not validation_source_ids:
            raise ValueError(
                "fold_count produced an empty validation fold; reduce fold_count"
            )
        validation_set = set(validation_source_ids)
        training_source_ids = tuple(
            source_id
            for source_id in all_sources
            if source_id not in validation_set
        )
        folds.append(
            SourceFold(
                fold_index=fold_index,
                training_source_ids=training_source_ids,
                validation_source_ids=validation_source_ids,
            )
        )
    return folds
