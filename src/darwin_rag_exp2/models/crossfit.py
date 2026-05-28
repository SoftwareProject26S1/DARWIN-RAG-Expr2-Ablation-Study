"""Crossfit classifier training for official Phase 6 category statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .calibration import calibrate_logits, softmax
from .category_stats import build_category_stats
from .classifier import (
    _NaiveBayesTextClassifier,
    _TrainingRow,
    _file_sha256,
    _metric,
    _prediction_rows,
    _read_training_rows,
    _write_json,
    _write_jsonl,
)
from .splits import SourceFold, build_source_folds


OOF_PROBABILITY_SOURCE = "out_of_fold"
MANIFEST_PROBABILITY_SOURCE = "out_of_fold_calibrated_probabilities"
LAMBDA_C_INTERPRETATION = "semantic_similarity_mixture_coefficient"
LAMBDA_C_NOT = "bert_confidence"


@dataclass(frozen=True)
class CrossfitClassifierResult:
    """Artifact metadata from the Phase 6 crossfit run."""

    manifest: dict[str, object]


def train_crossfit_classifier(
    chunks_path: Path,
    output_dir: Path,
    *,
    fold_count: int,
) -> CrossfitClassifierResult:
    """Train fold-local classifiers and write out-of-fold predictions."""

    rows = _read_training_rows(chunks_path)
    folds = build_source_folds(
        [
            {"source_id": row.source_id, "category": row.category}
            for row in rows
        ],
        fold_count=fold_count,
    )
    categories = tuple(sorted({row.category for row in rows}))
    prediction_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    model_references: list[dict[str, object]] = []

    for fold in folds:
        training_rows, validation_rows = _rows_for_fold(rows, fold)
        classifier = _NaiveBayesTextClassifier.fit(training_rows)
        if tuple(classifier.labels) != categories:
            raise ValueError(
                f"fold {fold.fold_index} training data does not contain every category"
            )

        logits = classifier.decision_function(
            row.classifier_text
            for row in validation_rows
        )
        label_to_index = {
            label: index
            for index, label in enumerate(classifier.labels)
        }
        labels = [label_to_index[row.category] for row in validation_rows]
        calibration = calibrate_logits(logits, labels)
        probabilities = softmax(logits, calibration.temperature)
        fold_predictions = _prediction_rows(
            validation_rows,
            classifier.labels,
            logits,
            probabilities,
        )
        for prediction in fold_predictions:
            prediction["fold_index"] = fold.fold_index
            prediction["probability_source"] = OOF_PROBABILITY_SOURCE
            prediction["temperature"] = _metric(calibration.temperature)
        prediction_rows.extend(fold_predictions)

        calibration_rows.append(
            {
                "fold_index": fold.fold_index,
                "training_source_count": len(fold.training_source_ids),
                "validation_source_count": len(fold.validation_source_ids),
                **calibration.to_dict(),
            }
        )
        model_references.append(
            {
                "fold_index": fold.fold_index,
                "training_source_count": len(fold.training_source_ids),
                "validation_source_count": len(fold.validation_source_ids),
                **classifier.to_model_reference(
                    purpose=(
                        "Phase 6 fold-local classifier for out-of-fold "
                        "probability artifacts"
                    )
                ),
            }
        )

    prediction_rows = sorted(
        prediction_rows,
        key=lambda row: (str(row["source_id"]), str(row["chunk_id"])),
    )
    category_stats = build_category_stats(
        prediction_rows,
        categories=categories,
        smoke_only=False,
    )
    for row in category_stats:
        row["probability_source"] = OOF_PROBABILITY_SOURCE
        row["lambda_c_interpretation"] = LAMBDA_C_INTERPRETATION
        row["lambda_c_not"] = LAMBDA_C_NOT

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "folds.json", {"folds": [_fold_to_dict(fold) for fold in folds]})
    _write_json(output_dir / "calibration_by_fold.json", {"folds": calibration_rows})
    _write_json(output_dir / "model_references.json", {"folds": model_references})
    _write_jsonl(output_dir / "out_of_fold_predictions.jsonl", prediction_rows)
    _write_json(output_dir / "category_stats.json", {"rows": category_stats})
    _write_category_stats_parquet(output_dir / "category_stats.parquet", category_stats)

    manifest: dict[str, object] = {
        "phase": 6,
        "mode": "crossfit",
        "smoke_only": False,
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "fold_count": len(folds),
        "source_count": len({row.source_id for row in rows}),
        "prediction_chunk_count": len(prediction_rows),
        "category_count": len(categories),
        "categories": list(categories),
        "probability_source": MANIFEST_PROBABILITY_SOURCE,
        "lambda_c_interpretation": LAMBDA_C_INTERPRETATION,
        "lambda_c_not": LAMBDA_C_NOT,
        "artifact_files": [
            "folds.json",
            "calibration_by_fold.json",
            "model_references.json",
            "out_of_fold_predictions.jsonl",
            "category_stats.json",
            "category_stats.parquet",
            "manifest.json",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return CrossfitClassifierResult(manifest=manifest)


def _rows_for_fold(
    rows: Sequence[_TrainingRow],
    fold: SourceFold,
) -> tuple[list[_TrainingRow], list[_TrainingRow]]:
    training_sources = set(fold.training_source_ids)
    validation_sources = set(fold.validation_source_ids)
    training_rows = [
        row
        for row in rows
        if row.source_id in training_sources
    ]
    validation_rows = [
        row
        for row in rows
        if row.source_id in validation_sources
    ]
    if not training_rows:
        raise ValueError(f"fold {fold.fold_index} has no training rows")
    if not validation_rows:
        raise ValueError(f"fold {fold.fold_index} has no validation rows")
    return training_rows, validation_rows


def _fold_to_dict(fold: SourceFold) -> dict[str, object]:
    return {
        "fold_index": fold.fold_index,
        "training_source_ids": list(fold.training_source_ids),
        "validation_source_ids": list(fold.validation_source_ids),
    }


def _write_category_stats_parquet(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    schema = pa.schema(
        [
            pa.field("category", pa.string()),
            pa.field("chunk_count", pa.int64()),
            pa.field("source_count", pa.int64()),
            pa.field("mu_confidence", pa.float64()),
            pa.field("sigma_confidence", pa.float64()),
            pa.field("lambda_c", pa.float64()),
            pa.field("smoke_only", pa.bool_()),
            pa.field("probability_source", pa.string()),
            pa.field("lambda_c_interpretation", pa.string()),
            pa.field("lambda_c_not", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path)
