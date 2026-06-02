"""Crossfit classifier training for official Phase 6 category statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .calibration import calibrate_logits, softmax
from .category_stats import build_category_stats
from .classifier import (
    ProgressCallback,
    TRANSFORMER_MODEL_TYPE,
    TransformerTrainingConfig,
    _TrainingRow,
    _checkpoint_fingerprint,
    _emit_progress,
    _file_sha256,
    _fit_predict_transformer_classifier,
    _metric,
    _prediction_rows,
    _read_training_rows,
    _source_ids,
    _split_fit_calibration_rows,
    _write_json,
    _write_jsonl,
)
from .splits import SourceFold, build_source_folds


OOF_PROBABILITY_SOURCE = "out_of_fold"
MANIFEST_PROBABILITY_SOURCE = "out_of_fold_calibrated_probabilities"
LAMBDA_C_INTERPRETATION = "semantic_similarity_mixture_coefficient"
LAMBDA_C_NOT = "bert_confidence"
FOLD_CLASSIFIER_PURPOSE = (
    "Phase 6 fold-local BERT classifier for out-of-fold probability artifacts"
)


@dataclass(frozen=True)
class CrossfitClassifierResult:
    """Artifact metadata from the Phase 6 crossfit run."""

    manifest: dict[str, object]


def train_crossfit_classifier(
    chunks_path: Path,
    output_dir: Path,
    *,
    fold_count: int,
    training_config: TransformerTrainingConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
) -> CrossfitClassifierResult:
    """Train fold-local BERT classifiers and write out-of-fold predictions."""

    config = training_config or TransformerTrainingConfig()
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
    calibration_report_rows: list[dict[str, object]] = []
    model_references: list[dict[str, object]] = []
    fold_total = len(folds)
    _emit_progress(
        progress_callback,
        f"[train-classifier:crossfit] preparing {fold_total}-fold BERT crossfit",
    )

    for fold in folds:
        fold_number = fold.fold_index + 1
        _emit_progress(
            progress_callback,
            f"[train-classifier:crossfit] fold {fold_number}/{fold_total} started",
        )
        training_rows, validation_rows = _rows_for_fold(rows, fold)
        fit_rows, calibration_subset_rows = _split_fit_calibration_rows(
            training_rows,
            calibration_fraction=config.calibration_fraction,
            seed=config.seed + fold.fold_index,
        )
        fold_fingerprint = _checkpoint_fingerprint(
            training_rows=fit_rows,
            calibration_rows=calibration_subset_rows,
            prediction_rows=validation_rows,
            categories=categories,
            config=config,
            purpose=FOLD_CLASSIFIER_PURPOSE,
        )
        if resume:
            partial_payload = _load_fold_partial(
                output_dir,
                fold_index=fold.fold_index,
                expected_fingerprint=fold_fingerprint,
            )
            if partial_payload is not None:
                prediction_rows.extend(partial_payload["predictions"])
                calibration_report_rows.append(partial_payload["calibration_report"])
                model_references.append(partial_payload["model_reference"])
                _emit_progress(
                    progress_callback,
                    (
                        f"[train-classifier:crossfit] fold {fold_number}/{fold_total} "
                        "resumed from partial artifact"
                    ),
                )
                continue
        run = _fit_predict_transformer_classifier(
            training_rows=fit_rows,
            calibration_rows=calibration_subset_rows,
            prediction_rows=validation_rows,
            categories=categories,
            model_output_dir=output_dir / "models" / f"fold_{fold.fold_index:03d}",
            config=config,
            purpose=FOLD_CLASSIFIER_PURPOSE,
            progress_callback=progress_callback,
            progress_label=f"crossfit fold {fold_number}/{fold_total}",
            resume=resume,
        )
        if tuple(run.labels) != categories:
            raise ValueError(
                f"fold {fold.fold_index} classifier labels do not match categories"
            )

        calibration = calibrate_logits(run.calibration_logits, run.calibration_label_ids)
        probabilities = softmax(run.prediction_logits, calibration.temperature)
        fold_predictions = _prediction_rows(
            validation_rows,
            run.labels,
            run.prediction_logits,
            probabilities,
        )
        for prediction in fold_predictions:
            prediction["fold_index"] = fold.fold_index
            prediction["probability_source"] = OOF_PROBABILITY_SOURCE
            prediction["temperature"] = _metric(calibration.temperature)
        prediction_rows.extend(fold_predictions)

        calibration_report = {
            "fold_index": fold.fold_index,
            "fit_source_count": len(_source_ids(fit_rows)),
            "calibration_source_count": len(_source_ids(calibration_subset_rows)),
            "validation_source_count": len(fold.validation_source_ids),
            "fit_source_ids": _source_ids(fit_rows),
            "calibration_source_ids": _source_ids(calibration_subset_rows),
            **calibration.to_dict(),
        }
        model_reference = {
            "fold_index": fold.fold_index,
            "fit_source_count": len(_source_ids(fit_rows)),
            "calibration_source_count": len(_source_ids(calibration_subset_rows)),
            "validation_source_count": len(fold.validation_source_ids),
            **run.model_reference,
        }
        calibration_report_rows.append(calibration_report)
        model_references.append(model_reference)
        _write_fold_partial(
            output_dir,
            fold_index=fold.fold_index,
            payload={
                "schema_version": 1,
                "fingerprint": fold_fingerprint,
                "calibration_report": calibration_report,
                "model_reference": model_reference,
                "predictions": fold_predictions,
            },
        )
        _emit_progress(
            progress_callback,
            (
                f"[train-classifier:crossfit] fold {fold_number}/{fold_total} "
                f"partial artifact saved to {_fold_partial_path(output_dir, fold.fold_index)}"
            ),
        )
        _emit_progress(
            progress_callback,
            f"[train-classifier:crossfit] fold {fold_number}/{fold_total} finished",
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
    _write_json(output_dir / "calibration_by_fold.json", {"folds": calibration_report_rows})
    _write_json(output_dir / "model_references.json", {"folds": model_references})
    _write_jsonl(output_dir / "out_of_fold_predictions.jsonl", prediction_rows)
    _write_prediction_rows_parquet(output_dir / "predictions.parquet", prediction_rows)
    _write_json(output_dir / "category_stats.json", {"rows": category_stats})
    _write_category_stats_parquet(output_dir / "category_stats.parquet", category_stats)

    manifest: dict[str, object] = {
        "phase": 6,
        "mode": "crossfit",
        "smoke_only": False,
        "model_type": TRANSFORMER_MODEL_TYPE,
        "base_model": config.model_name,
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "fold_count": len(folds),
        "source_count": len({row.source_id for row in rows}),
        "prediction_chunk_count": len(prediction_rows),
        "category_count": len(categories),
        "categories": list(categories),
        "epochs": config.epochs,
        "training_hyperparameters": config.to_manifest(),
        "probability_source": MANIFEST_PROBABILITY_SOURCE,
        "lambda_c_interpretation": LAMBDA_C_INTERPRETATION,
        "lambda_c_not": LAMBDA_C_NOT,
        "artifact_files": [
            "models/",
            "folds.json",
            "calibration_by_fold.json",
            "model_references.json",
            "out_of_fold_predictions.jsonl",
            "predictions.parquet",
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


def _fold_partial_path(output_dir: Path, fold_index: int) -> Path:
    return output_dir / "partial" / f"fold_{fold_index:03d}.json"


def _write_fold_partial(
    output_dir: Path,
    *,
    fold_index: int,
    payload: Mapping[str, object],
) -> None:
    path = _fold_partial_path(output_dir, fold_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)


def _load_fold_partial(
    output_dir: Path,
    *,
    fold_index: int,
    expected_fingerprint: str,
) -> dict[str, object] | None:
    path = _fold_partial_path(output_dir, fold_index)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"incompatible partial fold schema at {path}")
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"incompatible partial fold artifact at {path}")
    return payload


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


def _write_prediction_rows_parquet(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    parquet_rows = []
    for row in rows:
        parquet_rows.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "source_id": str(row["source_id"]),
                "category": str(row["category"]),
                "predicted_category": str(row["predicted_category"]),
                "confidence": float(row["confidence"]),
                "fold_index": int(row["fold_index"]),
                "probability_source": str(row["probability_source"]),
                "temperature": float(row["temperature"]),
                "probabilities_json": json.dumps(
                    row["probabilities"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("category", pa.string()),
            pa.field("predicted_category", pa.string()),
            pa.field("confidence", pa.float64()),
            pa.field("fold_index", pa.int64()),
            pa.field("probability_source", pa.string()),
            pa.field("temperature", pa.float64()),
            pa.field("probabilities_json", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(parquet_rows, schema=schema), path)
