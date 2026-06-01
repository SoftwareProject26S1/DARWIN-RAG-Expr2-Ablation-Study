import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2.cli import main
import darwin_rag_exp2.models.crossfit as crossfit_module
from darwin_rag_exp2.models.classifier import (
    TransformerTrainingConfig,
    _checkpoint_fingerprint,
    _read_training_rows,
    _split_fit_calibration_rows,
)
from darwin_rag_exp2.models.splits import build_source_folds


def test_train_classifier_crossfit_writes_bert_out_of_fold_contract(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "classifier" / "crossfit"
    rows = [
        chunk_row("a1", "학사", "수강 신청 변경 기간 학사 공지"),
        chunk_row("a2", "학사", "강의 시간표와 학사 일정 안내"),
        chunk_row("a3", "학사", "졸업 요건 학점 이수 안내"),
        chunk_row("s1", "장학", "국가 장학금 신청 서류 안내"),
        chunk_row("s2", "장학", "성적 장학 선발 결과 공지"),
        chunk_row("s3", "장학", "교내 장학 추천서 제출 안내"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), chunks_path)
    calls = []

    def fake_fit_predict_transformer_classifier(**kwargs):
        calls.append(kwargs)
        progress = kwargs.get("progress_callback")
        if progress is not None:
            progress(f"[bert:{kwargs['progress_label']}] epoch 1/1 started")
            progress(f"[bert:{kwargs['progress_label']}] epoch 1/1 finished")
        labels = tuple(kwargs["categories"])
        prediction_rows = kwargs["prediction_rows"]
        calibration_rows = kwargs["calibration_rows"]
        return SimpleNamespace(
            labels=labels,
            model_reference={
                "model_type": "transformer_sequence_classification",
                "base_model": kwargs["config"].model_name,
                "labels": list(labels),
                "label2id": {label: index for index, label in enumerate(labels)},
                "id2label": {str(index): label for index, label in enumerate(labels)},
                "purpose": kwargs["purpose"],
            },
            calibration_logits=logits_for_rows(calibration_rows, labels),
            calibration_label_ids=[labels.index(row.category) for row in calibration_rows],
            prediction_logits=logits_for_rows(prediction_rows, labels),
        )

    monkeypatch.setattr(
        crossfit_module,
        "_fit_predict_transformer_classifier",
        fake_fit_predict_transformer_classifier,
        raising=False,
    )

    result = main(
        [
            "train-classifier",
            "--mode",
            "crossfit",
            "--chunks",
            str(chunks_path),
            "--folds",
            "3",
            "--output",
            str(output_path),
            "--model",
            "test-bert",
            "--epochs",
            "1",
            "--calibration-fraction",
            "0.5",
            "--log-every-batches",
            "9",
            "--resume",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    captured = capsys.readouterr().out
    assert "[train-classifier:crossfit] preparing 3-fold BERT crossfit" in captured
    for fold_number in range(1, 4):
        assert f"[train-classifier:crossfit] fold {fold_number}/3 started" in captured
        assert f"[bert:crossfit fold {fold_number}/3] epoch 1/1 started" in captured
        assert f"[bert:crossfit fold {fold_number}/3] epoch 1/1 finished" in captured
        assert f"[train-classifier:crossfit] fold {fold_number}/3 finished" in captured
    assert len(calls) == 3
    assert all(call["config"].log_every_batches == 9 for call in calls)
    assert all(call["resume"] is True for call in calls)
    assert [
        call["progress_label"]
        for call in calls
    ] == [
        "crossfit fold 1/3",
        "crossfit fold 2/3",
        "crossfit fold 3/3",
    ]
    manifest = json.loads((output_path / "manifest.json").read_text())
    folds = json.loads((output_path / "folds.json").read_text())["folds"]
    calibration_folds = json.loads((output_path / "calibration_by_fold.json").read_text())["folds"]
    model_references = json.loads((output_path / "model_references.json").read_text())["folds"]
    fold_by_index = {fold["fold_index"]: fold for fold in folds}
    predictions = [
        json.loads(line)
        for line in (output_path / "out_of_fold_predictions.jsonl").read_text().splitlines()
    ]
    stats = json.loads((output_path / "category_stats.json").read_text())["rows"]

    assert manifest["phase"] == 6
    assert manifest["mode"] == "crossfit"
    assert manifest["smoke_only"] is False
    assert manifest["model_type"] == "transformer_sequence_classification"
    assert manifest["base_model"] == "test-bert"
    assert manifest["probability_source"] == "out_of_fold_calibrated_probabilities"
    assert manifest["lambda_c_interpretation"] == "semantic_similarity_mixture_coefficient"
    assert manifest["lambda_c_not"] == "bert_confidence"
    assert len(predictions) == len(rows)
    assert {row["chunk_id"] for row in predictions} == {row["chunk_id"] for row in rows}

    for prediction in predictions:
        fold = fold_by_index[prediction["fold_index"]]
        assert prediction["source_id"] in fold["validation_source_ids"]
        assert prediction["source_id"] not in fold["training_source_ids"]
        assert prediction["probability_source"] == "out_of_fold"

    for fold in calibration_folds:
        fit_sources = set(fold["fit_source_ids"])
        calibration_sources = set(fold["calibration_source_ids"])
        validation_sources = set(fold_by_index[fold["fold_index"]]["validation_source_ids"])
        assert fit_sources.isdisjoint(calibration_sources)
        assert fit_sources.isdisjoint(validation_sources)
        assert calibration_sources.isdisjoint(validation_sources)

    assert all(
        reference["model_type"] == "transformer_sequence_classification"
        for reference in model_references
    )
    assert all(reference["base_model"] == "test-bert" for reference in model_references)
    assert {row["category"] for row in stats} == {"학사", "장학"}
    assert all(row["smoke_only"] is False for row in stats)
    assert all(row["probability_source"] == "out_of_fold" for row in stats)
    assert all(
        row["lambda_c_interpretation"] == "semantic_similarity_mixture_coefficient"
        for row in stats
    )
    assert all(row["lambda_c_not"] == "bert_confidence" for row in stats)
    assert (output_path / "category_stats.parquet").exists()
    assert (output_path / "predictions.parquet").exists()
    assert (output_path / "calibration_by_fold.json").exists()
    assert (output_path / "model_references.json").exists()


def test_train_classifier_crossfit_resume_reuses_completed_partial_fold(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "classifier" / "crossfit"
    rows = [
        chunk_row("a1", "학사", "수강 신청 변경 기간 학사 공지"),
        chunk_row("a2", "학사", "강의 시간표와 학사 일정 안내"),
        chunk_row("s1", "장학", "국가 장학금 신청 서류 안내"),
        chunk_row("s2", "장학", "성적 장학 선발 결과 공지"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), chunks_path)
    config = TransformerTrainingConfig(
        model_name="test-bert",
        epochs=1,
        calibration_fraction=0.5,
        device="cpu",
    )
    training_rows = _read_training_rows(chunks_path)
    folds = build_source_folds(
        [
            {"source_id": row.source_id, "category": row.category}
            for row in training_rows
        ],
        fold_count=2,
    )
    categories = tuple(sorted({row.category for row in training_rows}))
    first_fold = folds[0]
    fold_training_rows, validation_rows = crossfit_module._rows_for_fold(
        training_rows,
        first_fold,
    )
    fit_rows, calibration_rows = _split_fit_calibration_rows(
        fold_training_rows,
        calibration_fraction=config.calibration_fraction,
        seed=config.seed + first_fold.fold_index,
    )
    fingerprint = _checkpoint_fingerprint(
        training_rows=fit_rows,
        calibration_rows=calibration_rows,
        prediction_rows=validation_rows,
        categories=categories,
        config=config,
        purpose=crossfit_module.FOLD_CLASSIFIER_PURPOSE,
    )
    partial_predictions = prediction_dicts(validation_rows, categories, fold_index=0)
    crossfit_module._write_fold_partial(
        output_path,
        fold_index=0,
        payload={
            "schema_version": 1,
            "fingerprint": fingerprint,
            "calibration_report": {
                "fold_index": 0,
                "fit_source_count": len({row.source_id for row in fit_rows}),
                "calibration_source_count": len({row.source_id for row in calibration_rows}),
                "validation_source_count": len(first_fold.validation_source_ids),
                "fit_source_ids": sorted({row.source_id for row in fit_rows}),
                "calibration_source_ids": sorted({row.source_id for row in calibration_rows}),
                "temperature": 1.0,
                "ece_before": 0.0,
                "ece_after": 0.0,
            },
            "model_reference": {
                "fold_index": 0,
                "fit_source_count": len({row.source_id for row in fit_rows}),
                "calibration_source_count": len({row.source_id for row in calibration_rows}),
                "validation_source_count": len(first_fold.validation_source_ids),
                "model_type": "transformer_sequence_classification",
                "base_model": "test-bert",
            },
            "predictions": partial_predictions,
        },
    )
    calls = []

    def fake_fit_predict_transformer_classifier(**kwargs):
        calls.append(kwargs)
        labels = tuple(kwargs["categories"])
        prediction_rows = kwargs["prediction_rows"]
        calibration_rows = kwargs["calibration_rows"]
        return SimpleNamespace(
            labels=labels,
            model_reference={
                "model_type": "transformer_sequence_classification",
                "base_model": kwargs["config"].model_name,
                "labels": list(labels),
                "label2id": {label: index for index, label in enumerate(labels)},
                "id2label": {str(index): label for index, label in enumerate(labels)},
                "purpose": kwargs["purpose"],
            },
            calibration_logits=logits_for_rows(calibration_rows, labels),
            calibration_label_ids=[labels.index(row.category) for row in calibration_rows],
            prediction_logits=logits_for_rows(prediction_rows, labels),
        )

    monkeypatch.setattr(
        crossfit_module,
        "_fit_predict_transformer_classifier",
        fake_fit_predict_transformer_classifier,
        raising=False,
    )

    result = main(
        [
            "train-classifier",
            "--mode",
            "crossfit",
            "--chunks",
            str(chunks_path),
            "--folds",
            "2",
            "--output",
            str(output_path),
            "--model",
            "test-bert",
            "--epochs",
            "1",
            "--calibration-fraction",
            "0.5",
            "--resume",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    captured = capsys.readouterr().out
    assert "[train-classifier:crossfit] fold 1/2 resumed from partial artifact" in captured
    assert len(calls) == 1
    assert calls[0]["progress_label"] == "crossfit fold 2/2"
    predictions = [
        json.loads(line)
        for line in (output_path / "out_of_fold_predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == len(rows)


def chunk_row(source_id: str, category: str, classifier_text: str) -> dict[str, object]:
    return {
        "chunk_id": f"{source_id}::0000",
        "source_id": source_id,
        "chunk_index": 0,
        "category": category,
        "title": classifier_text.split()[0],
        "title_prefix": classifier_text.split()[0],
        "body_text": classifier_text,
        "classifier_text": classifier_text,
        "body_token_count": len(classifier_text.split()),
        "title_token_count": 1,
        "classifier_token_count": len(classifier_text.split()),
        "url": f"https://example.test/{source_id}",
        "slug": source_id,
        "date": "2026-05-01",
        "source": "test",
        "collected_at": "2026-05-01 00:00:00",
    }


def logits_for_rows(rows, labels: tuple[str, ...]) -> list[list[float]]:
    logits = []
    for row in rows:
        logits.append([4.0 if label == row.category else -1.0 for label in labels])
    return logits


def prediction_dicts(rows, labels: tuple[str, ...], *, fold_index: int) -> list[dict[str, object]]:
    predictions = []
    for row in rows:
        probabilities = {
            label: (1.0 if label == row.category else 0.0)
            for label in labels
        }
        predictions.append(
            {
                "chunk_id": row.chunk_id,
                "source_id": row.source_id,
                "category": row.category,
                "predicted_category": row.category,
                "confidence": 1.0,
                "probabilities": probabilities,
                "logits": [1.0 if label == row.category else 0.0 for label in labels],
                "fold_index": fold_index,
                "probability_source": "out_of_fold",
                "temperature": 1.0,
            }
        )
    return predictions
