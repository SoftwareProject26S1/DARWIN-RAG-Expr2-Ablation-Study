import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2.cli import main, _classifier_model_from_config
import darwin_rag_exp2.models.classifier as classifier_module
from darwin_rag_exp2.models.classifier import (
    TransformerTrainingConfig,
    _ClassifierDataset,
    _TrainingRow,
    _format_batch_progress,
)


def test_train_classifier_single_writes_bert_smoke_artifacts(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "classifier" / "single"
    rows = [
        chunk_row("a1", "학사", "수강 신청 변경 기간 학사 공지"),
        chunk_row("a2", "학사", "학사 일정과 수업 운영 안내"),
        chunk_row("s1", "장학", "장학금 신청 서류 제출 안내"),
        chunk_row("s2", "장학", "국가 장학 선발 결과 공지"),
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
        classifier_module,
        "_fit_predict_transformer_classifier",
        fake_fit_predict_transformer_classifier,
        raising=False,
    )

    result = main(
        [
            "train-classifier",
            "--mode",
            "single",
            "--chunks",
            str(chunks_path),
            "--output",
            str(output_path),
            "--model",
            "test-bert",
            "--epochs",
            "1",
            "--resume",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    captured = capsys.readouterr().out
    assert calls
    assert calls[0]["config"].log_every_batches == 25
    assert calls[0]["resume"] is True
    assert "[train-classifier:single] preparing BERT fine-tuning" in captured
    assert "[bert:single] epoch 1/1 started" in captured
    assert "[bert:single] epoch 1/1 finished" in captured
    manifest = json.loads((output_path / "manifest.json").read_text())
    model_reference = json.loads((output_path / "model_reference.json").read_text())
    assert manifest["smoke_only"] is True
    assert manifest["model_type"] == "transformer_sequence_classification"
    assert manifest["base_model"] == "test-bert"
    assert manifest["epochs"] == 1
    assert model_reference["model_type"] == "transformer_sequence_classification"
    assert model_reference["base_model"] == "test-bert"
    assert (output_path / "model_reference.json").exists()
    assert (output_path / "calibration.json").exists()
    assert (output_path / "sample_predictions.jsonl").exists()
    assert (output_path / "category_stats.parquet").exists()


def test_train_classifier_final_writes_full_corpus_query_model(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "classifier" / "final"
    rows = [
        chunk_row("a1", "학사", "수강 신청 변경 기간 학사 공지"),
        chunk_row("a2", "학사", "학사 일정과 수업 운영 안내"),
        chunk_row("s1", "장학", "장학금 신청 서류 제출 안내"),
        chunk_row("s2", "장학", "국가 장학 선발 결과 공지"),
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
        classifier_module,
        "_fit_predict_transformer_classifier",
        fake_fit_predict_transformer_classifier,
        raising=False,
    )

    result = main(
        [
            "train-classifier",
            "--mode",
            "final",
            "--chunks",
            str(chunks_path),
            "--output",
            str(output_path),
            "--model",
            "test-bert",
            "--epochs",
            "1",
            "--calibration-fraction",
            "0.5",
            "--log-every-batches",
            "7",
            "--resume",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    captured = capsys.readouterr().out
    assert calls
    assert calls[0]["config"].log_every_batches == 7
    assert calls[0]["resume"] is True
    assert "[train-classifier:final] preparing full-corpus BERT fine-tuning" in captured
    assert "[bert:final] epoch 1/1 started" in captured
    assert "[bert:final] epoch 1/1 finished" in captured
    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest["mode"] == "final"
    assert manifest["model_type"] == "transformer_sequence_classification"
    assert manifest["base_model"] == "test-bert"
    assert manifest["probability_source"] == "full_corpus_calibrated_query_classifier"
    assert (output_path / "model_reference.json").exists()
    assert (output_path / "calibration.json").exists()
    assert (output_path / "calibration_predictions.jsonl").exists()


def test_batch_progress_message_reports_status_metrics() -> None:
    message = _format_batch_progress(
        progress_label="single",
        stage="train epoch 1/3",
        batch_index=25,
        total_batches=100,
        processed_items=800,
        total_items=3200,
        elapsed_seconds=50.0,
        latest_loss=0.4321,
        learning_rate=0.0000123,
    )

    assert message.startswith("[bert:single] train epoch 1/3 batch 25/100")
    assert "25.0%" in message
    assert "loss=0.4321" in message
    assert "lr=0.0000123" in message
    assert "elapsed=50s" in message
    assert "eta=150s" in message
    assert "chunks/s=16.00" in message


def test_classifier_model_config_is_read_as_utf8() -> None:
    config_path = RecordingTextPath(
        'experiment:\n  primary_categories:\n    - "학사"\n'
        "models:\n  classifier: klue/bert-base\n"
    )

    model_name = _classifier_model_from_config(config_path)

    assert model_name == "klue/bert-base"
    assert config_path.encoding == "utf-8"


def test_classifier_dataset_tokenizes_without_global_padding() -> None:
    tokenizer = RecordingTokenizer()
    rows = [
        _TrainingRow(
            chunk_id="a::0000",
            source_id="a",
            category="학사",
            classifier_text="짧은 공지",
        )
    ]

    dataset = _ClassifierDataset(
        rows=rows,
        tokenizer=tokenizer,
        label_to_index={"학사": 0},
        max_length=512,
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item["label"] == 0
    assert tokenizer.calls == [
        {
            "text": "짧은 공지",
            "truncation": True,
            "max_length": 512,
        }
    ]


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


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }


class RecordingTextPath:
    def __init__(self, text: str) -> None:
        self.text = text
        self.encoding = None

    def read_text(self, *, encoding=None) -> str:
        self.encoding = encoding
        return self.text
