"""BERT-based classifier training pipelines for Phase 5 and query-time use."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import random

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .calibration import calibrate_logits, softmax
from .category_stats import build_category_stats


TRANSFORMER_MODEL_TYPE = "transformer_sequence_classification"
FINAL_PROBABILITY_SOURCE = "full_corpus_calibrated_query_classifier"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Hyperparameters for deterministic BERT fine-tuning."""

    model_name: str = "klue/bert-base"
    max_length: int = 512
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    calibration_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")

    def to_manifest(self) -> dict[str, object]:
        """Return JSON-serializable training settings."""

        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "epochs": self.epochs,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "calibration_fraction": self.calibration_fraction,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass(frozen=True)
class SingleClassifierResult:
    """Artifact metadata from the Phase 5 smoke run."""

    manifest: dict[str, object]
    calibration: dict[str, float]


@dataclass(frozen=True)
class FinalClassifierResult:
    """Artifact metadata from the full-corpus query classifier run."""

    manifest: dict[str, object]
    calibration: dict[str, float]


@dataclass(frozen=True)
class _TrainingRow:
    chunk_id: str
    source_id: str
    category: str
    classifier_text: str


@dataclass(frozen=True)
class _TransformerRunResult:
    labels: tuple[str, ...]
    model_reference: dict[str, object]
    calibration_logits: list[list[float]]
    calibration_label_ids: list[int]
    prediction_logits: list[list[float]]


def train_single_classifier(
    chunks_path: Path,
    output_dir: Path,
    *,
    max_sources_per_category: int = 12,
    max_chunks_per_category: int = 80,
    training_config: TransformerTrainingConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SingleClassifierResult:
    """Train the Phase 5 BERT single-model smoke pipeline."""

    config = training_config or TransformerTrainingConfig(epochs=1)
    _emit_progress(
        progress_callback,
        "[train-classifier:single] preparing BERT fine-tuning",
    )
    rows = _select_smoke_rows(
        _read_training_rows(chunks_path),
        max_sources_per_category=max_sources_per_category,
        max_chunks_per_category=max_chunks_per_category,
    )
    categories = tuple(sorted({row.category for row in rows}))
    fit_rows, calibration_rows = _split_fit_calibration_rows(
        rows,
        calibration_fraction=config.calibration_fraction,
        seed=config.seed,
    )
    run = _fit_predict_transformer_classifier(
        training_rows=fit_rows,
        calibration_rows=calibration_rows,
        prediction_rows=rows,
        categories=categories,
        model_output_dir=output_dir / "model",
        config=config,
        purpose="Phase 5 single-mode BERT fine-tuning smoke",
        progress_callback=progress_callback,
        progress_label="single",
    )
    calibration = calibrate_logits(run.calibration_logits, run.calibration_label_ids)
    probabilities = softmax(run.prediction_logits, calibration.temperature)
    prediction_rows = _prediction_rows(rows, run.labels, run.prediction_logits, probabilities)
    category_stats = build_category_stats(
        prediction_rows,
        categories=run.labels,
        smoke_only=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "model_reference.json", run.model_reference)
    _write_json(output_dir / "calibration.json", calibration.to_dict())
    _write_jsonl(output_dir / "sample_predictions.jsonl", prediction_rows)
    _write_json(output_dir / "category_stats.json", {"rows": category_stats})
    _write_category_stats_parquet(output_dir / "category_stats.parquet", category_stats)

    manifest: dict[str, object] = {
        "phase": 5,
        "mode": "single",
        "smoke_only": True,
        "model_type": TRANSFORMER_MODEL_TYPE,
        "base_model": config.model_name,
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "training_chunk_count": len(fit_rows),
        "training_source_count": len({row.source_id for row in fit_rows}),
        "calibration_chunk_count": len(calibration_rows),
        "calibration_source_count": len({row.source_id for row in calibration_rows}),
        "prediction_chunk_count": len(rows),
        "category_count": len(run.labels),
        "categories": list(run.labels),
        "epochs": config.epochs,
        "training_hyperparameters": config.to_manifest(),
        "temperature": calibration.temperature,
        "ece_before": calibration.ece_before,
        "ece_after": calibration.ece_after,
        "artifact_files": [
            "model/",
            "model_reference.json",
            "calibration.json",
            "sample_predictions.jsonl",
            "category_stats.json",
            "category_stats.parquet",
            "manifest.json",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return SingleClassifierResult(
        manifest=manifest,
        calibration=calibration.to_dict(),
    )


def train_final_classifier(
    chunks_path: Path,
    output_dir: Path,
    *,
    training_config: TransformerTrainingConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> FinalClassifierResult:
    """Train the full-corpus calibrated BERT classifier for query-time routing."""

    config = training_config or TransformerTrainingConfig()
    _emit_progress(
        progress_callback,
        "[train-classifier:final] preparing full-corpus BERT fine-tuning",
    )
    rows = _read_training_rows(chunks_path)
    categories = tuple(sorted({row.category for row in rows}))
    fit_rows, calibration_rows = _split_fit_calibration_rows(
        rows,
        calibration_fraction=config.calibration_fraction,
        seed=config.seed,
    )
    run = _fit_predict_transformer_classifier(
        training_rows=fit_rows,
        calibration_rows=calibration_rows,
        prediction_rows=calibration_rows,
        categories=categories,
        model_output_dir=output_dir / "model",
        config=config,
        purpose="Final full-corpus BERT classifier for query-time routing",
        progress_callback=progress_callback,
        progress_label="final",
    )
    calibration = calibrate_logits(run.calibration_logits, run.calibration_label_ids)
    probabilities = softmax(run.prediction_logits, calibration.temperature)
    calibration_predictions = _prediction_rows(
        calibration_rows,
        run.labels,
        run.prediction_logits,
        probabilities,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "model_reference.json", run.model_reference)
    _write_json(output_dir / "calibration.json", calibration.to_dict())
    _write_jsonl(output_dir / "calibration_predictions.jsonl", calibration_predictions)

    manifest: dict[str, object] = {
        "phase": 6,
        "mode": "final",
        "smoke_only": False,
        "model_type": TRANSFORMER_MODEL_TYPE,
        "base_model": config.model_name,
        "probability_source": FINAL_PROBABILITY_SOURCE,
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "training_chunk_count": len(fit_rows),
        "training_source_count": len({row.source_id for row in fit_rows}),
        "calibration_chunk_count": len(calibration_rows),
        "calibration_source_count": len({row.source_id for row in calibration_rows}),
        "category_count": len(run.labels),
        "categories": list(run.labels),
        "epochs": config.epochs,
        "training_hyperparameters": config.to_manifest(),
        "temperature": calibration.temperature,
        "ece_before": calibration.ece_before,
        "ece_after": calibration.ece_after,
        "artifact_files": [
            "model/",
            "model_reference.json",
            "calibration.json",
            "calibration_predictions.jsonl",
            "manifest.json",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return FinalClassifierResult(
        manifest=manifest,
        calibration=calibration.to_dict(),
    )


def _read_training_rows(chunks_path: Path) -> list[_TrainingRow]:
    table = pq.read_table(
        chunks_path,
        columns=["chunk_id", "source_id", "category", "classifier_text"],
    )
    rows = [
        _TrainingRow(
            chunk_id=str(row["chunk_id"]),
            source_id=str(row["source_id"]),
            category=str(row["category"]),
            classifier_text=str(row["classifier_text"]),
        )
        for row in table.to_pylist()
    ]
    if not rows:
        raise ValueError(f"no chunks found in {chunks_path}")
    return rows


def _select_smoke_rows(
    rows: Sequence[_TrainingRow],
    *,
    max_sources_per_category: int,
    max_chunks_per_category: int,
) -> list[_TrainingRow]:
    if max_sources_per_category <= 0:
        raise ValueError("max_sources_per_category must be positive")
    if max_chunks_per_category <= 0:
        raise ValueError("max_chunks_per_category must be positive")

    selected: list[_TrainingRow] = []
    rows_by_category: dict[str, list[_TrainingRow]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item.category, item.source_id, item.chunk_id)):
        rows_by_category[row.category].append(row)

    for category in sorted(rows_by_category):
        source_ids: set[str] = set()
        category_rows: list[_TrainingRow] = []
        for row in rows_by_category[category]:
            if row.source_id not in source_ids:
                if len(source_ids) >= max_sources_per_category:
                    continue
                source_ids.add(row.source_id)
            if row.source_id in source_ids:
                category_rows.append(row)
            if len(category_rows) >= max_chunks_per_category:
                break
        selected.extend(category_rows)
    return selected


def _split_fit_calibration_rows(
    rows: Sequence[_TrainingRow],
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[_TrainingRow], list[_TrainingRow]]:
    """Split rows by source so calibration text is not used for fitting."""

    if not rows:
        raise ValueError("classifier training requires rows")
    if not 0 <= calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in [0, 1)")

    rows_by_source: dict[str, list[_TrainingRow]] = defaultdict(list)
    category_by_source: dict[str, str] = {}
    for row in rows:
        rows_by_source[row.source_id].append(row)
        previous = category_by_source.get(row.source_id)
        if previous is not None and previous != row.category:
            raise ValueError(f"source_id {row.source_id!r} has multiple categories")
        category_by_source[row.source_id] = row.category

    sources_by_category: dict[str, list[str]] = defaultdict(list)
    for source_id, category in category_by_source.items():
        sources_by_category[category].append(source_id)

    rng = random.Random(seed)
    calibration_sources: set[str] = set()
    for category in sorted(sources_by_category):
        sources = sorted(sources_by_category[category])
        if len(sources) < 2:
            continue
        shuffled = list(sources)
        rng.shuffle(shuffled)
        requested = max(1, round(len(sources) * calibration_fraction))
        count = min(requested, len(sources) - 1)
        calibration_sources.update(shuffled[:count])

    if not calibration_sources:
        return list(rows), list(rows)

    fit_rows = [
        row
        for row in rows
        if row.source_id not in calibration_sources
    ]
    calibration_rows = [
        row
        for row in rows
        if row.source_id in calibration_sources
    ]
    if not fit_rows or not calibration_rows:
        return list(rows), list(rows)
    return fit_rows, calibration_rows


def _source_ids(rows: Sequence[_TrainingRow]) -> list[str]:
    return sorted({row.source_id for row in rows})


def _label_ids(rows: Sequence[_TrainingRow], labels: Sequence[str]) -> list[int]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    return [label_to_index[row.category] for row in rows]


def _fit_predict_transformer_classifier(
    *,
    training_rows: Sequence[_TrainingRow],
    calibration_rows: Sequence[_TrainingRow],
    prediction_rows: Sequence[_TrainingRow],
    categories: Sequence[str],
    model_output_dir: Path,
    config: TransformerTrainingConfig,
    purpose: str,
    progress_callback: ProgressCallback | None = None,
    progress_label: str = "classifier",
) -> _TransformerRunResult:
    """Fine-tune a Hugging Face sequence classifier and return raw logits."""

    if not training_rows:
        raise ValueError("transformer fine-tuning requires training rows")
    if not calibration_rows:
        raise ValueError("transformer calibration requires calibration rows")
    if not prediction_rows:
        raise ValueError("transformer prediction requires prediction rows")

    _emit_progress(
        progress_callback,
        f"[bert:{progress_label}] loading training dependencies",
    )
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - exercised by environment
        raise RuntimeError(
            "BERT fine-tuning requires the classifier dependency group: "
            "uv run --group classifier ..."
        ) from exc

    labels = tuple(categories)
    label_to_index = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_index.items()}
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    _emit_progress(
        progress_callback,
        f"[bert:{progress_label}] loading tokenizer/model {config.model_name}",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        label2id=label_to_index,
        id2label=id_to_label,
    )
    device = _resolve_torch_device(torch, config.device)
    model.to(device)

    class _ClassifierDataset(Dataset):
        def __init__(self, rows: Sequence[_TrainingRow]) -> None:
            self.encodings = tokenizer(
                [row.classifier_text for row in rows],
                truncation=True,
                padding=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(
                [label_to_index[row.category] for row in rows],
                dtype=torch.long,
            )

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, object]:
            item = {
                key: value[index]
                for key, value in self.encodings.items()
            }
            item["labels"] = self.labels[index]
            return item

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    training_loader = DataLoader(
        _ClassifierDataset(training_rows),
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, len(training_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    _emit_progress(
        progress_callback,
        (
            f"[bert:{progress_label}] training {len(training_rows)} chunks "
            f"for {config.epochs} epoch(s)"
        ),
    )
    model.train()
    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        _emit_progress(
            progress_callback,
            f"[bert:{progress_label}] epoch {epoch_number}/{config.epochs} started",
        )
        for batch in training_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_on_device = {
                key: value.to(device)
                for key, value in batch.items()
            }
            outputs = model(**batch_on_device)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
        _emit_progress(
            progress_callback,
            f"[bert:{progress_label}] epoch {epoch_number}/{config.epochs} finished",
        )

    _emit_progress(
        progress_callback,
        f"[bert:{progress_label}] saving model to {model_output_dir}",
    )
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    _emit_progress(
        progress_callback,
        f"[bert:{progress_label}] running calibration inference",
    )
    calibration_logits = _predict_transformer_logits(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        rows=calibration_rows,
        config=config,
        device=device,
    )
    _emit_progress(
        progress_callback,
        f"[bert:{progress_label}] running prediction inference",
    )
    prediction_logits = _predict_transformer_logits(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        rows=prediction_rows,
        config=config,
        device=device,
    )
    model_reference: dict[str, object] = {
        "model_type": TRANSFORMER_MODEL_TYPE,
        "base_model": config.model_name,
        "model_dir": str(model_output_dir),
        "purpose": purpose,
        "labels": list(labels),
        "label2id": label_to_index,
        "id2label": {str(index): label for index, label in id_to_label.items()},
        "training_chunk_count": len(training_rows),
        "training_source_count": len(_source_ids(training_rows)),
        "calibration_chunk_count": len(calibration_rows),
        "calibration_source_count": len(_source_ids(calibration_rows)),
        "prediction_chunk_count": len(prediction_rows),
        "resolved_device": str(device),
        "training_hyperparameters": config.to_manifest(),
    }
    return _TransformerRunResult(
        labels=labels,
        model_reference=model_reference,
        calibration_logits=calibration_logits,
        calibration_label_ids=_label_ids(calibration_rows, labels),
        prediction_logits=prediction_logits,
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _resolve_torch_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _predict_transformer_logits(
    *,
    torch,
    model,
    tokenizer,
    rows: Sequence[_TrainingRow],
    config: TransformerTrainingConfig,
    device,
) -> list[list[float]]:
    logits: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), config.eval_batch_size):
            batch_rows = rows[start : start + config.eval_batch_size]
            encoded = tokenizer(
                [row.classifier_text for row in batch_rows],
                truncation=True,
                padding=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
            output = model(**encoded)
            logits.extend(output.logits.detach().cpu().tolist())
    return logits


def _prediction_rows(
    rows: Sequence[_TrainingRow],
    labels: Sequence[str],
    logits: Sequence[Sequence[float]],
    probabilities: Sequence[Sequence[float]],
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for row, logit_row, probability_row in zip(
        rows,
        logits,
        probabilities,
        strict=True,
    ):
        predicted_index = max(range(len(probability_row)), key=probability_row.__getitem__)
        predictions.append(
            {
                "chunk_id": row.chunk_id,
                "source_id": row.source_id,
                "category": row.category,
                "predicted_category": labels[predicted_index],
                "confidence": _metric(probability_row[predicted_index]),
                "probabilities": {
                    label: _metric(probability)
                    for label, probability in zip(labels, probability_row, strict=True)
                },
                "logits": [
                    _metric(value)
                    for value in logit_row
                ],
            }
        )
    return predictions


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            output.write(b"\n")


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
        ]
    )
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(value: float) -> float:
    return round(float(value), 12)
