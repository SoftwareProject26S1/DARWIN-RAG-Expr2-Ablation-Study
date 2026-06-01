"""BERT-based classifier training pipelines for Phase 5 and query-time use."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import random
import shutil
import time

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .calibration import calibrate_logits, softmax
from .category_stats import build_category_stats


TRANSFORMER_MODEL_TYPE = "transformer_sequence_classification"
FINAL_PROBABILITY_SOURCE = "full_corpus_calibrated_query_classifier"
CHECKPOINT_SCHEMA_VERSION = 1
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
    log_every_batches: int = 25

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
        if self.log_every_batches < 0:
            raise ValueError("log_every_batches must be non-negative")

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
            "log_every_batches": self.log_every_batches,
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


class _ClassifierDataset:
    """Tokenized classifier rows with batch-level dynamic padding left to the collator."""

    def __init__(
        self,
        *,
        rows: Sequence[_TrainingRow],
        tokenizer,
        label_to_index: Mapping[str, int],
        max_length: int,
    ) -> None:
        self.features = [
            {
                **tokenizer(
                    row.classifier_text,
                    truncation=True,
                    max_length=max_length,
                ),
                "label": label_to_index[row.category],
            }
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, object]:
        return dict(self.features[index])


def train_single_classifier(
    chunks_path: Path,
    output_dir: Path,
    *,
    max_sources_per_category: int = 12,
    max_chunks_per_category: int = 80,
    training_config: TransformerTrainingConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
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
        resume=resume,
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
    resume: bool = False,
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
        resume=resume,
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


def _checkpoint_fingerprint(
    *,
    training_rows: Sequence[_TrainingRow],
    calibration_rows: Sequence[_TrainingRow],
    prediction_rows: Sequence[_TrainingRow],
    categories: Sequence[str],
    config: TransformerTrainingConfig,
    purpose: str,
) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "purpose": purpose,
        "categories": list(categories),
        "training_rows": _fingerprint_rows(training_rows),
        "calibration_rows": _fingerprint_rows(calibration_rows),
        "prediction_rows": _fingerprint_rows(prediction_rows),
        "training_hyperparameters": config.to_manifest(),
    }
    return sha256(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def _fingerprint_rows(rows: Sequence[_TrainingRow]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": row.chunk_id,
            "source_id": row.source_id,
            "category": row.category,
        }
        for row in rows
    ]


def _latest_checkpoint_dir(model_output_dir: Path) -> Path:
    return model_output_dir / "checkpoints" / "latest"


def _checkpoint_saved_message(
    *,
    progress_label: str,
    checkpoint_dir: Path,
    completed_epochs: int,
    total_epochs: int,
) -> str:
    return (
        f"[bert:{progress_label}] checkpoint saved to {checkpoint_dir} "
        f"after epoch {completed_epochs}/{total_epochs}"
    )


def _load_latest_checkpoint_metadata(
    model_output_dir: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, object] | None:
    checkpoint_dir = _latest_checkpoint_dir(model_output_dir)
    metadata_path = checkpoint_dir / "checkpoint.json"
    if not checkpoint_dir.exists():
        return None
    if not metadata_path.exists():
        raise ValueError(f"checkpoint at {checkpoint_dir} is missing checkpoint.json")
    metadata = orjson.loads(metadata_path.read_bytes())
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"incompatible checkpoint schema at {metadata_path}")
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"incompatible checkpoint at {checkpoint_dir}")
    return metadata


def _load_latest_checkpoint_training_state(
    *,
    torch,
    model_output_dir: Path,
) -> Mapping[str, object]:
    state_path = _latest_checkpoint_dir(model_output_dir) / "training_state.pt"
    if not state_path.exists():
        raise ValueError(f"checkpoint at {_latest_checkpoint_dir(model_output_dir)} is missing training_state.pt")
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def _write_latest_checkpoint(
    *,
    torch,
    model,
    tokenizer,
    optimizer,
    scheduler,
    generator,
    model_output_dir: Path,
    fingerprint: str,
    config: TransformerTrainingConfig,
    labels: Sequence[str],
    completed_epochs: int,
    global_step: int,
) -> None:
    checkpoint_dir = _latest_checkpoint_dir(model_output_dir)
    tmp_dir = checkpoint_dir.with_name("latest.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    torch.save(
        {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "dataloader_generator_state": generator.get_state(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
                else None
            ),
            "completed_epochs": completed_epochs,
            "global_step": global_step,
        },
        tmp_dir / "training_state.pt",
    )
    _write_json(
        tmp_dir / "checkpoint.json",
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "base_model": config.model_name,
            "labels": list(labels),
            "completed_epochs": completed_epochs,
            "total_epochs": config.epochs,
            "global_step": global_step,
            "training_hyperparameters": config.to_manifest(),
        },
    )
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def _epoch_indexes_to_run(
    *,
    total_epochs: int,
    completed_epochs: int,
) -> range:
    return range(completed_epochs, total_epochs)


def _move_optimizer_state_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)


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
    resume: bool = False,
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
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
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

    checkpoint_fingerprint = _checkpoint_fingerprint(
        training_rows=training_rows,
        calibration_rows=calibration_rows,
        prediction_rows=prediction_rows,
        categories=labels,
        config=config,
        purpose=purpose,
    )
    checkpoint_metadata = (
        _load_latest_checkpoint_metadata(
            model_output_dir,
            expected_fingerprint=checkpoint_fingerprint,
        )
        if resume
        else None
    )
    checkpoint_dir = _latest_checkpoint_dir(model_output_dir)
    model_load_path = checkpoint_dir if checkpoint_metadata is not None else config.model_name
    if checkpoint_metadata is not None:
        _emit_progress(
            progress_callback,
            (
                f"[bert:{progress_label}] resuming from checkpoint "
                f"{checkpoint_dir} after epoch {checkpoint_metadata['completed_epochs']}"
            ),
        )
    _emit_progress(
        progress_callback,
        f"[bert:{progress_label}] loading tokenizer/model {model_load_path}",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_load_path,
        num_labels=len(labels),
        label2id=label_to_index,
        id2label=id_to_label,
    )
    device = _resolve_torch_device(torch, config.device)
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    training_loader = DataLoader(
        _ClassifierDataset(
            rows=training_rows,
            tokenizer=tokenizer,
            label_to_index=label_to_index,
            max_length=config.max_length,
        ),
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
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
    completed_epochs = 0
    global_step = 0
    if checkpoint_metadata is not None:
        training_state = _load_latest_checkpoint_training_state(
            torch=torch,
            model_output_dir=model_output_dir,
        )
        optimizer.load_state_dict(training_state["optimizer_state"])
        scheduler.load_state_dict(training_state["scheduler_state"])
        _move_optimizer_state_to_device(optimizer, device)
        if "dataloader_generator_state" in training_state:
            generator.set_state(training_state["dataloader_generator_state"])
        if "python_random_state" in training_state:
            random.setstate(training_state["python_random_state"])
        if "torch_rng_state" in training_state:
            torch.set_rng_state(training_state["torch_rng_state"])
        if torch.cuda.is_available() and training_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
        completed_epochs = int(training_state.get("completed_epochs", 0))
        global_step = int(training_state.get("global_step", 0))

    _emit_progress(
        progress_callback,
        (
            f"[bert:{progress_label}] training {len(training_rows)} chunks "
            f"from {len(_source_ids(training_rows))} sources across {len(labels)} "
            f"categories for {config.epochs} epoch(s); "
            f"batch_size={config.train_batch_size}, total_batches={len(training_loader)}, "
            f"total_steps={total_steps}, warmup_steps={warmup_steps}, device={device}"
        ),
    )
    model.train()
    epoch_indexes = _epoch_indexes_to_run(
        total_epochs=config.epochs,
        completed_epochs=completed_epochs,
    )
    if not epoch_indexes:
        _emit_progress(
            progress_callback,
            (
                f"[bert:{progress_label}] checkpoint already completed "
                f"{completed_epochs}/{config.epochs} epoch(s); skipping training"
            ),
        )
    for epoch_index in epoch_indexes:
        epoch_number = epoch_index + 1
        epoch_started_at = time.monotonic()
        processed_chunks = 0
        _emit_progress(
            progress_callback,
            f"[bert:{progress_label}] epoch {epoch_number}/{config.epochs} started",
        )
        for batch_index, batch in enumerate(training_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch_on_device = {
                key: value.to(device)
                for key, value in batch.items()
            }
            outputs = model(**batch_on_device)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            processed_chunks += int(batch_on_device["labels"].shape[0])
            if _should_emit_batch_progress(
                batch_index=batch_index,
                total_batches=len(training_loader),
                log_every_batches=config.log_every_batches,
            ):
                _emit_progress(
                    progress_callback,
                    _format_batch_progress(
                        progress_label=progress_label,
                        stage=f"train epoch {epoch_number}/{config.epochs}",
                        batch_index=batch_index,
                        total_batches=len(training_loader),
                        processed_items=processed_chunks,
                        total_items=len(training_rows),
                        elapsed_seconds=time.monotonic() - epoch_started_at,
                        latest_loss=float(loss.detach().cpu().item()),
                        learning_rate=float(scheduler.get_last_lr()[0]),
                    ),
                )
        _emit_progress(
            progress_callback,
            f"[bert:{progress_label}] epoch {epoch_number}/{config.epochs} finished",
        )
        _write_latest_checkpoint(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            model_output_dir=model_output_dir,
            fingerprint=checkpoint_fingerprint,
            config=config,
            labels=labels,
            completed_epochs=epoch_number,
            global_step=global_step,
        )
        _emit_progress(
            progress_callback,
            _checkpoint_saved_message(
                progress_label=progress_label,
                checkpoint_dir=_latest_checkpoint_dir(model_output_dir),
                completed_epochs=epoch_number,
                total_epochs=config.epochs,
            ),
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
        progress_callback=progress_callback,
        progress_label=progress_label,
        stage="calibration inference",
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
        progress_callback=progress_callback,
        progress_label=progress_label,
        stage="prediction inference",
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


def _should_emit_batch_progress(
    *,
    batch_index: int,
    total_batches: int,
    log_every_batches: int,
) -> bool:
    if log_every_batches <= 0:
        return False
    return batch_index == total_batches or batch_index % log_every_batches == 0


def _format_batch_progress(
    *,
    progress_label: str,
    stage: str,
    batch_index: int,
    total_batches: int,
    processed_items: int,
    total_items: int,
    elapsed_seconds: float,
    latest_loss: float | None = None,
    learning_rate: float | None = None,
) -> str:
    progress_fraction = batch_index / max(total_batches, 1)
    remaining_batches = max(total_batches - batch_index, 0)
    seconds_per_batch = elapsed_seconds / max(batch_index, 1)
    eta_seconds = seconds_per_batch * remaining_batches
    throughput = processed_items / elapsed_seconds if elapsed_seconds > 0 else 0.0
    parts = [
        (
            f"[bert:{progress_label}] {stage} batch {batch_index}/{total_batches} "
            f"({progress_fraction * 100:.1f}%)"
        ),
        f"items={processed_items}/{total_items}",
    ]
    if latest_loss is not None:
        parts.append(f"loss={latest_loss:.4f}")
    if learning_rate is not None:
        parts.append(f"lr={_format_learning_rate(learning_rate)}")
    parts.extend(
        [
            f"elapsed={_format_duration(elapsed_seconds)}",
            f"eta={_format_duration(eta_seconds)}",
            f"chunks/s={throughput:.2f}",
        ]
    )
    return " | ".join(parts)


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    return f"{seconds_int}s"


def _format_learning_rate(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


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
    progress_callback: ProgressCallback | None = None,
    progress_label: str = "classifier",
    stage: str = "inference",
) -> list[list[float]]:
    logits: list[list[float]] = []
    model.eval()
    total_batches = max(1, (len(rows) + config.eval_batch_size - 1) // config.eval_batch_size)
    started_at = time.monotonic()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(rows), config.eval_batch_size), start=1):
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
            if _should_emit_batch_progress(
                batch_index=batch_index,
                total_batches=total_batches,
                log_every_batches=config.log_every_batches,
            ):
                _emit_progress(
                    progress_callback,
                    _format_batch_progress(
                        progress_label=progress_label,
                        stage=stage,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        processed_items=len(logits),
                        total_items=len(rows),
                        elapsed_seconds=time.monotonic() - started_at,
                    ),
                )
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
