"""Final classifier inference helpers for Phase 9 query routing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from darwin_rag_exp2.models.calibration import softmax


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class FinalQueryClassifierMetadata:
    """Frozen metadata needed for calibrated query classification."""

    labels: tuple[str, ...]
    temperature: float
    max_length: int
    batch_size: int


class FinalQueryClassifier:
    """Load the Phase 6 final classifier and predict query category probabilities."""

    def __init__(
        self,
        classifier_dir: Path,
        *,
        device: str = "auto",
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.classifier_dir = classifier_dir
        self.metadata = load_final_query_classifier_metadata(classifier_dir)
        self.batch_size = batch_size or self.metadata.batch_size
        self.progress_callback = progress_callback
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "final query classification requires the classifier dependency group: "
                "uv run --group classifier ..."
            ) from error

        self._torch = torch
        self._device = _resolve_torch_device(torch, device)
        model_dir = classifier_dir / "model"
        self._emit(f"[query-classifier] loading model from {model_dir}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.to(self._device)
        self._model.eval()

    def predict_probabilities(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return calibrated category probabilities for query texts."""

        logits: list[list[float]] = []
        total = len(texts)
        with self._torch.inference_mode():
            for start in range(0, total, self.batch_size):
                batch_texts = list(texts[start : start + self.batch_size])
                encoded = self._tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.metadata.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                }
                outputs = self._model(**encoded)
                logits.extend(outputs.logits.detach().cpu().tolist())
                self._emit(
                    f"[query-classifier] classified {min(start + self.batch_size, total)}/{total} queries"
                )
        return probability_dicts_from_logits(
            logits,
            labels=self.metadata.labels,
            temperature=self.metadata.temperature,
        )

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)


def load_final_query_classifier_metadata(
    classifier_dir: Path,
) -> FinalQueryClassifierMetadata:
    """Read labels, temperature, and inference limits from final classifier artifacts."""

    reference = _load_json(classifier_dir / "model_reference.json")
    calibration = _load_json(classifier_dir / "calibration.json")
    labels = reference.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("model_reference.json must contain non-empty labels")
    training_hyperparameters = reference.get("training_hyperparameters") or {}
    if not isinstance(training_hyperparameters, dict):
        raise ValueError("training_hyperparameters must be an object")
    return FinalQueryClassifierMetadata(
        labels=tuple(str(label) for label in labels),
        temperature=float(calibration["temperature"]),
        max_length=int(training_hyperparameters.get("max_length", 512)),
        batch_size=int(training_hyperparameters.get("eval_batch_size", 16)),
    )


def probability_dicts_from_logits(
    logits: Sequence[Sequence[float]],
    *,
    labels: Sequence[str],
    temperature: float,
) -> list[dict[str, float]]:
    """Convert logits into calibrated category probability mappings."""

    if not labels:
        raise ValueError("labels must not be empty")
    probabilities = softmax(logits, temperature)
    results: list[dict[str, float]] = []
    for row in probabilities:
        if len(row) != len(labels):
            raise ValueError("logit width must match labels")
        results.append(
            {
                str(label): float(probability)
                for label, probability in zip(labels, row, strict=True)
            }
        )
    return results


def _load_json(path: Path) -> dict[str, Any]:
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _resolve_torch_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
