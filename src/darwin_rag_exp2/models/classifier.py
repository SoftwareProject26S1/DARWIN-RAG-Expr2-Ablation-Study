"""Single-model classifier smoke pipeline for Phase 5."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import re

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .calibration import calibrate_logits, softmax
from .category_stats import build_category_stats


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass(frozen=True)
class SingleClassifierResult:
    """Artifact metadata from the Phase 5 smoke run."""

    manifest: dict[str, object]
    calibration: dict[str, float]


@dataclass(frozen=True)
class _TrainingRow:
    chunk_id: str
    source_id: str
    category: str
    classifier_text: str


class _NaiveBayesTextClassifier:
    """Small deterministic text classifier used only for smoke plumbing."""

    def __init__(
        self,
        *,
        labels: tuple[str, ...],
        class_doc_counts: Mapping[str, int],
        class_token_counts: Mapping[str, Counter[str]],
        alpha: float,
    ) -> None:
        self.labels = labels
        self.class_doc_counts = dict(class_doc_counts)
        self.class_token_counts = {
            label: Counter(counts)
            for label, counts in class_token_counts.items()
        }
        self.alpha = alpha
        self.vocabulary = sorted(
            {
                token
                for counts in self.class_token_counts.values()
                for token in counts
            }
        )
        self.class_token_totals = {
            label: sum(self.class_token_counts[label].values())
            for label in self.labels
        }

    @classmethod
    def fit(
        cls,
        rows: Sequence[_TrainingRow],
        *,
        alpha: float = 1.0,
    ) -> "_NaiveBayesTextClassifier":
        if not rows:
            raise ValueError("single classifier smoke training requires rows")
        labels = tuple(sorted({row.category for row in rows}))
        if len(labels) < 2:
            raise ValueError("single classifier smoke training requires >=2 categories")

        class_doc_counts: Counter[str] = Counter()
        class_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            class_doc_counts[row.category] += 1
            class_token_counts[row.category].update(_tokenize(row.classifier_text))

        return cls(
            labels=labels,
            class_doc_counts=class_doc_counts,
            class_token_counts=class_token_counts,
            alpha=alpha,
        )

    def decision_function(self, texts: Iterable[str]) -> list[list[float]]:
        """Return per-label log scores for each text."""

        return [self._logits(text) for text in texts]

    def to_model_reference(
        self,
        *,
        purpose: str = "Phase 5 single-mode plumbing smoke, not official results",
    ) -> dict[str, object]:
        """Return lightweight model metadata without storing full training text."""

        return {
            "model_type": "multinomial_naive_bayes_smoke",
            "purpose": purpose,
            "labels": list(self.labels),
            "alpha": self.alpha,
            "vocabulary_size": len(self.vocabulary),
            "class_doc_counts": self.class_doc_counts,
            "class_token_totals": self.class_token_totals,
            "top_tokens_by_class": {
                label: [
                    token
                    for token, _ in self.class_token_counts[label].most_common(20)
                ]
                for label in self.labels
            },
        }

    def _logits(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        token_counts = Counter(tokens)
        total_docs = sum(self.class_doc_counts.values())
        label_count = len(self.labels)
        vocabulary_size = max(len(self.vocabulary), 1)
        logits: list[float] = []

        for label in self.labels:
            prior = math.log(
                (self.class_doc_counts[label] + self.alpha)
                / (total_docs + self.alpha * label_count)
            )
            denominator = (
                self.class_token_totals[label] + self.alpha * vocabulary_size
            )
            log_score = prior
            for token, count in token_counts.items():
                token_probability = (
                    self.class_token_counts[label][token] + self.alpha
                ) / denominator
                log_score += count * math.log(token_probability)
            logits.append(log_score)
        return logits


def train_single_classifier(
    chunks_path: Path,
    output_dir: Path,
    *,
    max_sources_per_category: int = 12,
    max_chunks_per_category: int = 80,
) -> SingleClassifierResult:
    """Train the Phase 5 in-sample single classifier smoke pipeline."""

    rows = _select_smoke_rows(
        _read_training_rows(chunks_path),
        max_sources_per_category=max_sources_per_category,
        max_chunks_per_category=max_chunks_per_category,
    )
    classifier = _NaiveBayesTextClassifier.fit(rows)
    logits = classifier.decision_function(row.classifier_text for row in rows)
    label_to_index = {label: index for index, label in enumerate(classifier.labels)}
    labels = [label_to_index[row.category] for row in rows]
    calibration = calibrate_logits(logits, labels)
    probabilities = softmax(logits, calibration.temperature)
    prediction_rows = _prediction_rows(rows, classifier.labels, logits, probabilities)
    category_stats = build_category_stats(
        prediction_rows,
        categories=classifier.labels,
        smoke_only=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "model_reference.json", classifier.to_model_reference())
    _write_json(output_dir / "calibration.json", calibration.to_dict())
    _write_jsonl(output_dir / "sample_predictions.jsonl", prediction_rows)
    _write_json(output_dir / "category_stats.json", {"rows": category_stats})
    _write_category_stats_parquet(output_dir / "category_stats.parquet", category_stats)

    manifest: dict[str, object] = {
        "phase": 5,
        "mode": "single",
        "smoke_only": True,
        "chunks_path": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "training_chunk_count": len(rows),
        "training_source_count": len({row.source_id for row in rows}),
        "category_count": len(classifier.labels),
        "categories": list(classifier.labels),
        "temperature": calibration.temperature,
        "ece_before": calibration.ece_before,
        "ece_after": calibration.ece_after,
        "artifact_files": [
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


def _tokenize(text: str) -> list[str]:
    tokens = [
        match.group(0).lower()
        for match in _TOKEN_PATTERN.finditer(text)
    ]
    return tokens or ["__empty__"]


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
