"""Temperature scaling and calibration metrics for classifier smoke runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Iterable, Sequence


DEFAULT_TEMPERATURE_CANDIDATES = (
    0.25,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    5.0,
)


@dataclass(frozen=True)
class CalibrationReport:
    """Summary of one scalar temperature-scaling pass."""

    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def softmax(
    logits: Sequence[Sequence[float]],
    temperature: float = 1.0,
) -> list[list[float]]:
    """Convert logits to probabilities with a positive temperature."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    probabilities: list[list[float]] = []
    for row in logits:
        if not row:
            raise ValueError("logit rows must not be empty")
        scaled = [float(value) / temperature for value in row]
        row_max = max(scaled)
        exp_values = [math.exp(value - row_max) for value in scaled]
        denominator = sum(exp_values)
        probabilities.append([value / denominator for value in exp_values])
    return probabilities


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Compute confidence-binned ECE from class probabilities."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if not probabilities:
        return 0.0

    bins = [
        {"count": 0, "confidence": 0.0, "correct": 0}
        for _ in range(n_bins)
    ]
    for row, label in zip(probabilities, labels, strict=True):
        confidence = max(row)
        predicted = max(range(len(row)), key=row.__getitem__)
        index = min(n_bins - 1, int(confidence * n_bins))
        bins[index]["count"] += 1
        bins[index]["confidence"] += confidence
        bins[index]["correct"] += int(predicted == label)

    total = len(probabilities)
    ece = 0.0
    for bin_values in bins:
        count = int(bin_values["count"])
        if count == 0:
            continue
        accuracy = float(bin_values["correct"]) / count
        confidence = float(bin_values["confidence"]) / count
        ece += abs(accuracy - confidence) * count / total
    return ece


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> float:
    """Compute mean negative log likelihood for the gold labels."""

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if not probabilities:
        return 0.0
    losses = []
    for row, label in zip(probabilities, labels, strict=True):
        losses.append(-math.log(max(row[label], 1e-12)))
    return sum(losses) / len(losses)


def calibrate_logits(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature_candidates: Iterable[float] = DEFAULT_TEMPERATURE_CANDIDATES,
) -> CalibrationReport:
    """Fit one scalar temperature by grid-searching validation NLL."""

    if len(logits) != len(labels):
        raise ValueError("logits and labels must have equal length")
    if not logits:
        raise ValueError("calibration requires at least one row")

    candidates = [float(candidate) for candidate in temperature_candidates]
    if not candidates:
        raise ValueError("temperature_candidates must not be empty")
    if any(candidate <= 0 for candidate in candidates):
        raise ValueError("temperature candidates must be positive")

    best_temperature = min(
        candidates,
        key=lambda temperature: negative_log_likelihood(
            softmax(logits, temperature),
            labels,
        ),
    )
    before = softmax(logits)
    after = softmax(logits, best_temperature)
    return CalibrationReport(
        temperature=best_temperature,
        ece_before=expected_calibration_error(before, labels),
        ece_after=expected_calibration_error(after, labels),
        nll_before=negative_log_likelihood(before, labels),
        nll_after=negative_log_likelihood(after, labels),
    )
