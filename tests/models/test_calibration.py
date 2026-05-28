from darwin_rag_exp2.models.calibration import (
    calibrate_logits,
    expected_calibration_error,
    softmax,
)


def test_temperature_scaling_reduces_ece_for_correct_underconfident_logits() -> None:
    logits = [
        [0.3, 0.0],
        [0.0, 0.3],
        [0.4, 0.0],
        [0.0, 0.4],
    ]
    labels = [0, 1, 0, 1]

    report = calibrate_logits(
        logits,
        labels,
        temperature_candidates=[0.25, 0.5, 1.0, 2.0],
    )

    assert report.temperature < 1.0
    assert report.ece_after < report.ece_before
    assert report.ece_before == expected_calibration_error(softmax(logits), labels)


def test_expected_calibration_error_uses_confidence_weighted_bins() -> None:
    probabilities = [
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.55, 0.45],
    ]
    labels = [0, 1, 0, 1]

    ece = expected_calibration_error(probabilities, labels, n_bins=2)

    assert round(ece, 4) == 0.1625
