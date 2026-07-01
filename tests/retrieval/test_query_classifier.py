import json

import pytest

from darwin_rag_exp2.retrieval.query_classifier import (
    load_final_query_classifier_metadata,
    probability_dicts_from_logits,
)


def test_probability_dicts_from_logits_applies_temperature_and_labels() -> None:
    probabilities = probability_dicts_from_logits(
        [[2.0, 0.0]],
        labels=("학사", "장학"),
        temperature=2.0,
    )

    assert probabilities == [
        {
            "학사": pytest.approx(0.7310585786),
            "장학": pytest.approx(0.2689414214),
        }
    ]


def test_load_final_query_classifier_metadata_reads_reference_and_calibration(tmp_path) -> None:
    classifier_dir = tmp_path / "final"
    classifier_dir.mkdir()
    (classifier_dir / "model_reference.json").write_text(
        json.dumps(
            {
                "labels": ["학사", "장학"],
                "training_hyperparameters": {
                    "max_length": 256,
                    "eval_batch_size": 4,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (classifier_dir / "calibration.json").write_text(
        json.dumps({"temperature": 1.5}),
        encoding="utf-8",
    )

    metadata = load_final_query_classifier_metadata(classifier_dir)

    assert metadata.labels == ("학사", "장학")
    assert metadata.temperature == 1.5
    assert metadata.max_length == 256
    assert metadata.batch_size == 4
