import json

import pytest

from darwin_rag_exp2.models.classifier import (
    TransformerTrainingConfig,
    _TrainingRow,
    _checkpoint_fingerprint,
    _checkpoint_saved_message,
    _epoch_indexes_to_run,
    _latest_checkpoint_dir,
    _load_latest_checkpoint_metadata,
    _write_latest_checkpoint,
)


def test_checkpoint_helpers_write_and_validate_latest_metadata(tmp_path) -> None:
    model_output_dir = tmp_path / "model"
    fingerprint = _checkpoint_fingerprint(
        training_rows=[row("train", "학사")],
        calibration_rows=[row("cal", "학사")],
        prediction_rows=[row("pred", "학사")],
        categories=("학사",),
        config=TransformerTrainingConfig(model_name="test-bert", epochs=3),
        purpose="unit-test",
    )

    _write_latest_checkpoint(
        torch=FakeTorch(),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        optimizer=FakeStateful({"optimizer": 1}),
        scheduler=FakeStateful({"scheduler": 2}),
        generator=FakeGenerator("generator-state"),
        model_output_dir=model_output_dir,
        fingerprint=fingerprint,
        config=TransformerTrainingConfig(model_name="test-bert", epochs=3),
        labels=("학사",),
        completed_epochs=1,
        global_step=10,
    )

    metadata = _load_latest_checkpoint_metadata(
        model_output_dir,
        expected_fingerprint=fingerprint,
    )

    assert metadata is not None
    assert metadata["completed_epochs"] == 1
    assert metadata["total_epochs"] == 3
    assert metadata["global_step"] == 10
    assert (_latest_checkpoint_dir(model_output_dir) / "model.saved").exists()
    assert (_latest_checkpoint_dir(model_output_dir) / "tokenizer.saved").exists()
    assert (_latest_checkpoint_dir(model_output_dir) / "training_state.pt").exists()


def test_checkpoint_metadata_rejects_incompatible_fingerprint(tmp_path) -> None:
    checkpoint_dir = _latest_checkpoint_dir(tmp_path / "model")
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": "old",
                "completed_epochs": 1,
            }
        )
    )

    with pytest.raises(ValueError, match="incompatible checkpoint"):
        _load_latest_checkpoint_metadata(
            tmp_path / "model",
            expected_fingerprint="new",
        )


def test_epoch_indexes_to_run_resume_from_next_epoch() -> None:
    assert list(_epoch_indexes_to_run(total_epochs=3, completed_epochs=1)) == [1, 2]
    assert list(_epoch_indexes_to_run(total_epochs=3, completed_epochs=3)) == []


def test_checkpoint_saved_message_reports_checkpoint_path(tmp_path) -> None:
    message = _checkpoint_saved_message(
        progress_label="crossfit fold 1/5",
        checkpoint_dir=tmp_path / "model" / "checkpoints" / "latest",
        completed_epochs=2,
        total_epochs=3,
    )

    assert message == (
        "[bert:crossfit fold 1/5] checkpoint saved to "
        f"{tmp_path / 'model' / 'checkpoints' / 'latest'} after epoch 2/3"
    )


def row(source_id: str, category: str) -> _TrainingRow:
    return _TrainingRow(
        chunk_id=f"{source_id}::0000",
        source_id=source_id,
        category=category,
        classifier_text=f"{category} 공지",
    )


class FakeTorch:
    def save(self, payload, path) -> None:
        path.write_text(json.dumps(payload, sort_keys=True))

    def get_rng_state(self):
        return "torch-state"


class FakeModel:
    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.saved").write_text("saved")


class FakeTokenizer:
    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.saved").write_text("saved")


class FakeStateful:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


class FakeGenerator:
    def __init__(self, state):
        self.state = state

    def get_state(self):
        return self.state
