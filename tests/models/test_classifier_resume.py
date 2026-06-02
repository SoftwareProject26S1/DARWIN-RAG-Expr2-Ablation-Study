import json

import pytest

import darwin_rag_exp2.models.classifier as classifier_module
from darwin_rag_exp2.models.classifier import (
    TransformerTrainingConfig,
    _TrainingRow,
    _checkpoint_fingerprint,
    _checkpoint_saved_message,
    _checkpoint_training_completed,
    _epoch_indexes_to_run,
    _format_mps_memory_report,
    _latest_checkpoint_dir,
    _load_latest_checkpoint_metadata,
    _release_training_memory_for_inference,
    _resolve_torch_device,
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


def test_auto_device_resolution_prefers_cuda_over_mps() -> None:
    torch = FakeTorchDevice(cuda_available=True, mps_available=True)

    assert _resolve_torch_device(torch, "auto") == "cuda"


def test_auto_device_resolution_falls_back_to_mps_then_cpu() -> None:
    assert _resolve_torch_device(
        FakeTorchDevice(cuda_available=False, mps_available=True),
        "auto",
    ) == "mps"
    assert _resolve_torch_device(
        FakeTorchDevice(cuda_available=False, mps_available=False),
        "auto",
    ) == "cpu"


def test_explicit_device_resolution_uses_requested_device() -> None:
    assert _resolve_torch_device(
        FakeTorchDevice(cuda_available=True, mps_available=True),
        "cpu",
    ) == "cpu"


def test_training_memory_cleanup_releases_optimizer_and_mps_cache(monkeypatch) -> None:
    torch = FakeMpsMemoryTorch()
    optimizer = FakeOptimizer()
    collect_calls = []
    messages: list[str] = []
    monkeypatch.setattr(
        classifier_module.gc,
        "collect",
        lambda: collect_calls.append("collect"),
    )

    _release_training_memory_for_inference(
        torch=torch,
        device="mps",
        optimizer=optimizer,
        progress_callback=messages.append,
        progress_label="crossfit fold 1/5",
    )

    assert optimizer.zero_grad_calls == [True]
    assert collect_calls == ["collect"]
    assert torch.mps.calls == ["synchronize", "empty_cache"]
    assert messages == [
        "[bert:crossfit fold 1/5] releasing training memory before inference",
        "[bert:crossfit fold 1/5] MPS memory before cleanup: "
        "current=1.00 GiB, driver=3.00 GiB, recommended=4.00 GiB",
        "[bert:crossfit fold 1/5] MPS memory after cleanup: "
        "current=1.00 GiB, driver=2.00 GiB, recommended=4.00 GiB",
    ]


def test_training_memory_cleanup_skips_mps_cache_for_non_mps_device(monkeypatch) -> None:
    torch = FakeMpsMemoryTorch()
    optimizer = FakeOptimizer()
    collect_calls = []
    monkeypatch.setattr(
        classifier_module.gc,
        "collect",
        lambda: collect_calls.append("collect"),
    )

    _release_training_memory_for_inference(
        torch=torch,
        device="cpu",
        optimizer=optimizer,
        progress_callback=None,
        progress_label="classifier",
    )

    assert optimizer.zero_grad_calls == [True]
    assert collect_calls == ["collect"]
    assert torch.mps.calls == []


def test_checkpoint_training_completed_matches_completed_epoch_count() -> None:
    config = TransformerTrainingConfig(model_name="test-bert", epochs=3)

    assert _checkpoint_training_completed({"completed_epochs": 3}, config) is True
    assert _checkpoint_training_completed({"completed_epochs": 4}, config) is True
    assert _checkpoint_training_completed({"completed_epochs": 2}, config) is False
    assert _checkpoint_training_completed(None, config) is False


def test_mps_memory_report_formats_gib_values() -> None:
    torch = FakeMpsMemoryTorch()

    assert _format_mps_memory_report(
        torch=torch,
        progress_label="crossfit fold 1/5",
        stage="before cleanup",
    ) == (
        "[bert:crossfit fold 1/5] MPS memory before cleanup: "
        "current=1.00 GiB, driver=3.00 GiB, recommended=4.00 GiB"
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


class FakeTorchDevice:
    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = FakeCuda(cuda_available)
        self.backends = FakeBackends(mps_available)

    def device(self, name: str) -> str:
        return name


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = FakeMps(mps_available)


class FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls: list[bool] = []

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.zero_grad_calls.append(set_to_none)


class FakeMpsMemoryTorch:
    def __init__(self) -> None:
        self.mps = FakeMpsMemory()


class FakeMpsMemory:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def current_allocated_memory(self) -> int:
        return 1 * 1024**3

    def driver_allocated_memory(self) -> int:
        if "empty_cache" in self.calls:
            return 2 * 1024**3
        return 3 * 1024**3

    def recommended_max_memory(self) -> int:
        return 4 * 1024**3

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")
