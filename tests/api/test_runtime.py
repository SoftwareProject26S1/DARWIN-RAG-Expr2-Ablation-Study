from pathlib import Path

import pytest


def test_runtime_settings_use_env_overrides(tmp_path):
    from darwin_rag_exp2.api.runtime import load_runtime_settings

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  normalize_embeddings: true",
                "models:",
                "  embedder: config/embedder",
                "  generator: config/generator",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_runtime_settings(
        env={
            "DARWIN_EXP2_CONFIG": str(config_path),
            "DARWIN_EXP2_INDEXES_DIR": "custom/indexes",
            "DARWIN_EXP2_CHUNKS_PATH": "custom/chunks.parquet",
            "DARWIN_EXP2_SETTINGS": "custom/frozen.yaml",
            "DARWIN_EXP2_QUERY_CLASSIFIER_DIR": "custom/classifier",
            "DARWIN_EXP2_EMBEDDING_DEVICE": "cuda:1",
            "DARWIN_EXP2_LLM_PLATFORM": "ROCm",
            "DARWIN_EXP2_LLM_MODEL": "local/model",
            "DARWIN_EXP2_LLM_TOKENIZER": "local/tokenizer",
            "DARWIN_EXP2_LLM_HF_CONFIG_PATH": "local/config",
            "DARWIN_EXP2_LLM_MAX_MODEL_LEN": "2048",
            "DARWIN_EXP2_LLM_GPU_MEMORY_UTILIZATION": "0.85",
            "DARWIN_EXP2_LLM_ENFORCE_EAGER": "true",
            "DARWIN_EXP2_LLM_MAX_TOKENS": "128",
            "DARWIN_EXP2_LLM_TEMPERATURE": "0.1",
            "DARWIN_EXP2_LLM_THINKING_MODE": "think",
            "DARWIN_EXP2_LLM_TOP_P": "0.91",
            "DARWIN_EXP2_LLM_TOP_K": "30",
            "DARWIN_EXP2_LLM_MIN_P": "0.02",
            "DARWIN_EXP2_LLM_REPETITION_PENALTY": "1.05",
            "DARWIN_EXP2_LLM_PRESENCE_PENALTY": "0.2",
        },
        cwd=tmp_path,
    )

    assert settings.config_path == config_path
    assert settings.indexes_dir == tmp_path / "custom/indexes"
    assert settings.chunks_path == tmp_path / "custom/chunks.parquet"
    assert settings.settings_path == tmp_path / "custom/frozen.yaml"
    assert settings.query_classifier_dir == tmp_path / "custom/classifier"
    assert settings.resolve_llm_model() == "local/model"
    assert settings.llm_platform == "rocm"
    assert settings.llm_tokenizer == "local/tokenizer"
    assert settings.llm_hf_config_path == "local/config"
    assert settings.llm_max_model_len == 2048
    assert settings.llm_gpu_memory_utilization == 0.85
    assert settings.llm_enforce_eager is True
    assert settings.llm_max_tokens == 128
    assert settings.llm_temperature == 0.1
    assert settings.llm_thinking_mode == "think"
    assert settings.llm_top_p == 0.91
    assert settings.llm_top_k == 30
    assert settings.llm_min_p == 0.02
    assert settings.llm_repetition_penalty == 1.05
    assert settings.llm_presence_penalty == 0.2
    assert settings.embedding_model == "config/embedder"
    assert settings.embedding_device == "cuda:1"
    assert settings.classifier_device == "cuda:1"
    assert settings.normalize_embeddings is True


def test_runtime_settings_default_classifier_falls_back_to_single(tmp_path, monkeypatch):
    from darwin_rag_exp2.api import runtime
    from darwin_rag_exp2.api.runtime import load_runtime_settings

    config_path = tmp_path / "configs" / "experiment.default.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  normalize_embeddings: false",
                "models:",
                "  embedder: BAAI/bge-m3",
                "  generator: mlx-community/Qwen3-8B-4bit",
                "  generator_vllm: Qwen/Qwen2.5-7B-Instruct",
            ]
        ),
        encoding="utf-8",
    )
    single_dir = tmp_path / "artifacts/classifier/single"
    single_dir.mkdir(parents=True)

    monkeypatch.setattr(runtime.sys, "platform", "linux")

    settings = load_runtime_settings(env={}, cwd=tmp_path)

    assert settings.query_classifier_dir == single_dir
    assert settings.llm_platform == "rocm"
    assert settings.resolve_llm_model() == "Qwen/Qwen2.5-7B-Instruct"
    assert settings.normalize_embeddings is False
    assert settings.embedding_backend == "sentence-transformers"


def test_runtime_settings_use_vllm_model_for_rocm_default(tmp_path):
    from darwin_rag_exp2.api.runtime import load_runtime_settings

    config_path = tmp_path / "configs" / "experiment.default.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  normalize_embeddings: false",
                "models:",
                "  embedder: BAAI/bge-m3",
                "  generator: mlx-community/Qwen3-8B-4bit",
                "  generator_vllm: Qwen/Qwen2.5-7B-Instruct",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(
        env={"DARWIN_EXP2_LLM_PLATFORM": "ROCm"},
        cwd=tmp_path,
    )

    assert settings.resolve_llm_model() == "Qwen/Qwen2.5-7B-Instruct"


def test_runtime_settings_use_qwen_thinking_sampling_defaults(tmp_path):
    from darwin_rag_exp2.api.runtime import load_runtime_settings

    config_path = tmp_path / "configs" / "experiment.default.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  normalize_embeddings: false",
                "models:",
                "  embedder: BAAI/bge-m3",
                "  generator_vllm: Qwen/Qwen3-4B",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(
        env={
            "DARWIN_EXP2_LLM_PLATFORM": "ROCm",
            "DARWIN_EXP2_LLM_THINKING_MODE": "think",
        },
        cwd=tmp_path,
    )

    assert settings.llm_temperature == 0.6
    assert settings.llm_top_p == 0.95
    assert settings.llm_top_k == 20
    assert settings.llm_min_p == 0.0


def test_runtime_settings_reject_mlx_generator_for_vllm_platform(tmp_path):
    from darwin_rag_exp2.api.runtime import load_runtime_settings

    config_path = tmp_path / "configs" / "experiment.default.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  normalize_embeddings: false",
                "models:",
                "  embedder: BAAI/bge-m3",
                "  generator: mlx-community/Qwen3-8B-4bit",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(
        env={"DARWIN_EXP2_LLM_PLATFORM": "ROCm"},
        cwd=tmp_path,
    )

    with pytest.raises(ValueError, match="cannot use the MLX generator model"):
        settings.resolve_llm_model()


def test_runtime_builds_platform_specific_generators(tmp_path, monkeypatch):
    from darwin_rag_exp2.api import runtime
    from darwin_rag_exp2.api.runtime import ApiRuntimeSettings

    built = []

    class FakeMlxGenerator:
        def __init__(
            self,
            model_name,
            *,
            max_tokens,
            temperature,
            thinking_mode,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
        ):
            built.append(
                (
                    "mlx",
                    model_name,
                    max_tokens,
                    temperature,
                    thinking_mode,
                    top_p,
                    top_k,
                    min_p,
                    repetition_penalty,
                    presence_penalty,
                )
            )

    class FakeVllmGenerator:
        def __init__(
            self,
            model_name,
            *,
            max_tokens,
            temperature,
            tokenizer,
            hf_config_path,
            max_model_len,
            gpu_memory_utilization,
            enforce_eager,
            thinking_mode,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
        ):
            built.append(
                (
                    "vllm",
                    model_name,
                    max_tokens,
                    temperature,
                    tokenizer,
                    hf_config_path,
                    max_model_len,
                    gpu_memory_utilization,
                    enforce_eager,
                    thinking_mode,
                    top_p,
                    top_k,
                    min_p,
                    repetition_penalty,
                    presence_penalty,
                )
            )

    def settings_for(platform):
        return ApiRuntimeSettings(
            config_path=tmp_path / "config.yaml",
            indexes_dir=tmp_path / "indexes",
            chunks_path=tmp_path / "chunks.parquet",
            settings_path=tmp_path / "settings.yaml",
            query_classifier_dir=tmp_path / "classifier",
            embedding_backend="hash",
            embedding_model="unused",
            embedding_device=None,
            normalize_embeddings=True,
            classifier_device="auto",
            llm_platform=platform,
            llm_model="local/model",
            llm_tokenizer="local/tokenizer",
            llm_hf_config_path="local/config",
            llm_max_model_len=2048,
            llm_gpu_memory_utilization=0.85,
            llm_enforce_eager=True,
            llm_max_tokens=33,
            llm_temperature=0.4,
            llm_thinking_mode="think",
            llm_top_p=0.95,
            llm_top_k=20,
            llm_min_p=0.0,
            llm_repetition_penalty=1.05,
            llm_presence_penalty=0.2,
        )

    monkeypatch.setattr(runtime, "MlxLmGenerator", FakeMlxGenerator)
    monkeypatch.setattr(runtime, "VllmGenerator", FakeVllmGenerator)
    runtime._build_generator(settings_for("mlx"))
    runtime._build_generator(settings_for("rocm"))
    runtime._build_generator(settings_for("cuda"))

    assert built == [
        ("mlx", "local/model", 33, 0.4, "think", 0.95, 20, 0.0, 1.05, 0.2),
        (
            "vllm",
            "local/model",
            33,
            0.4,
            "local/tokenizer",
            "local/config",
            2048,
            0.85,
            True,
            "think",
            0.95,
            20,
            0.0,
            1.05,
            0.2,
        ),
        (
            "vllm",
            "local/model",
            33,
            0.4,
            "local/tokenizer",
            "local/config",
            2048,
            0.85,
            True,
            "think",
            0.95,
            20,
            0.0,
            1.05,
            0.2,
        ),
    ]


def test_runtime_warm_starts_vllm_before_retrieval_components(tmp_path, monkeypatch):
    from darwin_rag_exp2.api import runtime
    from darwin_rag_exp2.api.runtime import ApiRuntimeSettings

    events = []

    class FakeGenerator:
        def warm_start(self):
            events.append("warm_start")

    class FakeEmbedding:
        def __init__(self):
            events.append("embedding")

    class FakeClassifier:
        def __init__(self, *args, **kwargs):
            events.append("classifier")

    class FakeSearchBackend:
        def __init__(self, *args, **kwargs):
            events.append("search_backend")

    monkeypatch.setattr(runtime, "_build_generator", lambda settings: FakeGenerator())
    monkeypatch.setattr(runtime, "HashEmbeddingModel", FakeEmbedding)
    monkeypatch.setattr(runtime, "FinalQueryClassifier", FakeClassifier)
    monkeypatch.setattr(runtime, "FaissSearchBackend", FakeSearchBackend)
    monkeypatch.setattr(runtime, "load_primary_run_settings", lambda path: object())
    monkeypatch.setattr(
        runtime.ChunkStore,
        "from_parquet",
        classmethod(lambda cls, path: object()),
    )

    settings = ApiRuntimeSettings(
        config_path=tmp_path / "config.yaml",
        indexes_dir=tmp_path / "indexes",
        chunks_path=tmp_path / "chunks.parquet",
        settings_path=tmp_path / "settings.yaml",
        query_classifier_dir=tmp_path / "classifier",
        embedding_backend="hash",
        embedding_model="unused",
        embedding_device=None,
        normalize_embeddings=True,
        classifier_device="auto",
        llm_platform="rocm",
        llm_model="local/model",
        llm_tokenizer=None,
        llm_hf_config_path=None,
        llm_max_model_len=None,
        llm_gpu_memory_utilization=None,
        llm_enforce_eager=None,
        llm_max_tokens=33,
        llm_temperature=0.4,
        llm_thinking_mode="no_think",
    )

    runtime._build_service(settings)

    assert events[:3] == ["warm_start", "embedding", "classifier"]


def test_runtime_passes_embedding_device_to_sentence_transformer(tmp_path, monkeypatch):
    from darwin_rag_exp2.api import runtime
    from darwin_rag_exp2.api.runtime import ApiRuntimeSettings

    events = []

    class FakeGenerator:
        pass

    class FakeEmbedding:
        def __init__(self, model_name, *, device):
            events.append(("embedding", model_name, device))

    class FakeClassifier:
        def __init__(self, *args, **kwargs):
            events.append(("classifier", kwargs["device"]))

    monkeypatch.setattr(runtime, "_build_generator", lambda settings: FakeGenerator())
    monkeypatch.setattr(runtime, "SentenceTransformerEmbeddingModel", FakeEmbedding)
    monkeypatch.setattr(runtime, "FinalQueryClassifier", FakeClassifier)
    monkeypatch.setattr(runtime, "FaissSearchBackend", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "load_primary_run_settings", lambda path: object())
    monkeypatch.setattr(
        runtime.ChunkStore,
        "from_parquet",
        classmethod(lambda cls, path: object()),
    )

    settings = ApiRuntimeSettings(
        config_path=tmp_path / "config.yaml",
        indexes_dir=tmp_path / "indexes",
        chunks_path=tmp_path / "chunks.parquet",
        settings_path=tmp_path / "settings.yaml",
        query_classifier_dir=tmp_path / "classifier",
        embedding_backend="sentence-transformers",
        embedding_model="BAAI/bge-m3",
        embedding_device="cuda:1",
        normalize_embeddings=True,
        classifier_device="cuda:1",
        llm_platform="rocm",
        llm_model="local/model",
        llm_tokenizer=None,
        llm_hf_config_path=None,
        llm_max_model_len=None,
        llm_gpu_memory_utilization=None,
        llm_enforce_eager=None,
        llm_max_tokens=33,
        llm_temperature=0.4,
        llm_thinking_mode="no_think",
    )

    runtime._build_service(settings)

    assert events[:2] == [
        ("embedding", "BAAI/bge-m3", "cuda:1"),
        ("classifier", "cuda:1"),
    ]


def test_create_app_with_injected_service_does_not_build_runtime_service(monkeypatch):
    from darwin_rag_exp2.api import app as app_module

    class FakeService:
        def answer(self, query):
            return "ok"

    def fail_build_service():
        raise AssertionError("runtime service must not be built when service is injected")

    monkeypatch.setattr(app_module, "build_default_message_service", fail_build_service)

    app = app_module.create_app(service=FakeService())

    assert app.state.message_service is not None
