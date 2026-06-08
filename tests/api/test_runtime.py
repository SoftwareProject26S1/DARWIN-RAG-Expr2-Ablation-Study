from pathlib import Path


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
            "DARWIN_EXP2_LLM_PLATFORM": "ROCm",
            "DARWIN_EXP2_LLM_MODEL": "local/model",
            "DARWIN_EXP2_LLM_MAX_TOKENS": "128",
            "DARWIN_EXP2_LLM_TEMPERATURE": "0.1",
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
    assert settings.llm_max_tokens == 128
    assert settings.llm_temperature == 0.1
    assert settings.embedding_model == "config/embedder"
    assert settings.normalize_embeddings is True


def test_runtime_settings_default_classifier_falls_back_to_single(tmp_path):
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
    single_dir = tmp_path / "artifacts/classifier/single"
    single_dir.mkdir(parents=True)

    settings = load_runtime_settings(env={}, cwd=tmp_path)

    assert settings.query_classifier_dir == single_dir
    assert settings.resolve_llm_model() == "mlx-community/Qwen3-8B-4bit"
    assert settings.normalize_embeddings is False
    assert settings.embedding_backend == "sentence-transformers"


def test_runtime_builds_platform_specific_generators(tmp_path, monkeypatch):
    from darwin_rag_exp2.api import runtime
    from darwin_rag_exp2.api.runtime import ApiRuntimeSettings

    built = []

    class FakeMlxGenerator:
        def __init__(self, model_name, *, max_tokens, temperature):
            built.append(("mlx", model_name, max_tokens, temperature))

    class FakeVllmGenerator:
        def __init__(self, model_name, *, max_tokens, temperature):
            built.append(("vllm", model_name, max_tokens, temperature))

    def settings_for(platform):
        return ApiRuntimeSettings(
            config_path=tmp_path / "config.yaml",
            indexes_dir=tmp_path / "indexes",
            chunks_path=tmp_path / "chunks.parquet",
            settings_path=tmp_path / "settings.yaml",
            query_classifier_dir=tmp_path / "classifier",
            embedding_backend="hash",
            embedding_model="unused",
            normalize_embeddings=True,
            classifier_device="auto",
            llm_platform=platform,
            llm_model="local/model",
            llm_max_tokens=33,
            llm_temperature=0.4,
        )

    monkeypatch.setattr(runtime, "MlxLmGenerator", FakeMlxGenerator)
    monkeypatch.setattr(runtime, "VllmGenerator", FakeVllmGenerator)
    runtime._build_generator(settings_for("mlx"))
    runtime._build_generator(settings_for("rocm"))
    runtime._build_generator(settings_for("cuda"))

    assert built == [
        ("mlx", "local/model", 33, 0.4),
        ("vllm", "local/model", 33, 0.4),
        ("vllm", "local/model", 33, 0.4),
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
