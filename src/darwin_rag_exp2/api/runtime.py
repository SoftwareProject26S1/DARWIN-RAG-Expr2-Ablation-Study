"""Runtime wiring for the DARWIN-RAG Exp2 API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys

import yaml

from darwin_rag_exp2.indexing.embeddings import (
    HashEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from darwin_rag_exp2.retrieval.faiss_backend import FaissSearchBackend
from darwin_rag_exp2.retrieval.query_classifier import FinalQueryClassifier
from darwin_rag_exp2.retrieval.settings import load_primary_run_settings

from .generator import MlxLmGenerator, VllmGenerator
from .service import AnswerGenerator, ChunkStore, MessageService


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiRuntimeSettings:
    """Filesystem paths and model settings used to build the default service."""

    config_path: Path
    indexes_dir: Path
    chunks_path: Path
    settings_path: Path
    query_classifier_dir: Path
    embedding_backend: str
    embedding_model: str
    normalize_embeddings: bool
    classifier_device: str
    llm_platform: str
    llm_model: str | None
    llm_tokenizer: str | None
    llm_hf_config_path: str | None
    llm_max_model_len: int | None
    llm_gpu_memory_utilization: float | None
    llm_enforce_eager: bool | None
    llm_max_tokens: int
    llm_temperature: float

    def build_service(self) -> MessageService:
        """Build the default service using this runtime configuration."""

        return _build_service(self)

    def resolve_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        config = _load_yaml(self.config_path)
        models = config.get("models") if isinstance(config, dict) else None
        return _resolve_config_llm_model(models, self.llm_platform)


def load_runtime_settings(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> ApiRuntimeSettings:
    """Load API runtime settings from environment variables and config defaults."""

    active_env = os.environ if env is None else env
    root = Path.cwd() if cwd is None else cwd
    config_path = _resolve_path(
        active_env.get("DARWIN_EXP2_CONFIG", "configs/experiment.default.yaml"),
        root,
    )
    config = _load_yaml(config_path)
    retrieval = config.get("retrieval") if isinstance(config, dict) else {}
    models = config.get("models") if isinstance(config, dict) else {}
    embedding_model = active_env.get("DARWIN_EXP2_EMBEDDING_MODEL")
    if embedding_model is None and isinstance(models, dict):
        embedding_model = models.get("embedder")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("config must define models.embedder or DARWIN_EXP2_EMBEDDING_MODEL")

    normalize = True
    if isinstance(retrieval, dict) and "normalize_embeddings" in retrieval:
        normalize = bool(retrieval["normalize_embeddings"])

    return ApiRuntimeSettings(
        config_path=config_path,
        indexes_dir=_resolve_path(
            active_env.get("DARWIN_EXP2_INDEXES_DIR", "artifacts/indexes"),
            root,
        ),
        chunks_path=_resolve_path(
            active_env.get("DARWIN_EXP2_CHUNKS_PATH", "artifacts/chunks/chunks.parquet"),
            root,
        ),
        settings_path=_resolve_path(
            active_env.get("DARWIN_EXP2_SETTINGS", "artifacts/settings/primary/frozen.yaml"),
            root,
        ),
        query_classifier_dir=_resolve_classifier_dir(active_env, root),
        embedding_backend=active_env.get(
            "DARWIN_EXP2_EMBEDDING_BACKEND",
            "sentence-transformers",
        ),
        embedding_model=embedding_model,
        normalize_embeddings=normalize,
        classifier_device=active_env.get("DARWIN_EXP2_CLASSIFIER_DEVICE", "auto"),
        llm_platform=normalize_llm_platform(
            active_env.get("DARWIN_EXP2_LLM_PLATFORM"),
        ),
        llm_model=active_env.get("DARWIN_EXP2_LLM_MODEL") or None,
        llm_tokenizer=active_env.get("DARWIN_EXP2_LLM_TOKENIZER") or None,
        llm_hf_config_path=active_env.get("DARWIN_EXP2_LLM_HF_CONFIG_PATH") or None,
        llm_max_model_len=_parse_optional_int(
            active_env.get("DARWIN_EXP2_LLM_MAX_MODEL_LEN"),
            "DARWIN_EXP2_LLM_MAX_MODEL_LEN",
        ),
        llm_gpu_memory_utilization=_parse_optional_float(
            active_env.get("DARWIN_EXP2_LLM_GPU_MEMORY_UTILIZATION"),
            "DARWIN_EXP2_LLM_GPU_MEMORY_UTILIZATION",
        ),
        llm_enforce_eager=_parse_optional_bool(
            active_env.get("DARWIN_EXP2_LLM_ENFORCE_EAGER"),
            "DARWIN_EXP2_LLM_ENFORCE_EAGER",
        ),
        llm_max_tokens=int(active_env.get("DARWIN_EXP2_LLM_MAX_TOKENS", "512")),
        llm_temperature=float(active_env.get("DARWIN_EXP2_LLM_TEMPERATURE", "0.0")),
    )


def build_default_message_service() -> MessageService:
    """Build the default service using runtime env and frozen artifacts."""

    logger.info("loading API runtime settings cwd=%s", Path.cwd())
    try:
        settings = load_runtime_settings()
    except Exception:
        logger.exception("API runtime settings load failed")
        raise
    logger.info(
        "API runtime settings loaded config=%s indexes=%s chunks=%s settings=%s "
        "classifier=%s embedding_backend=%s classifier_device=%s llm_platform=%s",
        settings.config_path,
        settings.indexes_dir,
        settings.chunks_path,
        settings.settings_path,
        settings.query_classifier_dir,
        settings.embedding_backend,
        settings.classifier_device,
        settings.llm_platform,
    )
    return settings.build_service()


def _build_service(settings: ApiRuntimeSettings) -> MessageService:
    stage = "generator"
    try:
        logger.info("message service stage=%s started", stage)
        generator = _build_generator(settings)
        logger.info("message service stage=%s completed", stage)

        stage = "generator_warm_start"
        logger.info("message service stage=%s started", stage)
        _warm_start_generator(settings, generator)
        logger.info("message service stage=%s completed", stage)

        stage = "embedding_model"
        logger.info(
            "message service stage=%s started backend=%s model=%s",
            stage,
            settings.embedding_backend,
            settings.embedding_model,
        )
        if settings.embedding_backend == "sentence-transformers":
            embedding_model = SentenceTransformerEmbeddingModel(settings.embedding_model)
        elif settings.embedding_backend == "hash":
            embedding_model = HashEmbeddingModel()
        else:
            raise ValueError(
                "DARWIN_EXP2_EMBEDDING_BACKEND must be one of: sentence-transformers, hash"
            )
        logger.info("message service stage=%s completed", stage)

        stage = "query_classifier"
        logger.info(
            "message service stage=%s started path=%s device=%s",
            stage,
            settings.query_classifier_dir,
            settings.classifier_device,
        )
        classifier = FinalQueryClassifier(
            settings.query_classifier_dir,
            device=settings.classifier_device,
        )
        logger.info("message service stage=%s completed", stage)

        stage = "faiss_backend"
        logger.info("message service stage=%s started indexes=%s", stage, settings.indexes_dir)
        search_backend = FaissSearchBackend(settings.indexes_dir)
        logger.info("message service stage=%s completed", stage)

        stage = "primary_settings"
        logger.info("message service stage=%s started path=%s", stage, settings.settings_path)
        primary_settings = load_primary_run_settings(settings.settings_path)
        logger.info("message service stage=%s completed", stage)

        stage = "chunk_store"
        logger.info("message service stage=%s started path=%s", stage, settings.chunks_path)
        chunk_store = ChunkStore.from_parquet(settings.chunks_path)
        logger.info("message service stage=%s completed", stage)

        return MessageService(
            embedding_model=embedding_model,
            classifier=classifier,
            search_backend=search_backend,
            settings=primary_settings,
            chunk_store=chunk_store,
            generator=generator,
            normalize_embeddings=settings.normalize_embeddings,
        )
    except Exception:
        logger.exception("message service construction failed stage=%s", stage)
        raise


def _build_generator(settings: ApiRuntimeSettings) -> AnswerGenerator:
    model_name = settings.resolve_llm_model()
    logger.info(
        "building answer generator platform=%s model=%s max_tokens=%s temperature=%s",
        settings.llm_platform,
        model_name,
        settings.llm_max_tokens,
        settings.llm_temperature,
    )
    if settings.llm_platform == "mlx":
        return MlxLmGenerator(
            model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    if settings.llm_platform in {"rocm", "cuda"}:
        return VllmGenerator(
            model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            tokenizer=settings.llm_tokenizer,
            hf_config_path=settings.llm_hf_config_path,
            max_model_len=settings.llm_max_model_len,
            gpu_memory_utilization=settings.llm_gpu_memory_utilization,
            enforce_eager=settings.llm_enforce_eager,
        )
    raise ValueError("DARWIN_EXP2_LLM_PLATFORM must be one of: MLX, ROCm, CUDA")


def normalize_llm_platform(value: str | None) -> str:
    """Normalize user-facing generator platform names to runtime keys."""

    if value is None or not value.strip():
        return "mlx" if sys.platform == "darwin" else "rocm"
    normalized = (
        value.strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    if normalized in {"mlx", "macos", "darwin"}:
        return "mlx"
    if normalized in {"rocm", "amd", "linuxrocm"}:
        return "rocm"
    if normalized in {"cuda", "nvidia", "rocm(cuda)", "rocmcuda", "vllm"}:
        return "cuda"
    raise ValueError("supported platforms: MLX, ROCm, CUDA")


def _warm_start_generator(
    settings: ApiRuntimeSettings,
    generator: AnswerGenerator,
) -> None:
    if settings.llm_platform not in {"rocm", "cuda"}:
        return
    warm_start = getattr(generator, "warm_start", None)
    if callable(warm_start):
        warm_start()


def _resolve_config_llm_model(models: object, platform: str) -> str:
    if not isinstance(models, dict):
        raise ValueError("config must define a models mapping or DARWIN_EXP2_LLM_MODEL")

    if platform == "mlx":
        model_name = _first_model_name(models, ("generator_mlx", "generator"))
        if model_name is None:
            raise ValueError(
                "config must define models.generator or DARWIN_EXP2_LLM_MODEL"
            )
        return model_name

    model_name = _first_model_name(
        models,
        (f"generator_{platform}", "generator_vllm"),
    )
    if model_name is not None:
        return model_name

    fallback = _first_model_name(models, ("generator",))
    if fallback is None:
        raise ValueError(
            "vLLM platforms require DARWIN_EXP2_LLM_MODEL, "
            f"models.generator_{platform}, or models.generator_vllm"
        )
    if _looks_like_mlx_model(fallback):
        raise ValueError(
            "vLLM platforms cannot use the MLX generator model "
            f"{fallback!r}; set DARWIN_EXP2_LLM_MODEL or define "
            f"models.generator_{platform}/models.generator_vllm"
        )
    return fallback


def _first_model_name(models: Mapping[object, object], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = models.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_like_mlx_model(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized.startswith("mlx-community/") or "/mlx-" in normalized


def _resolve_classifier_dir(env: Mapping[str, str], root: Path) -> Path:
    override = env.get("DARWIN_EXP2_QUERY_CLASSIFIER_DIR")
    if override:
        return _resolve_path(override, root)
    candidates = [
        root / "artifacts/classifier/final",
        root / "artifacts/classifier/single",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _parse_optional_int(value: str | None, name: str) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _parse_optional_float(value: str | None, name: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a float") from error


def _parse_optional_bool(value: str | None, name: str) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return dict(payload)
