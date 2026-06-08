"""Runtime wiring for the DARWIN-RAG Exp2 API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

import yaml

from darwin_rag_exp2.indexing.embeddings import (
    HashEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from darwin_rag_exp2.retrieval.faiss_backend import FaissSearchBackend
from darwin_rag_exp2.retrieval.query_classifier import FinalQueryClassifier
from darwin_rag_exp2.retrieval.settings import load_primary_run_settings

from .generator import MlxLmGenerator
from .service import ChunkStore, MessageService


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
    llm_model: str | None
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
        model_name = models.get("generator") if isinstance(models, dict) else None
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("config must define models.generator or DARWIN_EXP2_LLM_MODEL")
        return model_name


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
        llm_model=active_env.get("DARWIN_EXP2_LLM_MODEL") or None,
        llm_max_tokens=int(active_env.get("DARWIN_EXP2_LLM_MAX_TOKENS", "512")),
        llm_temperature=float(active_env.get("DARWIN_EXP2_LLM_TEMPERATURE", "0.0")),
    )


def build_default_message_service() -> MessageService:
    """Build the default service using runtime env and frozen artifacts."""

    settings = load_runtime_settings()
    return settings.build_service()


def _build_service(settings: ApiRuntimeSettings) -> MessageService:
    if settings.embedding_backend == "sentence-transformers":
        embedding_model = SentenceTransformerEmbeddingModel(settings.embedding_model)
    elif settings.embedding_backend == "hash":
        embedding_model = HashEmbeddingModel()
    else:
        raise ValueError(
            "DARWIN_EXP2_EMBEDDING_BACKEND must be one of: sentence-transformers, hash"
        )
    return MessageService(
        embedding_model=embedding_model,
        classifier=FinalQueryClassifier(
            settings.query_classifier_dir,
            device=settings.classifier_device,
        ),
        search_backend=FaissSearchBackend(settings.indexes_dir),
        settings=load_primary_run_settings(settings.settings_path),
        chunk_store=ChunkStore.from_parquet(settings.chunks_path),
        generator=MlxLmGenerator(
            settings.resolve_llm_model(),
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        ),
        normalize_embeddings=settings.normalize_embeddings,
    )


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


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return dict(payload)
