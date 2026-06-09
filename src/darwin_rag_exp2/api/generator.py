"""Local LLM generation adapters for API responses."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any


logger = logging.getLogger(__name__)


LoadFn = Callable[[str], tuple[Any, Any]]
GenerateFn = Callable[..., str]
VllmLlmFactory = Callable[..., Any]
VllmSamplingParamsFactory = Callable[..., Any]
QWEN_THINKING_MODES = {"auto", "think", "no_think"}


class MlxLmGenerator:
    """Lazy MLX-LM wrapper with a small generate(prompt) surface."""

    def __init__(
        self,
        model_name: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        thinking_mode: str = "auto",
        loader: LoadFn | None = None,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_mode = thinking_mode
        self._loader = loader
        self._generate_fn = generate_fn
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str) -> str:
        """Generate an answer, loading the MLX model on first use."""

        try:
            model, tokenizer = self._load_model()
            generate_fn = self._resolve_generate_fn()
            answer = generate_fn(
                model,
                tokenizer,
                prompt=apply_qwen_thinking_mode(prompt, self.thinking_mode),
                max_tokens=self.max_tokens,
                temp=self.temperature,
            )
        except Exception:
            logger.exception("MLX generation failed model=%s", self.model_name)
            raise
        return str(answer).strip()

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            loader = self._resolve_loader()
            logger.info("loading MLX model model=%s", self.model_name)
            try:
                self._model, self._tokenizer = loader(self.model_name)
            except Exception:
                logger.exception("MLX model load failed model=%s", self.model_name)
                raise
            logger.info("MLX model load completed model=%s", self.model_name)
        return self._model, self._tokenizer

    def _resolve_loader(self) -> LoadFn:
        if self._loader is not None:
            return self._loader
        try:
            from mlx_lm import load
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mlx-lm is required for local LLM generation; "
                "run with the api dependency group"
            ) from error
        return load

    def _resolve_generate_fn(self) -> GenerateFn:
        if self._generate_fn is not None:
            return self._generate_fn
        try:
            from mlx_lm import generate
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mlx-lm is required for local LLM generation; "
                "run with the api dependency group"
            ) from error
        return generate


class VllmGenerator:
    """Lazy vLLM wrapper with a small generate(prompt) surface."""

    def __init__(
        self,
        model_name: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        tokenizer: str | None = None,
        hf_config_path: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        enforce_eager: bool | None = None,
        thinking_mode: str = "auto",
        llm_factory: VllmLlmFactory | None = None,
        sampling_params_factory: VllmSamplingParamsFactory | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.hf_config_path = hf_config_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.thinking_mode = thinking_mode
        self._llm_factory = llm_factory
        self._sampling_params_factory = sampling_params_factory
        self._llm = None

    def generate(self, prompt: str) -> str:
        """Generate an answer, loading the vLLM engine on first use."""

        try:
            llm = self._load_llm()
            sampling_params = self._build_sampling_params()
            outputs = llm.generate(
                [apply_qwen_thinking_mode(prompt, self.thinking_mode)],
                sampling_params,
            )
        except Exception:
            logger.exception("vLLM generation failed model=%s", self.model_name)
            raise
        return str(outputs[0].outputs[0].text).strip()

    def warm_start(self) -> None:
        """Load the vLLM engine before request-time retrieval touches Torch/CUDA."""

        logger.info("vLLM warm start requested model=%s", self.model_name)
        self._load_llm()
        logger.info("vLLM warm start completed model=%s", self.model_name)

    def _load_llm(self) -> Any:
        if self._llm is None:
            llm_factory = self._resolve_llm_factory()
            kwargs: dict[str, Any] = {"model": self.model_name}
            if self.tokenizer is not None:
                kwargs["tokenizer"] = self.tokenizer
            if self.hf_config_path is not None:
                kwargs["hf_config_path"] = self.hf_config_path
            if self.max_model_len is not None:
                kwargs["max_model_len"] = self.max_model_len
            if self.gpu_memory_utilization is not None:
                kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
            if self.enforce_eager is not None:
                kwargs["enforce_eager"] = self.enforce_eager
            logger.info("initializing vLLM engine kwargs=%s", _safe_vllm_kwargs(kwargs))
            try:
                self._llm = llm_factory(**kwargs)
            except Exception:
                logger.exception(
                    "vLLM engine initialization failed kwargs=%s",
                    _safe_vllm_kwargs(kwargs),
                )
                raise
            logger.info("vLLM engine initialization completed model=%s", self.model_name)
        return self._llm

    def _build_sampling_params(self) -> Any:
        sampling_params_factory = self._resolve_sampling_params_factory()
        return sampling_params_factory(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _resolve_llm_factory(self) -> VllmLlmFactory:
        if self._llm_factory is not None:
            return self._llm_factory
        try:
            from vllm import LLM
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "vllm is required for ROCm/CUDA local LLM generation; "
                "install the Linux API dependencies or use --platform MLX on macOS"
            ) from error
        return LLM

    def _resolve_sampling_params_factory(self) -> VllmSamplingParamsFactory:
        if self._sampling_params_factory is not None:
            return self._sampling_params_factory
        try:
            from vllm import SamplingParams
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "vllm is required for ROCm/CUDA local LLM generation; "
                "install the Linux API dependencies or use --platform MLX on macOS"
            ) from error
        return SamplingParams


def _safe_vllm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key
        in {
            "model",
            "tokenizer",
            "hf_config_path",
            "max_model_len",
            "gpu_memory_utilization",
            "enforce_eager",
        }
    }


def apply_qwen_thinking_mode(prompt: str, thinking_mode: str) -> str:
    """Append a Qwen3 prompt-level thinking switch when requested."""

    if thinking_mode == "auto":
        return prompt
    if thinking_mode == "think":
        suffix = "/think"
    elif thinking_mode == "no_think":
        suffix = "/no_think"
    else:
        raise ValueError(
            "thinking_mode must be one of: "
            + ", ".join(sorted(QWEN_THINKING_MODES))
        )
    clean_prompt = prompt.rstrip()
    return f"{clean_prompt}\n\n{suffix}" if clean_prompt else suffix
