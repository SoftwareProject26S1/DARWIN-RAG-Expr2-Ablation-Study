"""Local LLM generation adapters for API responses."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


LoadFn = Callable[[str], tuple[Any, Any]]
GenerateFn = Callable[..., str]
MlxSamplerFactory = Callable[..., Any]
MlxLogitsProcessorsFactory = Callable[..., list[Any]]
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
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        loader: LoadFn | None = None,
        generate_fn: GenerateFn | None = None,
        sampler_factory: MlxSamplerFactory | None = None,
        logits_processors_factory: MlxLogitsProcessorsFactory | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_mode = thinking_mode
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self._loader = loader
        self._generate_fn = generate_fn
        self._sampler_factory = sampler_factory
        self._logits_processors_factory = logits_processors_factory
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str) -> str:
        """Generate an answer, loading the MLX model on first use."""

        try:
            model, tokenizer = self._load_model()
            generate_fn = self._resolve_generate_fn()
            generation_kwargs: dict[str, Any] = {
                "prompt": apply_qwen_thinking_mode(prompt, self.thinking_mode),
                "max_tokens": self.max_tokens,
            }
            sampler = self._build_sampler()
            if sampler is not None:
                generation_kwargs["sampler"] = sampler
            logits_processors = self._build_logits_processors()
            if logits_processors:
                generation_kwargs["logits_processors"] = logits_processors
            answer = generate_fn(model, tokenizer, **generation_kwargs)
        except Exception:
            logger.exception("MLX generation failed model=%s", self.model_name)
            raise
        return clean_generated_answer(answer)

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

    def _build_sampler(self) -> Any | None:
        if (
            self.temperature == 0.0
            and self.top_p is None
            and self.top_k is None
            and self.min_p is None
        ):
            return None
        sampler_factory = self._resolve_sampler_factory()
        kwargs: dict[str, Any] = {"temp": self.temperature}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.min_p is not None:
            kwargs["min_p"] = self.min_p
        return sampler_factory(**kwargs)

    def _build_logits_processors(self) -> list[Any] | None:
        if self.repetition_penalty is None and self.presence_penalty is None:
            return None
        logits_processors_factory = self._resolve_logits_processors_factory()
        kwargs: dict[str, Any] = {}
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        return logits_processors_factory(**kwargs)

    def _resolve_sampler_factory(self) -> MlxSamplerFactory:
        if self._sampler_factory is not None:
            return self._sampler_factory
        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mlx-lm is required for local LLM generation; "
                "run with the api dependency group"
            ) from error
        return make_sampler

    def _resolve_logits_processors_factory(self) -> MlxLogitsProcessorsFactory:
        if self._logits_processors_factory is not None:
            return self._logits_processors_factory
        try:
            from mlx_lm.sample_utils import make_logits_processors
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mlx-lm is required for local LLM generation; "
                "run with the api dependency group"
            ) from error
        return make_logits_processors


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
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
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
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
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
        return clean_generated_answer(outputs[0].outputs[0].text)

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
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.min_p is not None:
            kwargs["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        kwargs = _filter_supported_kwargs(sampling_params_factory, kwargs)
        return sampling_params_factory(**kwargs)

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


def clean_generated_answer(answer: object) -> str:
    """Normalize model output to the final answer surface returned by the API."""

    text = str(answer).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    marker = "최종 답변:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    text = _remove_repeated_separator_sections(text)
    text = _collapse_adjacent_repeated_sentences(text)
    return _collapse_adjacent_repeated_words(text)


def _remove_repeated_separator_sections(text: str) -> str:
    sections = [section.strip() for section in re.split(r"\s*---\s*", text)]
    kept: list[str] = []
    seen: set[str] = set()
    for section in sections:
        normalized = re.sub(r"\s+", " ", section).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        if normalized.startswith("(최종 답변은") and "추측" in normalized:
            continue
        seen.add(normalized)
        kept.append(section)
    return " --- ".join(kept).strip()


def _collapse_adjacent_repeated_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    if len(sentences) <= 1:
        return text.strip()

    kept: list[str] = []
    previous = ""
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized:
            continue
        if normalized == previous:
            continue
        kept.append(sentence.strip())
        previous = normalized
    return " ".join(kept).strip()


def _collapse_adjacent_repeated_words(text: str) -> str:
    tokens = text.strip().split()
    if len(tokens) <= 1:
        return text.strip()

    kept: list[str] = []
    previous = ""
    for token in tokens:
        normalized = token.strip()
        comparable = normalized.strip(".,!?;:()[]{}\"'")
        if comparable and comparable == previous:
            continue
        kept.append(normalized)
        previous = comparable
    return " ".join(kept).strip()


def _filter_supported_kwargs(
    callable_obj: Callable[..., Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    supported = set(signature.parameters)
    dropped = sorted(key for key in kwargs if key not in supported)
    if dropped:
        logger.warning(
            "dropping unsupported vLLM SamplingParams options keys=%s",
            dropped,
        )
    return {key: value for key, value in kwargs.items() if key in supported}
