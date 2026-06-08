"""Local LLM generation adapters for API responses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


LoadFn = Callable[[str], tuple[Any, Any]]
GenerateFn = Callable[..., str]
VllmLlmFactory = Callable[..., Any]
VllmSamplingParamsFactory = Callable[..., Any]


class MlxLmGenerator:
    """Lazy MLX-LM wrapper with a small generate(prompt) surface."""

    def __init__(
        self,
        model_name: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        loader: LoadFn | None = None,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._loader = loader
        self._generate_fn = generate_fn
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str) -> str:
        """Generate an answer, loading the MLX model on first use."""

        model, tokenizer = self._load_model()
        generate_fn = self._resolve_generate_fn()
        answer = generate_fn(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )
        return str(answer).strip()

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            loader = self._resolve_loader()
            self._model, self._tokenizer = loader(self.model_name)
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
        llm_factory: VllmLlmFactory | None = None,
        sampling_params_factory: VllmSamplingParamsFactory | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llm_factory = llm_factory
        self._sampling_params_factory = sampling_params_factory
        self._llm = None

    def generate(self, prompt: str) -> str:
        """Generate an answer, loading the vLLM engine on first use."""

        llm = self._load_llm()
        sampling_params = self._build_sampling_params()
        outputs = llm.generate([prompt], sampling_params)
        return str(outputs[0].outputs[0].text).strip()

    def _load_llm(self) -> Any:
        if self._llm is None:
            llm_factory = self._resolve_llm_factory()
            self._llm = llm_factory(model=self.model_name)
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
