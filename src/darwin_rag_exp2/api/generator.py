"""Local MLX-LM generation adapter for API responses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


LoadFn = Callable[[str], tuple[Any, Any]]
GenerateFn = Callable[..., str]


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
