# DARWIN-RAG Exp2

`Exp2` is the reproducible Python experiment workspace for the DARWIN-RAG
ablation study. It compares adaptive category weighting against fixed-weight
and category-free retrieval baselines over the exported Soongsil notice
corpus.

## Scope

- Input begins with the repository-owned notice export at
  `data/raw/scatch_notices.jsonl`.
- Crawling, OCR, and attachment text extraction are outside this experiment.
- Primary retrieval analysis uses normalized global score merge.
- Weighted RRF is deferred to an optional extension after the primary
  score-merge experiment is complete and frozen.

## Documents

- [Implementation Plan](docs/IMPLEMENTATION-PLAN.md)
- [Implementation Steps](docs/IMPLEMENTATION-STEPS.md)
- [Working Rules](docs/WORKING-RULES.md)

## Phase 1 Smoke Check

```bash
uv run pytest
uv run darwin-exp2
```

Model and indexing dependencies are introduced in the phase that uses them,
rather than installed into the initial documentation scaffold.

## API LLM Thinking Mode

The FastAPI message server can steer Qwen3 hybrid thinking mode with:

```bash
DARWIN_EXP2_LLM_THINKING_MODE=think uv run darwin-exp2 serve-api
```

Supported values:

- `think`: append `/think` to the augmented P-score RAG prompt so Qwen3 reasons
  before its final answer.
- `no_think`: append `/no_think`; this is the server default to preserve the
  experiment's final-answer-only comparison protocol.
- `auto`: do not append either switch and let the model/runtime default apply.

For Qwen3 thinking mode, consider overriding decoding as well, for example:

```bash
DARWIN_EXP2_LLM_MODEL=Qwen/Qwen3-4B-MLX-4bit \
DARWIN_EXP2_LLM_THINKING_MODE=think \
DARWIN_EXP2_LLM_TEMPERATURE=0.6 \
DARWIN_EXP2_LLM_MAX_TOKENS=2048 \
uv run darwin-exp2 serve-api --platform MLX
```
