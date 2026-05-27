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
