# DARWIN-RAG Exp2 Implementation Steps

> **For agentic workers:** Implement one phase at a time, run the listed verification, and do not begin the next phase until artifacts from the current phase are frozen.

**Goal:** Build the reproducible Exp2 ablation pipeline for normalized global score-merge DARWIN-RAG comparisons.

**Architecture:** Offline data/model/index artifact builders feed a frozen query experiment runner. Required reporting evaluates adaptive score merge directly; Weighted RRF may be added only as an optional extension after the primary experiment is frozen.

**Tech Stack:** Python 3.12, uv, PyTorch/MPS, transformers, sentence-transformers, FAISS, MLX-LM, NumPy, SciPy, scikit-learn, PyArrow, Typer, PyYAML, Optuna, pytest.

---

## Execution Rules

- Follow `docs/WORKING-RULES.md`: conduct all ablation study work within this
  standalone `Exp2` repository; `Exp1` is a separate project, and the parent
  repository submodule gitlink is updated only under the synchronization rule
  in that document.
- Start each phase or isolated phase task on an `Exp2` branch using the
  canonical branch convention in `docs/WORKING-RULES.md` (for example,
  `feat/phase2-data-audit`).
- Use the canonical commit-message convention in `docs/WORKING-RULES.md`.
- Retain raw input `data/raw/scatch_notices.jsonl` unchanged.
- Add runtime dependency groups when their phase begins; Phase 1 installs only
  the package and test runner.
- Write a test that fails for the phase's new behavior before adding that
  behavior.
- Store generated data, model weights, indexes, and run outputs under ignored
  `artifacts/` or `runs/`, with versioned manifests.
- No test-query result may be used to adjust thresholds, parameters, prompts,
  or model choices.

## Phase Overview

| Phase | Work | Completion Evidence |
|---|---|---|
| 1 | Standalone repository, `Exp2` scaffold, official documents | Independent origin configured, package/docs present, smoke test passes |
| 2 | `scatch_notices.jsonl` importer and dataset audit | Audit JSON/Markdown reproduces Phase 1 baseline counts |
| 3 | Category admission and quality filtering | Primary 8-category corpus and exclusion report |
| 4 | Improved chunker and artifact schema | JSONL/Parquet chunks and token-cap tests |
| 5 | Classifier `single` smoke pipeline | Small training/calibration/ECE artifacts |
| 6 | Classifier `crossfit` official pipeline | Leakage-free OOF calibrated probabilities |
| 7 | BGE-M3 embeddings and FAISS indexes | Hashed frozen index manifest |
| 8 | Query candidate pool and annotation contract | Valid dev 80/test 240 query files |
| 9 | `B0`, `B1`, `B2-score`, `P-score` runners | Primary score-merge results |
| 10 | MLX Qwen generation and metrics | Cached primary-variant answers using identical Top-5 contexts |
| 11 | Statistical, latency, and paper report export | Primary tables, confidence intervals, and figures |
| 12 | Primary artifact freeze and reproducibility closeout | Final manifest and frozen experiment bundle |

After Phase 12, `B2-wrrf` and `P-wrrf` may be implemented as an optional
supplemental extension. They are not completion criteria for the primary
ablation study.

## Phase 1: Repository, Scaffold, And Protocol Documents

**Inputs**

- Git repository `https://github.com/SoftwareProject26S1/DARWIN-RAG-Expr2-Ablation-Study.git`
- Approved design decisions in this implementation plan
- Repository-owned read-only raw export at `data/raw/scatch_notices.jsonl`

**Files**

- Create: `.gitignore`
- Create: `.python-version`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `configs/experiment.default.yaml`
- Create: `docs/IMPLEMENTATION-PLAN.md`
- Create: `docs/IMPLEMENTATION-STEPS.md`
- Create: `docs/WORKING-RULES.md`
- Create: `src/darwin_rag_exp2/__init__.py`
- Create: `src/darwin_rag_exp2/cli.py`
- Create: `tests/test_cli_smoke.py`

**Actions**

1. Clone this standalone repository and, for a new Phase 1 run, create branch
   `feat/phase1-scaffold` following the canonical branch convention in
   `docs/WORKING-RULES.md`.
2. Create a Python 3.12 `uv` package with a `darwin-exp2` console script and
   pytest development dependency.
3. Add a failing smoke test for the CLI package import and phase readiness
   output; then add the minimal CLI implementation that returns `0`.
4. Add the official protocol document, this implementation checklist, and the
   project working rules.
5. Add the default YAML values that fix primary categories, token budgets,
   model IDs, candidate/report depths, query sizes, and random seed.

**Verification**

```bash
uv lock
uv run pytest
uv run darwin-exp2
git diff --name-only origin/main...HEAD
git remote get-url origin
```

Expected evidence:

- `uv run pytest` reports one passing smoke test.
- `uv run darwin-exp2` prints `DARWIN-RAG Exp2: Phase 1 scaffold ready.`
- Changes listed against `origin/main` belong only to this standalone project.
- `git remote get-url origin` prints the Exp2 ablation study repository URL.

## Phase 2: Raw Notice Importer And Audit

**Purpose**

Load the exported notices without modifying them and materialize an auditable
baseline that guards later category and chunk decisions.

**Files**

- Modify: `pyproject.toml` to add `pydantic`, `typer`, and `orjson`
- Create: `src/darwin_rag_exp2/data/schema.py`
- Create: `src/darwin_rag_exp2/data/importer.py`
- Create: `src/darwin_rag_exp2/data/audit.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `audit-data`
- Create: `tests/data/test_importer.py`
- Create: `tests/data/test_audit.py`

**Artifact contract**

```json
{
  "record_count": 12749,
  "invalid_json_count": 0,
  "duplicate_id_count": 0,
  "duplicate_url_count": 0,
  "text_length_mismatch_count": 0,
  "title_only_count": 641
}
```

**Verification**

```bash
uv run pytest tests/data
uv run darwin-exp2 audit-data --input data/raw/scatch_notices.jsonl --output artifacts/audit/raw
```

Expected artifacts: `artifacts/audit/raw/audit.json` and `audit.md` with the
baseline counts stated in `IMPLEMENTATION-PLAN.md`.

## Phase 3: Category Admission And Quality Filtering

**Purpose**

Convert the site's categories into the fixed primary study population and
separate unusable or interpretively weak documents before any model training.

**Files**

- Create: `configs/category_mapping.yaml`
- Create: `src/darwin_rag_exp2/data/filtering.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `prepare-corpus`
- Create: `tests/data/test_filtering.py`

**Rules**

- Admit only `채용`, `장학`, `비교과·행사`, `학사`, `봉사`, `국제교류`,
  `외국인유학생`, and `교원채용`.
- Exclude `기타`, `미분류`, and `교직` from primary outputs with explicit
  reasons.
- Exclude title-only or under-30-token body records.
- Reject a primary category after filtering if it contains fewer than 100
  source documents.

**Verification**

```bash
uv run pytest tests/data/test_filtering.py
uv run darwin-exp2 prepare-corpus --input data/raw/scatch_notices.jsonl --config configs/experiment.default.yaml --output artifacts/corpus
```

Expected artifacts: admitted and excluded source JSONL files, category counts,
filter-reason counts, and a manifest containing the source export hash.

## Phase 4: Improved Chunking And Chunk Artifacts

**Purpose**

Create stable retrieval units while respecting the category classifier input
limit and preserving the meaningful paragraph structure of Korean notices.

**Files**

- Modify: `pyproject.toml` to add `transformers` and `pyarrow`
- Create: `src/darwin_rag_exp2/data/chunking.py`
- Create: `src/darwin_rag_exp2/data/artifacts.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `chunk-corpus`
- Create: `tests/data/test_chunking.py`

**Fixed behavior**

- Use `klue/bert-base` tokenizer for token budgets.
- Use paragraph-first and Korean sentence-boundary splitting.
- Use body target 384 tokens, overlap 64 tokens, minimum 30 tokens, title
  prefix up to 64 tokens, and final classifier input maximum 512 tokens.
- Fall back to token windows only for single units that exceed the budget.
- Generate stable `chunk_id` values from `source_id` and `chunk_index`.

**Verification**

```bash
uv run pytest tests/data/test_chunking.py
uv run darwin-exp2 chunk-corpus --corpus artifacts/corpus/admitted.jsonl --output artifacts/chunks
```

Expected artifacts: `chunks.jsonl`, `chunks.parquet`, length histograms, and a
manifest proving no chunk violates the final classifier token cap.

## Phase 5: Single-Model Classifier Smoke Pipeline

**Purpose**

Verify training, calibrated inference, and category-statistics plumbing on a
small subset before paying the cost of crossfit training.

**Files**

- Modify: `pyproject.toml` to add `torch`, `scikit-learn`, `scipy`, and
  required classifier dependencies
- Create: `src/darwin_rag_exp2/models/classifier.py`
- Create: `src/darwin_rag_exp2/models/calibration.py`
- Create: `src/darwin_rag_exp2/models/category_stats.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `train-classifier --mode single`
- Create: `tests/models/test_calibration.py`
- Create: `tests/models/test_category_stats.py`

**Verification**

```bash
uv run pytest tests/models
uv run darwin-exp2 train-classifier --mode single --chunks artifacts/chunks/chunks.parquet --output artifacts/classifier/single
```

Expected artifacts: model reference, temperature value, before/after ECE,
sample calibrated predictions, and category-statistics table marked
`smoke_only: true`.

## Phase 6: Crossfit Classifier And Official Category Statistics

**Purpose**

Produce out-of-fold calibrated probabilities so that index assignment and
adaptive weights do not rely on in-sample classification confidence.

**Files**

- Create: `src/darwin_rag_exp2/models/crossfit.py`
- Create: `src/darwin_rag_exp2/models/splits.py`
- Modify: `src/darwin_rag_exp2/cli.py` to support `--mode crossfit`
- Create: `tests/models/test_splits.py`
- Create: `tests/models/test_crossfit_contract.py`

**Verification**

```bash
uv run pytest tests/models
uv run darwin-exp2 train-classifier --mode crossfit --chunks artifacts/chunks/chunks.parquet --folds 5 --output artifacts/classifier/crossfit
```

Expected evidence: every prediction row records a fold whose training
`source_id` list excludes that row's source; official `mu_c` and `sigma_c`
refer only to out-of-fold probabilities. Any derived `lambda_c` artifact is
explicitly labeled as a semantic-similarity mixture coefficient, not a BERT
confidence value.

## Phase 7: Embeddings And Frozen FAISS Indexes

**Purpose**

Build shared retrieval artifacts for all variants with identical embeddings and
explicit category-duplication provenance.

**Files**

- Modify: `pyproject.toml` to add `sentence-transformers` and `faiss-cpu`
- Create: `src/darwin_rag_exp2/indexing/embeddings.py`
- Create: `src/darwin_rag_exp2/indexing/faiss_store.py`
- Create: `src/darwin_rag_exp2/indexing/partitions.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `build-indexes`
- Create: `tests/indexing/test_partitions.py`

**Rules**

- Embed every admitted chunk once with normalized `BAAI/bge-m3` vectors.
- Store all chunks in a unified `IndexFlatIP`.
- Store each chunk in all category indexes satisfying
  `P_calibrated(c | d) >= K_ingest`; use top-1 fallback when none pass.
- Keep one stable `chunk_id` across duplicated category occurrences.

**Verification**

```bash
uv run pytest tests/indexing
uv run darwin-exp2 build-embeddings --chunks artifacts/chunks/chunks.parquet --output artifacts/embeddings
uv run darwin-exp2 build-indexes --chunks artifacts/chunks/chunks.parquet --predictions artifacts/classifier/crossfit/predictions.parquet --output artifacts/indexes
```

Expected artifacts: FAISS files, ID maps, partition-assignment Parquet, model
and corpus hashes, reusable chunk embedding artifacts, and an immutable index
manifest. If embeddings were already generated on a GPU server, reuse them with
`--embeddings artifacts/embeddings` to avoid recomputing `BAAI/bge-m3` chunk
vectors.

## Phase 8: Query Pool And Annotation Contract

**Purpose**

Create a controlled human-annotation input for tuning and final evaluation.

**Files**

- Create: `src/darwin_rag_exp2/evaluation/queries.py`
- Create: `src/darwin_rag_exp2/evaluation/pool.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `export-query-pool` and
  `validate-queries`
- Create: `tests/evaluation/test_queries.py`

**Contract**

- `queries_dev.jsonl`: 80 annotated queries.
- `queries_test.jsonl`: 240 annotated queries.
- Around 30 percent of each set is `multi_category` or `ambiguous`.
- Rows contain `query_id`, `query`, `gold_chunks`, `reference_answer`,
  `gold_categories`, and `query_type`.

**Verification**

```bash
uv run pytest tests/evaluation/test_queries.py
uv run darwin-exp2 validate-queries --dev data/annotations/queries_dev.jsonl --test data/annotations/queries_test.jsonl
```

Expected evidence: non-overlapping dev/test IDs, resolvable gold chunk IDs,
category/type distribution report, and frozen query hashes.

## Phase 9: Primary Score-Merge Retrieval Variants

**Purpose**

Implement and tune the variant family that directly preserves the proposed
document-level mixture score without an additional fusion algorithm.

**Fixed score contract**

```text
s_norm(q,d) = (sim(q,d) + 1) / 2

score_c(q,d) = lambda_c * s_norm(q,d)
             + (1 - lambda_c) * P_calibrated(c | q)

final_score(q,d) = max_c score_c(q,d)
```

`lambda_c` is the proportion of semantic similarity used in the final score;
it is not the BERT confidence itself. `s_norm` is a shared fixed mapping for
all score-merge variants, not a per-partition normalization or probability
calibration step.

**Files**

- Modify: `pyproject.toml` to add `optuna`
- Create: `src/darwin_rag_exp2/retrieval/types.py`
- Create: `src/darwin_rag_exp2/retrieval/routing.py`
- Create: `src/darwin_rag_exp2/retrieval/score_merge.py`
- Create: `src/darwin_rag_exp2/retrieval/variants.py`
- Create: `src/darwin_rag_exp2/evaluation/retrieval_metrics.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `tune-primary` and `run-primary`
- Create: `tests/retrieval/test_score_merge.py`
- Create: `tests/retrieval/test_vanilla_rrf_invariance.py`

**Verification**

```bash
uv run pytest tests/retrieval
uv run darwin-exp2 tune-primary --queries data/annotations/queries_dev.jsonl --indexes artifacts/indexes --output artifacts/settings/primary
uv run darwin-exp2 run-primary --queries data/annotations/queries_test.jsonl --settings artifacts/settings/primary/frozen.yaml --output runs/primary
```

Expected evidence: the vanilla RRF invariance fixture proves it cannot
distinguish fixed/adaptive lambda, while `B2-score` and `P-score` produce
paired Top-10 test result files under frozen settings.

## Phase 10: Local Generation And Automated Metrics

**Purpose**

Convert each frozen primary retrieval result into a comparable generated
answer without external model drift.

**Files**

- Modify: `pyproject.toml` to add `mlx-lm`, `rouge-score`, and `bert-score`
- Create: `src/darwin_rag_exp2/generation/prompt.py`
- Create: `src/darwin_rag_exp2/generation/local_qwen.py`
- Create: `src/darwin_rag_exp2/generation/cache.py`
- Create: `src/darwin_rag_exp2/evaluation/generation_metrics.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `generate-answers`
- Create: `tests/generation/test_prompt.py`
- Create: `tests/generation/test_cache.py`

**Verification**

```bash
uv run pytest tests/generation
uv run darwin-exp2 generate-answers --runs runs/primary --queries data/annotations/queries_test.jsonl --model mlx-community/Qwen3-8B-4bit --output runs/generation
```

Expected evidence: all primary variants use exactly five context chunks per
query, shared prompt/decoding configuration, cached answers, and automated
EM, token-F1, ROUGE, and BERTScore result files.

## Phase 11: Statistics, Latency, And Report Export

**Purpose**

Produce the tables and figures used to accept or reject the research
hypotheses without modifying frozen experiment inputs.

**Files**

- Create: `src/darwin_rag_exp2/evaluation/statistics.py`
- Create: `src/darwin_rag_exp2/evaluation/latency.py`
- Create: `src/darwin_rag_exp2/evaluation/reporting.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `report`
- Create: `tests/evaluation/test_statistics.py`
- Create: `tests/evaluation/test_reporting.py`

**Verification**

```bash
uv run pytest
uv run darwin-exp2 report --primary runs/primary --generation runs/generation --output runs/report
```

Expected artifacts:

- Primary table for `P-score` versus `B2-score` with paired differences,
  Wilcoxon p-value, and paired bootstrap 95 percent confidence interval.
- Primary category and query-type diagnostic breakdown tables.
- Retrieval-only latency median/p95/p99 tables and plots.
- Complete manifest linking raw-data, model, index, settings, query, and run
  hashes.

## Phase 12: Primary Artifact Freeze And Reproducibility Closeout

**Purpose**

Freeze the score-merge experiment as the completed mandatory study before any
supplemental fusion implementation is considered.

**Files**

- Modify: `src/darwin_rag_exp2/cli.py` to add `verify-manifest`
- Create: `src/darwin_rag_exp2/evaluation/manifest.py`
- Create: `tests/evaluation/test_manifest.py`
- Create: `runs/final/PRIMARY-RESULTS.md`
- Create: `runs/final/manifest.json`
- Create: `runs/final/checksums.txt`
- Create: `runs/final/environment.txt`

**Actions**

1. Copy or reference only frozen `B0`, `B1`, `B2-score`, and `P-score`
   retrieval and generation results in the final manifest.
2. Record dataset, query, model, index, settings, prompt, and report hashes.
3. Record the executed verification commands and environment versions.
4. Mark the primary study complete before beginning any optional extension.

**Verification**

```bash
uv run pytest
uv run darwin-exp2 verify-manifest --manifest runs/final/manifest.json
```

Expected artifacts: a hash-verifiable primary experiment bundle and a final
summary whose central hypothesis test is `P-score > B2-score`.

## Optional Extension A: Weighted RRF Sensitivity Variants

**Prerequisite**

Run this extension only after Phase 12 primary artifacts are frozen. Its
outputs must not alter primary settings, metrics, or conclusion tables.

**Purpose**

Assess separately whether retaining a rank-fusion formulation changes the
adaptive weighting observation.

**Files**

- Create: `src/darwin_rag_exp2/retrieval/weighted_rrf.py`
- Modify: `src/darwin_rag_exp2/retrieval/variants.py`
- Modify: `src/darwin_rag_exp2/cli.py` to add `tune-wrrf` and `run-wrrf`
- Create: `tests/retrieval/test_weighted_rrf.py`
- Create: `configs/optional.wrrf.yaml` with the fixed RRF constant

**Verification**

```bash
uv run pytest tests/retrieval/test_weighted_rrf.py
uv run darwin-exp2 tune-wrrf --queries data/annotations/queries_dev.jsonl --indexes artifacts/indexes --config configs/optional.wrrf.yaml --output artifacts/settings/wrrf
uv run darwin-exp2 run-wrrf --queries data/annotations/queries_test.jsonl --settings artifacts/settings/wrrf/frozen.yaml --output runs/wrrf
```

Expected evidence: `B2-wrrf` and `P-wrrf` outputs contain per-category
`semantic_evidence` and `w(q,c)` values alongside final rankings, reported as
supplemental results only.
