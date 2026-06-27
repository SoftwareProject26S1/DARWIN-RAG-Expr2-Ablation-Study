#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

QUERIES="${QUERIES:-data/annotations/queries_test_v2.jsonl}"
SETTINGS="${SETTINGS:-artifacts/settings/primary/frozen.yaml}"
INDEXES="${INDEXES:-artifacts/indexes}"
CHUNKS="${CHUNKS:-artifacts/chunks/chunks.parquet}"
CONFIG="${CONFIG:-configs/experiment.default.yaml}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-sentence-transformers}"
CLASSIFIER_DEVICE="${CLASSIFIER_DEVICE:-auto}"
UNIFIED_CANDIDATE_K="${UNIFIED_CANDIDATE_K:-100}"

STAMP="$(date +%Y%m%d-%H%M%S)"
SUITE_NAME="${PREFIX:-full-primary-suite-${STAMP}}"
RUN_ROOT="${RUN_ROOT:-runs/${SUITE_NAME}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/${SUITE_NAME}}"

run_and_analyze() {
  local name="$1"
  shift

  local run_name="${name}"
  local run_dir="${RUN_ROOT}/${run_name}"
  local analysis_dir="${ANALYSIS_ROOT}/${run_name}"

  echo "==> run-primary: ${run_name}"
  uv run darwin-exp2 run-primary \
    --queries "${QUERIES}" \
    --settings "${SETTINGS}" \
    --indexes "${INDEXES}" \
    --config "${CONFIG}" \
    --output "${run_dir}" \
    --embedding-backend "${EMBEDDING_BACKEND}" \
    --classifier-device "${CLASSIFIER_DEVICE}" \
    "$@"

  echo "==> analyze-primary: ${run_name}"
  uv run darwin-exp2 analyze-primary \
    --run "${run_dir}" \
    --output "${analysis_dir}" \
    --chunks "${CHUNKS}"
}

mkdir -p "${RUN_ROOT}" "${ANALYSIS_ROOT}"

run_and_analyze "primary"

run_and_analyze "oracle-router" \
  --router oracle

run_and_analyze "unified-rerank" \
  --search-mode unified-prior-rerank \
  --unified-candidate-k "${UNIFIED_CANDIDATE_K}"

run_and_analyze "core-single-category" \
  --query-type core-single-category

run_and_analyze "multi-category" \
  --query-type multi-category

run_and_analyze "ambiguous" \
  --query-type ambiguous

echo
echo "Done."
echo "Prefix:   ${PREFIX}"
echo "Runs:     ${RUN_ROOT}"
echo "Analysis: ${ANALYSIS_ROOT}"
