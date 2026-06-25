# DARWIN-RAG Exp2 Query Guidelines

이 문서는 Phase 8 v2 query annotation 파일인
`queries_dev_v2.jsonl`과 `queries_test_v2.jsonl`의 작성, 검증, 평가
계약을 정의한다. v2 schema는 overlap-aware rechunked 평가용
`eval_v3_overlap_aware_rechunked`이며, Phase 9 primary retrieval 평가의
공통 입력이다.

## Scope

- 대상 파일:
  - `data/annotations/queries_dev_v2.jsonl`
  - `data/annotations/queries_test_v2.jsonl`
- 입력 기준 artifact:
  - `artifacts/chunks/chunks.parquet`
  - `artifacts/indexes/manifest.json`
- 작성 형식:
  - JSONL
  - 한 줄에 하나의 query row
  - UTF-8 인코딩

현재 `.gitignore`는 `data/annotations/`를 제외한다. Annotation 파일은
일반 source commit 대상이 아니며, freeze 시점에는 manifest와 hash로
lineage를 기록한다.

## Split Targets

| Split | File | Count | Purpose |
|---|---|---:|---|
| dev | `queries_dev_v2.jsonl` | 80 | threshold, fixed lambda, adaptive parameter 선택 |
| test | `queries_test_v2.jsonl` | 240 | 최종 retrieval, generation, statistical reporting |

Dev split만 설정 선택에 사용한다. Test split은 `run-primary`와
`analyze-primary` 평가 입력이며, tuning/settings/prompt/model 선택에
사용하지 않는다.

## Row Schema

각 JSONL row는 v2 field를 가진다.

```json
{
  "query_id": "dev_q0001",
  "query": "이번 학기 수강신청 변경 기간은 언제인가요?",
  "reference_answer": "수강신청 변경 기간은 ...",
  "query_type": "core_single_category",
  "expected_categories": ["학사"],
  "source_id": "scatch_...",
  "gold_chunk_ids": [
    {
      "chunk_id": "scatch_...::0000",
      "source_id": "scatch_...",
      "chunk_index": 0,
      "relevance": 2,
      "role": "direct"
    }
  ],
  "neighbor_chunk_ids": ["scatch_...::0001"],
  "graded_relevance": {
    "scatch_...::0000": 1.0
  },
  "requires_multi_category": false,
  "gold_category_set": ["학사"],
  "gold_category_pure": true,
  "evidence_unit": "title+category+body",
  "schema_version": "eval_v3_overlap_aware_rechunked"
}
```

Downstream loader는 `gold_chunk_ids[*].chunk_id`를 canonical
`gold_chunks`로, `gold_category_set`을 canonical `gold_categories`로
정규화한다. 문서와 실행 명령은 v2 파일명을 기준으로 한다.

## Stable Identifiers

`query_id`는 split prefix를 포함한다.

- dev: `dev_q0001`, `dev_q0002`, ...
- test: `test_q0001`, `test_q0002`, ...
- dev/test 사이에 중복이 없어야 한다.
- 같은 split 안에서도 중복이 없어야 한다.
- annotation freeze 이후에는 동일 query의 표현을 소폭 다듬더라도 기존
  ID를 유지한다.

`source_id`는 대표 근거 notice ID이다. `gold_chunk_ids[*].source_id`는
각 gold chunk의 source를 기록하며, `chunk_id`와 `chunk_index`는
`artifacts/chunks/chunks.parquet`에서 resolve되어야 한다.

## Query Text

`query`는 사용자가 실제 RAG 시스템에 입력할 법한 한국어 자연어
질문이다.

- 공지 제목을 그대로 복사하지 않는다.
- 공지 번호, URL, source ID, chunk ID를 직접 묻지 않는다.
- category label을 노골적으로 붙이지 않는다.
- 사용자가 알고 싶은 행동, 조건, 기간, 자격, 제출물, 장소, 절차가
  드러나도록 작성한다.
- 하나의 query는 하나의 평가 의도를 가져야 한다.

## Gold Evidence

`gold_chunk_ids`는 retrieval 평가의 정답 근거 chunk 목록이다. 각 항목은
다음 필드를 가진다.

| Field | Rule |
|---|---|
| `chunk_id` | `artifacts/chunks/chunks.parquet`에 존재하는 chunk ID |
| `source_id` | 해당 chunk의 source notice ID |
| `chunk_index` | source 안의 0-based chunk index |
| `relevance` | `2`는 직접 근거, `1`은 보조 근거 |
| `role` | `direct`, `direct_overlap`, `support` 중 하나 |

작성 규칙:

- `gold_chunk_ids`는 비어 있으면 안 된다.
- 같은 query 안에서 `chunk_id`를 중복하지 않는다.
- `reference_answer`를 작성하는 데 필요한 최소 근거만 넣는다.
- `role=direct` 또는 `direct_overlap`은 답변 핵심 근거에 사용한다.
- `role=support`는 보조 조건, 세부 절차, 이어지는 chunk 근거에 사용한다.

`neighbor_chunk_ids`는 overlap-aware 진단용 인접 chunk 목록이다.

- 문자열 chunk ID 배열이다.
- `gold_chunk_ids`와 겹치면 안 된다.
- 중복 없이 `artifacts/chunks/chunks.parquet`에서 resolve되어야 한다.
- 정답으로 채점하지 않고, failure/neighbor 진단에 사용한다.

`graded_relevance`는 `gold_chunk_ids[*].chunk_id`를 정확히 같은 key 집합으로
가져야 한다. 값은 `0.0`부터 `1.0`까지의 numeric score이다. 현재 관례는
`relevance=2`를 `1.0`, `relevance=1`을 `0.5`로 기록한다.

## Reference Answer

`reference_answer`는 사람이 `gold_chunk_ids`를 읽고 작성한 generation 평가
정답이다.

- 질문에 직접 답한다.
- gold evidence에 없는 내용을 추론해서 추가하지 않는다.
- 날짜, 대상, 금액, 장소, URL, 신청 방법, 제출 서류처럼 평가에 중요한
  값을 명확히 적는다.
- 공지 원문이 불확실하거나 조건부 표현을 쓰면 답변에도 그 조건을
  보존한다.
- "자세한 내용은 공지 참고" 같은 회피형 답변만으로 끝내지 않는다.

## Category Fields

Category field는 routing 평가와 category breakdown에 사용한다.

| Field | Rule |
|---|---|
| `expected_categories` | query 의도상 기대되는 primary category 목록 |
| `gold_category_set` | gold evidence가 속한 primary category 목록 |
| `requires_multi_category` | `gold_category_set`이 2개 이상이면 `true` |
| `gold_category_pure` | gold evidence가 단일 primary category로 순수하면 `true` |
| `query_type` | 진단용 query type 문자열. v2에서는 비어 있으면 안 된다. |

`expected_categories`와 `gold_category_set`은 같은 category 집합이어야 한다.
허용 category는 다음 8개이다.

```text
채용
장학
비교과·행사
학사
봉사
국제교류
외국인유학생
교원채용
```

## Validation Checklist

Phase 8 완료 전에는 다음 조건을 만족해야 한다.

- `queries_dev_v2.jsonl`은 정확히 80 rows이다.
- `queries_test_v2.jsonl`은 정확히 240 rows이다.
- 모든 row의 `schema_version`은 `eval_v3_overlap_aware_rechunked`이다.
- dev/test 사이와 각 split 내부에 `query_id` 중복이 없다.
- 모든 `gold_chunk_ids[*].chunk_id`와 `neighbor_chunk_ids`는 chunk artifact에
  존재한다.
- `gold_chunk_ids`와 `neighbor_chunk_ids`는 중복되거나 서로 겹치지 않는다.
- `graded_relevance`는 gold chunk key를 정확히 cover한다.
- category fields는 8개 primary category 안에서 일관되어야 한다.
- test query는 설정 변경, threshold 조정, prompt 선택에 사용하지 않는다.

검증 명령:

```bash
uv run darwin-exp2 validate-queries \
  --dev data/annotations/queries_dev_v2.jsonl \
  --test data/annotations/queries_test_v2.jsonl \
  --output artifacts/query_validation/v2
```

## Annotation-To-Evaluation Commands

Tuning은 dev v2만 사용한다.

```bash
uv run darwin-exp2 tune-primary \
  --queries data/annotations/queries_dev_v2.jsonl \
  --indexes artifacts/indexes \
  --output artifacts/settings/primary \
  --query-classifier artifacts/classifier/final
```

최종 retrieval 실행은 test v2를 사용하며 `B0`, `B1`, `B2-score`,
`P-score` 네 variant row를 query마다 생성한다. 각 row는
`retrieval_time_ms`를 포함한다.

```bash
uv run darwin-exp2 run-primary \
  --queries data/annotations/queries_test_v2.jsonl \
  --settings artifacts/settings/primary/frozen.yaml \
  --indexes artifacts/indexes \
  --output runs/primary \
  --query-classifier artifacts/classifier/final
```

분석은 frozen test run만 읽는다.

```bash
uv run darwin-exp2 analyze-primary \
  --run runs/primary \
  --chunks artifacts/chunks/chunks.parquet \
  --output runs/analysis/primary \
  --metric ndcg@10 \
  --top-failures 20
```

Expected analysis artifacts:

- `summary.json`
- `metrics_by_variant.csv`
- `retrieval_time_by_variant.csv`
- `retrieval_time_by_query_type.csv`
- `paired_comparison.json`
- `paired_deltas.csv`
- `failure_cases.jsonl`
- `report.html`

Timing CSVs report retrieval-only `retrieval_time_ms` count, mean, min, max,
and median by variant and by query type. Timing does not include generation,
analysis, API latency, or settings tuning.
