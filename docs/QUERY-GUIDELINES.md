# DARWIN-RAG Exp2 Query Guidelines

이 문서는 Phase 8 query annotation 파일인 `queries_dev.jsonl`과
`queries_test.jsonl`을 작성하고 검증하기 위한 기준을 정의한다. Phase
8 query는 단순 질의 목록이 아니라, 이후 Phase 9 retrieval 평가, Phase
10 generation 평가, Phase 11 category/query-type 진단 분석의 공통
입력이다.

## Scope

- 대상 파일:
  - `data/annotations/queries_dev.jsonl`
  - `data/annotations/queries_test.jsonl`
- 입력 기준 artifact:
  - `artifacts/chunks/chunks.parquet`
  - `artifacts/indexes/manifest.json`
- 작성 형식:
  - JSONL
  - 한 줄에 하나의 query row
  - UTF-8 인코딩

현재 `.gitignore`는 `data/annotations/`를 제외한다. 따라서 annotation
파일은 일반 source commit 대상이 아니며, freeze 시점에는 manifest와
hash로 lineage를 기록한다. source control에 포함해야 하는 정책 변경이
필요하면 `.gitignore`와 working rule을 의도적으로 함께 수정한다.

## Split Targets

현재 primary category 수는 `C=8`이다. query annotation 목표량은 다음과
같이 고정한다.

| Split | File | Count | Purpose |
|---|---|---:|---|
| dev | `queries_dev.jsonl` | 80 | threshold, fixed lambda, adaptive parameter 선택 |
| test | `queries_test.jsonl` | 240 | 최종 retrieval, generation, statistical reporting |

각 split의 약 30%는 `multi_category` 또는 `ambiguous` query로 작성한다.
권장 분포는 다음과 같다.

| Split | `single_category` | `multi_category` | `ambiguous` | Total |
|---:|---:|---:|---:|---:|
| dev | 56 | 12 | 12 | 80 |
| test | 168 | 36 | 36 | 240 |

`single_category` query는 8개 primary category에 최대한 균등하게
분배한다. dev split에서는 category당 7개, test split에서는 category당
21개를 기본 목표로 삼는다.

## Row Schema

각 JSONL row는 다음 6개 필드를 가진다.

```json
{
  "query_id": "dev_q0001",
  "query": "이번 학기 수강신청 변경 기간은 언제인가요?",
  "gold_chunks": ["scatch_...::0001"],
  "reference_answer": "수강신청 변경 기간은 ...",
  "gold_categories": ["학사"],
  "query_type": "single_category"
}
```

추가 metadata가 필요하면 Phase 8 validator와 downstream runner가 허용할
때만 도입한다. 기본 annotation 파일은 위 6개 필드를 기준으로 동결한다.

## `query_id`

`query_id`는 query row의 안정적인 식별자이다.

작성 규칙:

- split prefix를 포함한다.
  - dev: `dev_q0001`, `dev_q0002`, ...
  - test: `test_q0001`, `test_q0002`, ...
- dev와 test 사이에 중복이 없어야 한다.
- 같은 split 안에서도 중복이 없어야 한다.
- annotation freeze 이후에는 동일 query의 표현을 소폭 다듬더라도 기존
  ID를 유지한다.
- 완전히 다른 질문으로 교체할 때만 새 ID를 부여한다.

금지 예시:

```json
{"query_id": "q1"}
{"query_id": "학사질문1"}
{"query_id": "dev_1_duplicate"}
```

## `query`

`query`는 사용자가 실제 RAG 시스템에 입력할 법한 한국어 자연어
질문이다.

작성 규칙:

- 공지 제목을 그대로 복사하지 않는다.
- 공지 번호, URL, source ID, chunk ID를 직접 묻지 않는다.
- category label을 노골적으로 붙이지 않는다.
- 사용자가 알고 싶은 행동, 조건, 기간, 자격, 제출물, 장소, 절차가
  드러나도록 작성한다.
- retrieval이 의미 기반으로 근거를 찾아야 하도록, 제목 복붙보다
  자연스러운 질문 문장으로 작성한다.
- 하나의 query는 하나의 평가 의도를 가져야 한다. 서로 무관한 질문을
  한 줄에 섞지 않는다.

좋은 예시:

```text
2026학년도 1학기 국가장학금 신청은 언제까지 해야 하나요?
교환학생 지원 자격과 제출 서류를 알려줘.
졸업 예정자가 수강신청 변경 기간에 확인해야 할 사항은 뭐야?
외국인 학생도 신청할 수 있는 장학 프로그램이 있나요?
```

피해야 할 예시:

```text
[학사] 2026학년도 1학기 수강신청 변경 안내
공지사항 12345번 내용 알려줘
장학 카테고리 문서 찾아줘
scatch_12345::0000에 뭐라고 써 있어?
```

## `gold_chunks`

`gold_chunks`는 retrieval 평가의 정답 chunk ID 목록이다. 이 값은
`source_id`가 아니라 `artifacts/chunks/chunks.parquet`에 존재하는
`chunk_id`여야 한다.

작성 규칙:

- 모든 ID는 `artifacts/chunks/chunks.parquet`에서 resolve되어야 한다.
- `reference_answer`를 작성하는 데 필요한 최소 근거 chunk만 포함한다.
- 단순히 관련 있어 보이는 chunk를 모두 넣지 않는다.
- 일반적으로 1-3개를 권장한다.
- `multi_category` query라도 가능하면 5개 이내로 제한한다.
- 같은 query 안에서 중복 chunk ID를 넣지 않는다.
- answer에 필요한 날짜, 대상, 신청 방법, 제출 서류 등이 서로 다른
  chunk에 흩어져 있으면 필요한 chunk를 모두 포함한다.

예시:

```json
"gold_chunks": [
  "scatch_12345::0000",
  "scatch_67890::0001"
]
```

판정 기준:

- Top-k retrieval metric에서는 `gold_chunks` 중 하나 이상을 맞히는지,
  또는 얼마나 높은 순위에 배치하는지를 평가한다.
- generation 평가에서는 `reference_answer`가 이 chunk들에 의해
  뒷받침되어야 한다.

## `reference_answer`

`reference_answer`는 generation 평가의 정답 답변이다. 사람이
`gold_chunks`를 읽고 작성한다.

작성 규칙:

- 질문에 직접 답한다.
- `gold_chunks`에 없는 내용을 추론해서 추가하지 않는다.
- 날짜, 대상, 금액, 장소, URL, 신청 방법, 제출 서류처럼 평가에 중요한
  값을 명확히 적는다.
- 원문이 표나 목록이면 자연어로 압축해도 된다.
- 공지 원문이 불확실하거나 조건부 표현을 쓰면 답변에도 그 조건을
  보존한다.
- 보통 1-4문장으로 작성한다.
- "자세한 내용은 공지 참고" 같은 회피형 답변만으로 끝내지 않는다.

좋은 예시:

```text
수강신청 변경 기간은 2026년 3월 4일부터 3월 8일까지입니다. 해당 기간에는 수강 과목 추가와 삭제가 가능하며, 세부 절차는 학사 공지의 안내를 따라야 합니다.
```

피해야 할 예시:

```text
공지에 나온 기간에 신청하면 됩니다.
자세한 내용은 학교 홈페이지를 참고하세요.
아마 3월 초쯤 신청하면 될 것 같습니다.
```

## `gold_categories`

`gold_categories`는 정답 근거가 속한 primary category 목록이다. 값은
반드시 다음 8개 category 중 하나 이상이어야 한다.

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

작성 규칙:

- `single_category` query는 정확히 1개 category를 가진다.
- `multi_category` query는 2개 이상의 category를 가진다.
- `ambiguous` query는 1개 이상 category를 가진다.
- category는 query 표현이 아니라 정답 근거 chunk의 category를 기준으로
  확정한다.
- 같은 category를 중복해서 넣지 않는다.
- category 순서는 안정적으로 유지한다. 권장 순서는 config의
  `primary_categories` 순서이다.

예시:

```json
"gold_categories": ["학사"]
```

```json
"gold_categories": ["장학", "외국인유학생"]
```

## `query_type`

`query_type`은 routing과 diagnostic breakdown을 위한 query 난이도
라벨이다.

허용 값:

| Type | Meaning | `gold_categories` rule |
|---|---|---|
| `single_category` | 한 category의 공지만으로 답변 가능 | 정확히 1개 |
| `multi_category` | 둘 이상의 category 근거를 함께 사용해야 답변 가능 | 2개 이상 |
| `ambiguous` | 사용자 표현만 보면 category 경계가 불명확하거나 여러 routing 후보가 자연스러움 | 1개 이상 |

`ambiguous`는 정답이 불명확한 query가 아니다. 사람이 annotation할 때는
`gold_chunks`, `reference_answer`, `gold_categories`가 명확해야 한다.
다만 query 표현이 짧거나 경계적이어서 classifier/routing이 어려운
경우를 뜻한다.

예시:

```json
{
  "query_id": "dev_q0017",
  "query": "외국인 학생도 신청할 수 있는 장학 프로그램이 있나요?",
  "gold_chunks": ["scatch_12345::0000", "scatch_67890::0000"],
  "reference_answer": "외국인유학생 대상 장학 신청은 공지에 명시된 지원 자격을 충족해야 하며, 신청 기간 내에 요구 서류를 제출해야 합니다. 세부 장학명과 제출 방식은 해당 장학 공지의 안내를 기준으로 확인해야 합니다.",
  "gold_categories": ["장학", "외국인유학생"],
  "query_type": "multi_category"
}
```

```json
{
  "query_id": "test_q0042",
  "query": "해외 프로그램 신청 전에 확인해야 할 지원 조건이 뭐야?",
  "gold_chunks": ["scatch_24680::0000"],
  "reference_answer": "해외 프로그램 신청 전에는 모집 대상, 지원 자격, 제출 서류, 신청 기간을 확인해야 합니다. 공지에 명시된 조건을 충족하지 않으면 선발 대상에서 제외될 수 있습니다.",
  "gold_categories": ["국제교류"],
  "query_type": "ambiguous"
}
```

## Annotation Workflow

권장 작업 순서는 다음과 같다.

1. `artifacts/chunks/chunks.parquet`에서 candidate chunks를 category별로
   검토한다.
2. source notice의 제목, 본문, 날짜, URL을 확인한다.
3. 실제 사용자가 물을 법한 `query`를 먼저 작성한다.
4. 답변 근거가 되는 최소 `gold_chunks`를 확정한다.
5. `gold_chunks`만 보고 `reference_answer`를 작성한다.
6. `gold_categories`와 `query_type`을 부여한다.
7. dev/test split별 수량, category 분포, query type 분포를 점검한다.
8. `validate-queries`로 schema, 중복, chunk resolve, 분포 조건을 검증한다.

## Validation Checklist

Phase 8 완료 전에는 다음 조건을 만족해야 한다.

- `queries_dev.jsonl`은 정확히 80 rows이다.
- `queries_test.jsonl`은 정확히 240 rows이다.
- dev/test 사이에 `query_id` 중복이 없다.
- 각 split 내부에 `query_id` 중복이 없다.
- 모든 row는 6개 필드를 가진다.
- `query`는 비어 있지 않은 한국어 자연어 질문이다.
- 모든 `gold_chunks`는 `artifacts/chunks/chunks.parquet`에 존재한다.
- 각 row의 `gold_chunks`는 1개 이상이다.
- `reference_answer`는 비어 있지 않다.
- 모든 `gold_categories`는 primary category 집합 안에 있다.
- `query_type`은 `single_category`, `multi_category`, `ambiguous` 중 하나이다.
- `single_category`는 정확히 1개 `gold_categories`를 가진다.
- `multi_category`는 2개 이상 `gold_categories`를 가진다.
- 각 split에서 `multi_category`와 `ambiguous` 합계가 약 30%이다.
- test query는 설정 변경, threshold 조정, prompt 선택에 사용하지 않는다.

검증 명령:

```bash
uv run darwin-exp2 validate-queries \
  --dev data/annotations/queries_dev.jsonl \
  --test data/annotations/queries_test.jsonl
```
