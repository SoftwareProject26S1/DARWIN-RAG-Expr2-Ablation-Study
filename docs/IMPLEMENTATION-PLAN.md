# DARWIN-RAG Exp2 Implementation Plan

> 작성일: 2026-05-27
> 상태: Phase 1에서 확정한 구현 및 실험 프로토콜
> 입력 원천: `../../scatch_notices.jsonl`

## 연구 목적과 Exp2 범위

`Exp2`는 DARWIN-RAG의 핵심 제안인 카테고리별 적응 가중치
`lambda_c`가 검색 결과와 생성 응답 품질에 미치는 영향을 검증하는
Python 기반 ablation experiment 프로젝트이다. 실험 코드는 하나의
코퍼스, 하나의 분류기/임베더/생성기, 동일한 평가 질의와 메트릭을
공유하고, retrieval/fusion 전략만 바꾸어 비교한다.

이 프로젝트가 담당하는 범위는 다음과 같다.

- 학교 공지 export JSONL 검증, 품질 audit, category admission
- 공지 본문의 재현 가능한 청킹
- category classifier 학습, temperature scaling, crossfit 확률 산출
- 공통 임베딩 및 통합/카테고리별 FAISS artifact 생성
- score-merge 및 Weighted RRF 계열 ablation 실행
- 고정된 로컬 LLM을 사용한 생성 평가 및 통계 리포트 출력

다음 항목은 `Exp2`가 구현하지 않는다.

- 학교 홈페이지 크롤링 또는 export 생성
- OCR 또는 첨부 파일에서 추가 본문을 추출하는 파이프라인
- 실서비스 API, 외부 데이터베이스, 배포 서버
- 외부 API LLM 호출 기반의 본 평가

`Exp2`는 기존 실험 저장소의 `Exp1/`과 나란히 존재하는 독립 `uv`
프로젝트이며, `Exp1/`의 수집/학습 실험 결과를 수정하지 않는다.

## 초기 스코어링 함수와 RRF 설계 보정

### 원래 scoring 정의

원래 제안한 query-time score는 카테고리 `c`의 인덱스에서 검색된
문서 `d`에 대해 다음과 같다.

```text
score_c(q,d) = lambda_c * sim(q,d)
             + (1 - lambda_c) * P_calibrated(c | q)
```

- `sim(q,d)`: 질의와 문서의 임베딩 코사인 유사도
- `P_calibrated(c | q)`: 보정된 BERT가 질의 `q`를 카테고리 `c`로
  판단한 확률
- `lambda_c`: ingestion 과정에서 카테고리별 문서 분류 confidence
  분포로부터 사전 계산한 가중치

`B2`는 모든 카테고리에 동일한 `lambda_fixed`를 쓰고, `P`는
카테고리마다 다른 `lambda_c`를 쓴다는 것이 원래의 비교 의도이다.

### Vanilla RRF와 결합할 때 발생하는 무효성

고정된 질의 `q`와 하나의 카테고리 인덱스 `c` 안에서는
`P_calibrated(c | q)`와 `lambda_c`가 문서에 따라 변하지 않는다.
따라서 같은 인덱스 안의 문서 두 개에 대한 점수 차이는 다음과 같다.

```text
score_c(q,d1) - score_c(q,d2)
  = lambda_c * (sim(q,d1) - sim(q,d2))
```

`0 < lambda_c < 1`이면 `score_c` 순위는 항상 `sim` 순위와 같다.
즉, 고정 가중치와 적응 가중치가 파티션 내부 후보 순서를 바꾸지
않는다.

Vanilla Reciprocal Rank Fusion은 다음처럼 파티션별 순위만 합산한다.

```text
RRF(d) = sum_c 1 / (60 + rank_c(d))
```

이 방식은 파티션 간의 절대 score 차이, 곧 `lambda_c`와 query
category confidence가 반영한 차이를 폐기한다. 동일한 soft routing과
동일한 후보 검색 깊이를 사용하는 `B2`와 `P`는 vanilla RRF 이후
동일한 결과를 낼 수밖에 없다. 따라서 기존 구조로는
`P > B2`라는 adaptive weighting 가설을 시험할 수 없다.

vanilla RRF는 구현 시 synthetic regression test로 이 동치를
재현하는 대상이며, primary experiment variant로 사용하지 않는다.

### Primary: 정규화 점수 전역 병합

원래 document-level mixture 식을 유지하려면 파티션별 score를 순위로
버리지 않고 전역에서 비교해야 한다. 공통 임베더가 생성한 L2
정규화 벡터의 inner-product/cosine score를 다음과 같이 고정
변환한다.

```text
s_norm(q,d) = (sim(q,d) + 1) / 2

score_c(q,d) = lambda_c * s_norm(q,d)
             + (1 - lambda_c) * P_calibrated(c | q)

final_score(q,d) = max_c score_c(q,d)
```

실행 규칙은 다음과 같다.

1. soft routing으로 선택된 각 카테고리 인덱스에서
   `candidate_k_per_partition=50` 후보를 가져온다.
2. 모든 파티션에 같은 `s_norm` 변환을 적용한다. 파티션별 min-max
   정규화는 각 리스트의 기준을 바꾸므로 금지한다.
3. 각 `(chunk_id, c)` occurrence에 `score_c`를 계산한다.
4. 동일 `chunk_id`가 여러 파티션에 있으면 가장 높은 score만 남기고,
   어떤 category occurrence가 승자가 되었는지 provenance에 저장한다.
5. `final_score`로 전역 정렬해 검색 평가용 Top-10과 생성용 Top-5를
   분리 저장한다.

이 계열의 두 조건은 다음과 같다.

- `B2-score`: dev query에서 선택한 단일 `lambda_fixed` 사용
- `P-score`: ingestion 통계에서 얻은 category-specific `lambda_c` 사용

본 연구의 primary hypothesis는 test query에서
`P-score > B2-score`이다.

### Secondary: Weighted RRF

RRF의 rank 기반 결합 특성을 유지하는 추가 실험에서는 카테고리
리스트 자체에 신뢰도 가중치를 부여한다. 한 카테고리의 top result
하나에 과도하게 의존하지 않도록 top-5 의미 유사도 평균을 사용한다.

```text
semantic_evidence(q,c) = mean(top 5 s_norm values in category c)

w(q,c) = lambda_c * semantic_evidence(q,c)
       + (1 - lambda_c) * P_calibrated(c | q)

WRRF(d) = sum_c w(q,c) / (60 + rank_c(d))
```

이 구조에서는 `lambda_c`가 개별 문서의 mixture 계수가 아니라,
카테고리 검색 리스트가 전역 순위에 행사하는 투표 강도의 mixture
계수로 해석된다. 그러므로 WRRF는 원래 score 식을 직접 검증하는
primary 모델이 아니라 fusion 방식에 대한 sensitivity analysis이다.

### 최종 variant와 튜닝 통제

| Variant | Routing | Fusion / scoring | 역할 |
|---|---|---|---|
| `B0` | 없음, 통합 index | similarity 전역 순위 | naive baseline |
| `B1` | query top-1 category | similarity 파티션 순위 | hard-routing baseline |
| `B2-score` | soft routing | fixed-lambda score merge | primary comparator |
| `P-score` | soft routing | adaptive-lambda score merge | primary proposal |
| `B2-wrrf` | soft routing | fixed-lambda Weighted RRF | secondary comparator |
| `P-wrrf` | soft routing | adaptive-lambda Weighted RRF | secondary proposal |

모든 soft-routing variant는 같은 `K_ingest`와 `theta_route`를 사용한다.
먼저 `B2-score`의 dev `nDCG@10` 기준으로 두 threshold를 선택해
고정하고, `lambda_fixed`는 `{0.0, 0.1, ..., 1.0}` grid 중 dev에서
선택한다. adaptive 모델의

```text
lambda_c = sigmoid(alpha * (mu_c - tau) - rho * sigma_c)
```

파라미터 `alpha`, `rho`, `tau`도 dev에서만 제한된 탐색으로 정한다.
어떤 설정도 test 결과를 확인한 뒤 변경하지 않는다.

## 학교 공지사항 데이터셋 청크 분할

### 입력 레코드와 현재 audit

실험의 원천 파일은 프로젝트 상위에 보존된 `scatch_notices.jsonl`이다.
한 줄은 하나의 웹 공지 원문이며, 다음 필드를 가진다.

| 필드 | 용도 |
|---|---|
| `id` | 고정 `source_id`; split과 lineage의 기본 키 |
| `url`, `slug` | 원문 추적 및 중복 검사 |
| `title` | 청크 metadata 및 제한된 context prefix |
| `date` | 결과 provenance 및 시점별 분석 후보 |
| `category` | 사이트 제공 category; primary gold label 후보 |
| `text` | HTML에서 추출된 본문; 청킹 대상 |
| `text_length` | 입력 무결성 audit용; loader가 다시 계산해 비교 |
| `source`, `collected_at` | 수집 lineage; retrieval scoring 입력에는 사용하지 않음 |

Phase 1에서 원본을 읽기 전용으로 검사한 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 전체 레코드 | 12,749 |
| invalid JSON | 0 |
| duplicate `id` / duplicate `url` | 0 / 0 |
| `text_length` 불일치 | 0 |
| 본문이 제목과 동일한 레코드 | 641 (5.03%) |
| 본문 문자 길이 median / p95 / max | 659 / 2,142 / 99,553 |

사이트 category 분포는 다음과 같다.

| Primary category 후보 | 원문 수 |
|---|---:|
| `채용` | 2,770 |
| `장학` | 2,239 |
| `비교과·행사` | 1,898 |
| `학사` | 1,857 |
| `봉사` | 1,170 |
| `국제교류` | 980 |
| `외국인유학생` | 508 |
| `교원채용` | 406 |

`기타` 666건, `미분류` 217건, `교직` 38건은 의미 경계가 모호하거나
표본이 적으므로 primary study에서 제외한다. 필터 적용 이후에도
primary category별 usable source document 수가 100건 이상인지 다시
검증하며, 부족하면 primary category에서 제외하고 manifest에 사유를
기록한다.

### 청킹에 사용하는 모델

| 역할 | 모델 | 선택 이유 |
|---|---|---|
| 토큰 budget 및 category classifier | [`klue/bert-base`](https://huggingface.co/klue/bert-base) | 한국어 BERT 분류 모델이며 최종 classifier 입력 한도 기준을 제공 |
| dense retrieval embedding | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | 다국어 dense retrieval 및 긴 문맥 지원으로 공지의 의미 단위를 보존 |

기존 `Exp1/src/chunking.py` 계열은 `klue/bert-base` tokenizer를
기준으로 본문 목표 384 tokens, overlap 64 tokens를 적용했다. 이
문단/문장 우선 전략은 유지한다. 반면 짧은 최대 입력 길이를 갖는
경량 sentence encoder를 384-token 청크와 결합하면 검색 임베딩에서
뒤쪽 정보가 truncate될 수 있으므로, `Exp2` retrieval 기본 embedder는
`BAAI/bge-m3`로 고정한다.

### 품질 필터와 분할 절차

청킹 전에 원문 수준의 품질 필터를 적용한다.

1. `text`가 없거나 정규화 후 비어 있는 원문은 제외한다.
2. 본문이 제목과 동일한 원문은 첨부 의존 또는 정보량 부족 대상으로
   제외 리포트에 남긴다.
3. 제목 중복 prefix를 제거한 본문이 `klue/bert-base` tokenizer 기준
   30 tokens 미만이면 primary corpus에서 제외한다.
4. `category`가 primary category 목록에 포함되지 않으면 제외 리포트
   대상이며, 학습/검색 artifact에는 넣지 않는다.
5. 필터 결과는 source-level counts와 사유별 counts로 JSON 및 Markdown
   audit report에 기록한다.

통과한 원문은 다음 순서로 청킹한다.

1. 제목은 metadata에 원형 보존한다. 모델 입력용 title prefix는 최대
   64 tokens이고, 본문이 이미 제목으로 시작하면 다시 붙이지 않는다.
2. 본문을 빈 줄 기준 paragraph로 나눈다.
3. paragraph가 body budget보다 길면 한국어 문장 종결 경계를
   우선하여 문장 단위로 누적한다.
4. 각 청크의 목표 본문 길이는 KLUE tokenizer 기준 `384` tokens이다.
5. 인접 청크는 이전 본문의 마지막 `64` tokens를 overlap으로 가진다.
6. 제목 prefix, separator, special tokens를 포함한 classifier 입력
   전체가 `512` tokens를 넘으면 body를 줄여 다시 생성한다.
7. 문장 자체가 허용 body budget을 넘는 경우에만 tokenizer 기반
   sliding window로 분리한다.

산출물의 핵심 계약은 다음과 같다.

```json
{
  "chunk_id": "scatch_...::0000",
  "source_id": "scatch_...",
  "chunk_index": 0,
  "title": "공지 제목",
  "text": "청크 본문",
  "gold_category": "학사",
  "token_count": 384,
  "url": "https://...",
  "date": "2026-01-01",
  "source": "scatch"
}
```

`chunks.jsonl`은 검수/annotation에 사용하고, 동일 내용을 담은
`chunks.parquet`은 임베딩, 인덱싱, 집계 작업에서 사용한다.

## Ablation Study 데이터셋 분할

### 서로 다른 두 분할 문제

이 실험은 두 종류의 평가 단위를 가진다.

1. category classifier는 원문에서 만들어진 청크를 분류한다.
2. RAG experiment는 사용자의 query에 대해 관련 청크를 찾고 답을
   생성한다.

같은 공지에서 만들어진 overlap 청크들은 독립 표본이 아니므로,
classifier split을 청크 단위로 수행하면 거의 같은 문장이 train과
validation/test에 동시에 들어가는 leakage가 발생한다. 따라서 모든
corpus split은 먼저 `source_id` 단위로 만들고, 청킹 artifact에 해당
split label을 전달한다.

### Classifier 공식 artifact: source-level crossfit

공식 인덱스와 `lambda_c` 통계는 stratified 5-fold crossfit으로
생성한다.

1. 품질 필터를 통과한 primary source documents를 category별 비율을
   유지하며 5 folds로 분리한다.
2. 각 반복에서 4 folds로 `klue/bert-base` classifier를 학습한다.
3. 학습 자료 내부의 source-level calibration subset으로 temperature
   scaling 값 `T*`를 구한다.
4. 학습에 포함되지 않은 held-out fold에만 calibrated category
   probability를 기록한다.
5. 모든 fold의 held-out prediction을 합쳐 모든 원문/청크에 대한
   out-of-fold probability table을 만든다.

이 out-of-fold 확률만 다음 artifact 생성에 사용한다.

- `P_calibrated(c | d) >= K_ingest`인 복수 category index에 동일
  `chunk_id`를 등록하는 partition assignment
- 어떤 threshold도 만족하지 못하는 문서의 top-1 fallback category
- category별 `mu_c`, `sigma_c`, adaptive `lambda_c`

질의 분류용 최종 classifier는 admitted corpus 전체에서 별도 학습하고
calibration을 적용한다. 개발 중 end-to-end wiring을 빨리 검증하기
위해 단일 모델의 in-sample 확률을 쓰는 `single` smoke mode도
구현하지만, 공식 결과표에는 `crossfit` artifacts만 사용할 수 있다.

### RAG query dev/test 분할

현재 primary category 수는 `C=8`이다. query annotation 목표량은
다음과 같이 고정한다.

| Split | 계산식 | 현재 목표 | 용도 |
|---|---:|---:|---|
| `queries_dev.jsonl` | `max(60, 10*C)` | 80 | threshold, fixed lambda, adaptive parameter 선택 |
| `queries_test.jsonl` | `max(200, 30*C)` | 240 | 최종 retrieval/generation/statistical reporting |

각 query row는 다음 필드를 갖는다.

```json
{
  "query_id": "q0001",
  "query": "이번 학기 수강신청 변경 기간은 언제인가요?",
  "gold_chunks": ["scatch_...::0001"],
  "reference_answer": "수강신청 변경 기간은 ...",
  "gold_categories": ["학사"],
  "query_type": "single_category"
}
```

- 질의의 약 30%는 `multi_category` 또는 `ambiguous`로 작성한다.
- `gold_chunks`와 `reference_answer`는 사람이 확정한다.
- dev query만 `K_ingest`, `theta_route`, `lambda_fixed`,
  `alpha/rho/tau`, model/prompt sanity selection에 사용한다.
- test query는 최종 frozen run에만 사용하고 설정을 변경하는 근거로
  사용하지 않는다.
- 별도의 `queries_train`은 현재 도입하지 않는다. 향후 query-level
  learn-to-rank 모델을 추가할 때만 필요한 split이다.

### 평가 통제와 통계

- 모든 variant는 같은 청크 corpus, 같은 embedding, 같은 query
  classifier, 같은 candidate depth, 같은 query 순서와 seed를 사용한다.
- retrieval report는 공통 Top-10을 사용하고, generation context는
  모든 variant에서 공통 Top-5로 고정한다.
- primary 비교는 test query별 `P-score` 대 `B2-score` paired
  difference에 대해 Wilcoxon signed-rank test와 paired bootstrap
  95% confidence interval을 보고한다.
- `P-wrrf` 대 `B2-wrrf`, fusion family 간 결과, confidence/category
  breakdown은 secondary analysis이며 Holm correction을 적용한다.
- latency 주 분석은 LLM 생성 시간을 제외한 retrieval path이다.

## RAG 생성 LLM 후보와 추천

### 실행 제약

실험 머신은 Apple M5 Pro와 24 GB unified memory를 가진 MacBook
Pro이다. 목적은 생성 모델 자체의 경쟁이 아니라 retrieval variant의
비교이므로, 모든 variant는 동일 로컬 generator, 동일 prompt
template, 동일 decoding parameter와 동일 Top-5 context를 사용한다.
외부 API는 버전 변경과 네트워크 변동을 추가하므로 본 평가에서
사용하지 않는다.

### 후보 비교

| 후보 | 장점 | 제약 | 사용 위치 |
|---|---|---|---|
| [`mlx-community/Qwen3-8B-4bit`](https://huggingface.co/mlx-community/Qwen3-8B-4bit) | MLX 4-bit artifact 제공, Apple Silicon에서 직접 실행 가능, Apache-2.0 | 본 실행 전에 한국어 context-following sanity check 필요 | **기본 generator** |
| [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) | MLX 4-bit instruction 모델, Apache-2.0, 더 보수적인 fallback | 공식 metadata에 한국어 특화 표기가 약함 | fallback |
| [`LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct) | 공식 metadata에 한국어 지원 명시 | license 조건과 MLX/양자화 실행 경로를 먼저 검증해야 함 | optional comparison |

기본 선택은 `mlx-community/Qwen3-8B-4bit`이다. 8B 규모의 생성 능력을
유지하면서 M5 MacBook Pro에서 실행할 수 있는 MLX 4-bit artifact가
이미 제공되고, Apache-2.0 라이선스로 실험 재현 절차를 명확히 기록할
수 있기 때문이다.

### 고정 generation protocol

- generator: `mlx-community/Qwen3-8B-4bit`
- decoding: greedy, `temperature=0.0`
- reasoning format: retrieval 비교를 위한 최종 답변만 출력하도록
  non-thinking prompt mode 사용
- context: 모든 variant에서 검색 결과 Top-5만 제공
- output: `(run_manifest_hash, variant, query_id)` 키로 캐시
- generation metrics: Exact Match, token F1, ROUGE, BERTScore
- excluded metric: 이번 primary/secondary 결과에서는 LLM-as-judge
  사용하지 않음

## 구현 산출물과 재현성 계약

`Exp2`는 다음 단계에서 생성될 artifact가 바뀌지 않도록 각 실행의
설정, 모델 ID, 데이터 hash, seed, 선택된 parameter를 manifest에
기록한다.

```text
data export -> audit/admission -> chunks -> crossfit classifier outputs
            -> embeddings and FAISS indexes -> dev-frozen settings
            -> test retrieval runs -> cached generation -> reports
```

세부 파일·명령·검증 순서는 `docs/IMPLEMENTATION-STEPS.md`를 공식
실행 체크리스트로 사용한다.
