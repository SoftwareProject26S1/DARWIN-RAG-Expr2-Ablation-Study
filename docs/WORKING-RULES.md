# DARWIN-RAG Exp2 Working Rules

이 문서는 DARWIN-RAG ablation study 구현과 검증 작업을 수행하는 모든
작업자와 자동화 에이전트가 따라야 할 프로젝트 규칙을 정의한다.

## 저장소 기준과 Submodule 동기화

- `Exp2`는 자체 `origin`을 가진 독립 Git 저장소이며, 상위
  `Experiments` 저장소에는 `Exp2` submodule로 포함되어 있다.
- 사용자가 상위 저장소의 gitlink 갱신을 명시적으로 지시하지 않는 한,
  ablation study의 브랜치 생성, 커밋, push, PR은 모두 이 `Exp2`
  저장소와 해당 원격 저장소를 기준으로 수행한다.
- `Exp2`에서 작업 중일 때 상위 저장소가 submodule을 modified로
  표시하는 것은 정상이다. 이를 이유로 진행 중인 Phase 변경을 상위
  저장소 커밋에 포함하지 않는다.
- 다음과 같이 상위 저장소에서 참조할 만한 큰 작업 단위가 완료되고
  `Exp2` 결과가 고정되면 submodule gitlink 동기화 시점으로 보고한다.
  사용자가 상위 저장소 작업을 지시한 경우에만 별도 상위 저장소
  브랜치, 커밋, PR로 gitlink를 갱신한다.

| 동기화 시점 | 완료 범위 |
|---|---|
| 데이터 처리 기반 완료 | Phase 2-4: audit, category/quality filtering, chunk artifact 계약 동결 |
| RAG 검색 구조 완료 | Phase 5-9: 보정 분류기, index, primary retrieval variant 결과 동결 |
| 주 실험 완료 | Phase 10-12: 생성 평가, 통계 보고서, 최종 manifest 동결 |

선택적 Weighted RRF 확장을 수행하고 그 결과를 상위 프로젝트가
참조해야 하는 경우에는, primary study gitlink와 분리된 추가 동기화
시점으로 취급한다.

## 작업 범위와 산출물 경계

- Ablation Study와 관련된 코드, 설정, 문서, 테스트, 입력 데이터,
  실험 산출물 작업은 이 `Exp2` 저장소 내부에서 수행한다.
- `Exp1`은 별개의 실험 저장소이며 `Exp2` 작업 범위에 포함하지 않는다.
- 원본 입력인 `data/raw/scatch_notices.jsonl`은 `Exp2`가 소유하는
  read-only baseline 입력이다. 원본 파일을 수정, 재포맷 또는
  대체하지 않는다.
- 생성되는 정제 데이터, 모델, 인덱스, 실험 실행 결과는
  `artifacts/` 또는 `runs/` 아래에 저장하고, 각 단계 문서에서
  요구하는 manifest와 hash로 원본 및 설정 lineage를 기록한다.

## 단계별 브랜치 규칙

모든 단계별 작업은 `Exp2` 저장소에서 구현을 시작하기 전에 별도의
Git 브랜치로 분리한다. 한 브랜치에는 한 단계 또는 그 단계의
명확히 구분되는 작업만 포함한다.

```text
<task-type>/<step>-<taskname>
```

예시:

```text
feat/phase1-scaffold
feat/phase2-data-audit
fix/phase4-token-cap
docs/phase1-protocol
```

- `<task-type>`은 변경 성격에 맞는 Conventional Commit type을 사용한다.
  기본 후보는 `feat`, `fix`, `docs`, `test`, `refactor`, `chore`이다.
- `<step>`은 `phase1`부터 `phase12`까지의 단계 식별자를 사용한다.
- `<taskname>`은 영문 소문자와 하이픈으로 간결하게 작성한다.
- 브랜치 이름과 커밋 메시지 형식은 이 문서를 canonical source로
  삼으며, 단계별 실행 문서는 이 규칙을 재정의하지 않는다.

## 커밋 규칙

모든 `Exp2` 작업 커밋은 Conventional Commit 형식을 사용한다. 상위
저장소의 submodule gitlink 커밋은 위 동기화 규칙에 따라 별도 작업으로
수행한다.

```text
<commit-type>: <한국어 메시지>
```

- `<commit-type>`은 `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
  등의 Conventional Commit type 중 변경 목적에 맞는 값을 선택한다.
- 메시지 본문인 `<한국어 메시지>`는 항상 한국어로 작성한다.
- 한 커밋은 하나의 검증 가능한 목적을 가져야 하며, 커밋 전 해당
  단계 문서에 명시된 검증을 실행한다.

예시:

```text
feat: Exp2 1단계 스캐폴드와 실험 계획 문서를 추가한다
test: 데이터 감사 통계 회귀 테스트를 추가한다
fix: 청크 토큰 상한 계산 오류를 수정한다
docs: 검색 병합 평가 절차를 명확히 기록한다
```

## 단계 완료 규칙

- 각 단계는 `docs/IMPLEMENTATION-STEPS.md`의 입력, 산출물, 검증
  조건을 기준으로 수행한다.
- 다음 단계 브랜치를 시작하기 전에 현재 단계의 테스트와 문서 검증을
  통과시키고, 생성 artifact 또는 manifest를 고정한다.
- test query 결과를 기준으로 threshold, parameter, prompt 또는 model
  선택을 변경하지 않는다.
