# DARWIN-RAG Exp2 Working Rules

이 문서는 DARWIN-RAG ablation study 구현과 검증 작업을 수행하는 모든
작업자와 자동화 에이전트가 따라야 할 프로젝트 규칙을 정의한다.

## 작업 경로

- Ablation Study와 관련된 모든 코드, 설정, 문서, 테스트, 입력 데이터,
  실험 산출물 작업은 반드시 이 `Exp2` 저장소 내부에서 수행한다.
- `Exp1`은 별개의 실험 저장소이며 Exp2 작업 범위에 포함하지 않는다.
- 원본 입력인 `data/raw/scatch_notices.jsonl`은 Exp2가 소유하는
  read-only baseline 입력이다. 생성되는 정제 데이터와 실험 결과는
  `artifacts/` 또는 `runs/` 경로에 저장한다.

## 단계별 브랜치 규칙

모든 단계별 작업은 구현을 시작하기 전에 별도의 Git 브랜치로
분리한다. 한 브랜치에는 한 단계 또는 그 단계의 명확히 구분되는
작업만 포함한다.

요청된 표기 의도는 다음과 같다.

```text
[task type]: exp2-{step}-{taskname}
```

그러나 Git ref 이름에는 `:`와 공백을 사용할 수 없으므로, 실제 브랜치
이름은 다음 canonical 형식을 사용한다.

```text
<task-type>/exp2-<step>-<taskname>
```

예시:

```text
feat/exp2-phase1-scaffold
feat/exp2-phase2-data-audit
fix/exp2-phase4-token-cap
docs/exp2-phase1-protocol
```

- `<task-type>`은 변경 성격에 맞는 Conventional Commit type을 사용한다.
  기본 후보는 `feat`, `fix`, `docs`, `test`, `refactor`, `chore`이다.
- `<step>`은 `phase1`부터 `phase12`까지의 단계 식별자를 사용한다.
- `<taskname>`은 영문 소문자와 하이픈으로 간결하게 작성한다.
- 현재 Phase 1 브랜치 `feat/exp2-phase1-scaffold`는 이 canonical
  형식을 따른다.

## 커밋 규칙

모든 작업 커밋은 Conventional Commit 형식을 사용한다.

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
