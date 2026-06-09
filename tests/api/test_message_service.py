import pytest

from darwin_rag_exp2.retrieval.types import PrimaryRunSettings, SearchHit


class FakeEmbeddingModel:
    def encode(self, texts):
        assert texts == ["수강신청 기간은 언제야?"]
        return [[3.0, 4.0]]


class FakeClassifier:
    def predict_probabilities(self, texts):
        assert texts == ["수강신청 기간은 언제야?"]
        return [{"학사": 0.8, "장학": 0.3}]


class PScoreOnlySearchBackend:
    def __init__(self):
        self.category_calls = []

    def search_unified(self, query_embedding, *, top_k):
        raise AssertionError("message API must use P-score category search only")

    def search_category(self, category, query_embedding, *, top_k):
        self.category_calls.append(category)
        if category == "학사":
            return [SearchHit("c1", "s1", "학사", 0.91, 1)]
        if category == "장학":
            return [SearchHit("c2", "s2", "장학", 0.88, 1)]
        return []


class PromptCapturingGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "  최종 답변입니다.  "


def _settings():
    return PrimaryRunSettings(
        candidate_k_per_partition=2,
        report_top_k=2,
        generation_context_top_n=2,
        theta_route=0.2,
        lambda_fixed=1.0,
        lambda_by_category={"학사": 0.9, "장학": 0.7},
    )


def test_message_service_runs_p_score_and_generates_from_retrieved_contexts():
    from darwin_rag_exp2.api.service import ChunkStore, MessageService

    search_backend = PScoreOnlySearchBackend()
    generator = PromptCapturingGenerator()
    service = MessageService(
        embedding_model=FakeEmbeddingModel(),
        classifier=FakeClassifier(),
        search_backend=search_backend,
        settings=_settings(),
        chunk_store=ChunkStore.from_rows(
            [
                {
                    "chunk_id": "c1",
                    "title": "수강신청 변경 안내",
                    "category": "학사",
                    "date": "2026-03-01",
                    "url": "https://example.test/academic",
                    "body_text": "수강신청 변경 기간은 3월 4일부터 3월 8일까지입니다.",
                },
                {
                    "chunk_id": "c2",
                    "title": "장학 신청 안내",
                    "category": "장학",
                    "date": "2026-03-02",
                    "url": "https://example.test/scholarship",
                    "body_text": "장학 신청은 별도 공지를 확인해야 합니다.",
                },
            ]
        ),
        generator=generator,
        normalize_embeddings=True,
    )

    answer = service.answer("  수강신청 기간은 언제야?  ")

    assert answer == "최종 답변입니다."
    assert search_backend.category_calls == ["학사", "장학"]
    prompt = generator.prompts[0]
    assert "사용자 질문:\n수강신청 기간은 언제야?" in prompt
    assert "[1] 제목: 수강신청 변경 안내" in prompt
    assert "수강신청 변경 기간은 3월 4일부터 3월 8일까지입니다." in prompt
    assert "[2] 제목: 장학 신청 안내" in prompt


def test_message_service_rejects_blank_query():
    from darwin_rag_exp2.api.service import ChunkStore, MessageService, MessageValidationError

    service = MessageService(
        embedding_model=FakeEmbeddingModel(),
        classifier=FakeClassifier(),
        search_backend=PScoreOnlySearchBackend(),
        settings=_settings(),
        chunk_store=ChunkStore.from_rows([]),
        generator=PromptCapturingGenerator(),
    )

    with pytest.raises(MessageValidationError, match="query must not be blank"):
        service.answer("   ")
