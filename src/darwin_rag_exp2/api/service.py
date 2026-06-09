"""Request-time P-score RAG service for the REST API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from darwin_rag_exp2.indexing.embeddings import EmbeddingModel, l2_normalize
from darwin_rag_exp2.retrieval.types import (
    PrimaryRunSettings,
    QueryFeatures,
    RankedChunk,
    SearchBackend,
)
from darwin_rag_exp2.retrieval.variants import run_p_score


class MessageValidationError(ValueError):
    """Raised when a message request cannot be served."""


class QueryClassifier(Protocol):
    """Minimal classifier interface used by the API service."""

    def predict_probabilities(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return category probabilities for each input text."""


class AnswerGenerator(Protocol):
    """Minimal local LLM interface used by the API service."""

    def generate(self, prompt: str) -> str:
        """Generate an answer from an augmented prompt."""


@dataclass(frozen=True)
class RetrievedContext:
    """Hydrated retrieval context passed into the answer prompt."""

    chunk_id: str
    title: str
    category: str
    date: str
    url: str
    body_text: str


class ChunkStore:
    """In-memory chunk lookup keyed by chunk_id."""

    def __init__(self, rows_by_chunk_id: Mapping[str, Mapping[str, object]]) -> None:
        self._rows_by_chunk_id = {
            str(chunk_id): dict(row)
            for chunk_id, row in rows_by_chunk_id.items()
        }

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, object]]) -> "ChunkStore":
        return cls({str(row["chunk_id"]): row for row in rows})

    @classmethod
    def from_parquet(cls, path) -> "ChunkStore":
        import pyarrow.parquet as pq

        return cls.from_rows(pq.read_table(path).to_pylist())

    def contexts_for(self, ranked_chunks: Sequence[RankedChunk]) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        for ranked in ranked_chunks:
            row = self._rows_by_chunk_id.get(ranked.chunk_id)
            if row is None:
                raise ValueError(f"missing chunk text for chunk_id {ranked.chunk_id!r}")
            contexts.append(
                RetrievedContext(
                    chunk_id=ranked.chunk_id,
                    title=str(row.get("title", "")),
                    category=str(row.get("category", "")),
                    date=str(row.get("date", "")),
                    url=str(row.get("url", "")),
                    body_text=str(row.get("body_text", "")),
                )
            )
        return contexts


class MessageService:
    """Run one query through P-score retrieval and local LLM generation."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        classifier: QueryClassifier,
        search_backend: SearchBackend,
        settings: PrimaryRunSettings,
        chunk_store: ChunkStore,
        generator: AnswerGenerator,
        normalize_embeddings: bool = True,
    ) -> None:
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.search_backend = search_backend
        self.settings = settings
        self.chunk_store = chunk_store
        self.generator = generator
        self.normalize_embeddings = normalize_embeddings

    def answer(self, query: str) -> str:
        """Return an LLM answer augmented by P-score retrieval contexts."""

        clean_query = query.strip()
        if not clean_query:
            raise MessageValidationError("query must not be blank")

        embedding = self._embed_query(clean_query)
        probabilities = self._classify_query(clean_query)
        query_features = QueryFeatures(
            query_id="api-message",
            query=clean_query,
            embedding=embedding,
            probabilities=probabilities,
        )
        retrieval = run_p_score(
            query_features,
            search_backend=self.search_backend,
            settings=self.settings,
        )
        contexts = self.chunk_store.contexts_for(retrieval.top5_contexts)
        prompt = build_augmented_prompt(clean_query, contexts)
        return self.generator.generate(prompt).strip()

    def _embed_query(self, query: str) -> list[float]:
        vectors = self.embedding_model.encode([query])
        if len(vectors) != 1:
            raise ValueError("embedding model must return exactly one query vector")
        if self.normalize_embeddings:
            vectors = l2_normalize(vectors)
        return [float(value) for value in vectors[0]]

    def _classify_query(self, query: str) -> dict[str, float]:
        predictions = self.classifier.predict_probabilities([query])
        if len(predictions) != 1:
            raise ValueError("query classifier must return exactly one probability row")
        probabilities = {
            str(category): float(probability)
            for category, probability in predictions[0].items()
        }
        if not probabilities:
            raise ValueError("query classifier returned no category probabilities")
        return probabilities


def build_augmented_prompt(
    query: str,
    contexts: Sequence[RetrievedContext],
) -> str:
    """Build the Korean answer prompt from retrieved contexts."""

    context_blocks = "\n\n".join(
        _context_block(index, context)
        for index, context in enumerate(contexts, start=1)
    )
    if not context_blocks:
        context_blocks = "검색된 근거가 없습니다."
    return "\n".join(
        [
            "너는 숭실대학교 공지사항 기반 질의응답 assistant다.",
            "아래 검색된 근거만 사용해서 한국어로 답변하라.",
            "근거에 없는 내용은 추측하지 말고 확인이 필요하다고 말하라.",
            "",
            "사용자 질문:",
            query,
            "",
            "검색된 근거:",
            context_blocks,
            "",
            "최종 답변:",
        ]
    )


def _context_block(index: int, context: RetrievedContext) -> str:
    body_text = context.body_text.strip()
    return "\n".join(
        [
            f"[{index}] 제목: {context.title}",
            f"카테고리: {context.category}",
            f"날짜: {context.date}",
            f"URL: {context.url}",
            f"내용: {body_text}",
        ]
    )
