from darwin_rag_exp2.retrieval.types import (
    PrimaryRunSettings,
    QueryFeatures,
    SearchHit,
)
from darwin_rag_exp2.retrieval.variants import run_primary_variants


class FakeSearchBackend:
    def search_unified(self, query_embedding: list[float], *, top_k: int) -> list[SearchHit]:
        return [
            SearchHit(
                chunk_id="c3",
                source_id="s3",
                source_category="비교과·행사",
                similarity=0.9,
                rank=1,
            ),
            SearchHit(
                chunk_id="c1",
                source_id="s1",
                source_category="학사",
                similarity=0.7,
                rank=2,
            ),
        ][:top_k]

    def search_category(
        self,
        category: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        hits_by_category = {
            "학사": [
                SearchHit(
                    chunk_id="c1",
                    source_id="s1",
                    source_category="학사",
                    similarity=0.8,
                    rank=1,
                ),
                SearchHit(
                    chunk_id="c2",
                    source_id="s2",
                    source_category="장학",
                    similarity=0.5,
                    rank=2,
                ),
            ],
            "장학": [
                SearchHit(
                    chunk_id="c2",
                    source_id="s2",
                    source_category="장학",
                    similarity=0.9,
                    rank=1,
                ),
                SearchHit(
                    chunk_id="c1",
                    source_id="s1",
                    source_category="학사",
                    similarity=0.6,
                    rank=2,
                ),
            ],
        }
        return hits_by_category.get(category, [])[:top_k]


class TrackingSearchBackend(FakeSearchBackend):
    def __init__(self) -> None:
        self.unified_top_k: list[int] = []
        self.category_calls: list[str] = []

    def search_unified(self, query_embedding: list[float], *, top_k: int) -> list[SearchHit]:
        self.unified_top_k.append(top_k)
        return [
            SearchHit(
                chunk_id="c1",
                source_id="s1",
                source_category="학사",
                similarity=0.9,
                rank=1,
            ),
            SearchHit(
                chunk_id="c2",
                source_id="s2",
                source_category="장학",
                similarity=0.85,
                rank=2,
            ),
        ][:top_k]

    def search_category(
        self,
        category: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        self.category_calls.append(category)
        return super().search_category(category, query_embedding, top_k=top_k)


def test_run_primary_variants_applies_variant_specific_routing_and_scoring() -> None:
    query = QueryFeatures(
        query_id="dev_q0001",
        query="수강신청과 장학 일정을 같이 알려줘",
        embedding=[1.0, 0.0],
        probabilities={"학사": 0.75, "장학": 0.62, "국제교류": 0.1},
        gold_chunks=("c1",),
        gold_categories=("학사",),
        query_type="single_category",
    )
    settings = PrimaryRunSettings(
        candidate_k_per_partition=2,
        report_top_k=2,
        generation_context_top_n=1,
        theta_route=0.6,
        lambda_fixed=0.5,
        lambda_by_category={"학사": 0.1, "장학": 0.95},
    )

    results = run_primary_variants(
        query,
        search_backend=FakeSearchBackend(),
        settings=settings,
    )

    assert list(results) == ["B0", "B1", "B2-score", "P-score"]
    assert [item.chunk_id for item in results["B0"].top10] == ["c3", "c1"]
    assert [item.chunk_id for item in results["B1"].top10] == ["c1", "c2"]
    assert results["B2-score"].top10[0].chunk_id == "c1"
    assert results["P-score"].top10[0].chunk_id == "c2"
    assert results["P-score"].top5_contexts == results["P-score"].top10[:1]


def test_unified_prior_rerank_uses_unified_candidates_for_b2_and_p() -> None:
    query = QueryFeatures(
        query_id="dev_q0001",
        query="수강신청과 장학 일정을 같이 알려줘",
        embedding=[1.0, 0.0],
        probabilities={"학사": 0.2, "장학": 0.9},
        gold_chunks=("c2",),
        gold_categories=("장학",),
        query_type="single_category",
    )
    settings = PrimaryRunSettings(
        candidate_k_per_partition=2,
        report_top_k=2,
        generation_context_top_n=1,
        theta_route=0.6,
        lambda_fixed=0.5,
        lambda_by_category={"학사": 0.9, "장학": 0.1},
    )
    backend = TrackingSearchBackend()

    results = run_primary_variants(
        query,
        search_backend=backend,
        settings=settings,
        search_mode="unified-prior-rerank",
        unified_candidate_k=100,
    )

    assert backend.unified_top_k == [2, 100, 100]
    assert backend.category_calls == ["장학"]
    assert results["B2-score"].top10[0].scoring_method == "unified-prior-rerank"
    assert results["B2-score"].top10[0].partition_category == "장학"
    assert results["P-score"].top10[0].chunk_id == "c2"
