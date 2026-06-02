import json

from darwin_rag_exp2.retrieval.runner import run_primary_queries, write_primary_run
from darwin_rag_exp2.retrieval.types import PrimaryRunSettings, QueryFeatures, SearchHit


class OneQuerySearchBackend:
    def search_unified(self, query_embedding: list[float], *, top_k: int) -> list[SearchHit]:
        return [
            SearchHit("c1", "s1", "학사", 0.9, 1),
            SearchHit("c2", "s2", "장학", 0.7, 2),
        ][:top_k]

    def search_category(
        self,
        category: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        if category == "학사":
            return [SearchHit("c1", "s1", "학사", 0.9, 1)]
        if category == "장학":
            return [SearchHit("c2", "s2", "장학", 0.8, 1)]
        return []


def test_primary_runner_writes_variant_rows_metrics_and_manifest(tmp_path) -> None:
    settings = PrimaryRunSettings(
        candidate_k_per_partition=2,
        report_top_k=2,
        generation_context_top_n=1,
        theta_route=0.6,
        lambda_fixed=0.5,
        lambda_by_category={"학사": 0.8, "장학": 0.7},
    )
    queries = [
        QueryFeatures(
            query_id="test_q0001",
            query="수강신청 변경 기간은 언제야?",
            embedding=[1.0, 0.0],
            probabilities={"학사": 0.9, "장학": 0.7},
            gold_chunks=("c1",),
            gold_categories=("학사",),
            query_type="single_category",
        )
    ]

    rows = run_primary_queries(
        queries,
        search_backend=OneQuerySearchBackend(),
        settings=settings,
    )
    write_primary_run(
        output_dir=tmp_path,
        result_rows=rows,
        settings=settings,
        run_metadata={"queries_path": "queries_test.jsonl"},
    )

    result_lines = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert len(result_lines) == 4
    assert {row["variant"] for row in result_lines} == {
        "B0",
        "B1",
        "B2-score",
        "P-score",
    }
    assert result_lines[0]["metrics"]["hit@2"] == 1.0
    assert result_lines[0]["top10"][0]["chunk_id"] == "c1"
    assert result_lines[0]["top5_contexts"][0]["chunk_id"] == "c1"
    assert manifest["query_count"] == 1
    assert manifest["variant_count"] == 4
    assert manifest["settings"]["theta_route"] == 0.6
