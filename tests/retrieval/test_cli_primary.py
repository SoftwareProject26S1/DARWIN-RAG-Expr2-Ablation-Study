import json

from darwin_rag_exp2 import cli
from darwin_rag_exp2.retrieval.types import SearchHit


class CliSearchBackend:
    def __init__(self, indexes_dir) -> None:
        self.indexes_dir = indexes_dir

    def search_unified(self, query_embedding: list[float], *, top_k: int) -> list[SearchHit]:
        return [SearchHit("c1", "s1", "학사", 0.9, 1)][:top_k]

    def search_category(
        self,
        category: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        if category == "학사":
            return [SearchHit("c1", "s1", "학사", 0.9, 1)][:top_k]
        return []


def test_run_primary_cli_writes_four_variant_rows_with_precomputed_probabilities(
    tmp_path,
    monkeypatch,
) -> None:
    queries_path = tmp_path / "queries.jsonl"
    settings_path = tmp_path / "frozen.yaml"
    indexes_path = tmp_path / "indexes"
    output_path = tmp_path / "run"
    indexes_path.mkdir()
    queries_path.write_text(
        (
            '{"query_id":"test_q0001","query":"수강신청 변경 기간은?",'
            '"gold_chunks":["c1"],"reference_answer":"3월입니다.",'
            '"gold_categories":["학사"],"query_type":"single_category",'
            '"probabilities":{"학사":0.9,"장학":0.1}}\n'
        ),
        encoding="utf-8",
    )
    settings_path.write_text(
        "\n".join(
            [
                "candidate_k_per_partition: 2",
                "report_top_k: 1",
                "generation_context_top_n: 1",
                "theta_route: 0.6",
                "lambda_fixed: 0.5",
                "lambda_by_category:",
                "  학사: 0.8",
                "  장학: 0.7",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "FaissSearchBackend", CliSearchBackend)

    result = cli.main(
        [
            "run-primary",
            "--queries",
            str(queries_path),
            "--settings",
            str(settings_path),
            "--indexes",
            str(indexes_path),
            "--output",
            str(output_path),
            "--embedding-backend",
            "hash",
        ]
    )

    rows = [
        json.loads(line)
        for line in (output_path / "results.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((output_path / "manifest.json").read_text())

    assert result == 0
    assert len(rows) == 4
    assert {row["variant"] for row in rows} == {
        "B0",
        "B1",
        "B2-score",
        "P-score",
    }
    assert manifest["query_count"] == 1


def test_tune_primary_cli_writes_frozen_settings(tmp_path, monkeypatch) -> None:
    queries_path = tmp_path / "queries_dev.jsonl"
    indexes_path = tmp_path / "indexes"
    output_path = tmp_path / "settings"
    category_stats_path = tmp_path / "category_stats.json"
    indexes_path.mkdir()
    queries_path.write_text(
        (
            '{"query_id":"dev_q0001","query":"수강신청 변경 기간은?",'
            '"gold_chunks":["c1"],"reference_answer":"3월입니다.",'
            '"gold_categories":["학사"],"query_type":"single_category",'
            '"probabilities":{"학사":0.9,"장학":0.1}}\n'
        ),
        encoding="utf-8",
    )
    category_stats_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "category": "학사",
                        "mu_confidence": 0.9,
                        "sigma_confidence": 0.1,
                    },
                    {
                        "category": "장학",
                        "mu_confidence": 0.7,
                        "sigma_confidence": 0.4,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "FaissSearchBackend", CliSearchBackend)

    cli.main(
        [
            "tune-primary",
            "--queries",
            str(queries_path),
            "--indexes",
            str(indexes_path),
            "--output",
            str(output_path),
            "--category-stats",
            str(category_stats_path),
            "--embedding-backend",
            "hash",
            "--theta-grid",
            "0.6",
            "--fixed-lambda-grid",
            "0.5",
        ]
    )

    settings_payload = (output_path / "frozen.yaml").read_text(encoding="utf-8")
    diagnostics = json.loads((output_path / "tuning.json").read_text(encoding="utf-8"))

    assert "lambda_fixed: 0.5" in settings_payload
    assert "theta_route: 0.6" in settings_payload
    assert diagnostics["best_variant"] == "B2-score"
