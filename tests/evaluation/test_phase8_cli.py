from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2 import cli


def test_phase8_cli_validates_queries_and_exports_pool(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev.jsonl"
    test_path = tmp_path / "queries_test.jsonl"
    validation_output = tmp_path / "query-validation"
    pool_output = tmp_path / "query-pool"
    _write_config(config_path)
    _write_chunks(chunks_path)
    _write_queries(dev_path, "dev_q0001")
    _write_queries(test_path, "test_q0001")

    validate_result = cli.main(
        [
            "validate-queries",
            "--dev",
            str(dev_path),
            "--test",
            str(test_path),
            "--chunks",
            str(chunks_path),
            "--config",
            str(config_path),
            "--output",
            str(validation_output),
        ]
    )
    pool_result = cli.main(
        [
            "export-query-pool",
            "--chunks",
            str(chunks_path),
            "--config",
            str(config_path),
            "--output",
            str(pool_output),
            "--per-category",
            "1",
            "--seed",
            "42",
        ]
    )

    assert validate_result == 0
    assert pool_result == 0
    assert (validation_output / "validation.json").exists()
    assert (pool_output / "query_pool.jsonl").exists()
    validation = json.loads(
        (validation_output / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["valid"] is True


def _write_config(path) -> None:
    path.write_text(
        "\n".join(
            [
                "experiment:",
                "  primary_categories:",
                '    - "학사"',
                "data:",
                "  query_type_non_single_fraction: 0.0",
                "evaluation:",
                "  dev_queries: 1",
                "  test_queries: 1",
            ]
        ),
        encoding="utf-8",
    )


def _write_queries(path, query_id: str) -> None:
    row = {
        "query_id": query_id,
        "query": "수강신청 변경 기간은 언제인가요?",
        "gold_chunks": ["c1"],
        "reference_answer": "수강신청 변경 기간은 공지된 기간을 따릅니다.",
        "gold_categories": ["학사"],
        "query_type": "single_category",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_chunks(path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "chunk_id": "c1",
                "source_id": "s1",
                "category": "학사",
                "title": "수강신청 변경 안내",
                "body_text": "수강신청 변경 기간과 절차를 안내합니다.",
                "date": "2026-06-01",
                "url": "https://example.test/notice/1",
            }
        ]
    )
    pq.write_table(table, path)
