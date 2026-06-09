from __future__ import annotations

import json

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2.evaluation.queries import (
    QueryValidationConfig,
    validate_query_splits,
    write_query_validation_artifacts,
)


PRIMARY_CATEGORIES = ("학사", "장학", "채용")


def test_validate_query_splits_accepts_valid_dev_and_test_files(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev.jsonl"
    test_path = tmp_path / "queries_test.jsonl"
    _write_chunks(chunks_path)
    _write_queries(
        dev_path,
        [
            _query("dev_q0001", "c1", ["학사"], "single_category"),
            _query("dev_q0002", "c2", ["장학", "학사"], "multi_category"),
            _query("dev_q0003", "c3", ["채용"], "ambiguous"),
            _query("dev_q0004", "c1", ["학사"], "single_category"),
        ],
    )
    _write_queries(
        test_path,
        [
            _query("test_q0001", "c1", ["학사"], "single_category"),
            _query("test_q0002", "c2", ["장학", "학사"], "multi_category"),
            _query("test_q0003", "c3", ["채용"], "ambiguous"),
            _query("test_q0004", "c1", ["학사"], "single_category"),
        ],
    )

    report = validate_query_splits(
        dev_path=dev_path,
        test_path=test_path,
        chunks_path=chunks_path,
        config=QueryValidationConfig(
            primary_categories=PRIMARY_CATEGORIES,
            expected_dev_count=4,
            expected_test_count=4,
            non_single_fraction=0.5,
            non_single_tolerance=0.05,
        ),
    )

    assert report["valid"] is True
    assert report["splits"]["dev"]["row_count"] == 4
    assert report["splits"]["test"]["query_type_counts"] == {
        "ambiguous": 1,
        "multi_category": 1,
        "single_category": 2,
    }
    assert len(report["query_hashes"]["queries"]) == 8


def test_validate_query_splits_rejects_extra_fields(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev.jsonl"
    test_path = tmp_path / "queries_test.jsonl"
    _write_chunks(chunks_path)
    _write_queries(
        dev_path,
        [
            {
                **_query("dev_q0001", "c1", ["학사"], "single_category"),
                "probabilities": {"학사": 0.9},
            }
        ],
    )
    _write_queries(test_path, [_query("test_q0001", "c1", ["학사"], "single_category")])

    with pytest.raises(ValueError, match="unexpected fields"):
        validate_query_splits(
            dev_path=dev_path,
            test_path=test_path,
            chunks_path=chunks_path,
            config=QueryValidationConfig(
                primary_categories=PRIMARY_CATEGORIES,
                expected_dev_count=1,
                expected_test_count=1,
                non_single_fraction=0.0,
                non_single_tolerance=0.05,
            ),
        )


@pytest.mark.parametrize(
    ("query_id", "gold_chunk", "gold_categories", "query_type", "message"),
    [
        ("bad_q0001", "c1", ["학사"], "single_category", "must start with dev_q"),
        ("dev_q0001", "missing", ["학사"], "single_category", "unknown gold_chunks"),
        ("dev_q0001", "c1", ["기타"], "single_category", "invalid gold_categories"),
        ("dev_q0001", "c1", ["학사"], "bad_type", "invalid query_type"),
        ("dev_q0001", "c1", ["학사", "장학"], "single_category", "single_category"),
        ("dev_q0001", "c1", ["학사"], "multi_category", "multi_category"),
    ],
)
def test_validate_query_splits_rejects_invalid_rows(
    tmp_path,
    query_id,
    gold_chunk,
    gold_categories,
    query_type,
    message,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev.jsonl"
    test_path = tmp_path / "queries_test.jsonl"
    _write_chunks(chunks_path)
    bad_row = _query(query_id, gold_chunk, gold_categories, query_type)
    _write_queries(dev_path, [bad_row])
    _write_queries(test_path, [_query("test_q0001", "c1", ["학사"], "single_category")])

    with pytest.raises(ValueError, match=message):
        validate_query_splits(
            dev_path=dev_path,
            test_path=test_path,
            chunks_path=chunks_path,
            config=QueryValidationConfig(
                primary_categories=PRIMARY_CATEGORIES,
                expected_dev_count=1,
                expected_test_count=1,
                non_single_fraction=0.0,
                non_single_tolerance=0.05,
            ),
        )


def test_validate_query_splits_rejects_duplicate_ids_across_splits(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev.jsonl"
    test_path = tmp_path / "queries_test.jsonl"
    _write_chunks(chunks_path)
    _write_queries(dev_path, [_query("dev_q0001", "c1", ["학사"], "single_category")])
    _write_queries(test_path, [_query("dev_q0001", "c1", ["학사"], "single_category")])

    with pytest.raises(ValueError, match="overlap"):
        validate_query_splits(
            dev_path=dev_path,
            test_path=test_path,
            chunks_path=chunks_path,
            config=QueryValidationConfig(
                primary_categories=PRIMARY_CATEGORIES,
                expected_dev_count=1,
                expected_test_count=1,
                non_single_fraction=0.0,
                non_single_tolerance=0.05,
            ),
        )


def test_write_query_validation_artifacts_writes_report_files(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev.jsonl"
    test_path = tmp_path / "queries_test.jsonl"
    output_path = tmp_path / "validation"
    _write_chunks(chunks_path)
    _write_queries(dev_path, [_query("dev_q0001", "c1", ["학사"], "single_category")])
    _write_queries(test_path, [_query("test_q0001", "c2", ["장학"], "single_category")])
    report = validate_query_splits(
        dev_path=dev_path,
        test_path=test_path,
        chunks_path=chunks_path,
        config=QueryValidationConfig(
            primary_categories=PRIMARY_CATEGORIES,
            expected_dev_count=1,
            expected_test_count=1,
            non_single_fraction=0.0,
            non_single_tolerance=0.05,
        ),
    )

    write_query_validation_artifacts(output_path, report)

    assert {
        "validation.json",
        "distribution.csv",
        "query_hashes.json",
        "manifest.json",
    }.issubset({path.name for path in output_path.iterdir()})
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "query_validation"


def _query(
    query_id: str,
    gold_chunk: str,
    gold_categories: list[str],
    query_type: str,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": f"{query_id} 일정과 조건을 알려주세요.",
        "gold_chunks": [gold_chunk],
        "reference_answer": "정답 답변입니다.",
        "gold_categories": gold_categories,
        "query_type": query_type,
    }


def _write_queries(path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_chunks(path) -> None:
    table = pa.Table.from_pylist(
        [
            {"chunk_id": "c1", "source_id": "s1", "category": "학사", "title": "학사 공지", "body_text": "본문"},
            {"chunk_id": "c2", "source_id": "s2", "category": "장학", "title": "장학 공지", "body_text": "본문"},
            {"chunk_id": "c3", "source_id": "s3", "category": "채용", "title": "채용 공지", "body_text": "본문"},
        ]
    )
    pq.write_table(table, path)
