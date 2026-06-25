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


def test_validate_query_splits_keeps_v1_non_single_fraction_gate(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev.jsonl"
    test_path = tmp_path / "queries_test.jsonl"
    _write_chunks(chunks_path)
    _write_queries(dev_path, [_query("dev_q0001", "c1", ["학사"], "single_category")])
    _write_queries(test_path, [_query("test_q0001", "c2", ["장학"], "single_category")])

    with pytest.raises(ValueError, match="non-single query fraction"):
        validate_query_splits(
            dev_path=dev_path,
            test_path=test_path,
            chunks_path=chunks_path,
            config=QueryValidationConfig(
                primary_categories=PRIMARY_CATEGORIES,
                expected_dev_count=1,
                expected_test_count=1,
                non_single_fraction=1.0,
                non_single_tolerance=0.0,
            ),
        )


def test_validate_query_splits_accepts_v2_rows_and_reports_schema_summary(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev_v2.jsonl"
    test_path = tmp_path / "queries_test_v2.jsonl"
    _write_chunks(chunks_path)
    dev_row = _v2_query("dev_q0001", ["c1", "c2"], ["학사", "장학"])
    dev_row["gold_category_set"] = ["장학", "학사"]
    dev_row["graded_relevance"]["c3"] = 0.15
    _write_queries(dev_path, [dev_row])
    _write_queries(test_path, [_v2_query("test_q0001", ["c2"], ["장학"])])

    report = validate_query_splits(
        dev_path=dev_path,
        test_path=test_path,
        chunks_path=chunks_path,
        config=QueryValidationConfig(
            primary_categories=PRIMARY_CATEGORIES,
            expected_dev_count=1,
            expected_test_count=1,
            non_single_fraction=1.0,
            non_single_tolerance=0.0,
        ),
    )

    assert report["valid"] is True
    assert report["splits"]["dev"]["row_count"] == 1
    assert report["splits"]["dev"]["schema_version_counts"] == {
        "eval_v3_overlap_aware_rechunked": 1,
    }
    assert report["splits"]["dev"]["category_counts"] == {"장학": 1, "학사": 1}


def test_validate_query_splits_accepts_v2_source_ids(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev_v2.jsonl"
    test_path = tmp_path / "queries_test_v2.jsonl"
    _write_chunks(chunks_path)
    dev_row = _v2_query("dev_q0001", ["c1", "c2"], ["학사", "장학"])
    dev_row["source_ids"] = ["s1", "s2"]
    _write_queries(dev_path, [dev_row])
    _write_queries(test_path, [_v2_query("test_q0001", ["c2"], ["장학"])])

    report = validate_query_splits(
        dev_path=dev_path,
        test_path=test_path,
        chunks_path=chunks_path,
        config=QueryValidationConfig(
            primary_categories=PRIMARY_CATEGORIES,
            expected_dev_count=1,
            expected_test_count=1,
            non_single_fraction=1.0,
            non_single_tolerance=0.0,
        ),
    )

    assert report["valid"] is True


def test_validate_query_splits_rejects_malformed_v2_source_ids(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev_v2.jsonl"
    test_path = tmp_path / "queries_test_v2.jsonl"
    _write_chunks(chunks_path)
    bad_row = _v2_query("dev_q0001", ["c1"], ["학사"])
    bad_row["source_ids"] = ["s1", ""]
    _write_queries(dev_path, [bad_row])
    _write_queries(test_path, [_v2_query("test_q0001", ["c2"], ["장학"])])

    with pytest.raises(ValueError, match="source_ids"):
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


def test_validate_query_splits_rejects_unknown_v2_fields(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev_v2.jsonl"
    test_path = tmp_path / "queries_test_v2.jsonl"
    _write_chunks(chunks_path)
    bad_row = _v2_query("dev_q0001", ["c1"], ["학사"])
    bad_row["probabilities"] = {"학사": 0.9}
    _write_queries(dev_path, [bad_row])
    _write_queries(test_path, [_v2_query("test_q0001", ["c2"], ["장학"])])

    with pytest.raises(ValueError, match="unexpected v2 fields"):
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


def test_validate_query_splits_rejects_v2_graded_relevance_missing_gold_chunk(
    tmp_path,
) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev_v2.jsonl"
    test_path = tmp_path / "queries_test_v2.jsonl"
    _write_chunks(chunks_path)
    bad_row = _v2_query("dev_q0001", ["c1", "c2"], ["학사", "장학"])
    bad_row["graded_relevance"] = {"c1": 1.0}
    _write_queries(dev_path, [bad_row])
    _write_queries(test_path, [_v2_query("test_q0001", ["c2"], ["장학"])])

    with pytest.raises(ValueError, match="graded_relevance"):
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


def test_validate_query_splits_rejects_unknown_schema_version(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    dev_path = tmp_path / "queries_dev_v2.jsonl"
    test_path = tmp_path / "queries_test_v2.jsonl"
    _write_chunks(chunks_path)
    bad_row = _v2_query("dev_q0001", ["c1"], ["학사"])
    bad_row["schema_version"] = "future_schema"
    _write_queries(dev_path, [bad_row])
    _write_queries(test_path, [_v2_query("test_q0001", ["c2"], ["장학"])])

    with pytest.raises(ValueError, match="schema_version"):
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


def _v2_query(
    query_id: str,
    gold_chunks: list[str],
    gold_categories: list[str],
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": f"{query_id} 일정과 조건을 알려주세요.",
        "reference_answer": "정답 답변입니다.",
        "query_type": "core_multi_category" if len(gold_categories) > 1 else "core_single_category",
        "expected_categories": gold_categories,
        "source_id": "s1",
        "gold_chunk_ids": [
            {
                "chunk_id": chunk_id,
                "source_id": f"s{index}",
                "chunk_index": index - 1,
                "relevance": 2 if index == 1 else 1,
                "role": "direct" if index == 1 else "support",
            }
            for index, chunk_id in enumerate(gold_chunks, start=1)
        ],
        "neighbor_chunk_ids": ["c3"] if "c3" not in gold_chunks else [],
        "graded_relevance": {
            chunk_id: 1.0 if index == 1 else 0.5
            for index, chunk_id in enumerate(gold_chunks, start=1)
        },
        "requires_multi_category": len(gold_categories) > 1,
        "gold_category_set": gold_categories,
        "gold_category_pure": len(gold_categories) == 1,
        "evidence_unit": "title+category+body",
        "schema_version": "eval_v3_overlap_aware_rechunked",
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
