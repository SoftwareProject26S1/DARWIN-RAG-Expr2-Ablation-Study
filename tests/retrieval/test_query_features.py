import json

from darwin_rag_exp2.retrieval.query_features import (
    build_query_features,
    embed_query_rows,
    load_query_rows,
    oracle_probabilities_from_query_rows,
    probabilities_from_query_rows,
)
from darwin_rag_exp2.indexing.embeddings import HashEmbeddingModel


def test_load_query_rows_and_build_features_from_model_outputs(tmp_path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        "\n".join(
            [
                '{"query_id":"q1","query":"수강신청 기간은?","gold_chunks":["c1"],'
                '"reference_answer":"3월입니다.","gold_categories":["학사"],'
                '"query_type":"single_category"}'
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_query_rows(queries_path)
    features = build_query_features(
        rows,
        embeddings_by_query_id={"q1": [1.0, 0.0]},
        probabilities_by_query_id={"q1": {"학사": 0.9, "장학": 0.1}},
    )

    assert features[0].query_id == "q1"
    assert features[0].query == "수강신청 기간은?"
    assert features[0].embedding == [1.0, 0.0]
    assert features[0].probabilities == {"학사": 0.9, "장학": 0.1}
    assert features[0].gold_chunks == ("c1",)
    assert features[0].gold_categories == ("학사",)
    assert features[0].query_type == "single_category"


def test_load_query_rows_normalizes_v2_metadata_for_features(tmp_path) -> None:
    queries_path = tmp_path / "queries_v2.jsonl"
    queries_path.write_text(
        json.dumps(
            {
                "query_id": "dev_q1",
                "query": "수강신청과 장학 일정을 알려줘",
                "reference_answer": "답변입니다.",
                "query_type": "multi_category",
                "expected_categories": ["학사", "장학"],
                "source_id": "doc1",
                "gold_chunk_ids": [
                    {
                        "chunk_id": "c1",
                        "source_id": "doc1",
                        "chunk_index": 0,
                        "relevance": 2,
                        "role": "direct",
                    },
                    {
                        "chunk_id": "c2",
                        "source_id": "doc1",
                        "chunk_index": 1,
                        "relevance": 1,
                        "role": "support",
                    },
                ],
                "neighbor_chunk_ids": ["c3"],
                "graded_relevance": {"c1": 1.0, "c2": 0.5},
                "requires_multi_category": True,
                "gold_category_set": ["학사", "장학"],
                "gold_category_pure": False,
                "evidence_unit": "chunk",
                "schema_version": "eval_v3_overlap_aware_rechunked",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_query_rows(queries_path)
    features = build_query_features(
        rows,
        embeddings_by_query_id={"dev_q1": [1.0, 0.0]},
        probabilities_by_query_id={"dev_q1": {"학사": 0.6, "장학": 0.4}},
    )

    assert rows[0]["gold_chunks"] == ["c1", "c2"]
    assert rows[0]["gold_categories"] == ["학사", "장학"]
    assert rows[0]["graded_relevance"] == {"c1": 1.0, "c2": 0.5}
    assert features[0].graded_relevance == {"c1": 1.0, "c2": 0.5}
    assert features[0].neighbor_chunk_ids == ("c3",)
    assert features[0].source_id == "doc1"
    assert features[0].evidence_unit == "chunk"
    assert features[0].schema_version == "eval_v3_overlap_aware_rechunked"
    assert features[0].requires_multi_category is True
    assert features[0].gold_category_pure is False


def test_embed_query_rows_and_extract_precomputed_probabilities() -> None:
    rows = [
        {
            "query_id": "q1",
            "query": "수강신청 기간은?",
            "gold_chunks": ["c1"],
            "reference_answer": "3월입니다.",
            "gold_categories": ["학사"],
            "query_type": "single_category",
            "probabilities": {"학사": 0.8, "장학": 0.2},
        }
    ]

    embeddings = embed_query_rows(
        rows,
        embedding_model=HashEmbeddingModel(dimension=4),
        normalize_embeddings=True,
    )
    probabilities = probabilities_from_query_rows(rows)

    assert set(embeddings) == {"q1"}
    assert len(embeddings["q1"]) == 4
    assert probabilities == {"q1": {"학사": 0.8, "장학": 0.2}}


def test_oracle_probabilities_share_weight_across_gold_categories() -> None:
    rows = [
        {
            "query_id": "q1",
            "query": "수강신청과 장학 일정을 알려줘",
            "gold_chunks": ["c1"],
            "reference_answer": "답변입니다.",
            "gold_categories": ["학사", "장학"],
            "query_type": "multi_category",
        }
    ]

    probabilities = oracle_probabilities_from_query_rows(
        rows,
        categories=["학사", "장학", "국제교류"],
    )

    assert probabilities == {
        "q1": {"학사": 0.5, "장학": 0.5, "국제교류": 0.0}
    }
