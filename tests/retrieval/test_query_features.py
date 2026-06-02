from darwin_rag_exp2.retrieval.query_features import (
    build_query_features,
    embed_query_rows,
    load_query_rows,
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
