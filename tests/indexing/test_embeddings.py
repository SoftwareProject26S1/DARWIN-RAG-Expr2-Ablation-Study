import math

import pytest

from darwin_rag_exp2.indexing.embeddings import HashEmbeddingModel, l2_normalize


def test_l2_normalize_makes_each_vector_unit_length() -> None:
    vectors = l2_normalize([[3.0, 4.0], [5.0, 12.0]])

    assert vectors == [[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]]
    assert [round(math.sqrt(sum(value * value for value in row)), 12) for row in vectors] == [
        1.0,
        1.0,
    ]


def test_l2_normalize_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        l2_normalize([[0.0, 0.0]])


def test_hash_embedding_model_is_deterministic_and_normalized() -> None:
    model = HashEmbeddingModel(dimension=8)

    first = model.encode(["학사 일정 안내", "장학 신청 안내"])
    second = model.encode(["학사 일정 안내", "장학 신청 안내"])

    assert first == second
    assert first[0] != first[1]
    assert all(
        round(math.sqrt(sum(value * value for value in row)), 12) == 1.0
        for row in first
    )
