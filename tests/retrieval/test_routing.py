import pytest

from darwin_rag_exp2.retrieval.routing import (
    route_categories_for_query,
    soft_route_categories,
)
from darwin_rag_exp2.retrieval.types import PrimaryRunSettings, QueryFeatures


def _settings() -> PrimaryRunSettings:
    return PrimaryRunSettings(
        candidate_k_per_partition=2,
        report_top_k=2,
        generation_context_top_n=1,
        theta_route=0.8,
        lambda_fixed=0.5,
        lambda_by_category={
            "학사": 0.9,
            "장학": 0.8,
            "국제교류": 0.7,
            "채용": 0.6,
        },
    )


def _query(
    *,
    query: str = "수강신청 일정을 알려줘",
    query_type: str = "core_single_category",
    probabilities: dict[str, float] | None = None,
    requires_multi_category: bool | None = None,
) -> QueryFeatures:
    return QueryFeatures(
        query_id="test_q0001",
        query=query,
        embedding=[1.0, 0.0],
        probabilities=(
            {"학사": 0.9, "장학": 0.1, "국제교류": 0.05, "채용": 0.04}
            if probabilities is None
            else probabilities
        ),
        query_type=query_type,
        requires_multi_category=requires_multi_category,
    )


def test_route_categories_for_query_uses_top3_when_query_type_is_multi_category() -> None:
    # Given
    query = _query(query_type="multi_category")

    # When
    decision = route_categories_for_query(query, _settings())

    # Then
    assert decision.mode == "multi_min_top3"
    assert decision.categories == ("학사", "장학", "국제교류")


def test_route_categories_for_query_uses_top3_when_metadata_requires_multi() -> None:
    # Given
    query = _query(requires_multi_category=True)

    # When
    decision = route_categories_for_query(query, _settings())

    # Then
    assert decision.mode == "multi_min_top3"
    assert decision.categories == ("학사", "장학", "국제교류")


@pytest.mark.parametrize(
    "text",
    [
        "수강신청과 장학 일정을 같이 알려줘",
        "두 공지의 대상과 마감일을 각각 구분해 줘",
        "학사와 장학 공지를 비교해 줘",
    ],
)
def test_route_categories_for_query_uses_top3_for_multi_intent_text(
    text: str,
) -> None:
    # Given
    query = _query(query=text)

    # When
    decision = route_categories_for_query(query, _settings())

    # Then
    assert decision.mode == "multi_min_top3"
    assert decision.categories == ("학사", "장학", "국제교류")


def test_route_categories_for_query_uses_top3_when_top1_confidence_is_low() -> None:
    # Given
    query = _query(
        probabilities={"학사": 0.7, "장학": 0.2, "국제교류": 0.1, "채용": 0.05},
    )

    # When
    decision = route_categories_for_query(query, _settings())

    # Then
    assert decision.mode == "low_confidence_min_top3"
    assert decision.categories == ("학사", "장학", "국제교류")


def test_route_categories_for_query_uses_top3_when_top_margin_is_small() -> None:
    # Given
    query = _query(
        probabilities={"학사": 0.9, "장학": 0.75, "국제교류": 0.1, "채용": 0.05},
    )

    # When
    decision = route_categories_for_query(query, _settings())

    # Then
    assert decision.mode == "small_margin_min_top3"
    assert decision.categories == ("학사", "장학", "국제교류")


def test_route_categories_for_query_keeps_high_confidence_single_threshold_route() -> None:
    # Given
    query = _query()

    # When
    decision = route_categories_for_query(query, _settings())

    # Then
    assert decision.mode == "soft_threshold"
    assert decision.categories == ("학사",)


def test_route_categories_for_query_keeps_high_confidence_single_top1_fallback() -> None:
    # Given
    settings = PrimaryRunSettings(
        candidate_k_per_partition=2,
        report_top_k=2,
        generation_context_top_n=1,
        theta_route=0.95,
        lambda_fixed=0.5,
        lambda_by_category={"학사": 0.9, "장학": 0.8},
    )
    query = _query(probabilities={"학사": 0.9, "장학": 0.05})

    # When
    decision = route_categories_for_query(query, settings)

    # Then
    assert decision.mode == "top1_fallback"
    assert decision.categories == ("학사",)


def test_route_categories_for_query_rejects_empty_probabilities() -> None:
    # Given
    query = _query(probabilities={})

    # When / Then
    with pytest.raises(ValueError, match="query probabilities"):
        route_categories_for_query(query, _settings())


def test_soft_route_categories_rejects_invalid_theta() -> None:
    # Given / When / Then
    with pytest.raises(ValueError, match="theta_route"):
        soft_route_categories({"학사": 0.9}, theta_route=1.1)
