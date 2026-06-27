"""Query routing helpers for Phase 9 retrieval variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Final

from .types import PrimaryRunSettings, QueryFeatures


MULTI_QUERY_TYPES: Final = frozenset(
    ("multi_category", "core_multi_category", "multi-category")
)
MULTI_INTENT_TERMS: Final = ("각각", "같이", "구분")
MULTI_CONNECTOR_PATTERN: Final = re.compile(r"[\w가-힣]+(?:와|과)\s+[\w가-힣]+")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    categories: tuple[str, ...]
    mode: str
    top1_category: str


def top1_category(probabilities: Mapping[str, float]) -> str:
    """Return the highest-probability category with deterministic tie-breaking."""

    return _ranked_probabilities(probabilities)[0][0]


def soft_route_categories(
    probabilities: Mapping[str, float],
    *,
    theta_route: float,
) -> tuple[str, ...]:
    """Route to all threshold-passing categories or a top-1 fallback."""

    _validate_theta_route(theta_route)
    ranked = _ranked_probabilities(probabilities)

    selected = [
        (category, probability)
        for category, probability in ranked
        if probability >= theta_route
    ]
    if not selected:
        return (ranked[0][0],)
    return tuple(category for category, _ in selected)


def route_categories_for_query(
    query: QueryFeatures,
    settings: PrimaryRunSettings,
) -> RouteDecision:
    _validate_theta_route(settings.theta_route)
    ranked = _ranked_probabilities(query.probabilities)
    top1, top1_probability = ranked[0]

    if _has_multi_intent(query):
        return _top_n_decision(
            ranked,
            settings.min_multi_route_width,
            "multi_min_top3",
        )
    if top1_probability < settings.low_confidence_top1_threshold:
        return _top_n_decision(
            ranked,
            settings.min_multi_route_width,
            "low_confidence_min_top3",
        )
    if _has_small_top_margin(ranked, settings.small_margin_threshold):
        return _top_n_decision(
            ranked,
            settings.min_multi_route_width,
            "small_margin_min_top3",
        )

    selected = tuple(
        category
        for category, probability in ranked
        if probability >= settings.theta_route
    )
    if selected:
        return RouteDecision(
            categories=selected,
            mode="soft_threshold",
            top1_category=top1,
        )
    return RouteDecision(
        categories=(top1,),
        mode="top1_fallback",
        top1_category=top1,
    )


def stable_category_order(categories: Sequence[str]) -> tuple[str, ...]:
    """Return a deterministic category order without duplicates."""

    return tuple(dict.fromkeys(str(category) for category in categories))


def _ranked_probabilities(
    probabilities: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if not probabilities:
        raise ValueError("query probabilities must not be empty")
    return tuple(
        sorted(
            (
                (str(category), float(probability))
                for category, probability in probabilities.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _validate_theta_route(theta_route: float) -> None:
    if theta_route < 0.0 or theta_route > 1.0:
        raise ValueError("theta_route must be between 0 and 1")


def _has_multi_intent(query: QueryFeatures) -> bool:
    return (
        query.requires_multi_category is True
        or query.query_type in MULTI_QUERY_TYPES
        or any(term in query.query for term in MULTI_INTENT_TERMS)
        or MULTI_CONNECTOR_PATTERN.search(query.query) is not None
    )


def _has_small_top_margin(
    ranked: tuple[tuple[str, float], ...],
    threshold: float,
) -> bool:
    if len(ranked) < 2:
        return False
    return ranked[0][1] - ranked[1][1] < threshold


def _top_n_decision(
    ranked: tuple[tuple[str, float], ...],
    width: int,
    mode: str,
) -> RouteDecision:
    selected = ranked[: max(1, width)]
    return RouteDecision(
        categories=tuple(category for category, _ in selected),
        mode=mode,
        top1_category=ranked[0][0],
    )
