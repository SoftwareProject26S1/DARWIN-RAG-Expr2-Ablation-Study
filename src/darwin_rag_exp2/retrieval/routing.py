"""Query routing helpers for Phase 9 retrieval variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def top1_category(probabilities: Mapping[str, float]) -> str:
    """Return the highest-probability category with deterministic tie-breaking."""

    if not probabilities:
        raise ValueError("query probabilities must not be empty")
    return max(
        ((str(category), float(probability)) for category, probability in probabilities.items()),
        key=lambda item: (item[1], item[0]),
    )[0]


def soft_route_categories(
    probabilities: Mapping[str, float],
    *,
    theta_route: float,
) -> tuple[str, ...]:
    """Route to all threshold-passing categories or a top-1 fallback."""

    if theta_route < 0.0 or theta_route > 1.0:
        raise ValueError("theta_route must be between 0 and 1")
    if not probabilities:
        raise ValueError("query probabilities must not be empty")

    selected = [
        (str(category), float(probability))
        for category, probability in probabilities.items()
        if float(probability) >= theta_route
    ]
    if not selected:
        return (top1_category(probabilities),)
    return tuple(
        category
        for category, _ in sorted(selected, key=lambda item: (-item[1], item[0]))
    )


def stable_category_order(categories: Sequence[str]) -> tuple[str, ...]:
    """Return a deterministic category order without duplicates."""

    return tuple(dict.fromkeys(str(category) for category in categories))
