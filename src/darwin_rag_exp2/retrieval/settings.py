"""Frozen settings helpers for Phase 9 primary retrieval runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import exp
from pathlib import Path
from typing import Any

import orjson
import yaml

from .types import PrimaryRunSettings


def build_lambda_by_category(
    category_stats_rows: Sequence[Mapping[str, object]],
    *,
    alpha: float,
    rho: float,
    tau: float,
) -> dict[str, float]:
    """Compute adaptive semantic-mixture coefficients from category stats."""

    lambdas: dict[str, float] = {}
    for row in category_stats_rows:
        category = str(row["category"])
        mu_confidence = float(row["mu_confidence"])
        sigma_confidence = float(row["sigma_confidence"])
        value = alpha * (mu_confidence - tau) - rho * sigma_confidence
        lambdas[category] = _metric(_sigmoid(value))
    return lambdas


def load_primary_run_settings(
    settings_path: Path,
    *,
    category_stats_path: Path | None = None,
) -> PrimaryRunSettings:
    """Load a frozen Phase 9 settings YAML file."""

    payload = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("settings YAML root must be a mapping")
    lambda_by_category = payload.get("lambda_by_category")
    if lambda_by_category is None:
        if category_stats_path is None:
            raise ValueError(
                "settings must include lambda_by_category or category_stats_path"
            )
        adaptive = payload.get("adaptive_lambda") or {}
        if not isinstance(adaptive, dict):
            raise ValueError("adaptive_lambda must be a mapping")
        lambda_by_category = build_lambda_by_category(
            _load_category_stats_rows(category_stats_path),
            alpha=float(adaptive.get("alpha", 8.0)),
            rho=float(adaptive.get("rho", 4.0)),
            tau=float(adaptive.get("tau", 0.5)),
        )
    if not isinstance(lambda_by_category, dict):
        raise ValueError("lambda_by_category must be a mapping")
    return PrimaryRunSettings(
        candidate_k_per_partition=int(payload["candidate_k_per_partition"]),
        report_top_k=int(payload["report_top_k"]),
        generation_context_top_n=int(payload["generation_context_top_n"]),
        theta_route=float(payload["theta_route"]),
        lambda_fixed=float(payload["lambda_fixed"]),
        lambda_by_category={
            str(category): float(value)
            for category, value in lambda_by_category.items()
        },
    )


def write_primary_run_settings(
    path: Path,
    *,
    candidate_k_per_partition: int,
    report_top_k: int,
    generation_context_top_n: int,
    theta_route: float,
    lambda_fixed: float,
    lambda_by_category: Mapping[str, float],
    tuning_metadata: Mapping[str, object] | None = None,
) -> None:
    """Write frozen Phase 9 settings as YAML."""

    payload: dict[str, object] = {
        "candidate_k_per_partition": candidate_k_per_partition,
        "report_top_k": report_top_k,
        "generation_context_top_n": generation_context_top_n,
        "theta_route": theta_route,
        "lambda_fixed": lambda_fixed,
        "lambda_by_category": {
            str(category): float(value)
            for category, value in lambda_by_category.items()
        },
    }
    if tuning_metadata:
        payload["tuning_metadata"] = dict(tuning_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_category_stats_rows(path: Path) -> list[dict[str, object]]:
    payload = orjson.loads(path.read_bytes())
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("category stats must contain a rows list")
    return [dict(row) for row in rows]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    numerator = exp(value)
    return numerator / (1.0 + numerator)


def _metric(value: float) -> float:
    return round(float(value), 6)
