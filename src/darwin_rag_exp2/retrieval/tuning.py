"""Dev-query tuning helpers for Phase 9 primary retrieval settings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from darwin_rag_exp2.evaluation.retrieval_metrics import retrieval_metrics_at_k

from .settings import build_lambda_by_category
from .types import PrimaryRunSettings, QueryFeatures, SearchBackend
from .variants import run_b2_score, run_p_score


def tune_primary_settings(
    queries: Sequence[QueryFeatures],
    *,
    search_backend: SearchBackend,
    candidate_k_per_partition: int,
    report_top_k: int,
    generation_context_top_n: int,
    theta_candidates: Sequence[float],
    lambda_fixed_candidates: Sequence[float],
    lambda_by_category: Mapping[str, float],
    metric_key: str = "ndcg@10",
) -> tuple[PrimaryRunSettings, dict[str, object]]:
    """Select theta_route and lambda_fixed by B2-score dev retrieval metric."""

    if not queries:
        raise ValueError("tuning requires at least one dev query")
    if not theta_candidates:
        raise ValueError("theta_candidates must not be empty")
    if not lambda_fixed_candidates:
        raise ValueError("lambda_fixed_candidates must not be empty")

    trials: list[dict[str, object]] = []
    best_settings: PrimaryRunSettings | None = None
    best_metric = float("-inf")
    for theta_route in theta_candidates:
        for lambda_fixed in lambda_fixed_candidates:
            settings = PrimaryRunSettings(
                candidate_k_per_partition=candidate_k_per_partition,
                report_top_k=report_top_k,
                generation_context_top_n=generation_context_top_n,
                theta_route=float(theta_route),
                lambda_fixed=float(lambda_fixed),
                lambda_by_category={
                    str(category): float(value)
                    for category, value in lambda_by_category.items()
                },
            )
            metric_value = _average_b2_metric(
                queries,
                search_backend=search_backend,
                settings=settings,
                metric_key=metric_key,
            )
            trials.append(
                {
                    "theta_route": settings.theta_route,
                    "lambda_fixed": settings.lambda_fixed,
                    "metric": metric_value,
                }
            )
            if metric_value > best_metric:
                best_metric = metric_value
                best_settings = settings

    if best_settings is None:
        raise ValueError("no tuning trials were evaluated")
    diagnostics = {
        "best_variant": "B2-score",
        "metric_key": metric_key,
        "best_metric": best_metric,
        "trials": trials,
    }
    return best_settings, diagnostics


def tune_adaptive_lambda_parameters(
    queries: Sequence[QueryFeatures],
    *,
    search_backend: SearchBackend,
    base_settings: PrimaryRunSettings,
    category_stats_rows: Sequence[Mapping[str, object]],
    alpha_candidates: Sequence[float],
    rho_candidates: Sequence[float],
    tau_candidates: Sequence[float],
    metric_key: str = "ndcg@10",
) -> tuple[PrimaryRunSettings, dict[str, object]]:
    """Select adaptive lambda parameters by P-score dev retrieval metric."""

    if not queries:
        raise ValueError("adaptive tuning requires at least one dev query")
    if not alpha_candidates:
        raise ValueError("alpha_candidates must not be empty")
    if not rho_candidates:
        raise ValueError("rho_candidates must not be empty")
    if not tau_candidates:
        raise ValueError("tau_candidates must not be empty")

    trials: list[dict[str, object]] = []
    best_settings: PrimaryRunSettings | None = None
    best_parameters: dict[str, float] | None = None
    best_metric = float("-inf")
    for alpha in alpha_candidates:
        for rho in rho_candidates:
            for tau in tau_candidates:
                lambda_by_category = build_lambda_by_category(
                    category_stats_rows,
                    alpha=float(alpha),
                    rho=float(rho),
                    tau=float(tau),
                )
                settings = PrimaryRunSettings(
                    candidate_k_per_partition=base_settings.candidate_k_per_partition,
                    report_top_k=base_settings.report_top_k,
                    generation_context_top_n=base_settings.generation_context_top_n,
                    theta_route=base_settings.theta_route,
                    lambda_fixed=base_settings.lambda_fixed,
                    lambda_by_category=lambda_by_category,
                )
                metric_value = _average_p_metric(
                    queries,
                    search_backend=search_backend,
                    settings=settings,
                    metric_key=metric_key,
                )
                parameters = {
                    "alpha": float(alpha),
                    "rho": float(rho),
                    "tau": float(tau),
                }
                trials.append(
                    {
                        **parameters,
                        "metric": metric_value,
                    }
                )
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_settings = settings
                    best_parameters = parameters

    if best_settings is None or best_parameters is None:
        raise ValueError("no adaptive tuning trials were evaluated")
    diagnostics = {
        "best_variant": "P-score",
        "metric_key": metric_key,
        "best_metric": best_metric,
        "best_parameters": best_parameters,
        "trials": trials,
    }
    return best_settings, diagnostics


def _average_b2_metric(
    queries: Sequence[QueryFeatures],
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
    metric_key: str,
) -> float:
    values: list[float] = []
    for query in queries:
        result = run_b2_score(
            query,
            search_backend=search_backend,
            settings=settings,
        )
        metrics = retrieval_metrics_at_k(
            ranked_chunk_ids=[row.chunk_id for row in result.top10],
            gold_chunk_ids=query.gold_chunks,
            k=settings.report_top_k,
        )
        if metric_key not in metrics:
            raise ValueError(f"unknown retrieval metric {metric_key!r}")
        values.append(float(metrics[metric_key]))
    return sum(values) / len(values)


def _average_p_metric(
    queries: Sequence[QueryFeatures],
    *,
    search_backend: SearchBackend,
    settings: PrimaryRunSettings,
    metric_key: str,
) -> float:
    values: list[float] = []
    for query in queries:
        result = run_p_score(
            query,
            search_backend=search_backend,
            settings=settings,
        )
        metrics = retrieval_metrics_at_k(
            ranked_chunk_ids=[row.chunk_id for row in result.top10],
            gold_chunk_ids=query.gold_chunks,
            k=settings.report_top_k,
        )
        if metric_key not in metrics:
            raise ValueError(f"unknown retrieval metric {metric_key!r}")
        values.append(float(metrics[metric_key]))
    return sum(values) / len(values)
