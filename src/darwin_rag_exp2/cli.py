"""Command-line entrypoint for the Exp2 experiment package."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Annotated

import orjson
import typer
import yaml

from .data.artifacts import write_chunk_artifacts
from .data.audit import audit_notice_export, write_audit_artifacts
from .data.chunking import build_chunks, load_chunking_config
from .data.filtering import (
    load_corpus_filter_config,
    prepare_corpus,
    write_corpus_artifacts,
)
from .evaluation.pool import build_query_pool, write_query_pool_artifacts
from .evaluation.queries import (
    load_query_validation_config,
    validate_query_splits,
    write_query_validation_artifacts,
)
from .evaluation.retrieval_analysis import (
    analyze_primary_results,
    load_chunk_lookup,
    load_primary_result_rows,
    write_primary_analysis,
)
from .indexing.artifacts import build_index_artifacts, load_indexing_config
from .indexing.embedding_artifacts import build_embedding_artifacts
from .indexing.embeddings import HashEmbeddingModel, SentenceTransformerEmbeddingModel
from .indexing.faiss_store import FaissIndexWriter
from .models.classifier import (
    TransformerTrainingConfig,
    train_final_classifier,
    train_single_classifier,
)
from .models.crossfit import train_crossfit_classifier
from .retrieval.faiss_backend import FaissSearchBackend
from .retrieval.query_classifier import FinalQueryClassifier
from .retrieval.query_features import (
    build_query_features,
    embed_query_rows,
    load_query_rows,
    oracle_probabilities_from_query_rows,
    probabilities_from_query_rows,
)
from .retrieval.runner import run_primary_queries, write_primary_run
from .retrieval.settings import (
    build_lambda_by_category,
    load_primary_run_settings,
    write_primary_run_settings,
)
from .retrieval.tuning import (
    tune_adaptive_lambda_parameters,
    tune_primary_settings,
)
from .retrieval.variants import SEARCH_MODES


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.callback(invoke_without_command=True)
def root(context: typer.Context) -> None:
    """Run reproducible DARWIN-RAG Exp2 artifact-building tasks."""

    if context.invoked_subcommand is None:
        typer.echo("DARWIN-RAG Exp2: Phase 1 scaffold ready.")


@app.command("audit-data")
def audit_data(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
) -> None:
    """Generate the non-destructive audit baseline for the raw export."""

    report = audit_notice_export(input_path)
    write_audit_artifacts(report, output_path)
    typer.echo(f"Wrote raw-data audit to {output_path}")


@app.command("prepare-corpus")
def prepare_corpus_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, readable=True),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    mapping_path: Annotated[
        Path,
        typer.Option(
            "--mapping",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("configs/category_mapping.yaml"),
) -> None:
    """Prepare the Phase 3 primary corpus and exclusion report."""

    config = load_corpus_filter_config(config_path, mapping_path)
    result = prepare_corpus(input_path, config)
    write_corpus_artifacts(result, output_path)
    typer.echo(
        f"Wrote Phase 3 corpus to {output_path} "
        f"({len(result.admitted_records)} admitted, "
        f"{len(result.excluded_records)} excluded)"
    )


@app.command("chunk-corpus")
def chunk_corpus_command(
    corpus_path: Annotated[
        Path,
        typer.Option("--corpus", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
) -> None:
    """Build Phase 4 chunk JSONL/Parquet artifacts from the admitted corpus."""

    config = load_chunking_config(config_path)
    result = build_chunks(corpus_path, config)
    write_chunk_artifacts(result, output_path)
    typer.echo(
        f"Wrote Phase 4 chunks to {output_path} "
        f"({len(result.chunks)} chunks, "
        f"{result.manifest['violating_classifier_token_cap_count']} cap violations)"
    )


@app.command("train-classifier")
def train_classifier_command(
    mode: Annotated[str, typer.Option("--mode")],
    chunks_path: Annotated[
        Path,
        typer.Option("--chunks", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    max_sources_per_category: Annotated[
        int,
        typer.Option("--max-sources-per-category"),
    ] = 12,
    folds: Annotated[int, typer.Option("--folds")] = 5,
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
    model_name: Annotated[
        str | None,
        typer.Option("--model"),
    ] = None,
    max_length: Annotated[int, typer.Option("--max-length")] = 512,
    epochs: Annotated[int | None, typer.Option("--epochs")] = None,
    train_batch_size: Annotated[int, typer.Option("--train-batch-size")] = 8,
    eval_batch_size: Annotated[int, typer.Option("--eval-batch-size")] = 16,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 2e-5,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.01,
    warmup_ratio: Annotated[float, typer.Option("--warmup-ratio")] = 0.1,
    calibration_fraction: Annotated[
        float,
        typer.Option("--calibration-fraction"),
    ] = 0.1,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    device: Annotated[str, typer.Option("--device")] = "auto",
    log_every_batches: Annotated[int, typer.Option("--log-every-batches")] = 25,
    resume: Annotated[bool, typer.Option("--resume")] = False,
) -> None:
    """Train Phase 5/6 classifier artifacts."""

    resolved_model_name = model_name or _classifier_model_from_config(config_path)
    resolved_epochs = epochs if epochs is not None else (1 if mode == "single" else 3)
    training_config = TransformerTrainingConfig(
        model_name=resolved_model_name,
        max_length=max_length,
        epochs=resolved_epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        calibration_fraction=calibration_fraction,
        seed=seed,
        device=device,
        log_every_batches=log_every_batches,
    )

    if mode == "single":
        result = train_single_classifier(
            chunks_path,
            output_path,
            max_sources_per_category=max_sources_per_category,
            training_config=training_config,
            progress_callback=typer.echo,
            resume=resume,
        )
        typer.echo(
            f"Wrote Phase 5 BERT single classifier smoke artifacts to {output_path} "
            f"({result.manifest['training_chunk_count']} chunks, "
            f"T={result.calibration['temperature']})"
        )
        return
    if mode == "crossfit":
        result = train_crossfit_classifier(
            chunks_path,
            output_path,
            fold_count=folds,
            training_config=training_config,
            progress_callback=typer.echo,
            resume=resume,
        )
        typer.echo(
            f"Wrote Phase 6 BERT crossfit classifier artifacts to {output_path} "
            f"({result.manifest['prediction_chunk_count']} OOF predictions, "
            f"{result.manifest['fold_count']} folds)"
        )
        return
    if mode == "final":
        result = train_final_classifier(
            chunks_path,
            output_path,
            training_config=training_config,
            progress_callback=typer.echo,
            resume=resume,
        )
        typer.echo(
            f"Wrote final BERT query classifier artifacts to {output_path} "
            f"({result.manifest['training_chunk_count']} chunks, "
            f"T={result.calibration['temperature']})"
        )
        return
    raise typer.BadParameter("supported classifier modes: single, crossfit, final")


def _classifier_model_from_config(config_path: Path) -> str:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = payload.get("models") or {}
    model_name = models.get("classifier")
    if not isinstance(model_name, str) or not model_name.strip():
        raise typer.BadParameter("config must define models.classifier")
    return model_name


@app.command("build-embeddings")
def build_embeddings_command(
    chunks_path: Annotated[
        Path,
        typer.Option("--chunks", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
    embedding_backend: Annotated[
        str,
        typer.Option("--embedding-backend"),
    ] = "sentence-transformers",
    embedding_model_name: Annotated[
        str | None,
        typer.Option("--embedding-model"),
    ] = None,
) -> None:
    """Build reusable Phase 7 chunk embedding artifacts."""

    config = load_indexing_config(config_path)
    model_name = embedding_model_name or config.embedding_model
    embedding_model = _load_embedding_model(embedding_backend, model_name)
    result = build_embedding_artifacts(
        chunks_path=chunks_path,
        output_dir=output_path,
        embedding_model=embedding_model,
        embedding_model_name=model_name,
        normalize_embeddings=config.normalize_embeddings,
        similarity_metric=config.similarity_metric,
    )
    typer.echo(
        f"Wrote Phase 7 embeddings to {output_path} "
        f"({result.manifest['chunk_count']} chunks, "
        f"{result.manifest['embedding_dimension']} dimensions)"
    )


@app.command("build-indexes")
def build_indexes_command(
    chunks_path: Annotated[
        Path,
        typer.Option("--chunks", exists=True, dir_okay=False, readable=True),
    ],
    predictions_path: Annotated[
        Path,
        typer.Option("--predictions", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
    embeddings_path: Annotated[
        Path | None,
        typer.Option("--embeddings", exists=True, file_okay=False, readable=True),
    ] = None,
    ingest_threshold: Annotated[float, typer.Option("--ingest-threshold")] = 0.5,
    embedding_backend: Annotated[
        str,
        typer.Option("--embedding-backend"),
    ] = "sentence-transformers",
    embedding_model_name: Annotated[
        str | None,
        typer.Option("--embedding-model"),
    ] = None,
) -> None:
    """Build Phase 7 unified and category FAISS indexes."""

    config = load_indexing_config(config_path)
    model_name = embedding_model_name or config.embedding_model
    embedding_model = (
        None
        if embeddings_path is not None
        else _load_embedding_model(embedding_backend, model_name)
    )

    result = build_index_artifacts(
        chunks_path=chunks_path,
        predictions_path=predictions_path,
        output_dir=output_path,
        embedding_model=embedding_model,
        index_writer=FaissIndexWriter(),
        ingest_threshold=ingest_threshold,
        embedding_model_name=model_name,
        embedding_artifacts_dir=embeddings_path,
        normalize_embeddings=config.normalize_embeddings,
        similarity_metric=config.similarity_metric,
    )
    typer.echo(
        f"Wrote Phase 7 indexes to {output_path} "
        f"({result.manifest['chunk_count']} chunks, "
        f"{len(result.manifest['category_indexes'])} category indexes)"
    )


@app.command("export-query-pool")
def export_query_pool_command(
    chunks_path: Annotated[
        Path,
        typer.Option("--chunks", exists=True, dir_okay=False, readable=True),
    ] = Path("artifacts/chunks/chunks.parquet"),
    output_path: Annotated[Path, typer.Option("--output")] = Path(
        "artifacts/query-pool"
    ),
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
    per_category: Annotated[int, typer.Option("--per-category")] = 80,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    preview_chars: Annotated[int, typer.Option("--preview-chars")] = 240,
) -> None:
    """Export a Phase 8 chunk candidate pool for query annotation."""

    try:
        validation_config = load_query_validation_config(config_path)
        result = build_query_pool(
            chunks_path=chunks_path,
            primary_categories=validation_config.primary_categories,
            per_category=per_category,
            seed=seed,
            preview_chars=preview_chars,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    write_query_pool_artifacts(output_path, result)
    typer.echo(
        f"Wrote Phase 8 query pool to {output_path} "
        f"({result.manifest['row_count']} candidates)"
    )


@app.command("validate-queries")
def validate_queries_command(
    dev_path: Annotated[
        Path,
        typer.Option("--dev", exists=True, dir_okay=False, readable=True),
    ],
    test_path: Annotated[
        Path,
        typer.Option("--test", exists=True, dir_okay=False, readable=True),
    ],
    chunks_path: Annotated[
        Path,
        typer.Option("--chunks", exists=True, dir_okay=False, readable=True),
    ] = Path("artifacts/chunks/chunks.parquet"),
    output_path: Annotated[Path, typer.Option("--output")] = Path(
        "artifacts/query-validation"
    ),
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
    non_single_tolerance: Annotated[
        float,
        typer.Option("--non-single-tolerance"),
    ] = 0.05,
) -> None:
    """Validate Phase 8 dev/test query annotation JSONL files."""

    try:
        validation_config = load_query_validation_config(
            config_path,
            non_single_tolerance=non_single_tolerance,
        )
        report = validate_query_splits(
            dev_path=dev_path,
            test_path=test_path,
            chunks_path=chunks_path,
            config=validation_config,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    write_query_validation_artifacts(output_path, report)
    typer.echo(
        f"Wrote Phase 8 query validation to {output_path} "
        f"(dev={report['splits']['dev']['row_count']}, "
        f"test={report['splits']['test']['row_count']})"
    )


@app.command("tune-primary")
def tune_primary_command(
    queries_path: Annotated[
        Path,
        typer.Option("--queries", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    indexes_path: Annotated[
        Path,
        typer.Option("--indexes", exists=True, file_okay=False, readable=True),
    ] = Path("artifacts/indexes"),
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
    category_stats_path: Annotated[
        Path,
        typer.Option("--category-stats", dir_okay=False, readable=True),
    ] = Path("artifacts/classifier/crossfit/category_stats.json"),
    query_classifier_path: Annotated[
        Path | None,
        typer.Option("--query-classifier", file_okay=False, readable=True),
    ] = Path("artifacts/classifier/final"),
    embedding_backend: Annotated[
        str,
        typer.Option("--embedding-backend"),
    ] = "sentence-transformers",
    embedding_model_name: Annotated[
        str | None,
        typer.Option("--embedding-model"),
    ] = None,
    classifier_device: Annotated[
        str,
        typer.Option("--classifier-device"),
    ] = "auto",
    theta_grid: Annotated[
        str,
        typer.Option("--theta-grid"),
    ] = "0.5,0.6,0.7,0.8,0.9",
    fixed_lambda_grid: Annotated[
        str | None,
        typer.Option("--fixed-lambda-grid"),
    ] = None,
    adaptive_alpha: Annotated[float, typer.Option("--adaptive-alpha")] = 8.0,
    adaptive_rho: Annotated[float, typer.Option("--adaptive-rho")] = 4.0,
    adaptive_tau: Annotated[float, typer.Option("--adaptive-tau")] = 0.5,
    adaptive_alpha_grid: Annotated[
        str | None,
        typer.Option("--adaptive-alpha-grid"),
    ] = None,
    adaptive_rho_grid: Annotated[
        str | None,
        typer.Option("--adaptive-rho-grid"),
    ] = None,
    adaptive_tau_grid: Annotated[
        str | None,
        typer.Option("--adaptive-tau-grid"),
    ] = None,
    metric_key: Annotated[str, typer.Option("--metric")] = "ndcg@10",
) -> None:
    """Tune Phase 9 primary retrieval settings on dev queries."""

    if not category_stats_path.exists():
        raise typer.BadParameter(f"missing category stats: {category_stats_path}")
    config = load_indexing_config(config_path)
    retrieval_defaults = _retrieval_defaults_from_config(config_path)
    model_name = embedding_model_name or config.embedding_model
    query_rows = load_query_rows(queries_path)
    embedding_model = _load_embedding_model(embedding_backend, model_name)
    embeddings_by_query_id = embed_query_rows(
        query_rows,
        embedding_model=embedding_model,
        normalize_embeddings=config.normalize_embeddings,
    )
    probabilities_by_query_id, probability_source = _load_query_probabilities(
        query_rows,
        query_classifier_path=query_classifier_path,
        classifier_device=classifier_device,
    )
    query_features = build_query_features(
        query_rows,
        embeddings_by_query_id=embeddings_by_query_id,
        probabilities_by_query_id=probabilities_by_query_id,
    )
    category_stats_rows = _load_category_stats_rows(category_stats_path)
    lambda_by_category = build_lambda_by_category(
        category_stats_rows,
        alpha=adaptive_alpha,
        rho=adaptive_rho,
        tau=adaptive_tau,
    )
    search_backend = FaissSearchBackend(indexes_path)
    settings, diagnostics = tune_primary_settings(
        query_features,
        search_backend=search_backend,
        candidate_k_per_partition=int(retrieval_defaults["candidate_k_per_partition"]),
        report_top_k=int(retrieval_defaults["report_top_k"]),
        generation_context_top_n=int(retrieval_defaults["generation_context_top_n"]),
        theta_candidates=_parse_float_grid(theta_grid),
        lambda_fixed_candidates=(
            _parse_float_grid(fixed_lambda_grid)
            if fixed_lambda_grid is not None
            else [
                float(value)
                for value in retrieval_defaults["fixed_lambda_grid"]
            ]
        ),
        lambda_by_category=lambda_by_category,
        metric_key=metric_key,
    )
    settings, adaptive_diagnostics = tune_adaptive_lambda_parameters(
        query_features,
        search_backend=search_backend,
        base_settings=settings,
        category_stats_rows=category_stats_rows,
        alpha_candidates=(
            _parse_float_grid(adaptive_alpha_grid)
            if adaptive_alpha_grid is not None
            else [adaptive_alpha]
        ),
        rho_candidates=(
            _parse_float_grid(adaptive_rho_grid)
            if adaptive_rho_grid is not None
            else [adaptive_rho]
        ),
        tau_candidates=(
            _parse_float_grid(adaptive_tau_grid)
            if adaptive_tau_grid is not None
            else [adaptive_tau]
        ),
        metric_key=metric_key,
    )
    diagnostics = {
        **diagnostics,
        "query_count": len(query_features),
        "probability_source": probability_source,
        "adaptive_lambda": adaptive_diagnostics["best_parameters"],
        "p_score_tuning": adaptive_diagnostics,
    }
    write_primary_run_settings(
        output_path / "frozen.yaml",
        candidate_k_per_partition=settings.candidate_k_per_partition,
        report_top_k=settings.report_top_k,
        generation_context_top_n=settings.generation_context_top_n,
        theta_route=settings.theta_route,
        lambda_fixed=settings.lambda_fixed,
        lambda_by_category=settings.lambda_by_category,
        tuning_metadata=diagnostics,
    )
    _write_json(output_path / "tuning.json", diagnostics)
    typer.echo(
        f"Wrote Phase 9 primary settings to {output_path} "
        f"(theta_route={settings.theta_route}, "
        f"lambda_fixed={settings.lambda_fixed})"
    )


@app.command("run-primary")
def run_primary_command(
    queries_path: Annotated[
        Path,
        typer.Option("--queries", exists=True, dir_okay=False, readable=True),
    ],
    settings_path: Annotated[
        Path,
        typer.Option("--settings", exists=True, dir_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    indexes_path: Annotated[
        Path,
        typer.Option("--indexes", exists=True, file_okay=False, readable=True),
    ] = Path("artifacts/indexes"),
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiment.default.yaml"),
    category_stats_path: Annotated[
        Path | None,
        typer.Option("--category-stats", dir_okay=False, readable=True),
    ] = None,
    query_classifier_path: Annotated[
        Path | None,
        typer.Option("--query-classifier", file_okay=False, readable=True),
    ] = Path("artifacts/classifier/final"),
    embedding_backend: Annotated[
        str,
        typer.Option("--embedding-backend"),
    ] = "sentence-transformers",
    embedding_model_name: Annotated[
        str | None,
        typer.Option("--embedding-model"),
    ] = None,
    classifier_device: Annotated[
        str,
        typer.Option("--classifier-device"),
    ] = "auto",
    router: Annotated[
        str,
        typer.Option("--router"),
    ] = "final-classifier",
    search_mode: Annotated[
        str,
        typer.Option("--search-mode"),
    ] = "category-score-merge",
    unified_candidate_k: Annotated[
        int,
        typer.Option("--unified-candidate-k"),
    ] = 100,
) -> None:
    """Run Phase 9 B0/B1/B2-score/P-score retrieval variants."""

    _validate_run_primary_options(
        router=router,
        search_mode=search_mode,
        unified_candidate_k=unified_candidate_k,
    )
    config = load_indexing_config(config_path)
    model_name = embedding_model_name or config.embedding_model
    query_rows = load_query_rows(queries_path)
    settings = load_primary_run_settings(
        settings_path,
        category_stats_path=category_stats_path,
    )
    embedding_model = _load_embedding_model(embedding_backend, model_name)
    embeddings_by_query_id = embed_query_rows(
        query_rows,
        embedding_model=embedding_model,
        normalize_embeddings=config.normalize_embeddings,
    )
    probabilities_by_query_id, probability_source = _load_run_primary_probabilities(
        query_rows,
        router=router,
        query_classifier_path=query_classifier_path,
        classifier_device=classifier_device,
        categories=tuple(settings.lambda_by_category),
    )
    query_features = build_query_features(
        query_rows,
        embeddings_by_query_id=embeddings_by_query_id,
        probabilities_by_query_id=probabilities_by_query_id,
    )
    result_rows = run_primary_queries(
        query_features,
        search_backend=FaissSearchBackend(indexes_path),
        settings=settings,
        search_mode=search_mode,
        unified_candidate_k=unified_candidate_k,
    )
    write_primary_run(
        output_dir=output_path,
        result_rows=result_rows,
        settings=settings,
        run_metadata={
            "queries_path": str(queries_path),
            "settings_path": str(settings_path),
            "indexes_path": str(indexes_path),
            "config_path": str(config_path),
            "embedding_model": model_name,
            "normalize_embeddings": config.normalize_embeddings,
            "probability_source": probability_source,
            "router": router,
            "search_mode": search_mode,
            "unified_candidate_k": unified_candidate_k,
        },
    )
    typer.echo(
        f"Wrote Phase 9 primary retrieval results to {output_path} "
        f"({len(query_features)} queries, 4 variants)"
    )


@app.command("analyze-primary")
def analyze_primary_command(
    run_path: Annotated[
        Path,
        typer.Option("--run", exists=True, file_okay=False, readable=True),
    ],
    output_path: Annotated[Path, typer.Option("--output")],
    chunks_path: Annotated[
        Path | None,
        typer.Option("--chunks", exists=True, dir_okay=False, readable=True),
    ] = None,
    metric_key: Annotated[str, typer.Option("--metric")] = "ndcg@10",
    top_failures: Annotated[int, typer.Option("--top-failures")] = 20,
    bootstrap_samples: Annotated[
        int,
        typer.Option("--bootstrap-samples"),
    ] = 1000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Analyze Phase 9 primary retrieval results and write an HTML report."""

    result_rows = load_primary_result_rows(run_path)
    chunk_lookup = load_chunk_lookup(chunks_path)
    analysis = analyze_primary_results(
        result_rows,
        metric_key=metric_key,
        top_failures=top_failures,
        chunk_lookup=chunk_lookup,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    write_primary_analysis(
        output_dir=output_path,
        analysis=analysis,
        run_dir=run_path,
        chunks_path=chunks_path,
    )
    summary = analysis["summary"]
    typer.echo(
        f"Wrote Phase 9 retrieval analysis to {output_path} "
        f"({summary['query_count']} queries, metric={metric_key})"
    )


@app.command("serve-api")
def serve_api_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 5070,
    platform: Annotated[
        str | None,
        typer.Option(
            "--platform",
            help="LLM runtime platform: MLX, ROCm, or CUDA. ROCm/CUDA use vLLM.",
        ),
    ] = None,
) -> None:
    """Serve the REST API for one-query P-score RAG generation."""

    if platform is not None:
        from darwin_rag_exp2.api.runtime import normalize_llm_platform

        try:
            os.environ["DARWIN_EXP2_LLM_PLATFORM"] = normalize_llm_platform(platform)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    try:
        import uvicorn
    except ImportError as error:
        raise typer.BadParameter(
            "uvicorn is required; run with the api dependency group"
        ) from error
    uvicorn.run(
        "darwin_rag_exp2.api.app:app",
        host=host,
        port=port,
    )


def _load_query_probabilities(
    query_rows: Sequence[dict[str, object]],
    *,
    query_classifier_path: Path | None,
    classifier_device: str,
) -> tuple[dict[str, dict[str, float]], str]:
    try:
        return probabilities_from_query_rows(query_rows), "query_jsonl"
    except ValueError as error:
        if query_classifier_path is None or not query_classifier_path.exists():
            raise typer.BadParameter(
                "query rows do not contain probabilities; provide "
                "--query-classifier pointing to the final classifier artifact"
            ) from error
    return _predict_query_probabilities(
        query_rows,
        query_classifier_path=query_classifier_path,
        classifier_device=classifier_device,
    )


def _load_run_primary_probabilities(
    query_rows: Sequence[dict[str, object]],
    *,
    router: str,
    query_classifier_path: Path | None,
    classifier_device: str,
    categories: Sequence[str],
) -> tuple[dict[str, dict[str, float]], str]:
    if router == "oracle":
        try:
            return (
                oracle_probabilities_from_query_rows(
                    query_rows,
                    categories=categories,
                ),
                "oracle_gold_categories",
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    if router == "precomputed":
        try:
            return probabilities_from_query_rows(query_rows), "query_jsonl"
        except ValueError as error:
            raise typer.BadParameter(
                "query rows do not contain probabilities for --router precomputed"
            ) from error
    if router == "final-classifier":
        if query_classifier_path is None or not query_classifier_path.exists():
            raise typer.BadParameter(
                "--router final-classifier requires --query-classifier pointing "
                "to the final classifier artifact"
            )
        return _predict_query_probabilities(
            query_rows,
            query_classifier_path=query_classifier_path,
            classifier_device=classifier_device,
        )
    raise typer.BadParameter(
        "unknown router; expected one of: final-classifier, precomputed, oracle"
    )


def _predict_query_probabilities(
    query_rows: Sequence[dict[str, object]],
    *,
    query_classifier_path: Path,
    classifier_device: str,
) -> tuple[dict[str, dict[str, float]], str]:
    classifier = FinalQueryClassifier(
        query_classifier_path,
        device=classifier_device,
        progress_callback=typer.echo,
    )
    predictions = classifier.predict_probabilities(
        [str(row["query"]) for row in query_rows]
    )
    return {
        str(row["query_id"]): probabilities
        for row, probabilities in zip(query_rows, predictions, strict=True)
    }, str(query_classifier_path)


def _validate_run_primary_options(
    *,
    router: str,
    search_mode: str,
    unified_candidate_k: int,
) -> None:
    if router not in {"final-classifier", "precomputed", "oracle"}:
        raise typer.BadParameter(
            "unknown router; expected one of: final-classifier, precomputed, oracle"
        )
    if search_mode not in SEARCH_MODES:
        raise typer.BadParameter(
            f"unknown search mode; expected one of: {', '.join(SEARCH_MODES)}"
        )
    if unified_candidate_k <= 0:
        raise typer.BadParameter("--unified-candidate-k must be positive")


def _retrieval_defaults_from_config(config_path: Path) -> dict[str, object]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    retrieval = payload.get("retrieval") or {}
    if not isinstance(retrieval, dict):
        raise typer.BadParameter("config retrieval section must be a mapping")
    return {
        "candidate_k_per_partition": int(
            retrieval.get("candidate_k_per_partition", 50)
        ),
        "report_top_k": int(retrieval.get("report_top_k", 10)),
        "generation_context_top_n": int(
            retrieval.get("generation_context_top_n", 5)
        ),
        "fixed_lambda_grid": list(
            retrieval.get(
                "fixed_lambda_grid",
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        ),
    }


def _load_category_stats_rows(path: Path) -> list[dict[str, object]]:
    payload = orjson.loads(path.read_bytes())
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise typer.BadParameter("category stats must contain a rows list")
    return [dict(row) for row in rows]


def _parse_float_grid(value: str) -> list[float]:
    try:
        grid = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as error:
        raise typer.BadParameter(f"invalid float grid: {value}") from error
    if not grid:
        raise typer.BadParameter("grid must contain at least one value")
    return grid


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        + b"\n"
    )


def _load_embedding_model(
    embedding_backend: str,
    model_name: str,
) -> HashEmbeddingModel | SentenceTransformerEmbeddingModel:
    if embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbeddingModel(model_name)
    if embedding_backend == "hash":
        return HashEmbeddingModel()
    raise typer.BadParameter("supported embedding backends: sentence-transformers, hash")


def main(argv: Sequence[str] | None = None) -> int:
    """Invoke the Exp2 command-line application."""

    app(args=list(argv) if argv is not None else None, standalone_mode=False)
    return 0
