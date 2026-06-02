"""Command-line entrypoint for the Exp2 experiment package."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

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
