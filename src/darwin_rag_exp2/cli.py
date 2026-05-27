"""Command-line entrypoint for the Exp2 experiment package."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from .data.audit import audit_notice_export, write_audit_artifacts
from .data.filtering import (
    load_corpus_filter_config,
    prepare_corpus,
    write_corpus_artifacts,
)


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


def main(argv: Sequence[str] | None = None) -> int:
    """Invoke the Exp2 command-line application."""

    app(args=list(argv) if argv is not None else None, standalone_mode=False)
    return 0
