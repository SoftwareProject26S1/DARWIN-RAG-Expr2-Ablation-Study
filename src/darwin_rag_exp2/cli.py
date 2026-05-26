"""Command-line entrypoint for the Exp2 experiment package."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Report that the Phase 1 experiment scaffold is available."""
    del argv
    print("DARWIN-RAG Exp2: Phase 1 scaffold ready.")
    return 0
