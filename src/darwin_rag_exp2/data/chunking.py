"""Phase 4 paragraph-first chunking for admitted notice records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Protocol

import yaml

from .importer import load_notice_export
from .schema import NoticeRecord


_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


class Tokenizer(Protocol):
    """Minimal tokenizer surface required by the artifact builder."""

    name_or_path: str

    def encode(self, text: str, add_special_tokens: bool = False) -> list[Any]:
        """Return token ids or token-like values for budget accounting."""

    def decode(
        self,
        tokens: Sequence[Any],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """Return text for a previously encoded token sequence."""


@dataclass(frozen=True)
class ChunkingConfig:
    """Frozen Phase 4 token-budget settings."""

    tokenizer_name: str
    target_body_tokens: int
    overlap_body_tokens: int
    minimum_information_tokens: int
    title_prefix_max_tokens: int
    classifier_max_tokens: int


@dataclass(frozen=True)
class NoticeChunk:
    """One stable retrieval/classifier unit derived from a source notice."""

    chunk_id: str
    source_id: str
    chunk_index: int
    category: str
    title: str
    title_prefix: str
    body_text: str
    classifier_text: str
    body_token_count: int
    title_token_count: int
    classifier_token_count: int
    body_tokens: tuple[Any, ...]
    url: str
    slug: str
    date: str
    source: str
    collected_at: str


@dataclass(frozen=True)
class ChunkingResult:
    """Chunks plus deterministic manifest metadata."""

    chunks: tuple[NoticeChunk, ...]
    manifest: dict[str, object]
    length_histogram: dict[str, dict[str, int]]


def _file_sha256(source_path: Path) -> str:
    digest = sha256()
    with source_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        msg = f"YAML root must be a mapping: {path}"
        raise ValueError(msg)
    return payload


def load_chunking_config(config_path: Path) -> ChunkingConfig:
    """Load Phase 4 chunking settings from an experiment YAML file."""

    config = _load_yaml(config_path)
    chunking = config.get("chunking", {})
    if not isinstance(chunking, dict):
        raise ValueError("chunking config section must be a mapping")

    tokenizer_name = chunking.get("counting_tokenizer")
    if not isinstance(tokenizer_name, str):
        raise ValueError("chunking.counting_tokenizer must be a string")

    integer_fields = {
        "target_body_tokens": chunking.get("target_body_tokens"),
        "overlap_body_tokens": chunking.get("overlap_body_tokens"),
        "minimum_information_tokens": chunking.get("minimum_information_tokens"),
        "title_prefix_max_tokens": chunking.get("title_prefix_max_tokens"),
        "classifier_max_tokens": chunking.get("classifier_max_tokens"),
    }
    if not all(isinstance(value, int) for value in integer_fields.values()):
        raise ValueError("all chunking token budget fields must be integers")

    loaded = ChunkingConfig(tokenizer_name=tokenizer_name, **integer_fields)
    _validate_config(loaded)
    return loaded


def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    """Load the configured Hugging Face tokenizer lazily."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def build_chunks(
    corpus_path: Path,
    config: ChunkingConfig,
    tokenizer: Tokenizer | None = None,
) -> ChunkingResult:
    """Build stable Phase 4 chunks from a Phase 3 admitted corpus JSONL."""

    _validate_config(config)
    active_tokenizer = tokenizer or load_tokenizer(config.tokenizer_name)
    imported = load_notice_export(corpus_path)
    if imported.issues:
        msg = "Phase 4 chunking requires a schema-valid admitted corpus"
        raise ValueError(msg)

    chunks: list[NoticeChunk] = []
    for record in imported.records:
        chunks.extend(_chunk_record(record, config, active_tokenizer))

    body_histogram = Counter(chunk.body_token_count for chunk in chunks)
    classifier_histogram = Counter(chunk.classifier_token_count for chunk in chunks)
    body_cap_violations = [
        chunk for chunk in chunks if chunk.body_token_count > config.target_body_tokens
    ]
    short_chunks = [
        chunk
        for chunk in chunks
        if chunk.body_token_count < config.minimum_information_tokens
    ]
    violating_chunks = [
        chunk
        for chunk in chunks
        if chunk.classifier_token_count > config.classifier_max_tokens
    ]
    manifest: dict[str, object] = {
        "phase": 4,
        "source_corpus_path": str(corpus_path),
        "source_corpus_sha256": _file_sha256(corpus_path),
        "source_record_count": len(imported.records),
        "chunk_count": len(chunks),
        "tokenizer_name": config.tokenizer_name,
        "target_body_tokens": config.target_body_tokens,
        "overlap_body_tokens": config.overlap_body_tokens,
        "minimum_information_tokens": config.minimum_information_tokens,
        "title_prefix_max_tokens": config.title_prefix_max_tokens,
        "classifier_max_tokens": config.classifier_max_tokens,
        "max_body_tokens": max(body_histogram, default=0),
        "max_classifier_tokens": max(classifier_histogram, default=0),
        "violating_body_token_cap_count": len(body_cap_violations),
        "violating_classifier_token_cap_count": len(violating_chunks),
        "violating_minimum_information_token_count": len(short_chunks),
    }
    return ChunkingResult(
        chunks=tuple(chunks),
        manifest=manifest,
        length_histogram={
            "body_token_count": _string_keyed_counts(body_histogram),
            "classifier_token_count": _string_keyed_counts(classifier_histogram),
        },
    )


def _validate_config(config: ChunkingConfig) -> None:
    if config.target_body_tokens <= 0:
        raise ValueError("target_body_tokens must be positive")
    if config.minimum_information_tokens <= 0:
        raise ValueError("minimum_information_tokens must be positive")
    if config.overlap_body_tokens < 0:
        raise ValueError("overlap_body_tokens must not be negative")
    if config.overlap_body_tokens >= config.target_body_tokens:
        raise ValueError("overlap_body_tokens must be smaller than target_body_tokens")
    if config.title_prefix_max_tokens < 0:
        raise ValueError("title_prefix_max_tokens must not be negative")
    if config.classifier_max_tokens <= config.title_prefix_max_tokens:
        raise ValueError("classifier_max_tokens must exceed title_prefix_max_tokens")


def _string_keyed_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(key): counts[key] for key in sorted(counts)}


def _encode(tokenizer: Tokenizer, text: str) -> list[Any]:
    try:
        return list(
            tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                verbose=False,
            )
        )
    except TypeError:
        return list(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer: Tokenizer, tokens: Sequence[Any]) -> str:
    return tokenizer.decode(
        list(tokens),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _decode_to_token_cap(
    tokenizer: Tokenizer,
    tokens: Sequence[Any],
    max_tokens: int,
) -> str:
    capped_tokens = list(tokens)
    while capped_tokens:
        decoded = _decode(tokenizer, capped_tokens)
        if len(_encode(tokenizer, decoded)) <= max_tokens:
            return decoded
        capped_tokens = capped_tokens[:-1]
    return ""


def _normalized_space(value: str) -> str:
    return " ".join(value.split())


def _record_body_text(record: NoticeRecord) -> str:
    raw_text = record.text.strip()
    title = record.title.strip()
    normalized_text = _normalized_space(raw_text)
    normalized_title = _normalized_space(title)
    if not normalized_title:
        return raw_text
    if normalized_text == normalized_title:
        return ""
    if raw_text.startswith(title):
        return raw_text[len(title) :].strip()
    title_prefix = f"{normalized_title} "
    if normalized_text.startswith(title_prefix):
        return normalized_text[len(title_prefix) :].strip()
    return raw_text


def _split_paragraphs(body_text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n+", body_text.strip())
    return [_normalized_space(paragraph) for paragraph in paragraphs if paragraph.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    sentences = [
        _normalized_space(match.group(0))
        for match in _SENTENCE_PATTERN.finditer(paragraph)
        if match.group(0).strip()
    ]
    return sentences or [_normalized_space(paragraph)]


def _semantic_units(
    body_text: str,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[str]:
    units: list[str] = []
    for paragraph in _split_paragraphs(body_text):
        if len(_encode(tokenizer, paragraph)) <= config.target_body_tokens:
            units.append(paragraph)
            continue
        units.extend(_split_sentences(paragraph))
    return [unit for unit in units if unit]


def _token_windows(
    unit: str,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[str]:
    tokens = _encode(tokenizer, unit)
    windows: list[str] = []
    step = config.target_body_tokens - config.overlap_body_tokens
    start = 0
    while start < len(tokens):
        window_tokens = tokens[start : start + config.target_body_tokens]
        windows.append(
            _decode_to_token_cap(tokenizer, window_tokens, config.target_body_tokens)
        )
        if start + config.target_body_tokens >= len(tokens):
            break
        start += step
    return windows


def _fit_with_overlap(
    current_text: str,
    next_unit: str,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> str:
    if config.overlap_body_tokens == 0:
        return next_unit
    overlap_tokens = _encode(tokenizer, current_text)[-config.overlap_body_tokens :]
    overlap_text = _decode(tokenizer, overlap_tokens)
    candidate = f"{overlap_text} {next_unit}".strip()
    if len(_encode(tokenizer, candidate)) <= config.target_body_tokens:
        return candidate
    return next_unit


def _body_segments(
    body_text: str,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[str]:
    segments: list[str] = []
    current = ""

    for unit in _semantic_units(body_text, config, tokenizer):
        unit_token_count = len(_encode(tokenizer, unit))
        if unit_token_count > config.target_body_tokens:
            if current:
                if len(_encode(tokenizer, current)) < config.minimum_information_tokens:
                    segments.extend(
                        _token_windows(f"{current}\n\n{unit}", config, tokenizer)
                    )
                    current = ""
                    continue
                segments.append(current)
                current = ""
            segments.extend(_token_windows(unit, config, tokenizer))
            continue

        if not current:
            current = unit
            continue

        candidate = f"{current}\n\n{unit}".strip()
        if len(_encode(tokenizer, candidate)) <= config.target_body_tokens:
            current = candidate
            continue

        if len(_encode(tokenizer, current)) < config.minimum_information_tokens:
            segments.extend(_token_windows(candidate, config, tokenizer))
            current = ""
            continue

        segments.append(current)
        current = _fit_with_overlap(current, unit, config, tokenizer)

    if current:
        segments.append(current)

    return _rebalance_short_segments(segments, config, tokenizer)


def _rebalance_short_segments(
    segments: list[str],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[str]:
    rebalanced: list[str] = []
    for segment in segments:
        if not rebalanced:
            rebalanced.append(segment)
            continue
        if (
            len(_encode(tokenizer, rebalanced[-1]))
            >= config.minimum_information_tokens
        ):
            rebalanced.append(segment)
            continue

        combined = f"{rebalanced[-1]}\n\n{segment}".strip()
        if len(_encode(tokenizer, combined)) <= config.target_body_tokens:
            rebalanced[-1] = combined
            continue

        windows = _token_windows(combined, config, tokenizer)
        rebalanced[-1] = windows[0]
        rebalanced.extend(windows[1:])

    return _merge_short_tail(rebalanced, config, tokenizer)


def _merge_short_tail(
    segments: list[str],
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[str]:
    if len(segments) < 2:
        return segments
    tail_tokens = _encode(tokenizer, segments[-1])
    if len(tail_tokens) >= config.minimum_information_tokens:
        return segments
    combined = f"{segments[-2]}\n\n{segments[-1]}".strip()
    if len(_encode(tokenizer, combined)) <= config.target_body_tokens:
        return [*segments[:-2], combined]
    borrowed_tail = _decode_to_token_cap(
        tokenizer,
        _encode(tokenizer, combined)[-config.target_body_tokens :],
        config.target_body_tokens,
    )
    if len(_encode(tokenizer, borrowed_tail)) >= config.minimum_information_tokens:
        return [*segments[:-1], borrowed_tail]
    return segments


def _title_prefix(record: NoticeRecord, config: ChunkingConfig, tokenizer: Tokenizer) -> str:
    title_tokens = _encode(tokenizer, record.title)
    return _decode_to_token_cap(
        tokenizer,
        title_tokens[: config.title_prefix_max_tokens],
        config.title_prefix_max_tokens,
    )


def _classifier_text(title_prefix: str, body_text: str) -> str:
    if title_prefix:
        return f"{title_prefix}\n\n{body_text}"
    return body_text


def _chunk_record(
    record: NoticeRecord,
    config: ChunkingConfig,
    tokenizer: Tokenizer,
) -> list[NoticeChunk]:
    body_text = _record_body_text(record)
    title_prefix = _title_prefix(record, config, tokenizer)
    title_token_count = len(_encode(tokenizer, title_prefix))
    chunks: list[NoticeChunk] = []

    for chunk_index, segment in enumerate(_body_segments(body_text, config, tokenizer)):
        body_tokens = tuple(_encode(tokenizer, segment))
        classifier_text = _classifier_text(title_prefix, segment)
        chunks.append(
            NoticeChunk(
                chunk_id=f"{record.id}::{chunk_index:04d}",
                source_id=record.id,
                chunk_index=chunk_index,
                category=record.category,
                title=record.title,
                title_prefix=title_prefix,
                body_text=segment,
                classifier_text=classifier_text,
                body_token_count=len(body_tokens),
                title_token_count=title_token_count,
                classifier_token_count=len(_encode(tokenizer, classifier_text)),
                body_tokens=body_tokens,
                url=record.url,
                slug=record.slug,
                date=record.date,
                source=record.source,
                collected_at=record.collected_at,
            )
        )
    return chunks
