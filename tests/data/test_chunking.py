import json

from darwin_rag_exp2.cli import main
from darwin_rag_exp2.data.artifacts import write_chunk_artifacts
from darwin_rag_exp2.data.chunking import (
    ChunkingConfig,
    build_chunks,
    load_chunking_config,
)


class WhitespaceTokenizer:
    name_or_path = "test-whitespace"

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(
        self,
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(tokens)


class ExpandingDecodeTokenizer(WhitespaceTokenizer):
    name_or_path = "test-expanding-decode"

    def decode(
        self,
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join([*tokens, "확장"])


def notice_record(**overrides: object) -> dict[str, object]:
    body = overrides.pop("body", long_body("본문", 34))
    title = str(overrides.get("title", "공지 제목"))
    text = f"{title} {body}"
    record: dict[str, object] = {
        "id": "notice-1",
        "url": "https://example.test/notices/1",
        "slug": "notice-1",
        "title": title,
        "date": "2026-05-01",
        "category": "학사",
        "text": text,
        "text_length": len(text),
        "source": "scatch",
        "collected_at": "2026-05-01 10:00:00",
    }
    record.update(overrides)
    if "text_length" not in overrides:
        record["text_length"] = len(str(record["text"]))
    return record


def long_body(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def write_records(path, records: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in records)
    path.write_text(f"{payload}\n", encoding="utf-8")


def chunking_config(**overrides: object) -> ChunkingConfig:
    values = {
        "tokenizer_name": "test-whitespace",
        "target_body_tokens": 12,
        "overlap_body_tokens": 3,
        "minimum_information_tokens": 5,
        "title_prefix_max_tokens": 4,
        "classifier_max_tokens": 20,
    }
    values.update(overrides)
    return ChunkingConfig(**values)


def test_build_chunks_respects_token_caps_and_stable_ids(tmp_path) -> None:
    source = tmp_path / "admitted.jsonl"
    body = "\n\n".join(
        [
            long_body("첫문단", 8),
            long_body("둘째문단", 8),
            long_body("셋째문단", 8),
        ]
    )
    write_records(source, [notice_record(id="notice-42", title="긴 공지 제목", body=body)])

    result = build_chunks(
        source,
        chunking_config(),
        tokenizer=WhitespaceTokenizer(),
    )

    assert [chunk.chunk_id for chunk in result.chunks] == [
        "notice-42::0000",
        "notice-42::0001",
        "notice-42::0002",
    ]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2]
    assert all(chunk.body_token_count <= 12 for chunk in result.chunks)
    assert all(chunk.classifier_token_count <= 20 for chunk in result.chunks)
    assert all(chunk.body_token_count >= 5 for chunk in result.chunks)
    assert result.manifest["violating_classifier_token_cap_count"] == 0
    assert result.manifest["chunk_count"] == 3


def test_build_chunks_falls_back_to_overlapping_token_windows_for_long_units(
    tmp_path,
) -> None:
    source = tmp_path / "admitted.jsonl"
    write_records(
        source,
        [notice_record(id="long-unit", body=long_body("토큰", 27))],
    )

    result = build_chunks(
        source,
        chunking_config(),
        tokenizer=WhitespaceTokenizer(),
    )

    assert [chunk.body_token_count for chunk in result.chunks] == [12, 12, 9]
    assert result.chunks[0].body_tokens[-3:] == result.chunks[1].body_tokens[:3]
    assert result.chunks[1].body_tokens[-3:] == result.chunks[2].body_tokens[:3]
    assert result.manifest["max_body_tokens"] == 12


def test_build_chunks_does_not_emit_short_paragraph_fragments(tmp_path) -> None:
    source = tmp_path / "admitted.jsonl"
    body = "\n\n".join(
        [
            long_body("짧음", 2),
            long_body("긴문단", 12),
            long_body("다음문단", 12),
        ]
    )
    write_records(source, [notice_record(id="short-fragment", body=body)])

    result = build_chunks(
        source,
        chunking_config(),
        tokenizer=WhitespaceTokenizer(),
    )

    assert all(5 <= chunk.body_token_count <= 12 for chunk in result.chunks)
    assert result.chunks[0].body_tokens[:2] == ("짧음0", "짧음1")


def test_build_chunks_rechecks_token_cap_after_decoding_windows(tmp_path) -> None:
    source = tmp_path / "admitted.jsonl"
    write_records(
        source,
        [notice_record(id="decode-expands", body=long_body("토큰", 9))],
    )

    result = build_chunks(
        source,
        chunking_config(
            tokenizer_name="test-expanding-decode",
            target_body_tokens=4,
            overlap_body_tokens=1,
            minimum_information_tokens=2,
            title_prefix_max_tokens=2,
            classifier_max_tokens=10,
        ),
        tokenizer=ExpandingDecodeTokenizer(),
    )

    assert all(chunk.body_token_count <= 4 for chunk in result.chunks)


def test_write_chunk_artifacts_serializes_jsonl_parquet_histogram_and_manifest(
    tmp_path,
) -> None:
    import pyarrow.parquet as pq

    source = tmp_path / "admitted.jsonl"
    output = tmp_path / "chunks"
    write_records(source, [notice_record(body=long_body("본문", 18))])
    result = build_chunks(
        source,
        chunking_config(),
        tokenizer=WhitespaceTokenizer(),
    )

    write_chunk_artifacts(result, output)

    jsonl_rows = [
        json.loads(line)
        for line in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    parquet_rows = pq.read_table(output / "chunks.parquet").to_pylist()
    histogram = json.loads(
        (output / "length_histogram.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert jsonl_rows[0]["chunk_id"] == "notice-1::0000"
    assert parquet_rows[0]["chunk_id"] == "notice-1::0000"
    assert histogram["body_token_count"]["12"] == 1
    assert manifest["artifact_files"] == [
        "chunks.jsonl",
        "chunks.parquet",
        "length_histogram.json",
        "manifest.json",
    ]
    assert manifest["violating_classifier_token_cap_count"] == 0


def test_chunk_corpus_command_creates_phase4_artifacts(tmp_path, monkeypatch) -> None:
    source = tmp_path / "admitted.jsonl"
    config_path = tmp_path / "experiment.yaml"
    output = tmp_path / "chunks"
    write_records(source, [notice_record(body=long_body("본문", 18))])
    config_path.write_text(
        """
chunking:
  counting_tokenizer: test-whitespace
  target_body_tokens: 12
  overlap_body_tokens: 3
  minimum_information_tokens: 5
  title_prefix_max_tokens: 4
  classifier_max_tokens: 20
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "darwin_rag_exp2.data.chunking.load_tokenizer",
        lambda _: WhitespaceTokenizer(),
    )

    result = main(
        [
            "chunk-corpus",
            "--corpus",
            str(source),
            "--config",
            str(config_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert (output / "chunks.jsonl").exists()
    assert (output / "chunks.parquet").exists()
    assert (output / "length_histogram.json").exists()
    assert (output / "manifest.json").exists()


def test_load_chunking_config_reads_phase4_defaults(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
chunking:
  counting_tokenizer: klue/bert-base
  target_body_tokens: 384
  overlap_body_tokens: 64
  minimum_information_tokens: 30
  title_prefix_max_tokens: 64
  classifier_max_tokens: 512
""".lstrip(),
        encoding="utf-8",
    )

    config = load_chunking_config(config_path)

    assert config == ChunkingConfig(
        tokenizer_name="klue/bert-base",
        target_body_tokens=384,
        overlap_body_tokens=64,
        minimum_information_tokens=30,
        title_prefix_max_tokens=64,
        classifier_max_tokens=512,
    )
