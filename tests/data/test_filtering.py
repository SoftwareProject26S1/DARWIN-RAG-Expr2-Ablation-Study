import json

from darwin_rag_exp2.cli import main
from darwin_rag_exp2.data.filtering import (
    CorpusFilterConfig,
    prepare_corpus,
    write_corpus_artifacts,
)


def notice_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "notice-1",
        "url": "https://example.test/notices/1",
        "slug": "notice-1",
        "title": "공지 제목",
        "date": "2026-05-01",
        "category": "학사",
        "text": "공지 제목 " + long_body(),
        "text_length": len("공지 제목 " + long_body()),
        "source": "scatch",
        "collected_at": "2026-05-01 10:00:00",
    }
    record.update(overrides)
    if "text_length" not in overrides:
        record["text_length"] = len(str(record["text"]))
    return record


def long_body(prefix: str = "본문", count: int = 30) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def write_records(path, records: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in records)
    path.write_text(f"{payload}\n", encoding="utf-8")


def test_prepare_corpus_admits_primary_categories_and_records_filter_reasons(
    tmp_path,
) -> None:
    source = tmp_path / "notices.jsonl"
    write_records(
        source,
        [
            notice_record(id="haksa-1", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="haksa-2", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="etc-1", category="기타", text="기타 제목 " + long_body()),
            notice_record(id="title-only", title="제목만 있음", text="제목만 있음"),
            notice_record(
                id="short-body",
                title="짧은 본문",
                text="짧은 본문 너무 짧음",
            ),
            notice_record(
                id="unsupported",
                category="입학",
                text="입학 공지 " + long_body(),
            ),
        ],
    )
    config = CorpusFilterConfig(
        primary_categories=("학사",),
        excluded_category_reasons={"기타": "excluded_ambiguous_category"},
        minimum_body_tokens=30,
        min_primary_source_documents=2,
    )

    corpus = prepare_corpus(source, config)

    assert [record.id for record in corpus.admitted_records] == ["haksa-1", "haksa-2"]
    assert corpus.admitted_category_counts == {"학사": 2}
    assert corpus.filter_reason_counts == {
        "excluded_category": 1,
        "title_only": 1,
        "body_too_short": 1,
        "unsupported_category": 1,
    }
    assert {
        record.source_record.id: record.reason for record in corpus.excluded_records
    } == {
        "etc-1": "excluded_category",
        "title-only": "title_only",
        "short-body": "body_too_short",
        "unsupported": "unsupported_category",
    }


def test_prepare_corpus_rejects_primary_categories_below_minimum(tmp_path) -> None:
    source = tmp_path / "notices.jsonl"
    write_records(
        source,
        [
            notice_record(id="haksa-1", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="haksa-2", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="bongsa-1", category="봉사", text="봉사 제목 " + long_body()),
        ],
    )
    config = CorpusFilterConfig(
        primary_categories=("학사", "봉사"),
        excluded_category_reasons={},
        minimum_body_tokens=30,
        min_primary_source_documents=2,
    )

    corpus = prepare_corpus(source, config)

    assert [record.id for record in corpus.admitted_records] == ["haksa-1", "haksa-2"]
    assert corpus.rejected_primary_categories == {"봉사": 1}
    assert corpus.filter_reason_counts == {"category_below_minimum": 1}
    assert corpus.excluded_records[0].source_record.id == "bongsa-1"
    assert corpus.excluded_records[0].reason == "category_below_minimum"


def test_write_corpus_artifacts_serializes_outputs_and_manifest(tmp_path) -> None:
    source = tmp_path / "notices.jsonl"
    output = tmp_path / "corpus"
    write_records(
        source,
        [
            notice_record(id="haksa-1", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="haksa-2", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="etc-1", category="기타", text="기타 제목 " + long_body()),
        ],
    )
    corpus = prepare_corpus(
        source,
        CorpusFilterConfig(
            primary_categories=("학사",),
            excluded_category_reasons={"기타": "excluded_ambiguous_category"},
            minimum_body_tokens=30,
            min_primary_source_documents=2,
        ),
    )

    write_corpus_artifacts(corpus, output)

    admitted = [
        json.loads(line)
        for line in (output / "admitted.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    excluded = [
        json.loads(line)
        for line in (output / "excluded.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    category_counts = json.loads(
        (output / "category_counts.json").read_text(encoding="utf-8")
    )
    filter_reasons = json.loads(
        (output / "filter_reason_counts.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert [record["id"] for record in admitted] == ["haksa-1", "haksa-2"]
    assert excluded == [
        {
            "id": "etc-1",
            "category": "기타",
            "reason": "excluded_category",
            "reason_detail": "excluded_ambiguous_category",
        }
    ]
    assert category_counts == {"학사": 2}
    assert filter_reasons == {"excluded_category": 1}
    assert manifest["source_sha256"] == corpus.source_sha256
    assert manifest["minimum_body_tokens"] == 30
    assert manifest["min_primary_source_documents"] == 2


def test_prepare_corpus_command_creates_phase3_artifacts(tmp_path) -> None:
    source = tmp_path / "notices.jsonl"
    config_path = tmp_path / "experiment.yaml"
    mapping_path = tmp_path / "category_mapping.yaml"
    output = tmp_path / "corpus"
    write_records(
        source,
        [
            notice_record(id="haksa-1", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="haksa-2", category="학사", text="공지 제목 " + long_body()),
            notice_record(id="etc-1", category="기타", text="기타 제목 " + long_body()),
        ],
    )
    config_path.write_text(
        """
experiment:
  primary_categories:
    - "학사"
data:
  min_primary_source_documents: 2
chunking:
  minimum_information_tokens: 30
""".lstrip(),
        encoding="utf-8",
    )
    mapping_path.write_text(
        """
excluded_categories:
  "기타": "excluded_ambiguous_category"
""".lstrip(),
        encoding="utf-8",
    )

    result = main(
        [
            "prepare-corpus",
            "--input",
            str(source),
            "--config",
            str(config_path),
            "--mapping",
            str(mapping_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert (output / "admitted.jsonl").exists()
    assert (output / "excluded.jsonl").exists()
    assert (output / "manifest.json").exists()
