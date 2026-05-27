import hashlib
import json

from darwin_rag_exp2.data.importer import load_notice_export


def notice_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "notice-1",
        "url": "https://example.test/notices/1",
        "slug": "notice-1",
        "title": "공지 제목",
        "date": "2026-05-01",
        "category": "학사",
        "text": "공지 본문",
        "text_length": len("공지 본문"),
        "source": "scatch",
        "collected_at": "2026-05-01 10:00:00",
    }
    record.update(overrides)
    return record


def test_load_notice_export_preserves_valid_rows_and_reports_invalid_lines(tmp_path) -> None:
    source = tmp_path / "notices.jsonl"
    valid_line = json.dumps(notice_record(), ensure_ascii=False)
    schema_invalid_line = json.dumps(
        notice_record(id="notice-2", text_length="unknown"), ensure_ascii=False
    )
    source.write_text(
        f"{valid_line}\n{{invalid-json\n{schema_invalid_line}\n",
        encoding="utf-8",
    )

    result = load_notice_export(source)

    assert result.line_count == 3
    assert [record.id for record in result.records] == ["notice-1"]
    assert result.invalid_json_count == 1
    assert result.schema_error_count == 1
    assert [issue.line_number for issue in result.issues] == [2, 3]
    assert result.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
