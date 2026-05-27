import json

from darwin_rag_exp2.cli import main
from darwin_rag_exp2.data.audit import audit_notice_export, write_audit_artifacts


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


def write_records(path, records: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in records)
    path.write_text(f"{payload}\n", encoding="utf-8")


def test_audit_counts_data_quality_signals_and_writes_artifacts(tmp_path) -> None:
    source = tmp_path / "notices.jsonl"
    output = tmp_path / "audit"
    write_records(
        source,
        [
            notice_record(text="공지 제목", text_length=len("공지 제목")),
            notice_record(
                id="notice-1",
                url="https://example.test/notices/1",
                category="장학",
                text="장학 공지 본문",
                text_length=999,
            ),
            notice_record(
                id="notice-3",
                url="https://example.test/notices/3",
                category="장학",
                text="정상 본문입니다",
                text_length=len("정상 본문입니다"),
            ),
        ],
    )

    report = audit_notice_export(source)
    write_audit_artifacts(report, output)

    assert report.record_count == 3
    assert report.valid_record_count == 3
    assert report.duplicate_id_count == 1
    assert report.duplicate_url_count == 1
    assert report.text_length_mismatch_count == 1
    assert report.title_only_count == 1
    assert report.category_counts == {"장학": 2, "학사": 1}
    serialized = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert serialized["record_count"] == 3
    assert "Category Counts" in (output / "audit.md").read_text(encoding="utf-8")


def test_audit_data_command_creates_report_files(tmp_path) -> None:
    source = tmp_path / "notices.jsonl"
    output = tmp_path / "audit"
    write_records(source, [notice_record()])

    result = main(["audit-data", "--input", str(source), "--output", str(output)])

    assert result == 0
    assert (output / "audit.json").exists()
    assert (output / "audit.md").exists()


def test_audit_markdown_renders_schema_error_as_one_table_row(tmp_path) -> None:
    source = tmp_path / "notices.jsonl"
    output = tmp_path / "audit"
    write_records(source, [notice_record(text_length="unknown")])

    report = audit_notice_export(source)
    write_audit_artifacts(report, output)

    markdown = (output / "audit.md").read_text(encoding="utf-8")
    issue_rows = [line for line in markdown.splitlines() if "schema_error" in line]
    assert len(issue_rows) == 1
    assert "validation error for NoticeRecord" in issue_rows[0]
    assert "text_length" in issue_rows[0]
