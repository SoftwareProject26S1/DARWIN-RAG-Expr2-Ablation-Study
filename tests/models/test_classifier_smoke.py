import json

import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2.cli import main


def test_train_classifier_single_writes_smoke_artifacts(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "classifier" / "single"
    rows = [
        chunk_row("a1", "학사", "수강 신청 변경 기간 학사 공지"),
        chunk_row("a2", "학사", "학사 일정과 수업 운영 안내"),
        chunk_row("s1", "장학", "장학금 신청 서류 제출 안내"),
        chunk_row("s2", "장학", "국가 장학 선발 결과 공지"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), chunks_path)

    result = main(
        [
            "train-classifier",
            "--mode",
            "single",
            "--chunks",
            str(chunks_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert json.loads((output_path / "manifest.json").read_text())["smoke_only"] is True
    assert (output_path / "model_reference.json").exists()
    assert (output_path / "calibration.json").exists()
    assert (output_path / "sample_predictions.jsonl").exists()
    assert (output_path / "category_stats.parquet").exists()


def chunk_row(source_id: str, category: str, classifier_text: str) -> dict[str, object]:
    return {
        "chunk_id": f"{source_id}::0000",
        "source_id": source_id,
        "chunk_index": 0,
        "category": category,
        "title": classifier_text.split()[0],
        "title_prefix": classifier_text.split()[0],
        "body_text": classifier_text,
        "classifier_text": classifier_text,
        "body_token_count": len(classifier_text.split()),
        "title_token_count": 1,
        "classifier_token_count": len(classifier_text.split()),
        "url": f"https://example.test/{source_id}",
        "slug": source_id,
        "date": "2026-05-01",
        "source": "test",
        "collected_at": "2026-05-01 00:00:00",
    }
