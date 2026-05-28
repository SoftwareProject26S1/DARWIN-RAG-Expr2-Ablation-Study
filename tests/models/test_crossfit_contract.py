import json

import pyarrow as pa
import pyarrow.parquet as pq

from darwin_rag_exp2.cli import main


def test_train_classifier_crossfit_writes_out_of_fold_contract(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "classifier" / "crossfit"
    rows = [
        chunk_row("a1", "학사", "수강 신청 변경 기간 학사 공지"),
        chunk_row("a2", "학사", "강의 시간표와 학사 일정 안내"),
        chunk_row("a3", "학사", "졸업 요건 학점 이수 안내"),
        chunk_row("s1", "장학", "국가 장학금 신청 서류 안내"),
        chunk_row("s2", "장학", "성적 장학 선발 결과 공지"),
        chunk_row("s3", "장학", "교내 장학 추천서 제출 안내"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), chunks_path)

    result = main(
        [
            "train-classifier",
            "--mode",
            "crossfit",
            "--chunks",
            str(chunks_path),
            "--folds",
            "3",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    manifest = json.loads((output_path / "manifest.json").read_text())
    folds = json.loads((output_path / "folds.json").read_text())["folds"]
    fold_by_index = {fold["fold_index"]: fold for fold in folds}
    predictions = [
        json.loads(line)
        for line in (output_path / "out_of_fold_predictions.jsonl").read_text().splitlines()
    ]
    stats = json.loads((output_path / "category_stats.json").read_text())["rows"]

    assert manifest["phase"] == 6
    assert manifest["mode"] == "crossfit"
    assert manifest["smoke_only"] is False
    assert manifest["probability_source"] == "out_of_fold_calibrated_probabilities"
    assert manifest["lambda_c_interpretation"] == "semantic_similarity_mixture_coefficient"
    assert manifest["lambda_c_not"] == "bert_confidence"
    assert len(predictions) == len(rows)
    assert {row["chunk_id"] for row in predictions} == {row["chunk_id"] for row in rows}

    for prediction in predictions:
        fold = fold_by_index[prediction["fold_index"]]
        assert prediction["source_id"] in fold["validation_source_ids"]
        assert prediction["source_id"] not in fold["training_source_ids"]
        assert prediction["probability_source"] == "out_of_fold"

    assert {row["category"] for row in stats} == {"학사", "장학"}
    assert all(row["smoke_only"] is False for row in stats)
    assert all(row["probability_source"] == "out_of_fold" for row in stats)
    assert all(
        row["lambda_c_interpretation"] == "semantic_similarity_mixture_coefficient"
        for row in stats
    )
    assert all(row["lambda_c_not"] == "bert_confidence" for row in stats)
    assert (output_path / "category_stats.parquet").exists()
    assert (output_path / "calibration_by_fold.json").exists()
    assert (output_path / "model_references.json").exists()


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
