from darwin_rag_exp2.indexing.partitions import build_partition_assignments


def test_build_partition_assignments_uses_threshold_and_top1_fallback() -> None:
    prediction_rows = [
        {
            "chunk_id": "c1",
            "probabilities": {"학사": 0.8, "장학": 0.2},
        },
        {
            "chunk_id": "c2",
            "probabilities": {"학사": 0.7, "장학": 0.75},
        },
        {
            "chunk_id": "c3",
            "probabilities": {"학사": 0.45, "장학": 0.55},
        },
    ]

    assignments = build_partition_assignments(
        prediction_rows,
        ingest_threshold=0.6,
    )

    assert [
        (row["chunk_id"], row["category"], row["assignment_reason"])
        for row in assignments
    ] == [
        ("c1", "학사", "threshold"),
        ("c2", "장학", "threshold"),
        ("c2", "학사", "threshold"),
        ("c3", "장학", "top1_fallback"),
    ]
    assert [row["chunk_id"] for row in assignments if row["chunk_id"] == "c2"] == [
        "c2",
        "c2",
    ]


def test_build_partition_assignments_rejects_missing_probabilities() -> None:
    prediction_rows = [{"chunk_id": "c1", "probabilities": {}}]

    try:
        build_partition_assignments(prediction_rows, ingest_threshold=0.5)
    except ValueError as error:
        assert "probabilities" in str(error)
    else:
        raise AssertionError("expected missing probabilities to be rejected")
