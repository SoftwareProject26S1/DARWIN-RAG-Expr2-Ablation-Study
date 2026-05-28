import pytest

from darwin_rag_exp2.models.splits import build_source_folds


def test_build_source_folds_holds_out_each_source_once_without_train_leakage() -> None:
    rows = [
        {"source_id": "a1", "category": "학사"},
        {"source_id": "a1", "category": "학사"},
        {"source_id": "a2", "category": "학사"},
        {"source_id": "s1", "category": "장학"},
        {"source_id": "s2", "category": "장학"},
    ]

    folds = build_source_folds(rows, fold_count=2)

    validation_counts: dict[str, int] = {}
    for fold in folds:
        assert set(fold.training_source_ids).isdisjoint(fold.validation_source_ids)
        for source_id in fold.validation_source_ids:
            validation_counts[source_id] = validation_counts.get(source_id, 0) + 1

    assert [fold.fold_index for fold in folds] == [0, 1]
    assert validation_counts == {"a1": 1, "a2": 1, "s1": 1, "s2": 1}
    assert build_source_folds(rows, fold_count=2) == folds


def test_build_source_folds_rejects_conflicting_source_categories() -> None:
    rows = [
        {"source_id": "same-source", "category": "학사"},
        {"source_id": "same-source", "category": "장학"},
    ]

    with pytest.raises(ValueError, match="multiple categories"):
        build_source_folds(rows, fold_count=2)
