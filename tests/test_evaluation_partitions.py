from __future__ import annotations

from betalactamase_transfer.evaluation_partitions import (
    INNER_VALIDATION_FOLD_COUNT,
    OUTER_FOLD_COUNT,
    SOURCE_CV_FOLD_COUNT,
    _balanced_fold_map,
    _nested_assignments,
    _source_cv_assignments,
)


def test_balanced_fold_map_is_reproducible_and_balanced() -> None:
    groups = [f"group_{index}" for index in range(23)]
    first = _balanced_fold_map(groups, fold_count=5, seed="example")
    second = _balanced_fold_map(reversed(groups), fold_count=5, seed="example")
    assert first == second
    counts = [list(first.values()).count(fold) for fold in range(5)]
    assert max(counts) - min(counts) <= 1


def test_nested_assignments_keep_groups_disjoint() -> None:
    groups = [f"position_{index}" for index in range(25)]
    rows = _nested_assignments(
        scheme_id="example",
        task_id="example_task",
        group_type="position_group_id",
        group_ids=groups,
    )
    assert OUTER_FOLD_COUNT == 5
    assert INNER_VALIDATION_FOLD_COUNT == 5
    for iteration in range(OUTER_FOLD_COUNT):
        current = [row for row in rows if row["iteration"] == iteration]
        assert len(current) == len(groups)
        assert {row["group_id"] for row in current} == set(groups)
        roles = {role: set() for role in ("train", "validation", "test")}
        for row in current:
            roles[row["split_role"]].add(row["group_id"])
        assert not roles["train"] & roles["validation"]
        assert not roles["train"] & roles["test"]
        assert not roles["validation"] & roles["test"]

    test_counts = {group_id: 0 for group_id in groups}
    for row in rows:
        if row["split_role"] == "test":
            test_counts[row["group_id"]] += 1
    assert set(test_counts.values()) == {1}


def test_source_cv_assignments_keep_target_out_of_source_folds() -> None:
    source = [f"source_{index}" for index in range(20)]
    target = [f"target_{index}" for index in range(17)]
    rows = _source_cv_assignments(
        scheme_id="transfer",
        task_id="transfer_task",
        group_type="position_group_id",
        source_group_ids=source,
        target_group_ids=target,
        fold_universe_group_ids=source,
        grouping_key="example_source_cv",
    )
    assert SOURCE_CV_FOLD_COUNT == 5
    source_rows = [row for row in rows if row["split_role"] == "source_model_selection"]
    target_rows = [row for row in rows if row["split_role"] == "target_test"]
    assert {row["group_id"] for row in source_rows} == set(source)
    assert {row["group_id"] for row in target_rows} == set(target)
    assert all(row["cv_fold"] in range(5) for row in source_rows)
    assert all(row["cv_fold"] == "" for row in target_rows)
