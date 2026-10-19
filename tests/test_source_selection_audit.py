"""Tests for model-selection records and five-fold source partitions."""

from __future__ import annotations

import csv
from pathlib import Path

from betalactamase_transfer.source_selection_audit import (
    ASSIGNMENTS,
    SCHEMES,
    audit_source_selection,
)


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_partition_rows_with_five_folds_pass(tmp_path: Path) -> None:
    records = []
    for scheme in SCHEMES:
        for i in range(15):
            records.append(
                {
                    "scheme_id": scheme,
                    "group_id": f"SOURCE:{i}",
                    "split_role": "source_model_selection",
                    "cv_fold": str(i % 5),
                }
            )
        records.append(
            {
                "scheme_id": scheme,
                "group_id": "TARGET:1",
                "split_role": "target_test",
                "cv_fold": "",
            }
        )
    _write(
        tmp_path / ASSIGNMENTS,
        ["scheme_id", "group_id", "split_role", "cv_fold"],
        records,
    )
    table, summary = audit_source_selection(tmp_path)
    assert summary["failed"] == 0
    assert summary["passed"] == 2
    assert summary["source_only_model_selection_fully_proved"] is False
    assert "REVIEW" in set(table.status)


def test_partition_overlap_fails(tmp_path: Path) -> None:
    records = []
    for scheme in SCHEMES:
        for i in range(10):
            records.append(
                {
                    "scheme_id": scheme,
                    "group_id": f"P:{i}",
                    "split_role": "source_model_selection",
                    "cv_fold": str(i % 5),
                }
            )
        records.append(
            {
                "scheme_id": scheme,
                "group_id": "P:0",
                "split_role": "target_test",
                "cv_fold": "",
            }
        )
    _write(
        tmp_path / ASSIGNMENTS,
        ["scheme_id", "group_id", "split_role", "cv_fold"],
        records,
    )
    _, summary = audit_source_selection(tmp_path)
    assert summary["failed"] == 2
