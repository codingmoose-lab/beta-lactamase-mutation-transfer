"""Test source-only fitting inputs in Ridge and ESM-C evaluation code."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analysis/50_verify_source_ridge_selection.py"
)
spec = importlib.util.spec_from_file_location("source_ridge_checks", SCRIPT)
assert spec is not None and spec.loader is not None
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.mark.parametrize("name,module", checks.MODULES)
@pytest.mark.parametrize("scheme,source,target", checks.SCHEMES)
def test_each_source_only_evaluation(name, module, scheme, source, target):
    row = checks._run_case(name, module, scheme, source, target)
    assert row["status"] == "PASS"
    assert row["source_fold_count"] == 5
    assert row["final_source_rows"] == 10


def test_real_feature_transform_calculations():
    assert checks.check_transform_inputs() == {
        "baseline": "PASS",
        "experimental_structure": "PASS",
        "sequence_plm": "PASS",
    }


def test_target_rows_rejected_as_source_validation():
    _, _, _, source, target = checks._sample(*checks.SCHEMES[0])
    source_ids, target_ids = checks._ids(source), checks._ids(target)
    folds = []
    for i in range(5):
        selected = source.iloc[i * 2 : (i + 1) * 2]
        training = source.drop(selected.index)
        folds.append((training, selected))
    checks.check_source_folds(folds, source_ids, target_ids)
    corrupted = list(folds)
    corrupted[0] = (corrupted[0][0], pd.concat([target.iloc[[0]], source.iloc[[1]]]))
    with pytest.raises(checks.SourceSelectionCheckError):
        checks.check_source_folds(corrupted, source_ids, target_ids)


def test_target_rows_rejected_from_fitting():
    _, _, _, source, target = checks._sample(*checks.SCHEMES[0])
    checks.check_final_fit(source, target, checks._ids(source), checks._ids(target))
    training_with_target = pd.concat([source.iloc[:-1], target.iloc[[0]]])
    with pytest.raises(checks.SourceSelectionCheckError):
        checks.check_final_fit(
            training_with_target, target, checks._ids(source), checks._ids(target)
        )


def test_empty_validation_folds_rejected():
    _, _, _, source, target = checks._sample(*checks.SCHEMES[0])
    with pytest.raises(checks.SourceSelectionCheckError):
        checks.check_source_folds([], checks._ids(source), checks._ids(target))


def test_all_checks_current_repository():
    root = Path(__file__).resolve().parents[1]
    rows, summary = checks.run_checks(root)
    assert len(rows) == 6
    assert summary["passed"] == 6
    assert summary["failed"] == 0
    assert summary["historic_model_training_verified"] is False


def test_command_writes_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    output = tmp_path / "verification"
    command = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(root),
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "source_ridge_checks.csv").is_file()
    assert (output / "source_ridge_summary.json").is_file()
    assert (output / "source_ridge_report.txt").is_file()
