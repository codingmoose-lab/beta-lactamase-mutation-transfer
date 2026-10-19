"""Check that source-protein fitting checks detect target-observation use."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analysis/49_verify_source_fitting_boundaries.py"
)
spec = importlib.util.spec_from_file_location("fitting_boundary_checks", SCRIPT)
assert spec is not None and spec.loader is not None
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def _copy_source(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    target = tmp_path / "src/betalactamase_transfer"
    target.mkdir(parents=True)
    for name in (*checks.NEURAL_MODULES, *checks.RIDGE_MODULES):
        (target / f"{name}.py").write_bytes(
            (source / "src/betalactamase_transfer" / f"{name}.py").read_bytes()
        )
    return tmp_path


def test_complete_source_code_checks():
    root = Path(__file__).resolve().parents[1]
    rows, summary = checks.run_checks(root)
    assert summary["passed"] == 8
    assert summary["review"] == 3
    assert summary["failed"] == 0
    assert len(rows) == 11
    assert summary["training_executed"] is False
    assert summary["target_outcomes_used_for_model_selection_proved_absent"] is False


@pytest.mark.parametrize("scheme,source,target", checks.SCHEMES)
def test_cross_protein_direction(scheme, source, target):
    root = Path(__file__).resolve().parents[1]
    record = checks._records(scheme, source, target)
    for name in checks.NEURAL_MODULES:
        path = root / "src/betalactamase_transfer" / f"{name}.py"
        result = checks._selection_check(path, name, record)
        assert result["fold_count"] == 5
        assert result["selected_epoch"] == 4
        assert checks._fitting_check(path, name, record)["final_source_rows"] == 10


def test_mutated_validation_input_detected(tmp_path):
    repo = _copy_source(tmp_path)
    name = "residue_graph_neural_evaluation"
    path = repo / "src/betalactamase_transfer" / f"{name}.py"
    source = path.read_text(encoding="utf-8")
    needle = '            fold_record["validation"],\n'
    assert needle in source
    path.write_text(
        source.replace(needle, '            context["target"],\n', 1),
        encoding="utf-8",
    )
    rows, summary = checks.run_checks(repo)
    assert summary["failed"] >= 1
    assert any(row["status"] == "FAIL" and row["model_source"] == name for row in rows)


def test_model_transform_wrong_population_detected(tmp_path):
    repo = _copy_source(tmp_path)
    name = "residue_graph_neural_evaluation"
    path = repo / "src/betalactamase_transfer" / f"{name}.py"
    source = path.read_text(encoding="utf-8")
    needle = "_target_transform(fitting_rows)"
    assert needle in source
    path.write_text(
        source.replace(needle, "_target_transform(validation_rows)", 1),
        encoding="utf-8",
    )
    rows, summary = checks.run_checks(repo)
    assert summary["failed"] >= 1
    assert any(row["status"] == "FAIL" and row["model_source"] == name for row in rows)


def test_missing_source_module(tmp_path):
    with pytest.raises(checks.SourceFittingBoundaryError, match="Missing source"):
        checks.run_checks(tmp_path)
