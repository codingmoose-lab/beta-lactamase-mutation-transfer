"""Tests for comparing file contents rather than modification timestamps."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analysis/46_check_research_file_hashes.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("research_hash_checks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_detects_content_changes_but_ignores_timestamp_only(tmp_path) -> None:
    module = _module()
    data = tmp_path / "data/interim"
    data.mkdir(parents=True)
    path = data / "measurements.csv"
    path.write_text("a,b\n1,2\n")
    initial = module.snapshot(tmp_path)
    path.touch()
    touched = module.snapshot(tmp_path)
    assert module.compare(initial, touched) == {
        "added": [],
        "removed": [],
        "changed_content": [],
    }
    path.write_text("a,b\n1,3\n")
    modified = module.snapshot(tmp_path)
    assert module.compare(initial, modified)["changed_content"] == [
        "data/interim/measurements.csv"
    ]
