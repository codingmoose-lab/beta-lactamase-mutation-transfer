from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mkdssp_path() -> str | None:
    from_path = shutil.which("mkdssp")
    if from_path:
        return from_path
    local = Path.home() / ".local/bin/mkdssp"
    return str(local) if local.is_file() else None


def test_secondary_structure_real_data_if_available() -> None:
    mkdssp = _mkdssp_path()
    required = [
        REPO_ROOT / "data/interim/residue_evidence/residue_evidence.csv",
        REPO_ROOT
        / "data/interim/structural_descriptors/residue_structural_descriptors.csv",
        REPO_ROOT / "data/raw/pdb/3SPU/3SPU.cif",
        REPO_ROOT / "data/raw/pdb/5YD7/5YD7.cif",
    ]
    if mkdssp is None or not all(path.exists() for path in required):
        return

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/15_prepare_secondary_structure.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--mkdssp",
            mkdssp,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    output = REPO_ROOT / "data/interim/secondary_structure"
    table = pd.read_csv(output / "secondary_structure.csv")
    summary = json.loads(
        (output / "secondary_structure_summary.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "PASS"
    assert summary["modeled_dssp_counts"] == {"NDM-1": 238, "VIM-2": 231}
    assert summary["no_experimental_coordinate_count"] == 68
    assert summary["coordinate_identity_mismatch_count"] == 0
    assert summary["assay_identity_mismatch_count"] == 1
    assert summary["documented_assay_reference_mismatch_count"] == 1
    assert summary["dssp_accessibility_used_as_feature"] is False

    assert len(table) == 537
    assert not table.duplicated(["protein_id", "assay_position"]).any()
    modeled = table[table["secondary_structure_status"] == "dssp_assignment_available"]
    assert len(modeled) == 469
    assert modeled["dssp_class"].notna().all()
    assert modeled["secondary_structure_3state"].isin({"helix", "strand", "coil"}).all()
    assert modeled["dssp_coordinate_residue_identity"].all()

    vim_gly2 = table[
        (table["protein_id"] == "VIM-2") & (table["assay_position"] == 2)
    ].iloc[0]
    assert vim_gly2["secondary_structure_status"] == "no_experimental_coordinate"
    assert pd.isna(vim_gly2["dssp_class"])

    vim_186 = table[
        (table["protein_id"] == "VIM-2") & (table["assay_position"] == 186)
    ].iloc[0]
    assert bool(vim_186["known_assay_reference_difference"])
    assert not bool(vim_186["dssp_assay_residue_identity"])
    assert bool(vim_186["dssp_coordinate_residue_identity"])


def test_secondary_structure_cli_help_without_mkdssp() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/15_prepare_secondary_structure.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        env={"PATH": "", "HOME": os.environ.get("HOME", "")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_retained_secondary_structure_outputs_if_available() -> None:
    output = REPO_ROOT / "data/interim/secondary_structure"
    table_path = output / "secondary_structure.csv"
    summary_path = output / "secondary_structure_summary.json"
    if not table_path.exists() or not summary_path.exists():
        return

    table = pd.read_csv(table_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert str(summary["mkdssp_version"]).startswith("mkdssp version 4.")
    assert summary["modeled_dssp_counts"] == {"NDM-1": 238, "VIM-2": 231}
    assert summary["no_experimental_coordinate_count"] == 68
    assert summary["coordinate_identity_mismatch_count"] == 0
    assert summary["documented_assay_reference_mismatch_count"] == 1
    assert summary["dssp_accessibility_used_as_feature"] is False
    assert len(table) == 537
    assert not table.duplicated(["protein_id", "assay_position"]).any()
    assert (
        table["secondary_structure_status"] == "dssp_assignment_available"
    ).sum() == 469
