from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_structural_descriptors_real_data_if_available(tmp_path: Path) -> None:
    required = [
        REPO_ROOT / "data/interim/residue_evidence/residue_evidence.csv",
        REPO_ROOT / "data/raw/pdb/3SPU/3SPU.cif",
        REPO_ROOT / "data/raw/pdb/5YD7/5YD7.cif",
    ]
    if not all(path.exists() for path in required):
        return

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/14_prepare_structural_descriptors.py"),
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    output = REPO_ROOT / "data/interim/structural_descriptors"
    residue = pd.read_csv(output / "residue_structural_descriptors.csv")
    pairs = pd.read_csv(output / "residue_pair_geometry.csv")
    summary = json.loads(
        (output / "structural_descriptor_summary.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "PASS"
    assert summary["modeled_residue_counts"] == {"NDM-1": 238, "VIM-2": 231}
    assert summary["residue_pair_counts"] == {"NDM-1": 28203, "VIM-2": 26565}
    assert summary["residue_pair_row_count"] == 54768

    assert len(residue) == 537
    assert len(pairs) == 54768
    assert not residue.duplicated(["protein_id", "assay_position"]).any()
    assert (
        residue["descriptor_status"] == "experimental_coordinate_available"
    ).sum() == 469

    vim_gly2 = residue[
        (residue["protein_id"] == "VIM-2") & (residue["assay_position"] == 2)
    ].iloc[0]
    assert vim_gly2["descriptor_status"] == "no_experimental_coordinate"
    assert pd.isna(vim_gly2["sasa_angstrom2"])

    modeled = residue[
        residue["descriptor_status"] == "experimental_coordinate_available"
    ]
    assert modeled["sasa_angstrom2"].notna().all()
    assert modeled["relative_sasa_wilke"].notna().all()
    assert (modeled["sasa_angstrom2"] >= 0).all()
    assert (modeled["relative_sasa_wilke"] >= 0).all()
    assert modeled["ca_distance_to_catalytic_zinc_centroid_angstrom"].notna().all()

    assert (pairs["left_assay_position"] < pairs["right_assay_position"]).all()
    assert (pairs["ca_distance_angstrom"] > 0).all()
    assert (pairs["minimum_heavy_atom_distance_angstrom"] > 0).all()
    assert (
        summary["secondary_structure_status"]
        == "not_computed_by_structural_descriptor_preparation"
    )
    assert (
        summary["residue_depth_status"]
        == "not_computed_by_structural_descriptor_preparation"
    )


def test_structural_descriptor_cli_is_self_contained() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/14_prepare_structural_descriptors.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        env={"PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
