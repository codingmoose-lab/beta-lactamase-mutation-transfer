from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_residue_evidence_real_data_if_available(tmp_path: Path) -> None:
    required = [
        REPO_ROOT / "data/interim/sequence_mapping/residue_number_mapping.csv",
        REPO_ROOT / "data/interim/protein_structures/residue_coordinate_status.csv",
        REPO_ROOT / "data/interim/alphafold_preparation/assay_alphafold_mapping.csv",
        REPO_ROOT / "data/interim/alphafold_preparation/residue_confidence.csv",
        REPO_ROOT
        / "data/interim/mcsa_correspondence/catalytic_residue_correspondence.csv",
        REPO_ROOT / "data/interim/mutation_measurements/cross_protein_comparisons.csv",
    ]
    if not all(path.exists() for path in required):
        return

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/13_prepare_residue_evidence.py"),
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    output = REPO_ROOT / "data/interim/residue_evidence"
    residue = pd.read_csv(output / "residue_evidence.csv", low_memory=False)
    full_pairs = pd.read_csv(output / "homologue_position_correspondence.csv")
    pairs = pd.read_csv(output / "reported_homologue_position_pairs.csv")
    summary = json.loads(
        (output / "residue_evidence_summary.json").read_text(encoding="utf-8")
    )

    assert len(residue) == 537
    assert residue.groupby("protein_id").size().to_dict() == {
        "NDM-1": 270,
        "VIM-2": 267,
    }
    assert not residue.duplicated(["protein_id", "assay_position"]).any()
    assert set(residue["selected_pdb_chain"]) == {"B"}
    assert set(residue["selected_pdb_id"]) == {"3SPU", "5YD7"}

    vim_gly2 = residue[
        (residue["protein_id"] == "VIM-2") & (residue["assay_position"] == 2)
    ].iloc[0]
    assert vim_gly2["assay_residue"] == "G"
    assert bool(vim_gly2["assay_specific_insertion"])
    assert pd.isna(vim_gly2["uniprot_position"])
    assert pd.isna(vim_gly2["alphafold_position"])
    assert not bool(vim_gly2["pdb_modeled"])

    vim_186 = residue[
        (residue["protein_id"] == "VIM-2") & (residue["assay_position"] == 186)
    ].iloc[0]
    assert vim_186["assay_residue"] == "I"
    assert int(vim_186["uniprot_position"]) == 185
    assert vim_186["uniprot_residue"] == "V"
    assert bool(vim_186["known_assay_reference_difference"])
    assert not bool(vim_186["assay_uniprot_residue_identity"])

    assert len(full_pairs) == 217
    assert not full_pairs["vim_assay_position"].duplicated().any()
    assert not full_pairs["ndm_assay_position"].duplicated().any()
    assert not full_pairs["mapping_uses_outcome_values"].any()
    assert full_pairs["same_wild_type_residue"].sum() == 81
    assert (~full_pairs["same_wild_type_residue"]).sum() == 136
    assert set(full_pairs["position_class"]) == {
        "shared_wild_type",
        "differing_wild_type",
    }

    assert len(pairs) == 132
    assert not pairs["vim_assay_position"].duplicated().any()
    assert not pairs["ndm_assay_position"].duplicated().any()
    assert set(pairs["pair_scope"]) == {
        "reported_comparison_positions_not_full_alignment"
    }

    assert summary["status"] == "PASS"
    assert summary["residue_evidence_row_count"] == 537
    assert summary["full_homologue_position_pair_count"] == 217
    assert summary["full_homologue_shared_wild_type_pair_count"] == 81
    assert summary["full_homologue_differing_wild_type_pair_count"] == 136
    assert summary["reported_homologue_position_pair_count"] == 132
    assert summary["assay_specific_insertion_rows"] == 1
    assert summary["known_assay_reference_difference_rows"] == 1


def test_residue_evidence_cli_is_self_contained() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/13_prepare_residue_evidence.py"),
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
