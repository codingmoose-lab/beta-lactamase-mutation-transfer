import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from betalactamase_transfer.mcsa_correspondence import prepare_mcsa_correspondence

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_correspondence_cli_runs_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/10_prepare_mcsa_correspondence.py"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "M-CSA CORRESPONDENCE PASSED" in result.stdout


def test_prepare_mcsa_correspondence_against_acquired_sources() -> None:
    summary = prepare_mcsa_correspondence(REPO_ROOT)

    assert summary["status"] == "PASS"
    assert summary["biological_crosscheck_status"] == "PASS"
    assert summary["alignment_pair_count"] == 4
    assert summary["curated_reference_residue_count"] == 13
    assert summary["correspondence_row_count"] == 26
    assert summary["unique_role_annotation_count"] == 19
    assert summary["alignment_ambiguous_row_count"] == 0
    assert summary["consensus_gap_row_count"] == 0
    assert summary["dimetallic_reference_mcsa_id"] == 15
    assert summary["dimetallic_reference_unique_metal_ligand_count"] == 6
    assert summary["dimetallic_reference_target_metal_ligand_row_count"] == 12
    assert summary["dimetallic_reference_identity_failure_count"] == 0
    assert summary["dimetallic_reference_selected_pdb_contact_failure_count"] == 0

    output = REPO_ROOT / "data/interim/mcsa_correspondence"
    persisted = json.loads(
        (output / "correspondence_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary

    alignments = pd.read_csv(output / "alignment_summary.csv")
    counts = {
        (int(row.mcsa_id), row.target_protein): int(row.optimal_alignment_count)
        for row in alignments.itertuples()
    }
    assert counts == {
        (15, "NDM-1"): 3,
        (15, "VIM-2"): 3,
        (16, "NDM-1"): 6,
        (16, "VIM-2"): 1,
    }

    rows = pd.read_csv(output / "catalytic_residue_correspondence.csv")
    roles = pd.read_csv(output / "catalytic_role_annotations.csv")
    role_key_columns = [
        "mcsa_id",
        "reference_accession",
        "reference_position",
        "reference_residue",
        "function_type",
        "group_function",
        "function",
        "emo_id",
    ]
    assert len(roles) == 19
    assert not roles.duplicated(subset=role_key_columns).any()

    expected_dimetallic = {
        "NDM-1": {
            99: (120, "H"),
            101: (122, "H"),
            103: (124, "D"),
            162: (189, "H"),
            181: (208, "C"),
            184: (211, "K"),
            193: (220, "N"),
            223: (250, "H"),
        },
        "VIM-2": {
            99: (114, "H"),
            101: (116, "H"),
            103: (118, "D"),
            162: (179, "H"),
            181: (198, "C"),
            184: (201, "Y"),
            193: (210, "N"),
            223: (240, "H"),
        },
    }
    for protein_id, expected in expected_dimetallic.items():
        subset = rows[(rows["mcsa_id"] == 15) & (rows["target_protein"] == protein_id)]
        observed = {
            int(row.reference_position): (
                int(row.target_uniprot_position),
                row.target_residue,
            )
            for row in subset.itertuples()
        }
        assert observed == expected

    metal_ligands = rows[rows["has_mcsa_metal_ligand_role"] == True]  # noqa: E712
    assert metal_ligands["residue_identity_conserved"].all()
    assert metal_ligands["target_contacts_selected_catalytic_zinc"].all()

    vim_lys = rows[
        (rows["mcsa_id"] == 15)
        & (rows["target_protein"] == "VIM-2")
        & (rows["reference_position"] == 184)
    ]
    assert len(vim_lys) == 1
    assert int(vim_lys.iloc[0]["target_uniprot_position"]) == 201
    assert vim_lys.iloc[0]["target_residue"] == "Y"
    assert not bool(vim_lys.iloc[0]["residue_identity_conserved"])
    assert (
        vim_lys.iloc[0]["mcsa_role_correspondence_status"]
        == "reference_role_correspondence_nonidentical_residue"
    )
    assert (
        vim_lys.iloc[0]["mcsa_reference_unique_functions"]
        == "electrostatic stabiliser;steric role"
    )
