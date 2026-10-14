from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_processed_tables_real_data_if_available() -> None:
    required = [
        REPO_ROOT / "data/interim/mutation_measurements/mutation_measurements.csv",
        REPO_ROOT / "data/interim/assay_definitions/measurement_groups.csv",
        REPO_ROOT / "data/interim/residue_evidence/residue_evidence.csv",
        REPO_ROOT
        / "data/interim/structural_descriptors/residue_structural_descriptors.csv",
        REPO_ROOT / "data/interim/secondary_structure/secondary_structure.csv",
    ]
    if not all(path.exists() for path in required):
        return

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/16_prepare_processed_tables.py"),
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    output = REPO_ROOT / "data/processed"
    summary = json.loads(
        (output / "processed_summary.json").read_text(encoding="utf-8")
    )
    observations = pd.read_csv(output / "observations.csv", low_memory=False)
    missense = pd.read_csv(output / "missense_observations.csv", low_memory=False)
    controls = pd.read_csv(output / "control_observations.csv", low_memory=False)
    residues = pd.read_csv(output / "residue_features.csv", low_memory=False)
    pairs = pd.read_csv(output / "residue_pair_geometry.csv", low_memory=False)
    reported = pd.read_csv(
        output / "reported_cross_protein_comparisons.csv",
        low_memory=False,
    )
    ec50 = pd.read_csv(output / "ec50_observations.csv", low_memory=False)
    roles = pd.read_csv(output / "observation_column_roles.csv")

    assert summary["status"] == "PASS"
    assert len(observations) == 138518
    assert len(missense) == 125681
    assert len(controls) == 12837
    assert len(residues) == 537
    assert len(pairs) == 54768
    assert len(reported) == 132
    assert len(ec50) == 5828
    assert len(roles) == len(observations.columns)
    assert set(roles["column_name"]) == set(observations.columns)
    assert set(roles["role"]) == {
        "outcome",
        "mutation_input",
        "condition_input",
        "grouping",
        "evaluation_only",
        "source_quality_metadata",
    }
    allowed = set(roles.loc[roles["default_model_input_allowed"], "column_name"])
    assert allowed == {
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
    }
    assert not roles.loc[
        roles["role"].isin(
            {"outcome", "grouping", "evaluation_only", "source_quality_metadata"}
        ),
        "default_model_input_allowed",
    ].any()

    assert (
        observations["score_analysis"] >= observations["minimum_fitness_score"]
    ).all()
    at_or_above_minimum = (
        observations["score_original"] >= observations["minimum_fitness_score"]
    )
    assert observations.loc[at_or_above_minimum, "score_analysis"].equals(
        observations.loc[at_or_above_minimum, "score_original"]
    )
    below_minimum = (
        observations["score_original"] < observations["minimum_fitness_score"]
    )
    assert observations.loc[below_minimum, "score_analysis"].equals(
        observations.loc[below_minimum, "minimum_fitness_score"]
    )

    assert not observations["observation_id"].duplicated().any()
    assert not residues.duplicated(["protein_id", "assay_position"]).any()
    assert (missense["mutation_class"] == "missense").all()
    assert set(controls["mutation_class"]) == {
        "stop_control",
        "wild_type_control",
    }

    assert missense["sequence_task_eligible"].all()
    assert (
        missense["experimental_structure_task_eligible"]
        <= missense["experimental_structure_available"]
    ).all()
    assert (
        missense["alphafold_structure_task_eligible"]
        <= missense["alphafold_structure_available"]
    ).all()

    vim186 = residues[
        (residues["protein_id"] == "VIM-2") & (residues["assay_position"] == 186)
    ].iloc[0]
    assert vim186["assay_residue"] == "I"
    assert bool(vim186["known_assay_reference_difference"]) is True
    assert bool(vim186["experimental_structure_available"]) is True
    assert bool(vim186["experimental_structure_matches_assay_residue"]) is False
    assert (
        vim186["structure_feature_status"]
        == "experimental_coordinate_reference_residue_differs_from_assay"
    )

    gly2 = residues[
        (residues["protein_id"] == "VIM-2") & (residues["assay_position"] == 2)
    ].iloc[0]
    assert bool(gly2["assay_specific_insertion"]) is True
    assert bool(gly2["experimental_structure_available"]) is False
    assert bool(gly2["alphafold_structure_available"]) is False
    assert (
        gly2["structure_feature_status"]
        == "assay_specific_position_without_reference_coordinate"
    )

    assert residues["reported_homologue_pair_position"].sum() == 264
    assert (
        reported["pair_scope"]
        .eq("reported_comparison_positions_not_full_alignment")
        .all()
    )


def test_processed_table_cli_is_self_contained() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/16_prepare_processed_tables.py"),
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
