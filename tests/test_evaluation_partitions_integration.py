from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_partitions_real_data_if_available() -> None:
    required = [
        REPO_ROOT / "data/processed/missense_observations.csv",
        REPO_ROOT / "data/interim/task_definitions/task_definition_summary.json",
        REPO_ROOT / "data/interim/task_definitions/evaluation_regimes.csv",
    ]
    if not all(path.exists() for path in required):
        return

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/18_define_evaluation_partitions.py"),
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    output = REPO_ROOT / "data/interim/evaluation_partitions"
    summary = json.loads(
        (output / "evaluation_partition_summary.json").read_text(encoding="utf-8")
    )
    schemes = pd.read_csv(output / "partition_schemes.csv")
    scopes = pd.read_csv(output / "partition_scopes.csv")
    assignments = pd.read_csv(output / "group_assignments.csv")
    audit = pd.read_csv(output / "partition_audit.csv")

    assert summary["status"] == "PASS"
    assert "stage" not in summary
    assert summary["scheme_count"] == 17
    assert summary["nested_group_scheme_count"] == 3
    assert summary["source_cv_final_test_scheme_count"] == 14
    assert summary["temperature_shared_mutation_group_count"] == 5016
    assert summary["temperature_25C_only_mutation_group_count"] == 12
    assert summary["temperature_37C_only_mutation_group_count"] == 7
    assert summary["vim_2020_37C_common_mutation_group_count"] == 5023
    assert summary["vim_2022_complete_trajectory_mutation_group_count"] == 5000
    assert summary["vim_2022_incomplete_trajectory_mutation_group_count"] == 15
    assert "target sequence or structure" in summary["target_label_policy"]

    assert schemes["scheme_id"].is_unique
    assert not schemes["held_out_labels_used_for_model_selection"].any()
    assert not schemes["assignment_uses_outcome_values"].any()

    headline = schemes[schemes["priority"].eq("headline")]
    assert set(headline["scheme_id"]) == {
        "ndm1_to_vim2_amp_source_cv5",
        "vim2_to_ndm1_amp_source_cv5",
    }

    ndm_target = scopes[
        scopes["scheme_id"].eq("ndm1_to_vim2_amp_source_cv5")
        & scopes["scope_role"].eq("final_test")
    ].iloc[0]
    assert int(ndm_target["row_count"]) == 5023
    assert int(ndm_target["experimental_structure_eligible_count"]) == 4324
    assert int(ndm_target["alphafold_structure_eligible_count"]) == 4985

    temperature_scopes = scopes[
        scopes["scheme_id"].str.startswith("vim2_2020_temperature_")
    ]
    assert set(temperature_scopes["mutation_group_count"].astype(int)) == {5016}
    assert set(temperature_scopes["row_count"].astype(int)) == {15048}

    antibiotic_scopes = scopes[
        scopes["scheme_id"].str.startswith("vim2_2020_loo_antibiotic_")
    ]
    assert set(antibiotic_scopes["mutation_group_count"].astype(int)) == {5023}

    concentration_schemes = schemes[
        schemes["scheme_id"].str.startswith("vim2_2022_loo_concentration_")
    ]
    assert len(concentration_schemes) == 7
    edge = concentration_schemes[
        concentration_schemes["held_out_class"].eq("edge_extrapolation")
    ]
    assert set(edge["held_out_value"].astype(int)) == {4, 256}

    within = assignments[assignments["scheme_id"].eq("ndm1_position_nested5")]
    assert set(within["split_role"]) == {"train", "validation", "test"}
    test_counts = within[within["split_role"].eq("test")].groupby("group_id").size()
    assert len(test_counts) == 270
    assert set(test_counts) == {1}

    transfer = assignments[assignments["scheme_id"].eq("ndm1_to_vim2_amp_source_cv5")]
    source = transfer[transfer["split_role"].eq("source_model_selection")]
    target = transfer[transfer["split_role"].eq("final_test")]
    assert source["group_id"].nunique() == 270
    assert target["group_id"].nunique() == 267
    assert set(source["cv_fold"].dropna().astype(int)) == {0, 1, 2, 3, 4}

    antibiotic_maps = []
    for scheme_id in (
        "vim2_2020_loo_antibiotic_amp",
        "vim2_2020_loo_antibiotic_ctx",
        "vim2_2020_loo_antibiotic_mem",
    ):
        current = assignments[
            assignments["scheme_id"].eq(scheme_id)
            & assignments["split_role"].eq("source_model_selection")
        ]
        antibiotic_maps.append(
            dict(zip(current["group_id"], current["cv_fold"].astype(int)))
        )
    assert antibiotic_maps[0] == antibiotic_maps[1] == antibiotic_maps[2]

    temperature_maps = []
    for scheme_id in (
        "vim2_2020_temperature_25_to_37",
        "vim2_2020_temperature_37_to_25",
    ):
        current = assignments[
            assignments["scheme_id"].eq(scheme_id)
            & assignments["split_role"].eq("source_model_selection")
        ]
        temperature_maps.append(
            dict(zip(current["group_id"], current["cv_fold"].astype(int)))
        )
    assert temperature_maps[0] == temperature_maps[1]

    assert not audit.empty
    for path in output.glob("*.csv"):
        assert b"\r" not in path.read_bytes(), path


def test_evaluation_partition_cli_is_self_contained() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/18_define_evaluation_partitions.py"),
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
