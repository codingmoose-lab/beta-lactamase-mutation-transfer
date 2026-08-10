from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_task_definitions_real_data_if_available() -> None:
    required = [
        REPO_ROOT / "data/processed/missense_observations.csv",
        REPO_ROOT / "data/processed/reported_cross_protein_comparisons.csv",
        REPO_ROOT / "data/processed/observation_column_roles.csv",
    ]
    if not all(path.exists() for path in required):
        return

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/17_define_modelling_tasks.py"),
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    output = REPO_ROOT / "data/interim/task_definitions"
    summary = json.loads(
        (output / "task_definition_summary.json").read_text(encoding="utf-8")
    )
    tasks = pd.read_csv(output / "task_definitions.csv")
    scopes = pd.read_csv(output / "task_scopes.csv")
    regimes = pd.read_csv(output / "evaluation_regimes.csv")
    conditions = pd.read_csv(output / "available_conditions.csv")

    assert summary["status"] == "PASS"
    assert "stage" not in summary
    assert summary["task_count"] == 7
    assert summary["headline_cross_protein_task_count"] == 2
    assert summary["evaluation_regime_count"] == 8
    assert summary["deferred_evaluation_regime_count"] == 0
    assert summary["available_condition_count"] == 25
    assert summary["vim_2020_2022_shared_mutation_group_count"] == 5005
    assert "no random-row headline" in summary["random_row_split_policy"]

    assert tasks["task_id"].is_unique
    assert set(tasks.loc[tasks["priority"].eq("headline"), "task_id"]) == {
        "ndm1_to_vim2_amp_transfer",
        "vim2_to_ndm1_amp_transfer",
    }
    dms_tasks = tasks[tasks["task_family"].ne("paired_biological_analysis")]
    assert dms_tasks["target_field"].eq("score_analysis").all()
    assert dms_tasks["target_scale_policy"].str.contains("retain_").all()

    expected_scope_counts = {
        ("ndm1_within_position_regression", "analysis"): 45369,
        ("vim2_2020_within_position_regression", "analysis"): 45222,
        ("ndm1_to_vim2_amp_transfer", "source"): 5041,
        ("ndm1_to_vim2_amp_transfer", "target"): 5023,
        ("vim2_to_ndm1_amp_transfer", "source"): 5023,
        ("vim2_to_ndm1_amp_transfer", "target"): 5041,
        ("vim2_2020_condition_transfer", "analysis"): 45222,
        ("vim2_2022_concentration_response", "analysis"): 35090,
        ("chen2024_wt_reversion_analysis", "analysis"): 132,
    }
    observed_scope_counts = {
        (row.task_id, row.scope_role): int(row.row_count)
        for row in scopes.itertuples(index=False)
    }
    assert observed_scope_counts == expected_scope_counts

    ndm_source = scopes[
        scopes["task_id"].eq("ndm1_to_vim2_amp_transfer")
        & scopes["scope_role"].eq("source")
    ].iloc[0]
    vim_target = scopes[
        scopes["task_id"].eq("ndm1_to_vim2_amp_transfer")
        & scopes["scope_role"].eq("target")
    ].iloc[0]
    assert int(ndm_source["experimental_structure_eligible_count"]) == 4472
    assert int(vim_target["experimental_structure_eligible_count"]) == 4324
    assert int(ndm_source["alphafold_structure_eligible_count"]) == 5041
    assert int(vim_target["alphafold_structure_eligible_count"]) == 4985

    aligned = regimes[
        regimes["regime_id"].eq("aligned_shared_wild_type_transfer_reporting")
    ].iloc[0]
    assert aligned["status"] == "ready_for_reporting_population"
    assert "identical WT-to-mutant substitutions" in aligned["interpretation"]
    assert "81 shared-WT positions" in aligned["required_rule"]

    paired = pd.read_csv(output / "aligned_shared_wild_type_mutation_pairs.csv")
    assert len(paired) == 1513
    assert paired["homologue_position_id"].nunique() == 81
    assert paired["pairing_uses_outcome_values"].eq(False).all()
    assert not any("score" in column.lower() for column in paired.columns)

    assert len(conditions) == 25
    assert set(conditions["assay_id"]) == {
        "ndm1_2024_dms",
        "vim2_2020_dms",
        "vim2_2022_amp_concentration_dms",
    }


def test_task_definition_cli_is_self_contained() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/17_define_modelling_tasks.py"),
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
