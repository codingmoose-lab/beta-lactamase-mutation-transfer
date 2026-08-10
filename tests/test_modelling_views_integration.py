from __future__ import annotations

from pathlib import Path

import pandas as pd

from betalactamase_transfer.modelling_views import prepare_modelling_views

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_modelling_view_outputs_and_scientific_contracts() -> None:
    partition_dir = REPO_ROOT / "data/interim/evaluation_partitions"
    partition_hashes_before = {
        path.name: path.read_bytes()
        for path in partition_dir.iterdir()
        if path.is_file()
    }

    summary = prepare_modelling_views(REPO_ROOT)
    assert summary["status"] == "PASS"
    assert "stage" not in summary
    assert summary["evaluation_partition_scheme_count"] == 17
    assert summary["observation_column_contract_count"] == 62
    assert summary["residue_feature_contract_count"] == 74
    assert summary["residue_pair_feature_contract_count"] == 13
    assert summary["representation_contract_count"] == 4
    assert summary["scheme_observation_membership_row_count"] == 496_748
    assert summary["unique_membership_observation_count"] == 125_681
    assert summary["target_in_membership_table"] is False

    out = REPO_ROOT / "data/interim/modelling_views"
    membership = pd.read_csv(out / "scheme_observation_membership.csv")
    contracts = pd.read_csv(out / "observation_column_contracts.csv")
    residue_contracts = pd.read_csv(out / "residue_feature_contracts.csv")
    pair_contracts = pd.read_csv(out / "residue_pair_feature_contracts.csv")
    comparisons = pd.read_csv(out / "comparison_populations.csv")

    assert "score_original" not in membership.columns
    assert "score_analysis" not in membership.columns
    assert "score_uncertainty" not in membership.columns
    assert membership["observation_id"].nunique() == 125_681

    allowed = set(contracts.loc[contracts["direct_predictor_allowed"], "column_name"])
    assert allowed == {
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
    }
    lookup = set(
        contracts.loc[contracts["representation_lookup_allowed"], "column_name"]
    )
    assert lookup == {"protein_id", "assay_position"}

    biological = residue_contracts[residue_contracts["biological_validation_reference"]]
    assert not biological["primary_tabular_structural_input_allowed"].any()
    assert biological["column_name"].str.contains("mcsa|catalytic_zinc|reported_").any()
    assert len(pair_contracts) == 13
    assert pair_contracts["graph_construction_allowed"].all()
    assert not pair_contracts["direct_predictor_allowed"].any()

    ndm = comparisons[
        comparisons["scheme_id"].eq("ndm1_to_vim2_amp_source_cv5")
        & comparisons["scope_role"].eq("source_model_selection")
    ]
    counts = dict(zip(ndm["comparison_population"], ndm["row_count"]))
    assert counts["sequence_full"] == 5_041
    assert counts["experimental_structure_matched"] == 4_472
    assert counts["alphafold_matched"] == 5_041

    vim = comparisons[
        comparisons["scheme_id"].eq("ndm1_to_vim2_amp_source_cv5")
        & comparisons["scope_role"].eq("final_test")
    ]
    counts = dict(zip(vim["comparison_population"], vim["row_count"]))
    assert counts["sequence_full"] == 5_023
    assert counts["experimental_structure_matched"] == 4_324
    assert counts["alphafold_matched"] == 4_985

    assignments = pd.read_csv(
        partition_dir / "group_assignments.csv",
        low_memory=False,
    )
    for scheme_id, current in membership.groupby("scheme_id"):
        group_type = current["group_type"].iloc[0]
        groups = set(current["group_id"])
        assigned = set(
            assignments.loc[assignments["scheme_id"].eq(scheme_id), "group_id"]
        )
        assert groups.issubset(assigned)
        assert current[group_type].astype(str).eq(current["group_id"].astype(str)).all()

    partition_hashes_after = {
        path.name: path.read_bytes()
        for path in partition_dir.iterdir()
        if path.is_file()
    }
    assert partition_hashes_before == partition_hashes_after

    for path in out.glob("*.csv"):
        assert b"\r" not in path.read_bytes(), path
