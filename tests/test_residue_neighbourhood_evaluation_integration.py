from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from betalactamase_transfer.residue_neighbourhood_evaluation import (
    EXPECTED_FEATURE_ROW_COUNT,
    EXPECTED_PREDICTIONS_PER_MODEL,
    MODEL_IDS,
    NEIGHBOUR_CONTINUOUS_COLUMNS,
    RESIDUE_NEIGHBOURHOOD_MODEL,
    prepare_residue_neighbourhood_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _required_inputs() -> list[Path]:
    return [
        REPO_ROOT / "data/processed/missense_observations.csv",
        REPO_ROOT / "data/processed/residue_features.csv",
        REPO_ROOT / "data/interim/modelling_views/scheme_observation_membership.csv",
        REPO_ROOT / "data/interim/evaluation_partitions/partition_schemes.csv",
        REPO_ROOT / "data/interim/evaluation_partitions/group_assignments.csv",
        REPO_ROOT / "data/interim/residue_graphs/residue_graph_nodes.csv",
        REPO_ROOT / "data/interim/residue_graphs/residue_graph_edges.csv",
        REPO_ROOT / "data/interim/residue_graphs/residue_graph_summary.json",
        REPO_ROOT
        / "data/interim/experimental_structure_evaluation"
        / "experimental_structure_predictions.csv",
    ]


def test_real_residue_neighbourhood_evaluation_if_available() -> None:
    if not all(path.exists() for path in _required_inputs()):
        return
    features, predictions, metrics, selection, paired, summary = (
        prepare_residue_neighbourhood_evaluation(REPO_ROOT)
    )
    assert summary["status"] == "PASS"
    assert summary["primary_model"] == RESIDUE_NEIGHBOURHOOD_MODEL
    assert summary["neighbour_descriptor_count"] == 36
    assert summary["absolute_coordinates_used_as_direct_model_inputs"] is False
    assert summary["mechanistic_reference_inputs_used"] is False
    assert summary["target_test_labels_used_for_alpha_selection"] is False
    assert len(features) == EXPECTED_FEATURE_ROW_COUNT
    assert list(features.columns[2:]) == list(NEIGHBOUR_CONTINUOUS_COLUMNS)
    assert np.isfinite(
        features[list(NEIGHBOUR_CONTINUOUS_COLUMNS)].to_numpy(dtype=float)
    ).all()
    counts = predictions.groupby("model_id").size().to_dict()
    assert set(counts) == set(MODEL_IDS)
    assert all(value == EXPECTED_PREDICTIONS_PER_MODEL for value in counts.values())
    assert predictions["y_true"].notna().all()
    assert predictions["y_pred"].notna().all()
    assert np.isfinite(predictions["y_true"].to_numpy(dtype=float)).all()
    assert np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all()
    selected = selection[selection["selection_scope"].eq("selected_alpha")]
    assert len(selected) == 17
    assert selected["selected"].astype(bool).all()
    assert len(metrics) == 66
    assert len(paired) == 40
    assert paired["bootstrap_valid_replicates"].gt(0).all()
    output_dir = REPO_ROOT / "data/interim/residue_neighbourhood_evaluation"
    saved_summary = json.loads(
        (output_dir / "residue_neighbourhood_summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["status"] == "PASS"


def test_residue_neighbourhood_cli_is_self_contained() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/data/31_evaluate_residue_neighbourhood.py"),
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
