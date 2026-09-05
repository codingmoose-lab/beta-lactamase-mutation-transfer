from __future__ import annotations

from pathlib import Path

import pytest

from betalactamase_transfer import sequence_structure_complementarity as module

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sequence_structure_complementarity_input_contract() -> None:
    required = (
        REPO_ROOT / "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv",
        REPO_ROOT
        / "data/interim/residue_graph_neural_evaluation"
        / "residue_graph_neural_predictions.csv",
        REPO_ROOT
        / "data/interim/sequence_residue_graph_neural_evaluation"
        / "sequence_residue_graph_predictions.csv",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Full repository data are not present in this test copy")
    summary = module.inspect_sequence_structure_complementarity_inputs(REPO_ROOT)
    assert summary["status"] == "PASS"
    assert summary["matched_final_prediction_count"] == 118204
    assert summary["final_evaluation_labels_used_for_weight_selection"] is False
    assert summary["cross_protein_target_labels_used_for_weight_selection"] is False
