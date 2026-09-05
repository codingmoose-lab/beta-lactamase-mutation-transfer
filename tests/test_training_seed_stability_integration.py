from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module("betalactamase_transfer.training_seed_stability")
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_seed_stability_input_contract() -> None:
    required = [
        REPO_ROOT
        / "data/interim/residue_graph_neural_evaluation"
        / "residue_graph_neural_repeat_predictions.csv",
        REPO_ROOT
        / "data/interim/residue_geometry_neural_evaluation"
        / "residue_geometry_neural_repeat_predictions.csv",
        REPO_ROOT
        / "data/interim/sequence_residue_graph_neural_evaluation"
        / "sequence_residue_graph_repeat_predictions.csv",
        REPO_ROOT
        / "data/interim/biological_model_difference_analysis"
        / "biological_model_difference_summary.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Full repository data are not present in this test copy")

    summary = module.inspect_training_seed_stability_inputs(REPO_ROOT)
    assert summary["status"] == "PASS"
    assert summary["total_training_seed_count"] == 10
    assert summary["additional_training_seed_count"] == 7
    assert summary["evaluation_context_count"] == 17
    assert summary["selected_epoch_record_count"] == 68
    assert summary["model_selection_performed"] is False
    assert summary["validation_search_repeated"] is False
