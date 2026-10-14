from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module(
    "betalactamase_transfer.sequence_residue_graph_neural_evaluation"
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sequence_residue_graph_input_contract() -> None:
    required = (
        REPO_ROOT / "data/processed/missense_observations.csv",
        REPO_ROOT / "data/interim/sequence_plm_embeddings/mutant_site_embeddings.npy",
        REPO_ROOT
        / "data/interim/sequence_plm_embeddings/wild_type_residue_embeddings.npy",
        REPO_ROOT
        / "data/interim/residue_graph_neural_evaluation"
        / "residue_graph_neural_predictions.csv",
    )
    if not all(path.is_file() for path in required):
        pytest.skip(
            "Required generated outputs are unavailable in this test environment"
        )
    summary = module.inspect_sequence_residue_graph_inputs(REPO_ROOT)
    assert summary["status"] == "PASS"
    assert summary["evaluation_context_count"] == 17
    assert summary["evaluation_prediction_count"] == 118204
    assert summary["unique_evaluation_observation_count"] == 109408
    assert sum(record["test_row_count"] for record in summary["contexts"]) == 118204
    assert summary["missing_esmc_evaluation_rows"] == 0
    assert summary["graph_eval_population_match"] is True
    assert summary["test_labels_used_for_model_selection"] is False
    assert summary["cross_protein_target_labels_used_for_model_selection"] is False
