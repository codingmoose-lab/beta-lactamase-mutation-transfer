from __future__ import annotations

import json
from pathlib import Path

import pytest

from betalactamase_transfer import biological_model_difference_analysis as module

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_biological_model_difference_input_contract() -> None:
    required = [
        REPO_ROOT / "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv",
        REPO_ROOT
        / "data/interim/residue_graph_neural_evaluation"
        / "residue_graph_neural_predictions.csv",
        REPO_ROOT
        / "data/interim/residue_geometry_neural_evaluation"
        / "residue_geometry_neural_predictions.csv",
        REPO_ROOT
        / "data/interim/sequence_residue_graph_neural_evaluation"
        / "sequence_residue_graph_predictions.csv",
        REPO_ROOT
        / "data/interim/sequence_structure_complementarity"
        / "sequence_structure_summary.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Full repository data are not present in this test copy")

    sequence_structure_summary = json.loads(required[-1].read_text(encoding="utf-8"))
    if sequence_structure_summary.get("model_id") != module.MIXTURE_MODEL:
        pytest.skip("Sequence/graph combination terminology has not yet been updated")

    summary = module.inspect_biological_model_difference_inputs(REPO_ROOT)
    assert summary["status"] == "PASS"
    assert summary["matched_rows"] == 118204
    assert summary["model_selection_performed"] is False
    assert summary["causal_interpretation_permitted"] is False
