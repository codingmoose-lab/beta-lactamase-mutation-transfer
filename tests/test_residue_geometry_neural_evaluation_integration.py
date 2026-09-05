from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module(
    "betalactamase_transfer.residue_geometry_neural_evaluation"
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_residue_geometry_neural_input_contract() -> None:
    summary = module.inspect_residue_geometry_neural_inputs(REPO_ROOT)
    assert summary["status"] == "PASS"
    assert summary["evaluation_context_count"] == 17
    assert summary["structure_supported_observation_count"] == 109408
    assert summary["evaluation_prediction_count"] == 118204
    assert summary["graph_node_count"] == 469
    assert summary["graph_edge_count"] == 2416
    assert summary["rigid_motion_output_check"]["passed"] is True
