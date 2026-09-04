from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module(
    "betalactamase_transfer.residue_graph_neural_evaluation"
)
CORE_SCHEME_IDS = module.CORE_SCHEME_IDS
EXPECTED_GRAPH_EDGE_COUNT = module.EXPECTED_GRAPH_EDGE_COUNT
EXPECTED_GRAPH_NODE_COUNT = module.EXPECTED_GRAPH_NODE_COUNT
inspect_residue_graph_neural_inputs = module.inspect_residue_graph_neural_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_residue_graph_neural_input_contract() -> None:
    summary = inspect_residue_graph_neural_inputs(REPO_ROOT)
    assert summary["status"] == "PASS"
    assert summary["graph_node_count"] == EXPECTED_GRAPH_NODE_COUNT
    assert summary["graph_edge_count"] == EXPECTED_GRAPH_EDGE_COUNT
    assert summary["evaluation_context_count"] == 17
    assert tuple(summary["core_scheme_ids"]) == CORE_SCHEME_IDS
