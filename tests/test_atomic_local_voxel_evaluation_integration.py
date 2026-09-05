from __future__ import annotations

from pathlib import Path

import pytest

from betalactamase_transfer.atomic_local_voxel_evaluation import (
    EXPECTED_GRAPH_NODE_COUNT,
    EXPECTED_PREDICTIONS_PER_MODEL,
    inspect_atomic_local_voxel_inputs,
)


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in matmul:RuntimeWarning"
)
@pytest.mark.filterwarnings("ignore:overflow encountered in matmul:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in matmul:RuntimeWarning")
def test_atomic_local_voxel_input_review_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required = repo_root / "data/interim/residue_graphs/residue_graph_nodes.csv"
    if not required.is_file():
        pytest.skip(
            "full repository outputs are not available in this test environment"
        )

    summary = inspect_atomic_local_voxel_inputs(repo_root)

    assert summary["status"] == "PASS"
    assert summary["voxel_grid_count"] == EXPECTED_GRAPH_NODE_COUNT
    assert summary["evaluation_prediction_count"] == EXPECTED_PREDICTIONS_PER_MODEL
    assert summary["protein_atoms_only"] is True
    assert summary["hetero_atoms_included"] is False
    assert summary["metal_atoms_included"] is False
    assert summary["predicted_mutant_structures_used"] is False
    assert summary["target_outcome_used_for_voxel_construction"] is False
    assert summary["model_selection_uses_test_labels"] is False
