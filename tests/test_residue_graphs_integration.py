from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from betalactamase_transfer.residue_graphs import (
    EXPECTED_CURRENT_EDGE_COUNTS,
    EXPECTED_CURRENT_NODE_COUNTS,
    EXPECTED_CURRENT_TESTED_POSITION_COUNTS,
    FORBIDDEN_GRAPH_FIELDS,
    prepare_residue_graphs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _required_inputs() -> list[Path]:
    return [
        REPO_ROOT / "data/processed/residue_features.csv",
        REPO_ROOT / "data/processed/residue_pair_geometry.csv",
        REPO_ROOT / "data/interim/modelling_views/scheme_observation_membership.csv",
        REPO_ROOT / "data/interim/modelling_views/residue_feature_contracts.csv",
        REPO_ROOT / "data/interim/modelling_views/residue_pair_feature_contracts.csv",
        REPO_ROOT / "data/interim/protein_structures/selected_structures.json",
    ]


def test_real_residue_graphs_if_available() -> None:
    if not all(path.exists() for path in _required_inputs()):
        return

    summary = prepare_residue_graphs(REPO_ROOT)

    output_dir = REPO_ROOT / "data/interim/residue_graphs"
    nodes = pd.read_csv(output_dir / "residue_graph_nodes.csv")
    edges = pd.read_csv(output_dir / "residue_graph_edges.csv")
    saved_summary = json.loads(
        (output_dir / "residue_graph_summary.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "PASS"
    assert saved_summary["status"] == "PASS"
    assert summary["node_counts"] == EXPECTED_CURRENT_NODE_COUNTS
    assert summary["edge_counts"] == EXPECTED_CURRENT_EDGE_COUNTS
    assert summary["tested_position_counts"] == EXPECTED_CURRENT_TESTED_POSITION_COUNTS

    assert len(nodes) == 469
    assert len(edges) == 2416
    assert not nodes.duplicated(["protein_id", "assay_position"]).any()
    assert not edges.duplicated(
        ["protein_id", "source_node_index", "target_node_index"]
    ).any()

    assert set(nodes.columns).isdisjoint(FORBIDDEN_GRAPH_FIELDS)
    assert set(edges.columns).isdisjoint(FORBIDDEN_GRAPH_FIELDS)

    assert (edges["source_node_index"] != edges["target_node_index"]).all()
    assert (edges["ca_distance_angstrom"] <= 8.0).all()
    assert (edges["ca_distance_angstrom"] > 0.0).all()

    assert summary["context_only_assay_positions"] == {
        "NDM-1": [],
        "VIM-2": [186],
    }

    for protein_id in ("NDM-1", "VIM-2"):
        stats = summary["graph_statistics"][protein_id]
        assert stats["connected_component_count"] == 1
        assert stats["isolated_node_count"] == 0
        assert stats["largest_component_node_count"] == stats["node_count"]

    assert len(summary["residue_graph_nodes_sha256"]) == 64
    assert len(summary["residue_graph_edges_sha256"]) == 64


def test_residue_graph_cli_is_self_contained() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/30_prepare_residue_graphs.py"),
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
