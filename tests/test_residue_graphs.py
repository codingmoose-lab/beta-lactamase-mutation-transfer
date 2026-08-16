from __future__ import annotations

import pandas as pd

from betalactamase_transfer.residue_graphs import (
    _build_edges,
    _build_nodes,
    _component_sizes,
)


def _residue_rows() -> pd.DataFrame:
    rows = []
    for position, x in [(10, 0.0), (20, 4.0), (30, 12.0)]:
        rows.append(
            {
                "protein_id": "P",
                "assay_position": position,
                "assay_residue": "A",
                "selected_pdb_id": "TEST",
                "selected_pdb_chain": "A",
                "pdb_coordinate_present": True,
                "ca_x": x,
                "ca_y": 0.0,
                "ca_z": 0.0,
                "side_chain_centroid_x": x,
                "side_chain_centroid_y": 1.0,
                "side_chain_centroid_z": 0.0,
                "heavy_atom_count": 5,
                "sasa_angstrom2": 20.0,
                "relative_sasa_wilke": 0.2,
                "secondary_structure_3state": "coil",
            }
        )
    return pd.DataFrame(rows)


def _pair_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protein_id": "P",
                "pdb_id": "TEST",
                "pdb_chain": "A",
                "left_assay_position": 10,
                "right_assay_position": 20,
                "sequence_separation": 10,
                "peptide_adjacent": False,
                "ca_distance_angstrom": 4.0,
                "minimum_heavy_atom_distance_angstrom": 2.0,
                "heavy_atom_contact_4_5_angstrom": True,
            },
            {
                "protein_id": "P",
                "pdb_id": "TEST",
                "pdb_chain": "A",
                "left_assay_position": 10,
                "right_assay_position": 30,
                "sequence_separation": 20,
                "peptide_adjacent": False,
                "ca_distance_angstrom": 12.0,
                "minimum_heavy_atom_distance_angstrom": 10.0,
                "heavy_atom_contact_4_5_angstrom": False,
            },
            {
                "protein_id": "P",
                "pdb_id": "TEST",
                "pdb_chain": "A",
                "left_assay_position": 20,
                "right_assay_position": 30,
                "sequence_separation": 10,
                "peptide_adjacent": False,
                "ca_distance_angstrom": 8.0,
                "minimum_heavy_atom_distance_angstrom": 6.0,
                "heavy_atom_contact_4_5_angstrom": False,
            },
        ]
    )


def test_nodes_use_position_order_and_contiguous_indices() -> None:
    nodes = _build_nodes(_residue_rows().iloc[[2, 0, 1]].copy())

    assert nodes["assay_position"].tolist() == [10, 20, 30]
    assert nodes["node_index"].tolist() == [0, 1, 2]


def test_primary_edges_use_inclusive_eight_angstrom_cutoff() -> None:
    nodes = _build_nodes(_residue_rows())
    edges = _build_edges(_pair_rows(), nodes)

    assert len(edges) == 2
    assert edges["ca_distance_angstrom"].tolist() == [4.0, 8.0]
    assert edges[["source_node_index", "target_node_index"]].values.tolist() == [
        [0, 1],
        [1, 2],
    ]


def test_component_sizes_detect_connected_graph() -> None:
    nodes = _build_nodes(_residue_rows())
    edges = _build_edges(_pair_rows(), nodes)

    assert _component_sizes(len(nodes), edges) == [3]
