from __future__ import annotations

import numpy as np
import pandas as pd

from betalactamase_transfer import biological_model_difference_analysis as module


def test_safe_correlation_rejects_constant_input() -> None:
    value = module._safe_correlation(
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        "spearman",
    )
    assert np.isnan(value)


def test_position_summary_uses_position_as_structural_unit() -> None:
    matched = pd.DataFrame(
        {
            "scheme_id": ["ndm1_position_nested5"] * 4,
            "protein_id": ["NDM-1"] * 4,
            "assay_position": [10, 10, 11, 11],
            "observation_id": ["a", "b", "c", "d"],
            "evaluation_group_id": ["p10", "p10", "p11", "p11"],
            "graph_minus_centre_absolute_error": [-0.2, 0.0, 0.1, 0.3],
            "geometry_minus_graph_absolute_error": [0.0, 0.0, 0.0, 0.0],
            "sequence_minus_graph_absolute_error": [0.1, 0.1, 0.2, 0.2],
            "joint_minus_graph_absolute_error": [-0.1, -0.3, 0.2, 0.4],
            "validation_selected_minus_graph_absolute_error": [0.0, 0.0, 0.0, 0.0],
        }
    )
    residue = pd.DataFrame(
        {
            "protein_id": ["NDM-1", "NDM-1"],
            "assay_position": [10, 11],
            "experimental_structure_matches_assay_residue": [True, True],
            "reported_homologue_pair_position": [True, True],
            "reported_homologue_partner_assay_position": [9, 10],
            "relative_sasa_wilke": [0.2, 0.4],
            "heavy_atom_contact_count_4_5_angstrom": [3, 5],
            "ca_distance_to_catalytic_zinc_centroid_angstrom": [8.0, 12.0],
            "ca_neighbour_count_10_angstrom": [7, 9],
            (
                "minimum_atom_distance_to_nearest_mcsa_"
                "metal_ligand_correspondence_angstrom"
            ): [6.0, 10.0],
            "secondary_structure_3state": ["helix", "coil"],
            "mcsa_any_metal_ligand_role_correspondence": [False, False],
            "mcsa_any_selected_catalytic_zinc_contact": [False, False],
        }
    )
    result = module._position_summary(matched, residue)
    assert len(result) == 2
    first = result.loc[result["assay_position"].eq(10)].iloc[0]
    assert first["n_observations"] == 2
    assert np.isclose(first["mean_joint_minus_graph_absolute_error"], -0.2)


def test_contrast_sign_is_first_model_minus_reference_error() -> None:
    left = np.array([0.2, 0.4])
    right = np.array([0.5, 0.5])
    difference = left - right
    assert np.mean(difference) < 0
