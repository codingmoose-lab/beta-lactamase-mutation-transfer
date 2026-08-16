from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.biological_error_analysis import (
    ESMC_MODEL,
    ESMC_STRUCTURE_DIFFERENCE,
    EXPERIMENTAL_STRUCTURE_MODEL,
    REFERENCE_CONTINUOUS_COLUMNS,
    _esmc_structure_position_differences,
    _group_error_table,
    _position_error_table,
    _spearman,
)


def _small_prediction_errors() -> pd.DataFrame:
    rows = []
    models = (
        "structure_matched_mutation_condition_ridge",
        "structure_matched_sequence_neighbourhood_ridge",
        ESMC_MODEL,
        EXPERIMENTAL_STRUCTURE_MODEL,
    )
    for model_id in models:
        for _observation_id, group_id, position, absolute_error in [
            ("o1", "g1", 10, 2.0),
            ("o2", "g1", 10, 4.0),
            ("o3", "g2", 20, 1.0),
        ]:
            if model_id == ESMC_MODEL:
                absolute_error += 0.5
            if model_id == EXPERIMENTAL_STRUCTURE_MODEL:
                absolute_error -= 0.5
            rows.append(
                {
                    "scheme_id": "ndm1_position_nested5",
                    "model_id": model_id,
                    "evaluation_group_id": group_id,
                    "protein_id": "NDM-1",
                    "assay_position": position,
                    "signed_error": absolute_error,
                    "absolute_error": absolute_error,
                    "squared_error": absolute_error**2,
                    "minimum_atom_distance_to_catalytic_zinc_angstrom": float(position),
                    "ca_distance_to_catalytic_zinc_centroid_angstrom": float(
                        position + 1
                    ),
                    REFERENCE_CONTINUOUS_COLUMNS[2]: float(position + 2),
                    REFERENCE_CONTINUOUS_COLUMNS[3]: float(position + 3),
                    "sasa_angstrom2": float(position + 4),
                    "relative_sasa_wilke": 0.2,
                    "ca_neighbour_count_8_angstrom": 10,
                    "ca_neighbour_count_10_angstrom": 20,
                    "ca_neighbour_count_12_angstrom": 30,
                    "heavy_atom_contact_count_4_5_angstrom": 8,
                    "mcsa_reference_position": "other_position",
                    "mcsa_metal_ligand_position": "other_position",
                    "catalytic_zinc_contact_position": "other_position",
                    "homologue_position_class": "shared_wild_type",
                    "secondary_structure_3state": "coil",
                    "homologue_position_id": "h1" if group_id == "g1" else "h2",
                }
            )
    return pd.DataFrame(rows)


def test_spearman_handles_monotonic_and_constant_inputs() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert _spearman(x, x) == pytest.approx(1.0)
    assert _spearman(x, x[::-1]) == pytest.approx(-1.0)
    assert np.isnan(_spearman(np.ones(3), x))


def test_group_error_table_uses_evaluation_groups() -> None:
    frame = _small_prediction_errors()
    grouped = _group_error_table(frame)
    esmc = grouped[
        grouped["model_id"].eq(ESMC_MODEL) & grouped["evaluation_group_id"].eq("g1")
    ].iloc[0]
    assert esmc["n_observations"] == 2
    assert esmc["mean_absolute_error"] == pytest.approx(3.5)


def test_esmc_structure_difference_is_positive_when_structure_has_smaller_error() -> (
    None
):
    grouped = _group_error_table(_small_prediction_errors())
    positions = _position_error_table(grouped)
    differences = _esmc_structure_position_differences(positions)
    assert (differences[ESMC_STRUCTURE_DIFFERENCE] > 0).all()
    assert set(differences["difference_interpretation"]) == {
        "positive_favours_experimental_structure"
    }


def test_position_error_table_averages_mutation_groups_within_residue() -> None:
    frame = _small_prediction_errors()
    extra = frame[frame["evaluation_group_id"].eq("g2")].copy()
    extra["evaluation_group_id"] = "g3"
    extra["assay_position"] = 10
    extra["minimum_atom_distance_to_catalytic_zinc_angstrom"] = 10.0
    extra["ca_distance_to_catalytic_zinc_centroid_angstrom"] = 11.0
    extra["minimum_atom_distance_to_nearest_mcsa_reference_correspondence_angstrom"] = (
        12.0
    )
    extra[
        "minimum_atom_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom"
    ] = 13.0
    extra["sasa_angstrom2"] = 14.0
    extra["homologue_position_id"] = "h1"
    combined = pd.concat([frame, extra], ignore_index=True)

    grouped = _group_error_table(combined)
    positions = _position_error_table(grouped)
    esmc = positions[
        positions["model_id"].eq(ESMC_MODEL) & positions["assay_position"].eq(10)
    ].iloc[0]

    expected = grouped[
        grouped["model_id"].eq(ESMC_MODEL) & grouped["assay_position"].eq(10)
    ]["mean_absolute_error"].mean()
    assert esmc["n_evaluation_groups"] == 2
    assert esmc["mean_absolute_error"] == pytest.approx(expected)
