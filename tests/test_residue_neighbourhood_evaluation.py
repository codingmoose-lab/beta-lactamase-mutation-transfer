from __future__ import annotations

import pandas as pd
import pytest

from betalactamase_transfer.residue_neighbourhood_evaluation import (
    NEIGHBOUR_CONTINUOUS_COLUMNS,
    build_residue_neighbourhood_features,
)


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protein_id": "P",
                "node_index": 0,
                "assay_position": 10,
                "assay_residue": "A",
                "heavy_atom_count": 5,
                "sasa_angstrom2": 10.0,
                "relative_sasa_wilke": 0.1,
                "secondary_structure_3state": "helix",
            },
            {
                "protein_id": "P",
                "node_index": 1,
                "assay_position": 20,
                "assay_residue": "C",
                "heavy_atom_count": 7,
                "sasa_angstrom2": 30.0,
                "relative_sasa_wilke": 0.3,
                "secondary_structure_3state": "strand",
            },
            {
                "protein_id": "P",
                "node_index": 2,
                "assay_position": 30,
                "assay_residue": "D",
                "heavy_atom_count": 9,
                "sasa_angstrom2": 50.0,
                "relative_sasa_wilke": 0.5,
                "secondary_structure_3state": "coil",
            },
        ]
    )


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protein_id": "P",
                "source_node_index": 0,
                "target_node_index": 1,
                "ca_distance_angstrom": 4.0,
                "minimum_heavy_atom_distance_angstrom": 2.0,
                "peptide_adjacent": True,
                "heavy_atom_contact_4_5_angstrom": True,
            },
            {
                "protein_id": "P",
                "source_node_index": 1,
                "target_node_index": 2,
                "ca_distance_angstrom": 8.0,
                "minimum_heavy_atom_distance_angstrom": 6.0,
                "peptide_adjacent": False,
                "heavy_atom_contact_4_5_angstrom": False,
            },
        ]
    )


def test_neighbourhood_descriptor_contract_has_36_values() -> None:
    assert len(NEIGHBOUR_CONTINUOUS_COLUMNS) == 36


def test_direct_neighbours_are_summarised_without_the_centre_residue() -> None:
    features = build_residue_neighbourhood_features(_nodes(), _edges())
    middle = features.loc[features["assay_position"].eq(20)].iloc[0]
    assert middle["neighbour_count"] == 2
    assert middle["neighbour_amino_acid_fraction_A"] == pytest.approx(0.5)
    assert middle["neighbour_amino_acid_fraction_D"] == pytest.approx(0.5)
    assert middle["neighbour_amino_acid_fraction_C"] == pytest.approx(0.0)
    assert middle["neighbour_secondary_structure_fraction_helix"] == pytest.approx(0.5)
    assert middle["neighbour_secondary_structure_fraction_coil"] == pytest.approx(0.5)
    assert middle["neighbour_heavy_atom_count_mean"] == pytest.approx(7.0)
    assert middle["neighbour_heavy_atom_count_sd"] == pytest.approx(2.0)
    assert middle["neighbour_ca_distance_angstrom_mean"] == pytest.approx(6.0)
    assert middle["neighbour_ca_distance_angstrom_sd"] == pytest.approx(2.0)
    assert middle["neighbour_peptide_adjacent_fraction"] == pytest.approx(0.5)
    assert middle[
        "neighbour_heavy_atom_contact_4_5_angstrom_fraction"
    ] == pytest.approx(0.5)


def test_neighbourhood_composition_fractions_sum_to_one() -> None:
    features = build_residue_neighbourhood_features(_nodes(), _edges())
    amino_acid_columns = [
        column
        for column in features.columns
        if column.startswith("neighbour_amino_acid_fraction_")
    ]
    secondary_columns = [
        column
        for column in features.columns
        if column.startswith("neighbour_secondary_structure_fraction_")
    ]
    assert features[amino_acid_columns].sum(axis=1).tolist() == pytest.approx(
        [1.0] * len(features)
    )
    assert features[secondary_columns].sum(axis=1).tolist() == pytest.approx(
        [1.0] * len(features)
    )


def test_matched_predictions_accept_csv_target_round_trip(monkeypatch) -> None:
    import pandas as pd

    import betalactamase_transfer.residue_neighbourhood_evaluation as evaluation

    monkeypatch.setattr(evaluation, "EXPECTED_PREDICTIONS_PER_MODEL", 1)

    reference = {
        "scheme_id": "test_scheme",
        "evaluation_iteration": "0",
        "evaluation_role": "outer_test",
        "observation_id": "obs_1",
        "model_id": evaluation.EXPERIMENTAL_STRUCTURE_MODEL,
        "task_id": "test_task",
        "evaluation_group_id": "position_1",
        "protein_id": "NDM-1",
        "assay_position": 1,
        "wild_type": "A",
        "mutant": "V",
        "antibiotic": "AMP",
        "concentration": 1.0,
        "temperature_c": 37.0,
        "y_true": 3.9926300000000018,
        "y_pred": 0.0,
    }

    neighbourhood = {
        **reference,
        "model_id": "residue_neighbourhood_ridge",
        "y_true": 3.99263,
    }

    result = evaluation._validate_matched_predictions(
        pd.DataFrame([reference]),
        pd.DataFrame([neighbourhood]),
    )

    assert len(result) == 2


def test_matched_predictions_reject_material_target_difference(monkeypatch) -> None:
    import pandas as pd
    import pytest

    import betalactamase_transfer.residue_neighbourhood_evaluation as evaluation

    monkeypatch.setattr(evaluation, "EXPECTED_PREDICTIONS_PER_MODEL", 1)

    reference = {
        "scheme_id": "test_scheme",
        "evaluation_iteration": "0",
        "evaluation_role": "outer_test",
        "observation_id": "obs_1",
        "model_id": evaluation.EXPERIMENTAL_STRUCTURE_MODEL,
        "task_id": "test_task",
        "evaluation_group_id": "position_1",
        "protein_id": "NDM-1",
        "assay_position": 1,
        "wild_type": "A",
        "mutant": "V",
        "antibiotic": "AMP",
        "concentration": 1.0,
        "temperature_c": 37.0,
        "y_true": 3.99263,
        "y_pred": 0.0,
    }

    neighbourhood = {
        **reference,
        "model_id": "residue_neighbourhood_ridge",
        "y_true": 3.992631,
    }

    with pytest.raises(
        evaluation.ResidueNeighbourhoodEvaluationError,
        match="Matched field disagrees with Script-27 predictions: y_true",
    ):
        evaluation._validate_matched_predictions(
            pd.DataFrame([reference]),
            pd.DataFrame([neighbourhood]),
        )
