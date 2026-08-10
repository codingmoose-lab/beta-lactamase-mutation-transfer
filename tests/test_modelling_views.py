from __future__ import annotations

from betalactamase_transfer.modelling_views import (
    DIRECT_OBSERVATION_INPUTS,
    REPRESENTATION_CONTRACTS,
    RESIDUE_GENERIC_STRUCTURAL_INPUT_COLUMNS,
    RESIDUE_MECHANISTIC_REFERENCE_COLUMNS,
    RESIDUE_REPORTED_PAIR_COLUMNS,
    _residue_role,
)


def test_representation_contracts_keep_alphafold_supplementary() -> None:
    contracts = {row["representation_id"]: row for row in REPRESENTATION_CONTRACTS}
    assert contracts["experimental_structure"]["status"] == "primary"
    assert contracts["experimental_structure"]["coordinate_source"] == (
        "selected_experimental_pdb"
    )
    assert contracts["alphafold_structure_supplementary"]["status"] == ("supplementary")


def test_direct_observation_input_set_remains_small() -> None:
    assert {
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
    } == DIRECT_OBSERVATION_INPUTS


def test_residue_contracts_reserve_biological_reference_fields() -> None:
    columns = sorted(
        RESIDUE_GENERIC_STRUCTURAL_INPUT_COLUMNS
        | RESIDUE_MECHANISTIC_REFERENCE_COLUMNS
        | RESIDUE_REPORTED_PAIR_COLUMNS
    )
    lookup = {}
    for column in columns:
        role, direct, _, biological, _, _ = _residue_role(column)
        lookup[column] = (role, direct, biological)

    for column in RESIDUE_GENERIC_STRUCTURAL_INPUT_COLUMNS:
        assert lookup[column][1] is True
        assert lookup[column][2] is False

    references = RESIDUE_MECHANISTIC_REFERENCE_COLUMNS | RESIDUE_REPORTED_PAIR_COLUMNS
    for column in references:
        assert lookup[column][1] is False
        assert lookup[column][2] is True
