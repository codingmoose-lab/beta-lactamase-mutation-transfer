from __future__ import annotations

from pathlib import Path

from betalactamase_transfer.biological_error_analysis import (
    EXPECTED_PREDICTION_ROWS,
    EXPECTED_TESTED_RESIDUE_KEYS,
    MODEL_IDS,
    _group_error_table,
    _load_inputs,
    _position_error_table,
    _prediction_error_table,
    _validate_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_biological_error_inputs_if_available() -> None:
    required = [
        REPO_ROOT / "data/interim/structure_matched_model_comparison/"
        "structure_matched_predictions.csv",
        REPO_ROOT / "data/interim/structure_matched_model_comparison/"
        "structure_matched_summary.json",
        REPO_ROOT / "data/processed/residue_features.csv",
        REPO_ROOT / "data/interim/residue_evidence/"
        "homologue_position_correspondence.csv",
    ]
    if not all(path.exists() for path in required):
        return

    predictions, features, homologue, hashes = _load_inputs(REPO_ROOT)
    _validate_inputs(predictions, features, homologue)
    errors = _prediction_error_table(predictions, features, homologue)
    groups = _group_error_table(errors)
    positions = _position_error_table(groups)

    assert len(errors) == EXPECTED_PREDICTION_ROWS
    assert set(errors["model_id"]) == set(MODEL_IDS)
    assert (
        len(errors[["protein_id", "assay_position", "wild_type"]].drop_duplicates())
        == EXPECTED_TESTED_RESIDUE_KEYS
    )
    assert errors["absolute_error"].ge(0).all()
    assert errors["minimum_atom_distance_to_catalytic_zinc_angstrom"].notna().all()
    for scheme_id, scheme_predictions in predictions.groupby("scheme_id"):
        expected_positions = scheme_predictions[
            ["protein_id", "assay_position"]
        ].drop_duplicates()
        for model_id in MODEL_IDS:
            observed = positions[
                positions["scheme_id"].eq(scheme_id)
                & positions["model_id"].eq(model_id)
            ]
            assert len(observed) == len(expected_positions)
    assert len(hashes["structure_matched_predictions_sha256"]) == 64
