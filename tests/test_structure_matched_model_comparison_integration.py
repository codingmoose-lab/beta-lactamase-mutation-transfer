from __future__ import annotations

import json
from pathlib import Path

from betalactamase_transfer.structure_matched_model_comparison import (
    EXPECTED_PREDICTIONS_PER_MODEL,
    EXPECTED_SCHEME_COUNTS,
    EXPERIMENTAL_STRUCTURE_MODEL,
    STRUCTURE_BASELINE_MODEL,
    _load_existing_structure_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_existing_structure_results_define_the_matched_test_population() -> None:
    required = [
        REPO_ROOT / "data/interim/experimental_structure_evaluation/"
        "experimental_structure_predictions.csv",
        REPO_ROOT / "data/interim/experimental_structure_evaluation/"
        "experimental_structure_model_selection.csv",
        REPO_ROOT / "data/interim/experimental_structure_evaluation/"
        "experimental_structure_summary.json",
    ]
    if not all(path.exists() for path in required):
        return

    predictions, selection, summary, hashes = _load_existing_structure_results(
        REPO_ROOT
    )
    assert summary["status"] == "PASS"
    assert summary["target_labels_used_for_model_selection"] is False
    assert set(predictions["model_id"]) == {
        STRUCTURE_BASELINE_MODEL,
        EXPERIMENTAL_STRUCTURE_MODEL,
    }
    counts = predictions.groupby("model_id").size().to_dict()
    assert counts == {
        STRUCTURE_BASELINE_MODEL: EXPECTED_PREDICTIONS_PER_MODEL,
        EXPERIMENTAL_STRUCTURE_MODEL: EXPECTED_PREDICTIONS_PER_MODEL,
    }

    reference = predictions[predictions["model_id"].eq(STRUCTURE_BASELINE_MODEL)]
    assert reference.groupby("scheme_id").size().to_dict() == EXPECTED_SCHEME_COUNTS
    assert not reference.duplicated(
        [
            "scheme_id",
            "evaluation_iteration",
            "evaluation_role",
            "observation_id",
        ]
    ).any()
    assert selection["selected"].astype(bool).any()
    assert len(hashes["experimental_structure_predictions_sha256"]) == 64


def test_script_27_summary_reserves_biological_reference_inputs() -> None:
    path = (
        REPO_ROOT / "data/interim/experimental_structure_evaluation/"
        "experimental_structure_summary.json"
    )
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["mechanistic_reference_inputs_used"] is False
    assert summary["catalytic_zinc_distance_inputs_used"] is False
    assert summary["reported_homologue_pair_inputs_used"] is False
    assert summary["raw_coordinates_used_as_direct_predictors"] is False
