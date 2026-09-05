from __future__ import annotations

from pathlib import Path

from betalactamase_transfer.matched_homologue_model_analysis import (
    ADVANCED_MODELS,
    REFERENCE_MODELS,
    _assemble_predictions,
    _load_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage42_input_population_and_prediction_coverage() -> None:
    baseline, prediction_frames, _ = _load_inputs(REPO_ROOT)
    predictions = _assemble_predictions(baseline, prediction_frames)

    assert len(predictions) == 3026 * (len(ADVANCED_MODELS) + len(REFERENCE_MODELS))
    assert set(predictions["model_id"]) == set(ADVANCED_MODELS).union(REFERENCE_MODELS)
    assert (
        not predictions[["direction", "model_id", "target_observation_id"]]
        .duplicated()
        .any()
    )

    counts = predictions.groupby(["direction", "model_id"]).size().reset_index(name="n")
    assert counts["n"].eq(1513).all()

    nonfloor = predictions.loc[predictions["floor_class"].eq("neither_at_floor")]
    nonfloor_counts = (
        nonfloor.groupby(["direction", "model_id"]).size().reset_index(name="n")
    )
    assert nonfloor_counts["n"].eq(478).all()


def test_stage42_model_predictions_match_saved_targets() -> None:
    baseline, prediction_frames, _ = _load_inputs(REPO_ROOT)
    predictions = _assemble_predictions(baseline, prediction_frames)
    assert predictions["target_effect"].notna().all()
    assert predictions["y_pred"].notna().all()
    assert predictions["absolute_error"].ge(0).all()
