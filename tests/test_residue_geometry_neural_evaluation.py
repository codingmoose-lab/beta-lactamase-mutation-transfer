from __future__ import annotations

import importlib

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module(
    "betalactamase_transfer.residue_geometry_neural_evaluation"
)


def test_stage33_model_contract() -> None:
    assert module.MODEL_ID == "egnn_residue_regressor"
    assert module.HIDDEN_WIDTH == 128
    assert module.GEOMETRY_LAYER_COUNT == 2
    assert module.MAX_EPOCHS == 400
    assert module.EARLY_STOPPING_PATIENCE == 20
    assert module.TRAINING_REPEATS == 3
    assert module.SELECTION_SEED != module.stage32.SELECTION_SEED
    assert set(module.FINAL_REPEAT_SEEDS).isdisjoint(module.stage32.FINAL_REPEAT_SEEDS)


def test_rigid_motion_output_check() -> None:
    result = module.rigid_motion_output_check()
    assert result["passed"] is True
    assert (
        result["maximum_absolute_prediction_difference"] <= result["absolute_tolerance"]
    )


def test_stage32_hash_record_is_complete() -> None:
    assert set(module.STAGE32_OUTPUT_HASHES) == {
        "residue_graph_neural_metrics.csv",
        "residue_graph_neural_paired_comparisons.csv",
        "residue_graph_neural_predictions.csv",
        "residue_graph_neural_repeat_predictions.csv",
        "residue_graph_neural_summary.json",
        "residue_graph_neural_training_history.csv",
    }
    assert all(len(value) == 64 for value in module.STAGE32_OUTPUT_HASHES.values())


def test_fitting_rows_reconstructs_source_cv_population() -> None:
    frame = pd.DataFrame(
        {
            "observation_id": [
                "obs_1",
                "obs_2",
                "obs_3",
                "obs_4",
            ],
            "score_analysis": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )

    context = {
        "source_folds": [
            {
                "fold": 0,
                "train": frame.iloc[[0, 1]].copy(),
                "validation": frame.iloc[[2, 3]].copy(),
            },
            {
                "fold": 1,
                "train": frame.iloc[[2, 3]].copy(),
                "validation": frame.iloc[[0, 1]].copy(),
            },
        ]
    }

    result = module._fitting_rows(context)

    assert len(result) == 4
    assert set(result["observation_id"]) == {
        "obs_1",
        "obs_2",
        "obs_3",
        "obs_4",
    }


def test_fitting_rows_rejects_inconsistent_source_cv_population() -> None:
    frame = pd.DataFrame(
        {
            "observation_id": [
                "obs_1",
                "obs_2",
                "obs_3",
                "obs_4",
            ],
            "score_analysis": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )

    context = {
        "source_folds": [
            {
                "fold": 0,
                "train": frame.iloc[[0, 1]].copy(),
                "validation": frame.iloc[[2, 3]].copy(),
            },
            {
                "fold": 1,
                "train": frame.iloc[[0, 1]].copy(),
                "validation": frame.iloc[[2]].copy(),
            },
        ]
    }

    with pytest.raises(
        module.ResidueGeometryNeuralEvaluationError,
        match="do not reconstruct the same fitting population",
    ):
        module._fitting_rows(context)


def test_average_repeat_predictions_accepts_requested_repeat_count() -> None:
    frame = pd.DataFrame(
        {
            "scheme_id": ["scheme", "scheme"],
            "evaluation_iteration": ["0", "0"],
            "evaluation_role": ["outer_test", "outer_test"],
            "observation_id": ["obs", "obs"],
            "task_id": ["task", "task"],
            "evaluation_group_id": ["group", "group"],
            "protein_id": ["protein", "protein"],
            "assay_position": [1, 1],
            "wild_type": ["A", "A"],
            "mutant": ["V", "V"],
            "antibiotic": ["AMP", "AMP"],
            "concentration": [1.0, 1.0],
            "temperature_c": [37.0, 37.0],
            "y_true": [0.5, 0.5],
            "model_id": [module.MODEL_ID, module.MODEL_ID],
            "training_repeat": [1, 2],
            "y_pred": [0.4, 0.6],
        }
    )
    original = module.EXPECTED_PREDICTIONS_PER_MODEL
    module.EXPECTED_PREDICTIONS_PER_MODEL = 1
    try:
        averaged = module._average_repeat_predictions(frame, expected_repeats=2)
    finally:
        module.EXPECTED_PREDICTIONS_PER_MODEL = original
    assert len(averaged) == 1
    assert averaged.loc[0, "y_pred"] == pytest.approx(0.5)


def test_context_checkpoint_round_trip(tmp_path) -> None:
    context = {
        "scheme_id": "scheme",
        "evaluation_iteration": "0",
        "evaluation_role": "outer_test",
        "task_id": "task",
        "group_column": "group_id",
    }
    test_rows = pd.DataFrame(
        {
            "observation_id": ["obs"],
            module.TARGET_COLUMN: [0.5],
        }
    )
    repeat_rows = [
        {
            "scheme_id": "scheme",
            "evaluation_iteration": "0",
            "evaluation_role": "outer_test",
            "observation_id": "obs",
            "model_id": module.MODEL_ID,
            "training_repeat": repeat,
            "y_true": 0.5,
            "y_pred": 0.4 + 0.1 * repeat,
        }
        for repeat in (1, 2, 3)
    ]
    history_rows = [
        {
            "scheme_id": "scheme",
            "evaluation_iteration": "0",
            "model_id": module.MODEL_ID,
            "fit_role": "selected_epoch",
            "epoch": 7,
        }
    ]
    module._write_context_checkpoint(
        tmp_path, context, repeat_rows, history_rows, 7, test_rows, 3
    )
    loaded = module._load_context_checkpoint(tmp_path, context, test_rows, 3)
    assert loaded is not None
    repeat_frame, history_frame, selected_epoch = loaded
    assert len(repeat_frame) == 3
    assert len(history_frame) == 1
    assert selected_epoch == 7
