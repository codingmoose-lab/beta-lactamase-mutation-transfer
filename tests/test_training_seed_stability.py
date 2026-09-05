from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module("betalactamase_transfer.training_seed_stability")


def test_seed_plan_has_ten_declared_seeds_per_model() -> None:
    plan = module._seed_plan()
    assert len(plan) == 40
    counts = plan.groupby("model_id")["training_seed"].nunique()
    assert counts.eq(10).all()
    assert (
        plan.groupby("model_id")["seed_set_index"].apply(list).tolist()
        == [list(range(1, 11))] * 4
    )


def test_metric_difference_is_positive_when_model_a_is_better() -> None:
    assert module._metric_difference("mae", 0.7, 0.9) == pytest.approx(0.2)
    assert module._metric_difference("rmse", 1.0, 1.3) == pytest.approx(0.3)
    assert module._metric_difference("spearman", 0.8, 0.6) == pytest.approx(0.2)
    assert module._metric_difference("pearson", 0.7, 0.5) == pytest.approx(0.2)


def test_selected_epochs_are_read_only_from_selected_epoch_rows() -> None:
    frame = pd.DataFrame(
        {
            "scheme_id": ["scheme", "scheme", "scheme"],
            "evaluation_iteration": ["0", "0", "0"],
            "model_id": ["model", "model", "model"],
            "fit_role": ["validation_epoch_selection", "selected_epoch", "final_refit"],
            "epoch": [1, 17, 1],
        }
    )
    values = module._selected_epochs_from_history(frame, ("model",))
    assert values == {("scheme", "0", "model"): 17}


def test_comparison_method_for_geometry_uses_all_seed_combinations() -> None:
    rows = []
    for scheme_id in module.CORE_SCHEME_IDS:
        for model_id in module.MODEL_IDS:
            seeds = module.SEEDS_BY_MODEL[model_id]
            for index, seed in enumerate(seeds, start=1):
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_id": model_id,
                        "seed_set_index": index,
                        "training_seed": seed,
                        "n_observations": 2,
                        "n_groups": 1,
                        "spearman": 0.5 + 0.001 * index,
                        "pearson": 0.5 + 0.001 * index,
                        "mae": 1.0 - 0.001 * index,
                        "rmse": 1.2 - 0.001 * index,
                    }
                )
    comparisons = module._comparison_rows(pd.DataFrame(rows))
    geometry = comparisons[
        comparisons["model_a"].eq(module.MODEL_GEOMETRY)
        & comparisons["model_b"].eq(module.MODEL_GRAPH)
    ]
    assert set(geometry["comparison_method"]) == {"all_seed_combinations"}
    assert len(geometry) == len(module.CORE_SCHEME_IDS) * 100 * 4


def test_prediction_frame_records_actual_training_seed() -> None:
    reference = pd.DataFrame(
        {
            "scheme_id": ["scheme"],
            "evaluation_iteration": ["0"],
            "evaluation_role": ["test"],
            "observation_id": ["obs"],
            "task_id": ["task"],
            "evaluation_group_id": ["group"],
            "protein_id": ["NDM-1"],
            "assay_position": [10],
            "wild_type": ["A"],
            "mutant": ["V"],
            "antibiotic": ["AMP"],
            "concentration": [256.0],
            "temperature_c": [37.0],
            "y_true": [0.5],
        }
    )
    context = {"reference": reference}
    frame = module._prediction_frame(
        context,
        module.MODEL_GRAPH,
        4,
        32104,
        np.array([0.6]),
    )
    assert frame.loc[0, "seed_set_index"] == 4
    assert frame.loc[0, "training_seed"] == 32104
    assert frame.loc[0, "record_source"] == "stage37_additional_fit"
