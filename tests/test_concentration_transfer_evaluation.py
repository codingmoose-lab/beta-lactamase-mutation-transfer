from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer import (
    concentration_transfer_evaluation as concentration_eval,
)


def test_reference_choice_uses_linear_for_interpolation() -> None:
    primary, secondary = concentration_eval._reference_models("interpolation")
    assert primary == "source_concentration_linear_log2"
    assert secondary == "source_concentration_nearest"


def test_reference_choice_uses_nearest_for_boundary() -> None:
    primary, secondary = concentration_eval._reference_models("edge_extrapolation")
    assert primary == "source_concentration_nearest"
    assert secondary == "source_concentration_linear_log2"


def test_reference_choice_rejects_unknown_category() -> None:
    with pytest.raises(concentration_eval.ConcentrationTransferEvaluationError):
        concentration_eval._reference_models("unknown")


def test_metric_direction_makes_positive_better() -> None:
    assert (
        concentration_eval.condition_eval._metric_difference("spearman", 0.7, 0.6) > 0
    )
    assert concentration_eval.condition_eval._metric_difference("mae", 0.4, 0.5) > 0


def test_prediction_contract_rejects_population_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for model_id, observation_ids in {
        "source_concentration_linear_log2": ["a", "b"],
        "concentration_transfer_esmc_site_delta_ridge": ["a"],
    }.items():
        for observation_id in observation_ids:
            rows.append(
                {
                    "scheme_id": "vim2_2022_loo_concentration_8",
                    "population_scope": concentration_eval.FULL_POPULATION,
                    "model_id": model_id,
                    "observation_id": observation_id,
                    "y_true": 0.0,
                    "y_pred": 0.0,
                }
            )
    monkeypatch.setattr(
        concentration_eval,
        "CONCENTRATION_SCHEME_IDS",
        ("vim2_2022_loo_concentration_8",),
    )
    frame = pd.DataFrame(rows)
    matched = frame.copy()
    matched["population_scope"] = concentration_eval.STRUCTURE_POPULATION
    frame = pd.concat([frame, matched], ignore_index=True)
    with pytest.raises(concentration_eval.ConcentrationTransferEvaluationError):
        concentration_eval._validate_prediction_contract(
            frame,
            {"vim2_2022_loo_concentration_8": "source_concentration_linear_log2"},
        )


def test_metric_values_are_finite_for_nonconstant_predictions() -> None:
    values = concentration_eval.condition_eval._metric_values(
        np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.9, 2.2])
    )
    assert np.isfinite(list(values.values())).all()
