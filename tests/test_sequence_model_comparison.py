from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import betalactamase_transfer.sequence_model_comparison as comparison
from betalactamase_transfer.sequence_model_comparison import (
    MODEL_PAIRS,
    SequenceModelComparisonError,
    _matched_predictions,
    _metric_difference,
    _metric_values,
    _paired_comparisons,
    _validate_matched_population,
)


def _prediction_frame(model_id: str, predictions: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheme_id": ["scheme", "scheme"],
            "task_id": ["task", "task"],
            "model_id": [model_id, model_id],
            "evaluation_iteration": ["0", "0"],
            "evaluation_role": ["outer_test", "outer_test"],
            "observation_id": ["obs1", "obs2"],
            "evaluation_group_id": ["g1", "g2"],
            "protein_id": ["NDM-1", "NDM-1"],
            "assay_position": [1, 2],
            "wild_type": ["M", "A"],
            "mutant": ["A", "C"],
            "antibiotic": ["AMP", "AMP"],
            "concentration": [1.0, 1.0],
            "temperature_c": [37.0, 37.0],
            "y_true": [0.0, 1.0],
            "y_pred": predictions,
        }
    )


def _small_matched_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheme_id": ["scheme"] * 4,
            "evaluation_iteration": ["0", "0", "1", "1"],
            "evaluation_role": ["outer_test"] * 4,
            "observation_id": ["o1", "o2", "o3", "o4"],
            "task_id": ["task"] * 4,
            "evaluation_group_id": ["g1", "g1", "g2", "g2"],
            "protein_id": ["NDM-1"] * 4,
            "assay_position": [1, 1, 2, 2],
            "wild_type": ["M", "M", "A", "A"],
            "mutant": ["A", "C", "C", "D"],
            "antibiotic": ["AMP"] * 4,
            "concentration": [1.0] * 4,
            "temperature_c": [37.0] * 4,
            "y_true": [0.0, 1.0, 2.0, 3.0],
            "mutation_condition_ridge": [0.5, 0.5, 2.5, 2.5],
            "sequence_neighbourhood_ridge": [0.1, 0.9, 2.1, 2.9],
            "esmc_site_delta_ridge": [0.0, 1.0, 2.0, 3.0],
        }
    )


def test_metric_values_are_exact_for_identical_predictions() -> None:
    values = np.asarray([-2.0, -1.0, 0.0, 1.0])
    metrics = _metric_values(values, values.copy())
    assert metrics["spearman"] == 1.0
    assert metrics["pearson"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_constant_prediction_has_undefined_correlations() -> None:
    y_true = np.asarray([-2.0, -1.0, 0.0, 1.0])
    y_pred = np.zeros(4, dtype=float)
    metrics = _metric_values(y_true, y_pred)
    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["pearson"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])


def test_metric_difference_is_positive_when_model_a_is_better() -> None:
    assert _metric_difference("spearman", 0.8, 0.6) == pytest.approx(0.2)
    assert _metric_difference("pearson", 0.8, 0.6) == pytest.approx(0.2)
    assert _metric_difference("mae", 1.0, 1.5) == pytest.approx(0.5)
    assert _metric_difference("rmse", 1.0, 1.5) == pytest.approx(0.5)


def test_metric_difference_rejects_unknown_metric() -> None:
    with pytest.raises(SequenceModelComparisonError, match="Unsupported metric"):
        _metric_difference("r2", 0.0, 0.0)


def test_matched_predictions_requires_exact_population_identity() -> None:
    frames = {
        "mutation_condition_ridge": _prediction_frame(
            "mutation_condition_ridge", [0.1, 0.9]
        ),
        "sequence_neighbourhood_ridge": _prediction_frame(
            "sequence_neighbourhood_ridge", [0.2, 0.8]
        ),
        "esmc_site_delta_ridge": _prediction_frame("esmc_site_delta_ridge", [0.0, 1.0]),
    }
    frames["esmc_site_delta_ridge"].loc[1, "assay_position"] = 3
    with pytest.raises(SequenceModelComparisonError, match="Matched field disagrees"):
        _matched_predictions(frames)


def test_matched_predictions_rejects_duplicate_comparison_keys() -> None:
    frame = _prediction_frame("mutation_condition_ridge", [0.1, 0.9])
    frame.loc[1, "observation_id"] = "obs1"
    frame.loc[1, "assay_position"] = 1
    frame.loc[1, "wild_type"] = "M"
    frame.loc[1, "mutant"] = "A"
    frames = {
        "mutation_condition_ridge": frame,
        "sequence_neighbourhood_ridge": _prediction_frame(
            "sequence_neighbourhood_ridge", [0.2, 0.8]
        ),
        "esmc_site_delta_ridge": _prediction_frame("esmc_site_delta_ridge", [0.0, 1.0]),
    }
    with pytest.raises(SequenceModelComparisonError, match="Duplicate comparison keys"):
        _matched_predictions(frames)


def test_paired_comparisons_are_reproducible_and_use_declared_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "BOOTSTRAP_REPLICATES", 20)
    matched = _small_matched_frame()

    first = _paired_comparisons(matched)
    second = _paired_comparisons(matched)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(MODEL_PAIRS) * 4
    assert set(first["bootstrap_replicates"]) == {20}
    assert set(first["n_groups"]) == {2}
    assert set(first["resampling_unit"]) == {"evaluation_group_id"}
    assert set(first["interval_method"]) == {"percentile"}
    assert set(first["difference_interpretation"]) == {"positive_favours_model_a"}


def test_validate_matched_population_rejects_group_across_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matched = _small_matched_frame()
    monkeypatch.setattr(comparison, "EXPECTED_MATCHED_OBSERVATION_COUNT", 4)
    monkeypatch.setattr(
        comparison,
        "EXPECTED_SCHEME_OBSERVATION_COUNTS",
        {"scheme": 4},
    )
    monkeypatch.setattr(
        comparison,
        "EXPECTED_SCHEME_ROLES",
        {"scheme": "outer_test"},
    )
    matched.loc[2:, "evaluation_group_id"] = "g1"

    with pytest.raises(
        SequenceModelComparisonError,
        match="multiple test iterations",
    ):
        _validate_matched_population(matched)
