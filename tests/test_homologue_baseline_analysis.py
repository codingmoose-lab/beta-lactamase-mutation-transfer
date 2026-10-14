from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from betalactamase_transfer.homologue_baseline_analysis import (
    _metric_difference,
    _metric_values,
    _minimum_score_class,
)


def test_minimum_score_class_separates_all_four_states() -> None:
    source = pd.Series([-4.0, -4.0, -1.0, -1.0])
    target = pd.Series([-4.0, -1.0, -4.0, -1.0])

    assert _minimum_score_class(source, target).tolist() == [
        "both_at_minimum",
        "source_at_minimum",
        "target_at_minimum",
        "both_above_minimum",
    ]


def test_metric_values_are_exact_for_identical_predictions() -> None:
    values = np.asarray(
        [-4.0, -1.0, 0.0, 1.0],
        dtype=float,
    )

    metrics = _metric_values(
        values,
        values.copy(),
    )

    assert metrics["spearman"] == 1.0
    assert metrics["pearson"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_constant_prediction_has_undefined_correlations_without_warning() -> None:
    y_true = np.asarray(
        [-4.0, -1.0, 0.0, 1.0],
        dtype=float,
    )
    y_pred = np.full_like(
        y_true,
        -1.0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            RuntimeWarning,
        )
        metrics = _metric_values(
            y_true,
            y_pred,
        )

    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["pearson"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])


def test_metric_difference_positive_value_favours_direct() -> None:
    assert (
        _metric_difference(
            "spearman",
            0.4,
            0.2,
        )
        == 0.2
    )

    assert (
        _metric_difference(
            "mae",
            1.0,
            1.5,
        )
        == 0.5
    )
