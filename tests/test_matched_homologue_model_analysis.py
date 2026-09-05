from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from betalactamase_transfer.matched_homologue_model_analysis import (
    _correlation_or_nan,
    _metric_difference,
    _metric_values,
    _population_subset,
)


def test_metric_values_identical_predictions() -> None:
    values = np.asarray([-4.0, -1.0, 0.0, 2.0], dtype=float)
    result = _metric_values(values, values.copy())
    assert result == {
        "spearman": 1.0,
        "pearson": 1.0,
        "mae": 0.0,
        "rmse": 0.0,
    }


def test_constant_correlation_is_nan_without_runtime_warning() -> None:
    left = np.asarray([1.0, 1.0, 1.0], dtype=float)
    right = np.asarray([1.0, 2.0, 3.0], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _correlation_or_nan(left, right)
    assert np.isnan(result)


def test_metric_difference_positive_always_favours_model() -> None:
    assert np.isclose(_metric_difference("spearman", 0.6, 0.4), 0.2)
    assert np.isclose(_metric_difference("pearson", 0.5, 0.2), 0.3)
    assert np.isclose(_metric_difference("mae", 0.8, 1.0), 0.2)
    assert np.isclose(_metric_difference("rmse", 1.0, 1.5), 0.5)


def test_neither_at_floor_population_is_selected_without_other_floor_states() -> None:
    frame = pd.DataFrame(
        {
            "floor_class": [
                "neither_at_floor",
                "both_at_floor",
                "source_only_at_floor",
                "target_only_at_floor",
            ],
            "value": [1, 2, 3, 4],
        }
    )
    selected = _population_subset(frame, "neither_at_floor")
    assert selected["value"].tolist() == [1]
