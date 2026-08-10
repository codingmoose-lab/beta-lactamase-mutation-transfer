from __future__ import annotations

import numpy as np
import pandas as pd

from betalactamase_transfer.baseline_evaluation import (
    CONCENTRATION_BASELINE_ID,
    TARGET_COLUMN,
    _fit_ridge_transform,
    _metrics,
    _predict_baseline,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wild_type": ["A", "A", "G", "G"],
            "mutant": ["V", "L", "V", "L"],
            "concentration": [4.0, 8.0, 16.0, 32.0],
            TARGET_COLUMN: [1.0, 2.0, -1.0, 0.0],
        }
    )


def test_lookup_baselines_back_off_to_training_information_only() -> None:
    train = _frame().iloc[:3].copy()
    test = pd.DataFrame(
        {
            "wild_type": ["W"],
            "mutant": ["Y"],
            "concentration": [64.0],
            TARGET_COLUMN: [999.0],
        }
    )
    prediction = _predict_baseline("substitution_mean", train, test)
    assert prediction[0] == train[TARGET_COLUMN].mean()


def test_concentration_transform_is_fit_on_training_rows_only() -> None:
    train = _frame().iloc[:3].copy()
    transform = _fit_ridge_transform(train, include_log2_concentration=True)
    expected = np.log2(train["concentration"].to_numpy()).mean()
    assert transform.concentration_mean == expected


def test_ridge_prediction_does_not_depend_on_test_target() -> None:
    train = _frame().iloc[:3].copy()
    test = _frame().iloc[3:].copy()
    first = _predict_baseline(CONCENTRATION_BASELINE_ID, train, test, alpha=1.0)
    test[TARGET_COLUMN] = 12345.0
    second = _predict_baseline(CONCENTRATION_BASELINE_ID, train, test, alpha=1.0)
    np.testing.assert_allclose(first, second)


def test_metrics_are_exact_for_perfect_predictions() -> None:
    y = np.asarray([-2.0, -1.0, 1.0, 3.0])
    metrics = _metrics(y, y)
    assert metrics["spearman"] == 1.0
    assert np.isclose(metrics["pearson"], 1.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_linear_log2_trajectory_interpolates_source_measurements_only() -> None:
    from betalactamase_transfer.baseline_evaluation import (
        _linear_log2_trajectory_prediction,
    )

    source = pd.DataFrame(
        {
            "concentration": [4.0, 16.0],
            TARGET_COLUMN: [0.0, 2.0],
        }
    )
    prediction = _linear_log2_trajectory_prediction(source, 8.0)
    assert np.isclose(prediction, 1.0)
