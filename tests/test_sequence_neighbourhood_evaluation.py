from __future__ import annotations

import numpy as np

from betalactamase_transfer.sequence_neighbourhood_evaluation import (
    MUTATION_CONDITION_MODEL,
    SEQUENCE_NEIGHBOURHOOD_MODEL,
    _candidate_grid,
    _metric_values,
    _neighbourhood_states,
)


def test_neighbourhood_excludes_centre_and_uses_padding() -> None:
    states = _neighbourhood_states("ACDEFG", position=1, radius=2)
    assert states == ("PAD", "PAD", "C", "D")
    assert len(states) == 4


def test_candidate_grid_keeps_ablation_radius_free() -> None:
    assert all(
        radius is None for radius, _ in _candidate_grid(MUTATION_CONDITION_MODEL)
    )
    assert all(
        radius is not None
        for radius, _ in _candidate_grid(SEQUENCE_NEIGHBOURHOOD_MODEL)
    )


def test_metric_values_are_exact_for_identical_predictions() -> None:
    values = np.asarray([-4.0, -1.0, 0.0, 1.0], dtype=float)
    metrics = _metric_values(values, values.copy())
    assert metrics["spearman"] == 1.0
    assert metrics["pearson"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_constant_prediction_has_undefined_correlations() -> None:
    y_true = np.asarray([-4.0, -1.0, 0.0, 1.0], dtype=float)
    y_pred = np.full_like(y_true, -1.0)
    metrics = _metric_values(y_true, y_pred)
    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["pearson"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
