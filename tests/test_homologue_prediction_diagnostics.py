"""Tests for direct-vs-fitted mutation-effect comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.homologue_prediction_diagnostics import (
    HomologuePredictionDiagnosticError,
    diagnostic_statistics,
    grouped_model_comparison,
    rank_correlation,
)


def _paired_example() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "homologue_position_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "source_effect": [-4, -2, -1, -3, -1, 0, -2, 0, 1],
            "target_effect": [-4, -3, -1, -2, 0, 1, -3, 0, 2],
            "y_pred": [-3.9, -2.9, -1.1, -2.1, 0.1, 0.9, -3, 0.2, 2],
        }
    )


def test_rank_correlation_rejects_incompatible_arrays() -> None:
    with pytest.raises(HomologuePredictionDiagnosticError, match="shapes differ"):
        rank_correlation(np.array([1, 2, 3]), np.array([1, 2]))


def test_shared_target_diagnostics_do_not_assume_biochemical_explanation() -> None:
    frame = _paired_example()
    results = diagnostic_statistics(frame, repetitions=100, seed=144)
    assert results["n_pairs"] == 9
    assert results["n_homologue_positions"] == 3
    assert results["triangle_inequality_violations"] == 0
    assert results["model_mae"] < results["direct_mae"]
    assert results["null_correlations_calculated"] == 100
    assert "target measurement" in results["interpretation"]


def test_group_bootstrap_reports_paired_intervals() -> None:
    results = grouped_model_comparison(_paired_example(), repetitions=100, seed=432)
    assert results["group_bootstrap_repetitions"] == 100
    assert results["mae_improvement_ci_lower"] <= results["mae_improvement_ci_upper"]
    assert (
        results["spearman_difference_ci_lower"]
        <= results["spearman_difference_ci_upper"]
    )


def test_constant_rank_correlation_is_not_a_number() -> None:
    assert np.isnan(rank_correlation(np.ones(5), np.arange(5)))


def test_matched_prediction_loader_rejects_target_score_disagreement(tmp_path) -> None:
    from betalactamase_transfer.homologue_prediction_diagnostics import (
        BASELINE_PATH,
        DIRECTIONS,
        MODEL_PATH,
        MODELS,
        load_predictions,
    )

    observations = []
    models = []
    for direction in DIRECTIONS:
        for i in range(1513):
            position = i % 81
            record = {
                "direction": direction,
                "source_observation_id": f"{direction}/source/{i}",
                "target_observation_id": f"{direction}/target/{i}",
                "homologue_position_id": position,
                "source_effect": float(i % 7 - 4),
                "target_effect": float(i % 8 - 4),
                "minimum_score_class": "both_above_minimum"
                if i < 478
                else "both_at_minimum",
                "predicted_target_effect": float(i % 7 - 4),
            }
            observations.append(record)
            for model in MODELS:
                models.append(
                    {
                        **{
                            name: record[name]
                            for name in (
                                "direction",
                                "source_observation_id",
                                "target_observation_id",
                                "homologue_position_id",
                                "source_effect",
                                "target_effect",
                                "minimum_score_class",
                            )
                        },
                        "model_id": model,
                        "y_pred": float(i % 7 - 4),
                    }
                )
    baseline = tmp_path / BASELINE_PATH
    prediction = tmp_path / MODEL_PATH
    baseline.parent.mkdir(parents=True, exist_ok=True)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(observations).to_csv(baseline, index=False)
    pd.DataFrame(models).to_csv(prediction, index=False)
    matched, hashes = load_predictions(tmp_path)
    assert len(matched) == 1513 * 2 * len(MODELS)
    assert len(hashes) == 2
    rows = pd.DataFrame(models)
    rows.loc[0, "target_effect"] = 9.5
    rows.to_csv(prediction, index=False)
    with pytest.raises(
        HomologuePredictionDiagnosticError, match="target_effect differs"
    ):
        load_predictions(tmp_path)
