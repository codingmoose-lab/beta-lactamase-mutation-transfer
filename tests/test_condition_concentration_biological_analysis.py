from __future__ import annotations

import numpy as np
import pandas as pd

from betalactamase_transfer.condition_concentration_biological_analysis import (
    _mutation_trajectory_summary,
    _position_trajectory_summary,
)


def test_position_trajectory_summary_preserves_position_level() -> None:
    mutations = pd.DataFrame(
        {
            "assay_position": [10, 10, 20],
            "score_range": [1.0, 3.0, 2.0],
            "absolute_endpoint_change": [0.5, 1.5, 1.0],
            "endpoint_change_256_minus_4": [-0.5, -1.5, 1.0],
            "linear_slope_per_log2_concentration": [-0.1, -0.3, 0.2],
            "spearman_with_log2_concentration": [-1.0, -0.8, 0.9],
        }
    )
    result = _position_trajectory_summary(mutations)
    assert result["assay_position"].tolist() == [10, 20]
    row10 = result[result["assay_position"].eq(10)].iloc[0]
    assert row10["n_mutations"] == 2
    assert np.isclose(row10["median_score_range"], 2.0)
    assert np.isclose(row10["fraction_negative_endpoint_change"], 1.0)


def test_position_summary_does_not_pool_positions() -> None:
    mutations = pd.DataFrame(
        {
            "assay_position": [1, 2],
            "score_range": [1.0, 10.0],
            "absolute_endpoint_change": [1.0, 10.0],
            "endpoint_change_256_minus_4": [1.0, -10.0],
            "linear_slope_per_log2_concentration": [0.2, -2.0],
            "spearman_with_log2_concentration": [1.0, -1.0],
        }
    )
    result = _position_trajectory_summary(mutations)
    assert len(result) == 2
    assert result.loc[result["assay_position"].eq(1), "mean_score_range"].item() == 1.0
    assert result.loc[result["assay_position"].eq(2), "mean_score_range"].item() == 10.0


def test_mutation_trajectory_summary_handles_constant_scores_without_warning() -> None:
    predictions = pd.DataFrame(
        {
            "scheme_id": [
                "vim2_2022_loo_concentration_4",
                "vim2_2022_loo_concentration_8",
                "vim2_2022_loo_concentration_16",
                "vim2_2022_loo_concentration_32",
                "vim2_2022_loo_concentration_64",
                "vim2_2022_loo_concentration_128",
                "vim2_2022_loo_concentration_256",
            ],
            "population_scope": ["sequence_eligible"] * 7,
            "model_id": ["source_concentration_nearest"] * 7,
            "observation_id": [f"obs-{i}" for i in range(7)],
            "evaluation_group_id": ["mutation-1"] * 7,
            "assay_position": [10] * 7,
            "wild_type": ["A"] * 7,
            "mutant": ["V"] * 7,
            "concentration": [4, 8, 16, 32, 64, 128, 256],
            "antibiotic": ["meropenem"] * 7,
            "temperature_c": [37.0] * 7,
            "y_true": [1.0] * 7,
            "y_pred": [1.0] * 7,
        }
    )
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        result = _mutation_trajectory_summary(predictions)
    assert len(result) == 1
    assert np.isnan(result.iloc[0]["spearman_with_log2_concentration"])
    assert np.isclose(result.iloc[0]["linear_slope_per_log2_concentration"], 0.0)
