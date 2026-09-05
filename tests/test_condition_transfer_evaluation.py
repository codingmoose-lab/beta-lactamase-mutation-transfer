from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer import condition_transfer_evaluation as module


def test_metric_difference_positive_means_model_better() -> None:
    assert module._metric_difference("spearman", 0.7, 0.6) == pytest.approx(0.1)
    assert module._metric_difference("pearson", 0.7, 0.6) == pytest.approx(0.1)
    assert module._metric_difference("mae", 0.6, 0.7) == pytest.approx(0.1)
    assert module._metric_difference("rmse", 0.6, 0.7) == pytest.approx(0.1)


def test_condition_modes_do_not_encode_held_out_antibiotic() -> None:
    for scheme_id in (
        "vim2_2020_loo_antibiotic_amp",
        "vim2_2020_loo_antibiotic_ctx",
        "vim2_2020_loo_antibiotic_mem",
    ):
        assert module.CONDITION_MODE_BY_SCHEME[scheme_id] == "none"
    assert (
        module.CONDITION_MODE_BY_SCHEME["vim2_2020_temperature_25_to_37"]
        == "log2_concentration"
    )
    assert (
        module.CONDITION_MODE_BY_SCHEME["vim2_2020_temperature_37_to_25"]
        == "log2_concentration"
    )


def test_prediction_contract_requires_same_population_as_reference() -> None:
    rows = []
    for scheme_id in module.CONDITION_SCHEME_IDS:
        reference = module.PRIMARY_REFERENCE_BY_SCHEME[scheme_id]
        for population in (module.FULL_POPULATION, module.STRUCTURE_POPULATION):
            for model_id in (reference, "model_x"):
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "population_scope": population,
                        "model_id": model_id,
                        "observation_id": "obs1",
                        "y_true": 0.0,
                        "y_pred": 0.1,
                    }
                )
    module._validate_prediction_contract(pd.DataFrame(rows))


def test_metric_values_returns_expected_error_metrics() -> None:
    result = module._metric_values(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 2.0, 1.0]),
    )
    assert result["mae"] == pytest.approx(2.0 / 3.0)
    assert result["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert np.isfinite(result["spearman"])
    assert np.isfinite(result["pearson"])
