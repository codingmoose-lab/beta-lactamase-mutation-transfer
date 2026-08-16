from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import betalactamase_transfer.experimental_structure_evaluation as evaluation
from betalactamase_transfer.experimental_structure_evaluation import (
    ExperimentalStructureEvaluationError,
    _fit_predict,
    _metric_values,
    _normalise_secondary_structure,
    _select_setting,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": [f"o{i}" for i in range(8)],
            "protein_id": ["NDM-1"] * 8,
            "assay_position": list(range(1, 9)),
            "wild_type": list("ACDEFGHI"),
            "mutant": list("CDEFGHIK"),
            "antibiotic": ["AMP"] * 8,
            "concentration": [128.0] * 8,
            "temperature_c": [37.0] * 8,
            "score_analysis": np.linspace(-2.0, 2.0, 8),
            "group_id": [f"g{i}" for i in range(8)],
            "sasa_angstrom2": np.linspace(10.0, 80.0, 8),
            "relative_sasa_wilke": np.linspace(0.1, 0.8, 8),
            "ca_neighbour_count_8_angstrom": np.arange(8) + 2,
            "ca_neighbour_count_10_angstrom": np.arange(8) + 4,
            "ca_neighbour_count_12_angstrom": np.arange(8) + 8,
            "heavy_atom_contact_count_4_5_angstrom": np.arange(8) + 1,
            "secondary_structure_3state": [
                "helix",
                "strand",
                "coil",
                "H",
                "E",
                "C",
                "helix",
                "strand",
            ],
        }
    )


def test_secondary_structure_normalisation() -> None:
    assert _normalise_secondary_structure("H") == "helix"
    assert _normalise_secondary_structure("E") == "strand"
    assert _normalise_secondary_structure("C") == "coil"
    assert _normalise_secondary_structure("helix") == "helix"
    with pytest.raises(ExperimentalStructureEvaluationError):
        _normalise_secondary_structure("unknown")


def test_fit_predict_returns_finite_values() -> None:
    frame = _frame()
    prediction = _fit_predict(
        frame.iloc[:6],
        frame.iloc[6:],
        include_structure=True,
        condition_mode="categorical_condition",
        alpha=1.0,
    )
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_log2_concentration_model_returns_finite_values() -> None:
    frame = _frame()
    frame["concentration"] = [4, 8, 16, 32, 64, 128, 256, 4]
    prediction = _fit_predict(
        frame.iloc[:6],
        frame.iloc[6:],
        include_structure=True,
        condition_mode="log2_concentration",
        alpha=1.0,
    )
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_setting_selection_returns_same_result_for_same_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation, "RIDGE_ALPHAS", (0.1, 1.0, 10.0))
    frame = _frame()
    folds = [(frame.iloc[:6].copy(), frame.iloc[6:].copy())]
    first = _select_setting(
        folds,
        model_id=evaluation.EXPERIMENTAL_STRUCTURE_MODEL,
        condition_mode="categorical_condition",
    )
    second = _select_setting(
        folds,
        model_id=evaluation.EXPERIMENTAL_STRUCTURE_MODEL,
        condition_mode="categorical_condition",
    )
    assert first == second


def test_identical_predictions_have_zero_error() -> None:
    values = np.asarray([-1.0, 0.0, 1.0, 2.0])
    metrics = _metric_values(values, values.copy())
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
