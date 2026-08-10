from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from betalactamase_transfer.baseline_evaluation import prepare_baseline_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_baseline_predictions_preserve_scientific_contracts() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions, selection, direct, direct_metrics = prepare_baseline_predictions(
            REPO_ROOT
        )
    runtime_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert not runtime_warnings
    membership = pd.read_csv(
        REPO_ROOT / "data/interim/modelling_views/scheme_observation_membership.csv"
    )
    assert predictions["scheme_id"].nunique() == 17
    assert set(predictions["scheme_id"]) == set(membership["scheme_id"])
    assert predictions["observation_id"].isin(membership["observation_id"]).all()
    assert not {"score_original", "score_analysis", "score_uncertainty"} & set(
        predictions.columns
    )
    assert not selection.empty
    assert np.isfinite(predictions["y_pred"]).all()
    assert np.isfinite(predictions["y_true"]).all()
    selected = selection[selection["selection_scope"].eq("selected_setting")]
    assert np.isfinite(selected["validation_mae"]).all()
    assert len(direct) == 3026
    assert direct["homologue_position_id"].nunique() == 81
    assert direct.groupby("direction").size().eq(1513).all()
    assert len(direct_metrics) == 2
