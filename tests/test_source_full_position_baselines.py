"""Small checks of source-only fitting and paired comparison contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.source_full_position_baselines import (
    SourcePositionEvaluationError,
    alpha_selection,
    matrix,
    predict_reference,
)


def sample_positions() -> pd.DataFrame:
    records = []
    for position in range(15):
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            records.append(
                {
                    "assay_position": position,
                    "wild_type": "A",
                    "mutant": aa,
                    "score_analysis": position / 10 + (aa in "DE"),
                }
            )
    return pd.DataFrame(records)


def test_design_matrix_has_only_sequence_identity() -> None:
    source = sample_positions()
    assert matrix(source).shape == (300, 41)


def test_heldout_position_rejected() -> None:
    train = sample_positions()
    test = train.loc[train.assay_position.eq(0)].copy()
    with pytest.raises(SourcePositionEvaluationError, match="Held-out"):
        predict_reference(train, test, alpha=10.0)


def test_source_cv_and_new_prediction_does_not_access_target_scores() -> None:
    all_rows = sample_positions()
    training = all_rows.loc[all_rows.assay_position.ne(0)]
    heldout = all_rows.loc[all_rows.assay_position.eq(0)]
    alpha, trials = alpha_selection(training)
    assert alpha in (0.1, 1.0, 10.0, 100.0)
    assert len(trials) == 4
    preds = predict_reference(training, heldout, alpha)
    assert set(preds) == {
        "source_assay_mean",
        "source_assay_mutant_mean",
        "source_assay_substitution_mean",
        "source_assay_identity_ridge",
    }
    assert all(np.isfinite(p).all() and len(p) == 20 for p in preds.values())


def test_source_effect_change_changes_predictions_without_target() -> None:
    data = sample_positions()
    training = data.loc[data.assay_position.ne(0)].copy()
    testing = data.loc[data.assay_position.eq(0)]
    first = predict_reference(training, testing, 10.0)["source_assay_mean"]
    training.loc[:, "score_analysis"] += 0.5
    second = predict_reference(training, testing, 10.0)["source_assay_mean"]
    assert np.allclose(second - first, 0.5)


def test_identity_ridge_rejects_nonfinite_cv_predictions(monkeypatch) -> None:
    from betalactamase_transfer import source_full_position_baselines as module

    monkeypatch.setattr(module, "ridge_fit", lambda train, alpha: np.full(41, np.nan))
    with pytest.raises(SourcePositionEvaluationError, match="source CV|CV error"):
        alpha_selection(sample_positions())


def test_identity_ridge_rejects_nonfinite_final_predictions(monkeypatch) -> None:
    from betalactamase_transfer import source_full_position_baselines as module

    train = sample_positions().loc[lambda d: d.assay_position.ne(0)]
    test = sample_positions().loc[lambda d: d.assay_position.eq(0)]
    monkeypatch.setattr(module, "ridge_fit", lambda rows, alpha: np.full(41, np.nan))
    with pytest.raises(
        SourcePositionEvaluationError, match="source reference prediction"
    ):
        predict_reference(train, test, 1.0)
