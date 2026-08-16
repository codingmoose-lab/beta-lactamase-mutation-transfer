from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import betalactamase_transfer.structure_matched_model_comparison as matched_eval
from betalactamase_transfer.structure_matched_model_comparison import (
    MODEL_IDS,
    MODEL_PAIRS,
    StructureMatchedModelComparisonError,
    _metric_difference,
    _paired_comparisons,
)


def _small_matched_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheme_id": ["scheme"] * 4,
            "evaluation_iteration": ["0", "0", "1", "1"],
            "evaluation_role": ["outer_test"] * 4,
            "observation_id": ["o1", "o2", "o3", "o4"],
            "task_id": ["task"] * 4,
            "evaluation_group_id": ["g1", "g1", "g2", "g2"],
            "protein_id": ["NDM-1"] * 4,
            "assay_position": [1, 1, 2, 2],
            "wild_type": ["M", "M", "A", "A"],
            "mutant": ["A", "C", "C", "D"],
            "antibiotic": ["AMP"] * 4,
            "concentration": [1.0] * 4,
            "temperature_c": [37.0] * 4,
            "y_true": [0.0, 1.0, 2.0, 3.0],
            MODEL_IDS[0]: [0.5, 0.5, 2.5, 2.5],
            MODEL_IDS[1]: [0.2, 0.8, 2.2, 2.8],
            MODEL_IDS[2]: [0.1, 0.9, 2.1, 2.9],
            MODEL_IDS[3]: [0.0, 1.0, 2.0, 3.0],
        }
    )


def test_four_models_define_six_pairwise_contrasts() -> None:
    assert len(MODEL_IDS) == 4
    assert len(MODEL_PAIRS) == 6
    assert {frozenset(pair) for pair in MODEL_PAIRS} == {
        frozenset((model_a, model_b))
        for index, model_a in enumerate(MODEL_IDS)
        for model_b in MODEL_IDS[index + 1 :]
    }


def test_metric_difference_is_positive_when_model_a_is_better() -> None:
    assert _metric_difference("spearman", 0.8, 0.6) == pytest.approx(0.2)
    assert _metric_difference("pearson", 0.8, 0.6) == pytest.approx(0.2)
    assert _metric_difference("mae", 1.0, 1.5) == pytest.approx(0.5)
    assert _metric_difference("rmse", 1.0, 1.5) == pytest.approx(0.5)


def test_metric_difference_rejects_unknown_metric() -> None:
    with pytest.raises(
        StructureMatchedModelComparisonError, match="Unsupported metric"
    ):
        _metric_difference("r2", 0.0, 0.0)


def test_paired_comparisons_repeat_with_same_inputs_and_share_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched_eval, "BOOTSTRAP_REPLICATES", 20)
    matched = _small_matched_frame()

    first = _paired_comparisons(matched)
    second = _paired_comparisons(matched)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(MODEL_PAIRS) * 4
    assert set(first["bootstrap_replicates"]) == {20}
    assert set(first["n_groups"]) == {2}
    assert set(first["resampling_unit"]) == {"evaluation_group_id"}
    assert set(first["interval_method"]) == {"percentile"}
    assert set(first["difference_interpretation"]) == {"positive_favours_model_a"}
    assert np.isfinite(first["paired_difference"]).all()
