from __future__ import annotations

import numpy as np
import pandas as pd

from betalactamase_transfer import sequence_model_sensitivity as module


def test_feature_count_uses_esm2_hidden_size() -> None:
    transform = module.ESM2Transform(
        hidden_state_index=33,
        condition_mode="none",
        condition_categories=(),
        concentration_mean=0.0,
        concentration_sd=1.0,
        embedding_mean=np.zeros(module.ESM2_HIDDEN_SIZE),
        embedding_sd=np.ones(module.ESM2_HIDDEN_SIZE),
    )
    assert module._feature_count(transform) == 1 + len(module.SUBSTITUTIONS) + 1280


def test_metric_difference_positive_means_esm2_is_better() -> None:
    assert module._metric_difference("mae", 0.8, 1.0) > 0
    assert module._metric_difference("rmse", 1.0, 1.2) > 0
    assert module._metric_difference("spearman", 0.7, 0.6) > 0
    assert module._metric_difference("pearson", 0.8, 0.7) > 0


def test_paired_bootstrap_uses_evaluation_groups() -> None:
    frame = pd.DataFrame(
        {
            "scheme_id": ["s"] * 8,
            "evaluation_group_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "y_true": [0.0, 1.0] * 4,
            "esm2_prediction": [0.1, 0.9] * 4,
            "esmc_prediction": [0.3, 0.7] * 4,
        }
    )
    lower, upper, valid = module._paired_bootstrap_difference(
        frame,
        "mae",
        replicates=50,
    )
    assert valid == 50
    assert lower > 0
    assert upper > 0


def test_hidden_state_set_is_fixed_before_evaluation() -> None:
    assert module.EMBEDDING_HIDDEN_STATE_INDICES == (8, 16, 24, 33)
    assert module.ESM2_SOURCE_MODEL_ID == "facebook/esm2_t33_650M_UR50D"
