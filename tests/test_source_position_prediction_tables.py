"""Check population membership, minimum-score ranking, and grouped uncertainty."""

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.source_position_prediction_tables import (
    combine_predictions,
    grouped_interval,
    minimum_score_auc,
    population_mask,
)


def test_population_masks() -> None:
    classes = pd.Series(
        (
            "both_above_minimum",
            "both_at_minimum",
            "source_at_minimum",
            "target_at_minimum",
        )
    )
    assert population_mask(classes, "target_above_minimum").tolist() == (
        [True, False, True, False]
    )
    assert population_mask(classes, "source_above_minimum").tolist() == (
        [True, False, False, True]
    )
    assert population_mask(classes, "both_above_minimum").tolist() == (
        [True, False, False, False]
    )
    assert population_mask(classes, "discordant_minimum").tolist() == (
        [False, False, True, True]
    )
    assert population_mask(classes, "all_shared_wild_type_pairs").all()
    with pytest.raises(ValueError, match="Unknown"):
        population_mask(classes, "unknown")


def test_minimum_score_auc_lower_is_positive() -> None:
    assert (
        minimum_score_auc(np.array((-4, -4, -1, 0)), np.array((-4, -3, -1, 0))) == 1.0
    )
    assert (
        minimum_score_auc(np.array((-4, -4, -1, 0)), np.array((0, -1, -3, -4))) == 0.0
    )


def test_grouped_interval_deterministic_and_finite() -> None:
    values = np.array((0.4, 0.1, -0.2, 0.3, 0.8, -0.1))
    groups = np.array(("a", "a", "b", "b", "c", "c"))
    a = grouped_interval(values, groups, repetitions=300, seed_label="test")
    b = grouped_interval(values, groups, repetitions=300, seed_label="test")
    assert a == b
    assert np.isfinite(a).all()
    assert a[0] <= values.mean() <= a[1]


def test_matching_target_observations_and_folds() -> None:

    esmc_records = []
    full_records = []
    for direction in ("NDM-1_to_VIM-2", "VIM-2_to_NDM-1"):
        for index in range(1513):
            position = index % 81
            record = {
                "direction": direction,
                "target_observation_id": f"{direction}:{index}",
                "homologue_position_id": f"position_{position:03d}",
                "heldout_fold": position % 5,
                "target_effect": -4.0 if index % 2 else -1.0,
                "minimum_score_class": (
                    "both_at_minimum" if index % 2 else "both_above_minimum"
                ),
            }
            esmc_records.append(
                {**record, "model_prediction": -2.0, "reference_prediction": -2.5}
            )
            for model in (
                "source_assay_mean",
                "source_assay_mutant_mean",
                "source_assay_substitution_mean",
                "source_assay_identity_ridge",
            ):
                full_records.append({**record, "model_id": model, "y_pred": -2.5})
    esmc = pd.DataFrame(esmc_records)
    full = pd.DataFrame(full_records)
    checked = combine_predictions(esmc, full)
    assert len(checked) == 3026
    bad = full.copy()
    bad.loc[0, "target_effect"] = -3.0
    with pytest.raises(ValueError, match="Target measurement difference"):
        combine_predictions(esmc, bad)
    bad = full.copy()
    bad.loc[0, "heldout_fold"] = 4
    with pytest.raises(ValueError, match="heldout_fold mismatch"):
        combine_predictions(esmc, bad)


def test_minimum_score_auc_tied_predictions() -> None:
    assert (
        minimum_score_auc(np.array((-4, -4, -1, 0)), np.array((-2, -2, -2, -2))) == 0.5
    )
    assert (
        minimum_score_auc(np.array((-4, -4, -1, 0)), np.array((-3, -1, -1, 0))) == 0.875
    )
