"""Tests for position-withheld source-protein reference estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.position_withheld_homologue_baselines import (
    DIRECTIONS,
    MODEL_NAMES,
    PositionWithheldEvaluationError,
    _fit_reference_predictions,
    calculate_heldout_predictions,
    fold_for_positions,
    score_predictions,
)


def _matched_frame() -> pd.DataFrame:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    i = 0
    for pos in range(81):
        wt = amino_acids[pos % 20]
        mutants = [aa for aa in amino_acids if aa != wt]
        row_count = 19 if pos < 55 else 18
        for mutant in mutants[:row_count]:
            ndm_value = float(((i * 3) % 18) / 9.0 - 1.0)
            vim_value = float(((i * 7) % 17) / 8.5 - 1.0)
            rows.append(
                {
                    "homologue_position_id": f"H{pos:03d}",
                    "wild_type": wt,
                    "mutant": mutant,
                    "ndm": ndm_value,
                    "vim": vim_value,
                    "ndm_observation_id": f"N{i}",
                    "vim_observation_id": f"V{i}",
                    "minimum_score_class": "both_above_minimum",
                }
            )
            i += 1
    assert len(rows) == 1513
    # Exactly 478 samples at 63 positions are selected for the secondary analysis.
    by_position = {f"H{pos:03d}": [] for pos in range(63)}
    for j, item in enumerate(rows):
        if item["homologue_position_id"] in by_position:
            by_position[item["homologue_position_id"]].append(j)
    selected = set()
    for slot in range(7):
        for indices in by_position.values():
            selected.add(indices[slot])
    for indices in list(by_position.values())[:37]:
        selected.add(indices[7])
    assert len(selected) == 478
    assert len({rows[j]["homologue_position_id"] for j in selected}) == 63
    frames = []
    for direction in DIRECTIONS:
        for j, item in enumerate(rows):
            source_first = direction == DIRECTIONS[0]
            source = item["ndm"] if source_first else item["vim"]
            target = item["vim"] if source_first else item["ndm"]
            frames.append(
                {
                    "direction": direction,
                    "homologue_position_id": item["homologue_position_id"],
                    "wild_type": item["wild_type"],
                    "mutant": item["mutant"],
                    "source_observation_id": item["ndm_observation_id"]
                    if source_first
                    else item["vim_observation_id"],
                    "target_observation_id": item["vim_observation_id"]
                    if source_first
                    else item["ndm_observation_id"],
                    "source_effect": source,
                    "target_effect": target,
                    "minimum_score_class": "both_above_minimum"
                    if j in selected
                    else "both_at_minimum",
                }
            )
    return pd.DataFrame(frames)


def test_fold_assignment_is_repeatable_and_independent_of_effects() -> None:
    positions = pd.Series([f"H{i}" for i in range(81)])
    assigned = fold_for_positions(positions)
    assert assigned == fold_for_positions(positions.sample(frac=1, random_state=3))
    assert set(assigned.values()) == {0, 1, 2, 3, 4}
    assert max(pd.Series(list(assigned.values())).value_counts()) == 17


def test_fitted_values_exclude_heldout_source_effects() -> None:
    frame = _matched_frame().query("direction == @DIRECTIONS[0]").copy()
    assignments = fold_for_positions(frame.homologue_position_id)
    frame["fold"] = frame.homologue_position_id.map(assignments)
    train = frame.loc[frame.fold.ne(0)]
    test = frame.loc[frame.fold.eq(0)]
    normal = _fit_reference_predictions(train, test)
    test_perturbed = test.copy()
    test_perturbed["source_effect"] = 100000
    test_perturbed["target_effect"] = -100000
    changed = _fit_reference_predictions(train, test_perturbed)
    for model in MODEL_NAMES:
        np.testing.assert_array_equal(normal[model], changed[model])


def test_target_scores_cannot_change_prediction_values() -> None:
    frame = _matched_frame()
    first, folds = calculate_heldout_predictions(frame)
    changed_frame = frame.copy()
    changed_frame["target_effect"] += 999
    second, changed_folds = calculate_heldout_predictions(changed_frame)
    np.testing.assert_array_equal(first.y_pred.to_numpy(), second.y_pred.to_numpy())
    pd.testing.assert_frame_equal(folds, changed_folds)
    assert len(first) == 3026 * len(MODEL_NAMES)
    assert first.target_observation_id.nunique() == 3026


def test_only_source_changes_outside_heldout_position_affect_model() -> None:
    frame = _matched_frame().query("direction == @DIRECTIONS[0]").copy()
    assignments = fold_for_positions(frame.homologue_position_id)
    frame["fold"] = frame.homologue_position_id.map(assignments)
    train = frame.loc[frame.fold.ne(0)]
    test = frame.loc[frame.fold.eq(0)]
    normal = _fit_reference_predictions(train, test)
    changed_train = train.copy()
    changed_train["source_effect"] += 10
    changed = _fit_reference_predictions(changed_train, test)
    assert not np.array_equal(normal[MODEL_NAMES[0]], changed[MODEL_NAMES[0]])


def test_overlapping_positions_rejected() -> None:
    frame = _matched_frame().query("direction == @DIRECTIONS[0]")
    with pytest.raises(PositionWithheldEvaluationError, match="Held-out position"):
        _fit_reference_predictions(frame.head(12), frame.head(12))


def test_metrics_have_expected_populations() -> None:
    predictions, _ = calculate_heldout_predictions(_matched_frame())
    result = score_predictions(predictions, repetitions=100)
    assert len(result) == 12
    above = result.loc[result.population.eq("both_above_minimum")]
    assert set(above.n_pairs) == {478}
    assert set(above.n_homologue_positions) == {63}
    assert np.isfinite(result.mae).all()
    assert result.mae_improvement_ci_lower.le(result.mae_improvement_ci_upper).all()
