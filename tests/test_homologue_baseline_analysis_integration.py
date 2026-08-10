from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from betalactamase_transfer.homologue_baseline_analysis import (
    COMPARATOR_BASELINES,
    _prepare_direction_table,
    _prepare_nonfloor_analysis,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _matched_tables() -> list[pd.DataFrame]:
    baseline_dir = REPO_ROOT / "data/interim/baseline_evaluation"

    direct = pd.read_csv(baseline_dir / "homologue_direct_transfer_predictions.csv")
    predictions = pd.read_csv(
        baseline_dir / "baseline_predictions.csv",
        dtype={"evaluation_iteration": "string"},
    )
    observations = pd.read_csv(REPO_ROOT / "data/processed/missense_observations.csv")

    return [
        _prepare_direction_table(
            direction,
            direct,
            predictions,
            observations,
        )
        for direction in (
            "NDM-1_to_VIM-2",
            "VIM-2_to_NDM-1",
        )
    ]


def test_matched_population_uses_saved_predictions_and_valid_score_joins() -> None:
    for table in _matched_tables():
        assert len(table) == 1513
        assert table["homologue_position_id"].nunique() == 81
        assert not table["target_observation_id"].duplicated().any()

        for baseline_id in COMPARATOR_BASELINES:
            assert table[f"{baseline_id}_prediction"].notna().all()

        assert (
            table["floor_class"]
            .isin(
                {
                    "neither_at_floor",
                    "both_at_floor",
                    "source_only_at_floor",
                    "target_only_at_floor",
                }
            )
            .all()
        )


def test_nonfloor_and_discordant_floor_contracts() -> None:
    matched = pd.concat(
        _matched_tables(),
        ignore_index=True,
    )

    (
        metrics,
        comparisons,
        discordant,
        discordant_summary,
        summary,
    ) = _prepare_nonfloor_analysis(
        matched,
        bootstrap_replicates=100,
    )

    assert summary["nonfloor_pair_count_per_direction"] == 478
    assert summary["nonfloor_homologue_position_count"] == 63
    assert summary["discordant_floor_pair_count"] == 306

    assert set(metrics["direction"]) == {
        "NDM-1_to_VIM-2",
        "VIM-2_to_NDM-1",
    }

    assert set(metrics["baseline_id"]) == {
        "direct_homologue_source_effect",
        "substitution_mean",
    }

    assert comparisons["bootstrap_valid_replicates"].eq(100).all()

    assert np.isfinite(comparisons["paired_difference"]).all()

    assert len(discordant) == 306

    counts = dict(
        zip(
            discordant_summary["floor_switch_direction"],
            discordant_summary["n_pairs"],
        )
    )

    assert counts["NDM_floor_VIM_nonfloor"] == 118
    assert counts["NDM_nonfloor_VIM_floor"] == 188
