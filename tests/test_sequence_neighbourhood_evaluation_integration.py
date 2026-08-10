from __future__ import annotations

from pathlib import Path

import pandas as pd

from betalactamase_transfer.sequence_neighbourhood_evaluation import (
    CORE_SCHEME_IDS,
    _load_assay_sequences,
    _validate_observations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_assay_sequences_match_all_observation_wild_types() -> None:
    observations = pd.read_csv(REPO_ROOT / "data/processed/missense_observations.csv")
    assay_sequences = _load_assay_sequences(REPO_ROOT)
    _validate_observations(observations, assay_sequences)
    assert len(assay_sequences["NDM-1"]) == 270
    assert len(assay_sequences["VIM-2"]) == 267


def test_core_sequence_schemes_are_defined_and_ready() -> None:
    schemes = pd.read_csv(
        REPO_ROOT / "data/interim/evaluation_partitions/partition_schemes.csv"
    )
    selected = schemes[schemes["scheme_id"].isin(CORE_SCHEME_IDS)]
    assert set(selected["scheme_id"]) == set(CORE_SCHEME_IDS)
    assert selected["status"].eq("ready").all()
    assert not selected["held_out_labels_used_for_model_selection"].astype(bool).any()
    assert not selected["assignment_uses_outcome_values"].astype(bool).any()
