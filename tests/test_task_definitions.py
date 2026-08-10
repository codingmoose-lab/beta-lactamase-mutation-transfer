from __future__ import annotations

import pandas as pd

from betalactamase_transfer.task_definitions import (
    EVALUATION_REGIMES,
    TASK_DEFINITIONS,
    _filter_observations,
)


def test_filter_observations_applies_exact_condition_scope() -> None:
    frame = pd.DataFrame(
        {
            "assay_id": ["a", "a", "b"],
            "protein_id": ["P", "P", "Q"],
            "antibiotic": ["AMP", "CTX", "AMP"],
            "concentration": [128.0, 4.0, 128.0],
            "temperature_c": [37.0, 37.0, 37.0],
        }
    )
    scope = {
        "assay_id": "a",
        "protein_id": "P",
        "antibiotic": "AMP",
        "concentration": "128",
        "temperature_c": "37",
    }
    selected = _filter_observations(frame, scope)
    assert selected.index.tolist() == [0]


def test_bidirectional_headline_transfer_is_defined() -> None:
    headline = {
        task["task_id"] for task in TASK_DEFINITIONS if task["priority"] == "headline"
    }
    assert headline == {
        "ndm1_to_vim2_amp_transfer",
        "vim2_to_ndm1_amp_transfer",
    }


def test_random_row_headline_evaluation_is_not_defined() -> None:
    assert all("random" not in regime["regime_id"] for regime in EVALUATION_REGIMES)


def test_aligned_shared_wild_type_transfer_reporting_is_defined() -> None:
    regime = next(
        item
        for item in EVALUATION_REGIMES
        if item["regime_id"] == "aligned_shared_wild_type_transfer_reporting"
    )
    assert regime["status"] == "ready_for_reporting_population"
    assert "81 shared-WT positions" in regime["required_rule"]
    assert "target labels" in regime["required_rule"]


def test_antibiotic_transfer_is_restricted_to_shared_temperature() -> None:
    regime = next(
        item
        for item in EVALUATION_REGIMES
        if item["regime_id"] == "leave_one_antibiotic_out_vim2_2020"
    )
    assert "37 C" in regime["required_rule"]
    assert "AMP, CTX, and MEM" in regime["required_rule"]
