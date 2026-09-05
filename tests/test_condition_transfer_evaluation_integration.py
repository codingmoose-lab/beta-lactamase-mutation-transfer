from __future__ import annotations

from pathlib import Path

from betalactamase_transfer.condition_transfer_evaluation import (
    CONDITION_SCHEME_IDS,
    review_condition_transfer_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_live_condition_transfer_input_contract() -> None:
    required = REPO_ROOT / "data/interim/evaluation_partitions/partition_schemes.csv"
    if not required.is_file():
        return
    result = review_condition_transfer_inputs(REPO_ROOT)
    assert result["status"] == "PASS"
    assert result["scheme_count"] == len(CONDITION_SCHEME_IDS) == 5
    assert result["test_labels_used_for_model_selection"] is False
    assert result["antibiotic_condition_mode"] == "none"
    assert result["temperature_condition_mode"] == "log2_concentration"
