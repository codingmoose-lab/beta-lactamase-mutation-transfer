from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from betalactamase_transfer.condition_concentration_biological_analysis import (
    EXPECTED_INPUT_HASHES,
    review_condition_concentration_inputs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_live_condition_eval_concentration_eval_inputs_are_unchanged() -> None:
    root = Path.cwd()
    first = root / next(iter(EXPECTED_INPUT_HASHES))
    if not first.is_file():
        pytest.skip(
            "Live condition- and concentration-transfer outputs are not present "
            "in this test environment"
        )

    summary = review_condition_concentration_inputs(root)
    assert summary["status"] == "PASS"
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        assert _sha256(root / relative) == expected

    condition = json.loads(
        (
            root / "data/interim/condition_transfer_evaluation/"
            "condition_transfer_summary.json"
        ).read_text(encoding="utf-8")
    )
    concentration = json.loads(
        (
            root / "data/interim/concentration_response_evaluation/"
            "concentration_transfer_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert condition["status"] == "PASS"
    assert concentration["status"] == "PASS"
    assert concentration["log2_concentration_used"] is True
