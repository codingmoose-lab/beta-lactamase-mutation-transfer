from __future__ import annotations

import re
from pathlib import Path

from betalactamase_transfer.sequence_model_comparison import (
    EXPECTED_MATCHED_OBSERVATION_COUNT,
    EXPECTED_SCHEME_OBSERVATION_COUNTS,
    _load_predictions,
    _matched_predictions,
    _validate_matched_population,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_sequence_model_predictions_are_exactly_matched() -> None:
    frames, digests = _load_predictions(REPO_ROOT)
    matched = _matched_predictions(frames)
    _validate_matched_population(matched)

    assert len(matched) == EXPECTED_MATCHED_OBSERVATION_COUNT
    assert matched.groupby("scheme_id").size().to_dict() == (
        EXPECTED_SCHEME_OBSERVATION_COUNTS
    )
    assert set(digests) == {
        "sequence_neighbourhood_predictions_sha256",
        "sequence_plm_predictions_sha256",
    }
    for value in digests.values():
        assert re.fullmatch(r"[0-9a-f]{64}", value)
