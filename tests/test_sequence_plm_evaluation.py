from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.sequence_plm_evaluation import (
    SequencePLMEvaluationError,
    _site_delta,
    _validate_inputs,
)


def _embedding_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "protein_id": ["NDM-1", "VIM-2"],
            "assay_mutation": ["M1A", "M1A"],
            "assay_position": [1, 1],
            "wild_type": ["M", "M"],
            "mutant": ["A", "A"],
            "mutant_sequence_sha256": ["a", "b"],
        }
    )


def test_validate_inputs_requires_canonical_embedding_rows() -> None:
    index = _embedding_index()
    index["embedding_row"] = [1, 0]

    observations = pd.DataFrame({"score_analysis": [0.0]})
    membership = pd.DataFrame({"observation_id": ["obs_1"]})

    with pytest.raises(SequencePLMEvaluationError, match="canonical contiguous"):
        _validate_inputs(
            observations,
            membership,
            index,
            np.empty((0,)),
            np.empty((0,)),
        )


def test_validate_inputs_rejects_unexpected_embedding_protein() -> None:
    index = _embedding_index()
    index.loc[1, "protein_id"] = "OTHER"

    observations = pd.DataFrame({"score_analysis": [0.0]})
    membership = pd.DataFrame({"observation_id": ["obs_1"]})

    with pytest.raises(SequencePLMEvaluationError, match="Unexpected embedding-index"):
        _validate_inputs(
            observations,
            membership,
            index,
            np.empty((0,)),
            np.empty((0,)),
        )


def test_site_delta_rejects_unexpected_protein() -> None:
    frame = pd.DataFrame(
        {
            "embedding_row": [0],
            "protein_id": ["OTHER"],
            "assay_position": [1],
        }
    )
    mutant = np.zeros((1, 4, 1152), dtype=np.float16)
    wild_type = np.zeros((2, 4, 270, 1152), dtype=np.float16)

    with pytest.raises(SequencePLMEvaluationError, match="Unexpected protein"):
        _site_delta(frame, mutant, wild_type)


def test_site_delta_uses_vim2_residue_identity() -> None:
    frame = pd.DataFrame(
        {
            "embedding_row": [0],
            "protein_id": ["VIM-2"],
            "assay_position": [267],
        }
    )
    mutant = np.zeros((1, 4, 1152), dtype=np.float16)
    wild_type = np.zeros((2, 4, 270, 1152), dtype=np.float16)

    mutant[0, 3, :] = 3.0
    wild_type[1, 3, 266, :] = 1.0

    delta = _site_delta(frame, mutant, wild_type)

    assert delta.shape == (1, 1152)
    assert np.all(delta == 2.0)
