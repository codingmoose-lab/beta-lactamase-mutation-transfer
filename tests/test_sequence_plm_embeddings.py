from __future__ import annotations

import pytest

from betalactamase_transfer.sequence_plm_embeddings import (
    SequencePLMEmbeddingError,
    _make_mutant_sequence,
    _validate_hidden_state_indices,
)


def test_make_mutant_sequence_changes_only_requested_residue() -> None:
    wild_type = "ACDEFG"
    mutant = _make_mutant_sequence(wild_type, 3, "D", "Y")
    assert mutant == "ACYEFG"
    assert len(mutant) == len(wild_type)
    differences = [
        i for i, pair in enumerate(zip(wild_type, mutant)) if pair[0] != pair[1]
    ]
    assert differences == [2]


def test_make_mutant_sequence_rejects_wild_type_mismatch() -> None:
    with pytest.raises(SequencePLMEmbeddingError, match="Wild-type mismatch"):
        _make_mutant_sequence("ACDEFG", 3, "E", "Y")


def test_hidden_state_indices_are_validated() -> None:
    assert _validate_hidden_state_indices((9, 18, 27, 36)) == (9, 18, 27, 36)
    with pytest.raises(SequencePLMEmbeddingError, match="unique"):
        _validate_hidden_state_indices((36, 36))
    with pytest.raises(SequencePLMEmbeddingError, match="between"):
        _validate_hidden_state_indices((37,))
