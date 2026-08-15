from __future__ import annotations

from pathlib import Path

import numpy as np

from betalactamase_transfer.sequence_plm_evaluation import (
    _load_inputs,
    _validate_embedding_provenance,
    _validate_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_esmc_inputs_satisfy_evaluation_contract() -> None:
    (
        observations,
        _schemes,
        _assignments,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = _load_inputs(REPO_ROOT)

    _validate_inputs(
        observations,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    _validate_embedding_provenance(
        REPO_ROOT,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )


def test_real_esmc_arrays_are_finite_over_biological_residue_ranges() -> None:
    (
        _observations,
        _schemes,
        _assignments,
        _membership,
        _embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = _load_inputs(REPO_ROOT)

    assert np.isfinite(mutant_embeddings).all()
    assert np.isfinite(wild_type_embeddings[0, :, :270, :]).all()
    assert np.isfinite(wild_type_embeddings[1, :, :267, :]).all()

    assert wild_type_embeddings[1, :, 267:, :].shape == (4, 3, 1152)
