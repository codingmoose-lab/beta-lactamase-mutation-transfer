from __future__ import annotations

from pathlib import Path

from betalactamase_transfer.sequence_plm_embeddings import (
    EmbeddingConfig,
    build_mutation_embedding_index,
    prepare_sequence_plm_embedding_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_plm_embedding_index_contains_every_unique_missense_mutation() -> None:
    embedding_index = build_mutation_embedding_index(REPO_ROOT)
    counts = embedding_index.groupby("protein_id").size().to_dict()
    assert len(embedding_index) == 10086
    assert counts == {"NDM-1": 5041, "VIM-2": 5045}
    assert embedding_index["embedding_row"].tolist() == list(
        range(len(embedding_index))
    )


def test_embedding_index_preparation_is_outcome_blind_and_uses_verified_runtime() -> (
    None
):
    summary = prepare_sequence_plm_embedding_index(REPO_ROOT, EmbeddingConfig())
    assert summary["outcome_columns_loaded"] is False
    assert summary["batch_size"] == 8
    assert summary["precision"] == "float16"
    assert summary["hidden_state_indices"] == [9, 18, 27, 36]
    assert summary["primary_hidden_state_index"] == 36
