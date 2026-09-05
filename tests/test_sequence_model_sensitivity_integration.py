from __future__ import annotations

from pathlib import Path

from betalactamase_transfer import sequence_model_sensitivity as module


def test_embedding_inventory_matches_established_mutation_set(repo_root: Path) -> None:
    frame = module.build_mutation_embedding_index(repo_root)
    assert len(frame) == 10086
    assert frame["embedding_row"].tolist() == list(range(10086))
    assert set(frame["protein_id"]) == {"NDM-1", "VIM-2"}
