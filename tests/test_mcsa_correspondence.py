from __future__ import annotations

import json
from pathlib import Path

import pytest

from betalactamase_transfer.mcsa_correspondence import (
    MCSACorrespondenceError,
    _alignment_position_map,
    _optimal_alignment_consensus,
    _sequence_and_chain,
    _unique_roles,
)


def _write_uniprot(
    path: Path, accession: str, sequence: str, start: int, end: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "primaryAccession": accession,
                "sequence": {"value": sequence, "length": len(sequence)},
                "features": [
                    {
                        "type": "Chain",
                        "location": {
                            "start": {"value": start},
                            "end": {"value": end},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_sequence_and_chain_reads_exact_uniprot_interval(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    _write_uniprot(path, "PTEST", "MABCDEFGHIJK", 3, 10)
    sequence, start, end = _sequence_and_chain(path, "PTEST")
    assert sequence == "MABCDEFGHIJK"
    assert (start, end) == (3, 10)


def test_sequence_and_chain_rejects_multiple_chain_features(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    payload = {
        "primaryAccession": "PTEST",
        "sequence": {"value": "MAAAAA"},
        "features": [
            {
                "type": "Chain",
                "location": {"start": {"value": 1}, "end": {"value": 3}},
            },
            {
                "type": "Chain",
                "location": {"start": {"value": 4}, "end": {"value": 6}},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MCSACorrespondenceError, match="exactly one valid"):
        _sequence_and_chain(path, "PTEST")


def test_alignment_position_map_rejects_mismatched_block_counts() -> None:
    class FakeAlignment:
        aligned = (
            [[0, 1], [2, 3]],
            [[0, 1]],
        )

    with pytest.raises(MCSACorrespondenceError, match="unequal numbers"):
        _alignment_position_map(FakeAlignment(), 1, 1)


def test_optimal_alignment_consensus_maps_unambiguous_positions() -> None:
    consensus, summary = _optimal_alignment_consensus(
        "MABCDE",
        (2, 6),
        "XXABCDEYY",
        (3, 7),
        [2, 3, 6],
    )
    assert consensus == {
        2: ("consensus_mapped", 3),
        3: ("consensus_mapped", 4),
        6: ("consensus_mapped", 7),
    }
    assert summary["optimal_alignment_count"] >= 1
    assert summary["alignment_method"] == "global_mature_chain_blosum62_affine_gap"


def test_optimal_alignment_consensus_marks_alignment_ambiguity(monkeypatch) -> None:
    class FakeAlignment:
        def __init__(self, target_start: int) -> None:
            self.aligned = (
                [[0, 1]],
                [[target_start, target_start + 1]],
            )
            self.score = 1.0

    class FakeAlignments(list):
        pass

    class FakeAligner:
        def align(self, reference: str, target: str):
            return FakeAlignments([FakeAlignment(0), FakeAlignment(1)])

    monkeypatch.setattr(
        "betalactamase_transfer.mcsa_correspondence._make_aligner",
        lambda: FakeAligner(),
    )
    consensus, _ = _optimal_alignment_consensus(
        "A",
        (1, 1),
        "AA",
        (1, 2),
        [1],
    )
    assert consensus[1] == ("alignment_ambiguous", None)


def test_unique_roles_deduplicates_mechanism_repetitions() -> None:
    record = {
        "roles": [
            {
                "function_type": "interaction",
                "group_function": "metal ligand",
                "function": "metal ligand",
                "emo": "EMO_00116",
            },
            {
                "function_type": "interaction",
                "group_function": "metal ligand",
                "function": "metal ligand",
                "emo": "EMO_00116",
            },
        ]
    }
    assert _unique_roles(record) == [
        ("interaction", "metal ligand", "metal ligand", "EMO_00116")
    ]
