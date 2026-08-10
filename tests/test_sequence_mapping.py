from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from betalactamase_transfer.sequence_mapping import (
    GenBankTargetSpec,
    SequenceMappingError,
    alignment_consensus,
    alignment_metrics,
    alignment_position_map,
    extract_genbank_target,
)


def _write_genbank(path: Path, feature: SeqFeature, sequence: str) -> None:
    record = SeqRecord(Seq(sequence), id="fixture", name="fixture")
    record.annotations["molecule_type"] = "DNA"
    record.features = [feature]
    with path.open("w", encoding="utf-8") as handle:
        from Bio import SeqIO

        SeqIO.write(record, handle, "genbank")


def test_extracts_labelled_exon_without_translation(tmp_path: Path) -> None:
    path = tmp_path / "construct.gb"
    feature = SeqFeature(
        FeatureLocation(0, 12, strand=1),
        type="exon",
        qualifiers={"label": ["NDM1"]},
    )
    _write_genbank(path, feature, "ATGAAACCC TAA".replace(" ", ""))
    result = extract_genbank_target(
        path,
        GenBankTargetSpec(
            protein_id="NDM-1",
            feature_types=("exon",),
            required_label_terms=("NDM1",),
            expected_protein_length=3,
            record_id="fixture_ndm",
        ),
    )
    assert result.sequence == "MKP"
    assert result.terminal_stop_found is True
    assert result.translation_frame == 1


def test_respects_reverse_strand(tmp_path: Path) -> None:
    path = tmp_path / "reverse.gb"
    # Reverse complement of ATGAAACCCTAA.
    feature = SeqFeature(
        FeatureLocation(0, 12, strand=-1),
        type="CDS",
        qualifiers={"label": ["VIM 2 ORF"]},
    )
    _write_genbank(path, feature, "TTAGGGTTTCAT")
    result = extract_genbank_target(
        path,
        GenBankTargetSpec(
            protein_id="VIM-2",
            feature_types=("CDS",),
            required_label_terms=("VIM", "2", "ORF"),
            expected_protein_length=3,
            record_id="fixture_vim",
        ),
    )
    assert result.sequence == "MKP"


def test_no_longest_feature_fallback(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.gb"
    feature = SeqFeature(
        FeatureLocation(0, 12, strand=1),
        type="CDS",
        qualifiers={"label": ["unrelated protein"]},
    )
    _write_genbank(path, feature, "ATGAAACCCTAA")
    with pytest.raises(SequenceMappingError, match="labelled"):
        extract_genbank_target(
            path,
            GenBankTargetSpec(
                protein_id="VIM-2",
                feature_types=("CDS",),
                required_label_terms=("VIM", "2", "ORF"),
                expected_protein_length=3,
                record_id="fixture_vim",
            ),
        )


def test_alignment_reports_single_query_insertion() -> None:
    metrics = alignment_metrics("ABCDE", "AGBCDE")
    assert metrics["insertions_relative_to_reference"] == 1
    assert metrics["deletions_relative_to_reference"] == 0
    assert metrics["matches"] == 5


def test_alignment_position_map_marks_inserted_query_residue() -> None:
    _, query_to_ref, _ = alignment_position_map("ABCDE", "AGBCDE")
    assert query_to_ref[1] == 1
    assert query_to_ref[2] is None
    assert query_to_ref[3] == 2


def test_alignment_consensus_marks_tied_positions_ambiguous() -> None:
    consensus = alignment_consensus("AAA", "AA")
    assert consensus.optimal_alignment_count == 3
    assert consensus.ambiguous_reference_positions == (1, 2, 3)
    assert consensus.ambiguous_query_positions == (1, 2)
    assert all(value is None for value in consensus.reference_to_query.values())
    assert all(value is None for value in consensus.query_to_reference.values())
