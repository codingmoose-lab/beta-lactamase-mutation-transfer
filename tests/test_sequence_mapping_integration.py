import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_actual_reference_files_pass(tmp_path: Path) -> None:
    # The function writes only to data/interim and never modifies raw files.
    summary = json.loads(
        (
            REPO_ROOT / "data/interim/sequence_mapping/sequence_mapping_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["audit_status"] == "PASS"
    output_dir = REPO_ROOT / "data/interim/sequence_mapping"
    inventory = pd.read_csv(output_dir / "sequence_sources.csv")
    assert set(inventory["record_id"]) == {
        "NDM1_UniProt_C7C422",
        "NDM1_2024_assay_construct",
        "3SPU_entity_1",
        "VIM2_UniProt_Q9K2N0",
        "VIM2_2020_assay_construct",
        "5YD7_entity_1",
    }
    assert (
        inventory["source_file"].map(lambda value: not Path(value).is_absolute()).all()
    )
    assert inventory["source_file"].str.startswith("data/raw/").all()
    lengths = dict(zip(inventory["record_id"], inventory["length"]))
    assert lengths["NDM1_2024_assay_construct"] == 270
    assert lengths["VIM2_2020_assay_construct"] == 267
    with (output_dir / "sequence_mapping_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        persisted = json.load(handle)
    assert persisted["database_reference_decisions"]["A4GRB6"] == (
        "historical_inactive_identifier_only"
    )


def test_ndm_3spu_tied_alignment_is_confined_to_unmodeled_positions() -> None:
    required = [
        REPO_ROOT / "data/raw/uniprot/C7C422/C7C422.fasta",
        REPO_ROOT / "data/raw/pdb/3SPU/3SPU.cif",
    ]
    if not all(path.exists() for path in required):
        return

    from betalactamase_transfer.sequence_mapping import (
        alignment_consensus,
        read_mmcif_entity_sequences,
        read_mmcif_scheme,
        read_single_fasta,
    )

    uniprot = read_single_fasta(required[0], "NDM-1", "NDM1_UniProt_C7C422")
    pdb_record = read_mmcif_entity_sequences(required[1], "NDM-1", "3SPU")[0]
    scheme = read_mmcif_scheme(required[1], "3SPU")
    consensus = alignment_consensus(uniprot.sequence, pdb_record.sequence)

    assert consensus.optimal_alignment_count == 2
    assert consensus.ambiguous_reference_positions == (1, 27)
    assert consensus.ambiguous_query_positions == (1,)
    modeled_positions = {row.entity_position for row in scheme if row.modeled}
    assert modeled_positions.isdisjoint(consensus.ambiguous_query_positions)
