from __future__ import annotations

import json
from pathlib import Path

import pytest

from betalactamase_transfer.mcsa_reference_sequences import (
    MCSAReferenceSequenceError,
    _load_mcsa_reference_accessions,
    _parse_fasta,
    _parse_flat_file,
    _parse_json,
    acquire_mcsa_reference_sequences,
)


def _json_record(accession: str, sequence: str) -> bytes:
    return json.dumps(
        {
            "primaryAccession": accession,
            "uniProtkbId": f"TEST_{accession}",
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "organism": {"scientificName": "Test organism"},
            "sequence": {"value": sequence, "length": len(sequence)},
            "entryAudit": {
                "sequenceVersion": 3,
                "lastSequenceUpdateDate": "2024-01-01",
            },
        }
    ).encode()


def _fasta_record(accession: str, sequence: str) -> bytes:
    return f">sp|{accession}|TEST_{accession} Test protein\n{sequence}\n".encode()


def _txt_record(accession: str, sequence: str) -> bytes:
    grouped = " ".join(
        sequence[index : index + 10] for index in range(0, len(sequence), 10)
    )
    return (
        f"ID   TEST_{accession} Reviewed; {len(sequence)} AA.\n"
        f"AC   {accession};\n"
        f"SQ   SEQUENCE   {len(sequence)} AA;\n"
        f"     {grouped}        {len(sequence)}\n"
        "//\n"
    ).encode()


def _write_mcsa_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "results": [
                    {"mcsa_id": 15, "reference_uniprot_id": "P25910"},
                    {"mcsa_id": 16, "reference_uniprot_id": "P04190"},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_load_mcsa_reference_accessions(tmp_path: Path) -> None:
    path = tmp_path / "entries.json"
    _write_mcsa_fixture(path)
    assert _load_mcsa_reference_accessions(path) == {15: "P25910", 16: "P04190"}


def test_three_uniprot_formats_parse_same_sequence() -> None:
    accession = "P25910"
    sequence = "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGV"
    fasta_sequence, _ = _parse_fasta(_fasta_record(accession, sequence), accession)
    json_record = _parse_json(_json_record(accession, sequence), accession)
    flat_sequence, accessions = _parse_flat_file(
        _txt_record(accession, sequence), accession
    )
    assert fasta_sequence == sequence
    assert json_record["sequence"] == sequence
    assert flat_sequence == sequence
    assert accessions == [accession]


def test_parse_json_rejects_accession_mismatch() -> None:
    with pytest.raises(MCSAReferenceSequenceError, match="accession mismatch"):
        _parse_json(_json_record("P04190", "AAAA"), "P25910")


def test_acquisition_crosschecks_and_writes_sources(tmp_path: Path) -> None:
    _write_mcsa_fixture(tmp_path / "data/raw/mcsa/entries_15_16.json")
    sequences = {
        "P25910": "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGV",
        "P04190": "MKKSTLALVAAASLAAADAAHHHHHHHHHHHH",
    }
    release_headers = {
        "X-UniProt-Release": "2026_03",
        "X-UniProt-Release-Date": "22-July-2026",
    }

    def fetch(url: str, timeout: int):
        assert timeout == 30
        accession, suffix = url.rsplit("/", 1)[-1].split(".", 1)
        sequence = sequences[accession]
        if suffix == "fasta":
            return _fasta_record(accession, sequence), "text/plain", release_headers
        if suffix == "json":
            return (
                _json_record(accession, sequence),
                "application/json",
                release_headers,
            )
        if suffix == "txt":
            return _txt_record(accession, sequence), "text/plain", release_headers
        raise AssertionError(suffix)

    summary = acquire_mcsa_reference_sequences(
        tmp_path,
        timeout=30,
        fetch_response=fetch,
        retrieved_at="2026-08-07T10:30:00+00:00",
    )

    assert summary["status"] == "PASS"
    assert summary["accession_count"] == 2
    assert summary["downloaded_source_file_count"] == 6
    assert summary["sequence_representation_crosscheck"] == "PASS"
    assert summary["uniprot_release_consistency"] == "PASS"
    assert (tmp_path / "data/raw/uniprot/P25910/P25910.fasta").is_file()
    assert (tmp_path / "data/raw/uniprot/P04190/P04190.json").is_file()
    manifest = (tmp_path / "data/raw/uniprot/SHA256SUMS.txt").read_text()
    assert "P25910/P25910.fasta" in manifest
    assert "P04190/P04190.txt" in manifest


def test_acquisition_refuses_cross_format_sequence_disagreement(tmp_path: Path) -> None:
    _write_mcsa_fixture(tmp_path / "data/raw/mcsa/entries_15_16.json")
    release_headers = {
        "X-UniProt-Release": "2026_03",
        "X-UniProt-Release-Date": "22-July-2026",
    }

    def fetch(url: str, timeout: int):
        accession, suffix = url.rsplit("/", 1)[-1].split(".", 1)
        sequence = "AAAA"
        if suffix == "fasta":
            return _fasta_record(accession, sequence), "text/plain", release_headers
        if suffix == "json":
            return _json_record(accession, "AAAT"), "application/json", release_headers
        return _txt_record(accession, sequence), "text/plain", release_headers

    with pytest.raises(
        MCSAReferenceSequenceError, match="sequence representations disagree"
    ):
        acquire_mcsa_reference_sequences(tmp_path, fetch_response=fetch)


def test_acquisition_refuses_to_replace_changed_source(tmp_path: Path) -> None:
    _write_mcsa_fixture(tmp_path / "data/raw/mcsa/entries_15_16.json")
    target = tmp_path / "data/raw/uniprot/P25910/P25910.fasta"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("different existing bytes\n", encoding="utf-8")
    release_headers = {
        "X-UniProt-Release": "2026_03",
        "X-UniProt-Release-Date": "22-July-2026",
    }

    def fetch(url: str, timeout: int):
        accession, suffix = url.rsplit("/", 1)[-1].split(".", 1)
        sequence = "AAAA"
        if suffix == "fasta":
            return _fasta_record(accession, sequence), "text/plain", release_headers
        if suffix == "json":
            return (
                _json_record(accession, sequence),
                "application/json",
                release_headers,
            )
        return _txt_record(accession, sequence), "text/plain", release_headers

    with pytest.raises(MCSAReferenceSequenceError, match="Refusing to replace"):
        acquire_mcsa_reference_sequences(tmp_path, fetch_response=fetch)
