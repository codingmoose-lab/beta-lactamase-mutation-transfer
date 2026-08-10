import json
from pathlib import Path

import pytest

from betalactamase_transfer.mcsa_annotations import (
    MCSAAnnotationError,
    acquire_mcsa_annotations,
)


def _entry(mcsa_id: int, reference_uniprot: str) -> dict[str, object]:
    return {
        "mcsa_id": mcsa_id,
        "enzyme_name": "beta-lactamase (Class B1)",
        "reference_uniprot_id": reference_uniprot,
        "url": f"www.ebi.ac.uk/thornton-srv/m-csa/entry/{mcsa_id}/",
        "all_ecs": ["3.5.2.6"],
        "residues": [
            {
                "mcsa_id": mcsa_id,
                "roles_summary": "metal ligand",
                "residue_sequences": [
                    {
                        "uniprot_id": reference_uniprot,
                        "code": "His",
                        "resid": 99 if mcsa_id == 15 else 116,
                        "is_reference": True,
                    }
                ],
                "residue_chains": [
                    {
                        "pdb_id": "1znb" if mcsa_id == 15 else "1bc2",
                        "is_reference": True,
                    }
                ],
            }
        ],
        "reaction": {
            "ec": "3.5.2.6",
            "mechanisms": [{"mechanism_id": 1, "is_detailed": True}],
        },
    }


def _page(results: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "count": len(results),
            "next": None,
            "previous": None,
            "results": results,
        }
    ).encode()


def test_acquisition_writes_targeted_source_files(tmp_path: Path) -> None:
    entries = [_entry(15, "P25910"), _entry(16, "P04190")]
    residues = [entry["residues"][0] for entry in entries]

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        assert timeout == 30
        if "/entries/" in url:
            return _page(entries), "application/json"
        if "/residues/" in url:
            return _page(residues), "application/json"
        raise AssertionError(url)

    summary = acquire_mcsa_annotations(
        tmp_path,
        [15, 16],
        timeout=30,
        fetch_bytes=fake_fetch,
        retrieved_at="2026-08-07T00:00:00+00:00",
    )

    assert summary["status"] == "PASS"
    assert summary["entry_count"] == 2
    assert summary["curated_residue_api_record_count"] == 2
    assert summary["curated_residue_counts_by_mcsa_id"] == {"15": 1, "16": 1}
    assert summary["entry_residue_api_crosscheck"] == "PASS"
    assert summary["homologue_bulk_file_acquired"] is False

    raw_dir = tmp_path / "data/raw/mcsa"
    assert (raw_dir / "entries_15_16.json").is_file()
    assert (raw_dir / "residues_15_16.json").is_file()
    assert (raw_dir / "SHA256SUMS.txt").is_file()
    assert (tmp_path / "data/interim/mcsa_acquisition/entries.csv").is_file()


def test_acquisition_rejects_unexpected_entry_selection(tmp_path: Path) -> None:
    entries = [_entry(15, "P25910")]
    residues = [_entry(15, "P25910")["residues"][0]]

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        if "/entries/" in url:
            return _page(entries), "application/json"
        return _page(residues), "application/json"

    with pytest.raises(MCSAAnnotationError, match="entry mismatch"):
        acquire_mcsa_annotations(tmp_path, [15, 16], fetch_bytes=fake_fetch)


def test_acquisition_rejects_paginated_target_response(tmp_path: Path) -> None:
    entries = [_entry(15, "P25910"), _entry(16, "P04190")]
    paginated = json.dumps(
        {
            "count": 3,
            "next": "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?page=2",
            "previous": None,
            "results": [_entry(15, "P25910")["residues"][0]],
        }
    ).encode()

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        if "/entries/" in url:
            return _page(entries), "application/json"
        return paginated, "application/json"

    with pytest.raises(MCSAAnnotationError, match="paginated"):
        acquire_mcsa_annotations(tmp_path, [15, 16], fetch_bytes=fake_fetch)


def test_acquisition_refuses_conflicting_existing_source(tmp_path: Path) -> None:
    entries = [_entry(15, "P25910"), _entry(16, "P04190")]
    residues = [entry["residues"][0] for entry in entries]
    existing = tmp_path / "data/raw/mcsa/entries_15_16.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("different\n", encoding="utf-8")

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        if "/entries/" in url:
            return _page(entries), "application/json"
        return _page(residues), "application/json"

    with pytest.raises(MCSAAnnotationError, match="Refusing to replace"):
        acquire_mcsa_annotations(tmp_path, [15, 16], fetch_bytes=fake_fetch)


def test_acquisition_rejects_residue_view_with_unexpected_entry(tmp_path: Path) -> None:
    entries = [_entry(15, "P25910"), _entry(16, "P04190")]
    residues = [entry["residues"][0] for entry in entries]
    unexpected = dict(_entry(17, "P00000")["residues"][0])
    residues.append(unexpected)

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        if "/entries/" in url:
            return _page(entries), "application/json"
        return json.dumps(residues).encode(), "application/json"

    with pytest.raises(MCSAAnnotationError, match="unexpected entry 17"):
        acquire_mcsa_annotations(tmp_path, [15, 16], fetch_bytes=fake_fetch)


def test_acquisition_rejects_entry_residue_view_mismatch(tmp_path: Path) -> None:
    entries = [_entry(15, "P25910"), _entry(16, "P04190")]
    residues = [entry["residues"][0] for entry in entries]
    altered = dict(residues[0])
    altered["roles_summary"] = "different role"
    residues[0] = altered

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        if "/entries/" in url:
            return _page(entries), "application/json"
        return json.dumps(residues).encode(), "application/json"

    with pytest.raises(MCSAAnnotationError, match="residue identities disagree"):
        acquire_mcsa_annotations(tmp_path, [15, 16], fetch_bytes=fake_fetch)
