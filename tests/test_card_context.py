from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path

import pytest

from betalactamase_transfer.card_context import (
    CARDContextError,
    _ancestor_depths,
    _assay_crosswalk_rows,
    _validate_archive_members,
    parse_aro_obo,
)

MINI_OBO = """\
format-version: 1.2
date: 29:05:2025 10:16
saved-by: CARD
ontology: antibiotic_resistance

[Term]
id: ARO:0000020
name: carbapenem

[Term]
id: ARO:0000032
name: cephalosporin

[Term]
id: ARO:3000008
name: penicillin beta-lactam

[Term]
id: ARO:3009123
name: penicillin with extended spectrum
is_a: ARO:3000008 ! penicillin beta-lactam

[Term]
id: ARO:3009107
name: third-generation cephalosporin
is_a: ARO:0000032 ! cephalosporin

[Term]
id: ARO:3000637
name: ampicillin
is_a: ARO:3009123 ! penicillin with extended spectrum
synonym: "AMP" EXACT []

[Term]
id: ARO:3000645
name: cefotaxime
is_a: ARO:3009107 ! third-generation cephalosporin
synonym: "CTX" EXACT []

[Term]
id: ARO:0000073
name: meropenem
is_a: ARO:0000020 ! carbapenem

[Term]
id: ARO:3000057
name: NDM beta-lactamase
relationship: confers_resistance_to_drug_class ARO:0000020 ! carbapenem
relationship: confers_resistance_to_drug_class ARO:0000032 ! cephalosporin
relationship: confers_resistance_to_drug_class ARO:3000008 ! penicillin beta-lactam

[Term]
id: ARO:3000589
name: NDM-1
is_a: ARO:3000057 ! NDM beta-lactamase
relationship: confers_resistance_to_antibiotic ARO:0000073 ! meropenem
synonym: "NDM-1" EXACT CARD_Short_Name []
"""


def _write_obo(tmp_path: Path) -> Path:
    path = tmp_path / "aro.obo"
    path.write_text(MINI_OBO, encoding="utf-8")
    return path


def test_parse_aro_obo_preserves_relationships_and_synonyms(tmp_path: Path) -> None:
    terms = parse_aro_obo(_write_obo(tmp_path))
    ndm = terms["ARO:3000589"]
    assert ndm.name == "NDM-1"
    assert ndm.parents == ("ARO:3000057",)
    assert ndm.relationships == (("confers_resistance_to_antibiotic", "ARO:0000073"),)
    assert ndm.synonyms == ("NDM-1",)


def test_ancestor_depths_uses_minimum_is_a_distance(tmp_path: Path) -> None:
    terms = parse_aro_obo(_write_obo(tmp_path))
    depths = _ancestor_depths(terms, "ARO:3000637")
    assert depths["ARO:3009123"] == 1
    assert depths["ARO:3000008"] == 2


def test_assay_crosswalk_uses_aro_ancestry(tmp_path: Path) -> None:
    terms = parse_aro_obo(_write_obo(tmp_path))
    table = tmp_path / "condition_score_statistics.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["antibiotic"])
        writer.writeheader()
        for value in ("AMP", "CTX", "MEM"):
            writer.writerow({"antibiotic": value})
    rows = _assay_crosswalk_rows(table, terms)
    mapping = {
        row["assay_antibiotic_code"]: row["card_drug_class_name"] for row in rows
    }
    assert mapping == {
        "AMP": "penicillin beta-lactam",
        "CTX": "cephalosporin",
        "MEM": "carbapenem",
    }


def test_assay_crosswalk_rejects_unknown_code(tmp_path: Path) -> None:
    terms = parse_aro_obo(_write_obo(tmp_path))
    table = tmp_path / "condition_score_statistics.csv"
    table.write_text("antibiotic\nXYZ\n", encoding="utf-8")
    with pytest.raises(CARDContextError, match="without a CARD crosswalk"):
        _assay_crosswalk_rows(table, terms)


def test_archive_validation_rejects_parent_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        payload = b"unsafe"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with (
        tarfile.open(archive_path, "r:bz2") as archive,
        pytest.raises(CARDContextError, match="Unsafe path"),
    ):
        _validate_archive_members(archive)
