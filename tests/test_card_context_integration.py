from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from betalactamase_transfer import card_context

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_card_source_and_prepared_outputs_if_available() -> None:
    raw_dir = REPO_ROOT / "data/raw/card/ontology" / card_context.CARD_ONTOLOGY_VERSION
    context_dir = REPO_ROOT / "data/interim/card_context"
    archive = raw_dir / "card-ontology.tar.bz2"
    summary_path = context_dir / "context_summary.json"
    if not archive.exists() or not summary_path.exists():
        pytest.skip("CARD outputs have not been generated in this checkout")

    assert (
        card_context._sha256_path(archive) == card_context.CARD_ONTOLOGY_ARCHIVE_SHA256
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["target_count"] == 2
    assert summary["shared_expected_drug_class_count_per_target"] == 3
    assert summary["assay_drug_class_crosswalk_status"] == "PASS"
    assert summary["direct_vs_inherited_context_preserved"] is True
    assert summary["card_sequence_used_as_numbering_authority"] is False
    assert summary["card_fields_used_as_mutation_effect_labels"] is False

    with (context_dir / "assay_drug_class_crosswalk.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        row["assay_antibiotic_code"]: row["card_drug_class_name"] for row in rows
    }
    assert mapping == {
        "AMP": "penicillin beta-lactam",
        "CTX": "cephalosporin",
        "MEM": "carbapenem",
    }


def test_card_cli_scripts_are_self_contained() -> None:
    for script_name in (
        "11_acquire_card_ontology.py",
        "12_prepare_card_context.py",
    ):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/data" / script_name), "--help"],
            cwd=REPO_ROOT,
            env={"PATH": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
