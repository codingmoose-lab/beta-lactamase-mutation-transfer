import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_real_experimental_parsing_outputs_and_validation(tmp_path):
    summary = json.loads(
        (
            REPO_ROOT / "data/interim/mutation_measurements/measurement_summary.json"
        ).read_text(encoding="utf-8")
    )
    output = REPO_ROOT / "data/interim/mutation_measurements"
    assert summary["status"] == "PASS"
    assert summary["primary_observation_count"] > 100000
    assert summary["cross_homologue_comparison_count"] == 132
    assert summary["ec50_observation_count"] > 5000
    raw = _rows(output / "mutation_measurements_raw.csv")
    assert {row["protein_id"] for row in raw} == {"NDM-1", "VIM-2"}
    assert {row["antibiotic"] for row in raw} == {"AMP", "CTX", "MEM"}
    temperatures_by_study = {}
    for row in raw:
        temperatures_by_study.setdefault(row["study_id"], set()).add(
            row["temperature_c"]
        )
    assert temperatures_by_study["chen_2020_elife"] == {"25.0", "37.0"}
    assert temperatures_by_study["chen_2022_natee"] == {"37.0"}
    assert temperatures_by_study["chen_2024_natcomms"] == {"37.0"}
    validated = _rows(output / "mutation_measurements.csv")
    assert all(row["quality_status"] == "valid" for row in validated)
    structure_mapping = _rows(output / "mutation_structure_mapping.csv")
    preferred = {
        (row["protein_id"], row["pdb_chain"])
        for row in structure_mapping
        if row["provisional_preferred_chain"] == "True"
    }
    assert preferred == {("NDM-1", "B"), ("VIM-2", "B")}
    with (output / "validation_summary.json").open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    assert validation["sequence_validation_status"].get("wild_type_mismatch", 0) == 0
