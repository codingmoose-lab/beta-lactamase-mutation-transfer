import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_real_workbook_audit(tmp_path: Path) -> None:
    summary = json.loads(
        (REPO_ROOT / "data/interim/source_tables/source_table_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["audit_status"] == "PASS"
    assert summary["workbook_count"] == 5
    assert summary["sheet_count"] == 21

    output_dir = REPO_ROOT / "data/interim/source_tables"
    sheets = _read_csv(output_dir / "worksheet_metadata.csv")
    sheet_keys = {(row["workbook_id"], row["sheet_name"]) for row in sheets}
    assert ("chen_2024_sd3", "SD3_NDM_fitness_scores") in sheet_keys
    assert ("chen_2020_sf2", "SF2A Fitness Scores") in sheet_keys
    assert ("chen_2022_sd", "SD1 Fitness scores") in sheet_keys

    columns = _read_csv(output_dir / "column_definitions.csv")
    headers = {
        (row["workbook_id"], row["sheet_name"], row["header"]) for row in columns
    }
    assert (
        "chen_2024_sd3",
        "SD3_NDM_fitness_scores",
        "32.0_AMP_score",
    ) in headers
    assert (
        "chen_2020_sf2",
        "SF2A Fitness Scores",
        "128ug/mL_AMP_25C",
    ) in headers
    assert (
        "chen_2022_sd",
        "SD1 Fitness scores",
        "256.0ug/mL_fitness",
    ) in headers

    with (output_dir / "source_table_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        persisted = json.load(handle)
    assert persisted["sheet_count"] == 21
