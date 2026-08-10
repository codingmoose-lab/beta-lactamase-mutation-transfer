import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_define_assays_and_conditions() -> None:
    output = REPO_ROOT / "data/interim/assay_definitions"
    summary = json.loads((output / "assay_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["validated_measurement_count"] == 138518
    assert summary["cross_protein_link_count"] == 132

    links = pd.read_csv(output / "derived_value_relationships.csv")
    assert len(links) == 132
    assert links["vim_observation_id"].notna().all()
    assert links["ndm_observation_id"].notna().all()

    conditions = pd.read_csv(output / "cross_protein_conditions.csv")
    direct = conditions[conditions["direct_comparison_use"]]
    assert direct["antibiotic"].tolist() == ["AMP"]
    other = conditions[conditions["antibiotic"].isin(["CTX", "MEM"])]
    assert not other["direct_comparison_use"].any()
    assert set(other["comparability_status"]) == {"condition_specific_only"}
