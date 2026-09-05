from __future__ import annotations

from pathlib import Path

import pandas as pd

from betalactamase_transfer import concentration_transfer_evaluation as stage40


def test_concentration_scheme_inventory() -> None:
    assert len(stage40.CONCENTRATION_SCHEME_IDS) == 7
    assert stage40.CONDITION_MODE == "log2_concentration"
    assert stage40.CONCENTRATION_SCHEME_IDS[0].endswith("_4")
    assert stage40.CONCENTRATION_SCHEME_IDS[-1].endswith("_256")


def test_design_summary_uses_numeric_dose_order(tmp_path: Path) -> None:
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    rows = []
    for concentration in (4, 8, 16, 32, 64, 128, 256):
        rows.append(
            {
                "assay_id": "vim2_2022_amp_concentration_dms",
                "concentration": float(concentration),
                "mutation_group_id": f"m{concentration}",
            }
        )
    pd.DataFrame(rows).to_csv(processed / "missense_observations.csv", index=False)
    schemes = pd.DataFrame(
        [
            {
                "scheme_id": scheme_id,
                "test_value": scheme_id.rsplit("_", 1)[-1],
                "test_category": (
                    "edge_extrapolation"
                    if scheme_id.endswith("_4") or scheme_id.endswith("_256")
                    else "interpolation"
                ),
            }
            for scheme_id in stage40.CONCENTRATION_SCHEME_IDS
        ]
    )
    primary = stage40._primary_reference_by_scheme(schemes)
    secondary = stage40._secondary_reference_by_scheme(schemes)
    summary = stage40._concentration_design_summary(
        tmp_path, schemes, primary, secondary
    )
    row4 = summary[summary["target_concentration"].eq(4.0)].iloc[0]
    row32 = summary[summary["target_concentration"].eq(32.0)].iloc[0]
    assert row4["primary_reference_model_id"] == "source_concentration_nearest"
    assert row32["primary_reference_model_id"] == "source_concentration_linear_log2"
    assert float(row32["lower_bracket"]) == 16.0
    assert float(row32["upper_bracket"]) == 64.0
