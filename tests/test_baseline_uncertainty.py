from __future__ import annotations

import pandas as pd

from betalactamase_transfer.baseline_uncertainty import _grouped_bootstrap


def test_grouped_bootstrap_uses_declared_grouping_unit(monkeypatch) -> None:
    monkeypatch.setattr(
        "betalactamase_transfer.baseline_uncertainty.BOOTSTRAP_REPLICATES", 20
    )
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "truth": [-1.0, 0.0, 1.0, 2.0],
            "prediction": [-0.8, 0.1, 0.9, 2.2],
        }
    )
    rows = _grouped_bootstrap(
        frame,
        group_column="group",
        true_column="truth",
        prediction_column="prediction",
        seed_parts=("test",),
    )
    assert len(rows) == 4
    assert {row["resampling_unit"] for row in rows} == {"group"}
    assert {row["bootstrap_replicates"] for row in rows} == {20}
