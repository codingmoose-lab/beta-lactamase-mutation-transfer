"""Check the Figure 4D numerical contract without fitting prediction models."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/figures/generate_figure4_cross_homologue_prediction.py"
)
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
CONSTANTS = {
    "BLUE",
    "RUST",
    "DIRECTION_STYLE",
    "PANEL_D_DIRECTIONS",
    "PANEL_D_POPULATIONS",
    "FOCAL_MODEL",
    "REFERENCE",
}
FUNCTIONS = {"prepare_rows", "draw_in_axes"}
selected = []
for node in TREE.body:
    if (
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in node.targets
        )
        or isinstance(node, ast.FunctionDef)
        and node.name in FUNCTIONS
    ):
        selected.append(node)
namespace = {"np": np, "pd": pd, "plt": plt}
module = ast.Module(body=selected, type_ignores=[])
exec(compile(module, str(SOURCE), "exec"), namespace)
MOD = SimpleNamespace(
    DIRECTIONS=namespace["PANEL_D_DIRECTIONS"],
    POPULATIONS=namespace["PANEL_D_POPULATIONS"],
    MODEL=namespace["FOCAL_MODEL"],
    REFERENCE=namespace["REFERENCE"],
    prepare_rows=namespace["prepare_rows"],
    draw_in_axes=namespace["draw_in_axes"],
)


def make_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(81)
    rows = []
    selected_above = []
    for position in range(1, 82):
        for mutant in range(19 if position <= 55 else 18):
            rows.append((position, f"{position}_{mutant}"))
    # 55 * 19 + 26 * 18 = 1,513; 478 subset substitutions at 63 positions
    assert len(rows) == 1513
    for position in range(1, 64):
        candidates = [label for pos, label in rows if pos == position]
        selected_above.extend(candidates[: 8 if position <= 37 else 7])
    selected_above = selected_above[:478]
    assert len(selected_above) == 478
    assert len({int(s.split("_")[0]) for s in selected_above}) == 63
    predictions = []
    comparisons = []
    for direction in MOD.DIRECTIONS:
        for position, mutation in rows:
            source = rng.normal(-1.0, 1.0)
            target = rng.normal(-1.1, 1.1)
            y_pred = rng.normal(-1.2, 1.1)
            predictions.append(
                {
                    "direction": direction,
                    "model_id": MOD.MODEL,
                    "homologue_position_id": f"pair_{position:03d}",
                    "target_observation_id": f"{direction}_{mutation}",
                    "target_effect": target,
                    "source_effect": source,
                    "y_pred": y_pred,
                    "minimum_score_class": (
                        "both_above_minimum" if mutation in selected_above else "other"
                    ),
                }
            )
    frame = pd.DataFrame(predictions)
    for population in MOD.POPULATIONS:
        for direction in MOD.DIRECTIONS:
            d = frame.loc[frame.direction.eq(direction)]
            if population == "both_above_minimum":
                d = d[d.minimum_score_class.eq("both_above_minimum")]
            diff = (
                abs(d.target_effect - d.source_effect) - abs(d.target_effect - d.y_pred)
            ).mean()
            comparisons.append(
                {
                    "direction": direction,
                    "population": population,
                    "model_id": MOD.MODEL,
                    "reference_model_id": MOD.REFERENCE,
                    "metric": "mae",
                    "paired_difference": diff,
                    "ci_lower": diff - 0.15,
                    "ci_upper": diff + 0.15,
                }
            )
    return frame, pd.DataFrame(comparisons)


def test_four_numeric_populations_and_local_images(tmp_path):
    pred, comparison = make_tables()
    rows, metrics = MOD.prepare_rows(pred, comparison)
    assert len(rows) == 3026
    assert len(metrics) == 4
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7), sharex=True, sharey=True)
    MOD.draw_in_axes(axes, rows)
    output = tmp_path / "figure4d_test.pdf"
    fig.savefig(output)
    plt.close(fig)
    assert output.stat().st_size > 1000


def test_rejects_changed_saved_mae():
    pred, comparison = make_tables()
    comparison.loc[0, "paired_difference"] += 0.2
    with pytest.raises(ValueError, match="does not match saved"):
        MOD.prepare_rows(pred, comparison)


def test_rejects_missing_pairs():
    pred, comparison = make_tables()
    with pytest.raises(ValueError, match="expected 1513"):
        MOD.prepare_rows(pred.iloc[1:], comparison)
