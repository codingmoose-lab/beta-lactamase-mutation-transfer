"""Check defined and undefined position-level structural correlations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSOCIATIONS = (
    "data/interim/biological_model_difference_analysis/"
    "biological_model_difference_structural_associations.csv"
)
POSITIONS = (
    "data/interim/biological_model_difference_analysis/"
    "biological_model_difference_position_summary.csv"
)


def assess(associations: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    output = []
    for row in associations.itertuples(index=False):
        values = np.asarray([row.value, row.ci_lower, row.ci_upper], dtype=float)
        n_valid = int(row.bootstrap_valid_replicates)
        part = positions.loc[positions.scheme_id.eq(row.scheme_id)]
        effect_col = f"mean_{row.contrast}"
        descriptor_col = row.variable
        have_columns = {effect_col, descriptor_col}.issubset(part.columns)
        complete = (
            part[[effect_col, descriptor_col]].dropna()
            if have_columns
            else pd.DataFrame()
        )
        n_unique_effects = int(complete[effect_col].nunique()) if have_columns else -1
        n_unique_descriptors = (
            int(complete[descriptor_col].nunique()) if have_columns else -1
        )
        if np.isfinite(values).all():
            status = "PASS" if n_valid > 0 and values[1] <= values[2] else "REVIEW"
            explanation = "finite association and confidence limits"
        elif np.isnan(values).all() and n_valid == 0:
            if len(complete) > 1 and n_unique_effects == 1:
                status = "UNDEFINED_CONSTANT_EFFECT"
                explanation = "the position-level model difference is constant"
            elif len(complete) > 1 and n_unique_descriptors == 1:
                status = "UNDEFINED_CONSTANT_DESCRIPTOR"
                explanation = "the structural descriptor is constant"
            else:
                status = "REVIEW"
                explanation = "undefined association requires source inspection"
        else:
            status = "REVIEW"
            explanation = "inconsistent estimate and bootstrap validity"
        output.append(
            {
                "scheme_id": row.scheme_id,
                "contrast": row.contrast,
                "variable": row.variable,
                "status": status,
                "explanation": explanation,
                "n_positions_table": int(row.n_positions),
                "n_complete_position_rows": int(len(complete)),
                "n_unique_effects": n_unique_effects,
                "n_unique_descriptors": n_unique_descriptors,
                "bootstrap_valid_replicates": n_valid,
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(root):
        raise ValueError("Review results must be written outside the repository")
    a, p = root / ASSOCIATIONS, root / POSITIONS
    if not a.is_file() or not p.is_file():
        raise FileNotFoundError(f"Missing structural analysis inputs: {a}; {p}")
    report = assess(pd.read_csv(a), pd.read_csv(p))
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "structural_association_review.csv", index=False)
    counts = report.status.value_counts().to_dict()
    summary = {"association_count": int(len(report)), "statuses": counts}
    (output / "structural_association_review.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if "REVIEW" in counts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
