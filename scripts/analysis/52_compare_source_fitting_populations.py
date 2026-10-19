"""Compare source-matched-only and full-source estimates on identical test rows."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {
    "source_assay_mean": "source_global_mean",
    "source_assay_mutant_mean": "source_mutant_mean_shrunk",
    "source_assay_substitution_mean": "source_substitution_mean_shrunk",
}
KEYS = ["direction", "target_observation_id", "homologue_position_id"]


def compare(new: pd.DataFrame, old: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    need = set(
        KEYS
        + ["model_id", "y_pred", "target_effect", "heldout_fold", "minimum_score_class"]
    )
    for title, frame in [("full-source", new), ("matched-only", old)]:
        missing = need - set(frame.columns)
        if missing:
            raise ValueError(f"{title} missing columns: {sorted(missing)}")
    output = []
    for new_model, old_model in PAIRS.items():
        first = new.loc[new.model_id.eq(new_model)]
        second = old.loc[old.model_id.eq(old_model)]
        joined = first.merge(
            second,
            on=KEYS,
            validate="one_to_one",
            suffixes=("_full", "_matched"),
            indicator=True,
        )
        if (
            len(joined) != 3026
            or not joined._merge.eq("both").all()
            or not joined.heldout_fold_full.eq(joined.heldout_fold_matched).all()
            or not np.allclose(
                joined.target_effect_full,
                joined.target_effect_matched,
                atol=1e-12,
                rtol=0,
            )
        ):
            raise ValueError("Full-source and matched-only test populations differ")
        if not joined.minimum_score_class_full.eq(
            joined.minimum_score_class_matched
        ).all():
            raise ValueError("Lower-score categories differ")
        for direction, pop, subset in (
            (
                direction,
                pop,
                joined.loc[joined.direction.eq(direction)]
                if pop == "all_shared_wild_type_pairs"
                else joined.loc[
                    joined.direction.eq(direction)
                    & joined.minimum_score_class_full.eq("both_above_minimum")
                ],
            )
            for direction in ("NDM-1_to_VIM-2", "VIM-2_to_NDM-1")
            for pop in ("all_shared_wild_type_pairs", "both_above_minimum")
        ):
            source_error = np.abs(
                subset.target_effect_full.to_numpy(float)
                - subset.y_pred_full.to_numpy(float)
            )
            matched_error = np.abs(
                subset.target_effect_full.to_numpy(float)
                - subset.y_pred_matched.to_numpy(float)
            )
            positions = subset.homologue_position_id.astype(str).to_numpy()
            groups = [
                np.flatnonzero(positions == pos) for pos in sorted(set(positions))
            ]
            seed = int(
                hashlib.sha256(f"{direction}|{pop}|{new_model}".encode()).hexdigest()[
                    :8
                ],
                16,
            )
            rng = np.random.default_rng(seed)
            values = []
            for _ in range(repetitions):
                idx = np.concatenate(
                    [groups[k] for k in rng.integers(0, len(groups), size=len(groups))]
                )
                values.append(float(np.mean(matched_error[idx] - source_error[idx])))
            lower, upper = np.quantile(values, [0.025, 0.975])
            output.append(
                {
                    "direction": direction,
                    "population": pop,
                    "model_with_full_source_assay": new_model,
                    "matched_only_model": old_model,
                    "n_pairs": len(subset),
                    "n_positions": len(groups),
                    "full_source_mae": float(source_error.mean()),
                    "matched_only_mae": float(matched_error.mean()),
                    "mae_improvement_full_source": float(
                        (matched_error - source_error).mean()
                    ),
                    "mae_improvement_ci_lower": float(lower),
                    "mae_improvement_ci_upper": float(upper),
                    "bootstrap_replicates": repetitions,
                }
            )
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-predictions", required=True, type=Path)
    parser.add_argument("--previous-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    frame = compare(
        pd.read_csv(args.new_predictions),
        pd.read_csv(args.previous_predictions),
        args.repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, lineterminator="\n")
    print(f"SAME-POSITION SOURCE-POPULATION COMPARISON: {len(frame)} rows")


if __name__ == "__main__":
    main()
