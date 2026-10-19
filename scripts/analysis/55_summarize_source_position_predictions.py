import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from betalactamase_transfer.source_position_prediction_tables import sha256, summarize


def compare_existing(
    existing_dir: Path, filename: str, calculated: pd.DataFrame
) -> None:
    old = existing_dir / filename
    if not old.is_file():
        print(f"EXISTING: {filename}: not found; no comparison")
        return
    prior = pd.read_csv(old)
    keys = (
        ["direction", "method"]
        if filename.startswith("minimum_score")
        else ["direction", "population", "comparator"]
    )
    joined = prior.merge(
        calculated,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_existing", "_calculated"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"Different row keys in existing {filename}")
    print(f"EXISTING: {filename}: {len(joined)} matched rows")
    for col in prior.columns:
        if col in keys:
            continue
        if not pd.api.types.is_numeric_dtype(prior[col]):
            continue
        old_values = joined[f"{col}_existing"].to_numpy(dtype=float)
        new_values = joined[f"{col}_calculated"].to_numpy(dtype=float)
        difference = float(np.max(abs(old_values - new_values)))
        kind = (
            "bootstrap interval"
            if col.startswith("position_ci_")
            else "direct statistic"
        )
        print(f"  {col} ({kind}): largest absolute difference {difference:.10g}")
        if kind == "direct statistic" and difference > 1e-9:
            raise ValueError(f"Unexpected numerical discrepancy in {filename}.{col}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--esmc-predictions", type=Path, required=True)
    ap.add_argument("--source-full-predictions", type=Path, required=True)
    ap.add_argument("--existing-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--repetitions", type=int, default=10000)
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write new files to a fresh output directory",
    )
    args = ap.parse_args()
    repo = args.repo_root.resolve(strict=True)
    esm_file = args.esmc_predictions.resolve(strict=True)
    full_file = args.source_full_predictions.resolve(strict=True)
    if args.repetitions < 100:
        ap.error("At least 100 position bootstrap resamples required")
    if args.write and args.output_dir is None:
        ap.error("--write requires --output-dir")
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        allowed_repo_output = (
            repo / "data" / "interim" / "source_position_prediction_tables"
        )
        if (output == repo or repo in output.parents) and output != allowed_repo_output:
            ap.error(
                "Inside the repository, output is limited to "
                "data/interim/source_position_prediction_tables"
            )
        if output.exists():
            ap.error("Output directory must not already exist")
    esmc = pd.read_csv(esm_file)
    full = pd.read_csv(full_file)
    auc, groups = summarize(esmc, full, repetitions=args.repetitions)
    print(
        f"VALIDATION PASS: {len(auc)} AUC rows, {len(groups)} grouped comparison rows"
    )
    print(f"POSITION BOOTSTRAP REPETITIONS: {args.repetitions}")
    print("SOURCE INPUT SHA256:")
    for path in (esm_file, full_file):
        print(f"  {sha256(path)}  {path}")
    if args.existing_dir is not None:
        compare_existing(args.existing_dir, "minimum_score_roc_auc.csv", auc)
        compare_existing(
            args.existing_dir, "target_score_stratified_comparisons.csv", groups
        )
    if not args.write:
        print("READ-ONLY REVIEW PASS: no files written")
        return
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "minimum_score_roc_auc.csv": auc,
        "target_score_stratified_comparisons.csv": groups,
    }
    for filename, frame in files.items():
        frame.to_csv(output / filename, index=False)
    try:
        commit = subprocess.check_output(
            ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    summary = {
        "git_commit": commit,
        "source_input_sha256": {
            str(path): sha256(path) for path in (esm_file, full_file)
        },
        "position_bootstrap_repetitions": args.repetitions,
        "position_bootstrap_method": (
            "resample homologous positions with replacement; retain all "
            "observations from each sampled position; percentile 2.5% and 97.5%"
        ),
        "position_bootstrap_seed": (
            "first 8 hex digits of "
            "SHA-256(direction|population|comparator|source_position_tables)"
        ),
        "identity_ridge_reference": (
            "extended ESM-C output reference_prediction; "
            "not the separate source-full run"
        ),
        "output_sha256": {n: sha256(output / n) for n in files},
    }
    (output / "source_position_table_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"AUTHOR-RUN OUTPUTS WRITTEN: {output}")


if __name__ == "__main__":
    main()
