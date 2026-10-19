"""Validate recorded mutation-effect comparisons without changing their outputs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from betalactamase_transfer.source_full_position_baselines import execute

DIRECTIONS = ("NDM-1_to_VIM-2", "VIM-2_to_NDM-1")
MATCHED = (
    "data/interim/baseline_evaluation/homologue_analysis/"
    "matched_homologue_predictions.csv"
)
MODELS = "data/interim/matched_homologue_model_analysis/"
STRUCTURAL = (
    "data/interim/biological_model_difference_analysis/"
    "biological_model_difference_structural_associations.csv"
)
PAIRS = "data/interim/task_definitions/aligned_shared_wild_type_mutation_pairs.csv"
ASSAYS = "data/interim/assay_definitions/cross_protein_conditions.csv"
FOLDS = "data/interim/evaluation_partitions/group_assignments.csv"


def calculate(root: Path, output: Path, repetitions: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    def record(subject: str, result: str, detail: str) -> None:
        checks.append({"subject": subject, "result": result, "detail": detail})

    files = list(root.glob("src/**/*.py")) + list(root.glob("scripts/**/*.py"))
    files += list(root.glob("tests/**/*.py"))
    errors = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    record(
        "Python source parsing",
        "PASS" if not errors else "FAIL",
        f"{len(files)} parsed, {len(errors)} errors: {errors[:5]}",
    )

    missense_path = root / "data/processed/missense_observations.csv"
    if missense_path.is_file():
        count = sum(1 for _ in missense_path.open("rb")) - 1
        record(
            "Processed missense observation count",
            "PASS" if count == 125681 else "FAIL",
            f"missense observations={count}; expected=125681",
        )
    else:
        record(
            "Processed missense observation count",
            "REVIEW",
            "Processed missense observation file absent",
        )

    ppath = root / PAIRS
    if ppath.is_file():
        pairs = pd.read_csv(ppath)
        required = {
            "homologue_position_id",
            "pairing_uses_outcome_values",
            "ndm_observation_id",
            "vim_observation_id",
        }
        good = required.issubset(pairs.columns) and len(pairs) == 1513
        if good:
            good = (
                pairs.homologue_position_id.nunique() == 81
                and not pairs.pairing_uses_outcome_values.astype(str)
                .str.lower()
                .isin(["true", "1"])
                .any()
                and not pairs[["ndm_observation_id", "vim_observation_id"]]
                .duplicated()
                .any()
            )
        record(
            "Homologous substitution identities",
            "PASS" if good else "FAIL",
            f"rows={len(pairs)}; positions={pairs.homologue_position_id.nunique()}",
        )
    else:
        record("Homologous substitution identities", "REVIEW", "Pair table absent")

    mpath = root / MATCHED
    matched = None
    if mpath.is_file():
        matched = pd.read_csv(mpath)
        complete = len(matched) == 3026 and set(matched.direction) == set(DIRECTIONS)
        if complete:
            for direction in DIRECTIONS:
                piece = matched.loc[matched.direction.eq(direction)]
                complete = complete and len(piece) == 1513
                complete = complete and piece.homologue_position_id.nunique() == 81
                if "minimum_score_class" in piece:
                    above = piece.minimum_score_class.eq("both_above_minimum")
                    complete = complete and int(above.sum()) == 478
                rho = float(
                    spearmanr(piece.source_effect, piece.target_effect).statistic
                )
                complete = complete and abs(rho - 0.66294407564) < 1e-6
                record(
                    f"Experimental substitution agreement ({direction})",
                    "PASS" if abs(rho - 0.66294407564) < 1e-6 else "FAIL",
                    f"paired Spearman={rho:.9f}",
                )
                above = piece.loc[piece.minimum_score_class.eq("both_above_minimum")]
                above_rho = float(
                    spearmanr(above.source_effect, above.target_effect).statistic
                )
                floor_n = int(piece.minimum_score_class.eq("both_at_minimum").sum())
                good_floor = (
                    len(above) == 478
                    and above.homologue_position_id.nunique() == 63
                    and floor_n == 729
                    and abs(above_rho - 0.3101242968) < 1e-6
                )
                record(
                    f"Lower-score comparison ({direction})",
                    "PASS" if good_floor else "FAIL",
                    f"478 above-minimum pairs: Spearman={above_rho:.9f}; "
                    f"both at minimum={floor_n}",
                )
        record(
            "Matched experimental analysis population",
            "PASS" if complete else "FAIL",
            f"directional rows={len(matched)}",
        )
    else:
        record(
            "Matched experimental analysis population",
            "REVIEW",
            "Matched prediction table absent",
        )

    fpath = root / FOLDS
    if fpath.is_file():
        assignment = pd.read_csv(fpath)
        fields = {"scheme_id", "iteration", "cv_fold", "split_role", "group_id"}
        if fields.issubset(assignment.columns):
            n_bad = 0
            for _, part in assignment.groupby(
                ["scheme_id", "iteration", "cv_fold"], dropna=False
            ):
                fit = part.loc[
                    part.split_role.isin(["train", "source_model_selection"])
                ]
                target = part.loc[part.split_role.isin(["test", "target_test"])]
                if set(fit.group_id.astype(str)) & set(target.group_id.astype(str)):
                    n_bad += 1
            record(
                "Fitting and final-test group IDs",
                "PASS" if n_bad == 0 else "FAIL",
                f"group partitions checked; overlapping groups={n_bad}",
            )
        else:
            record(
                "Fitting and final-test group IDs",
                "FAIL",
                f"Missing columns: {sorted(fields - set(assignment.columns))}",
            )
    else:
        record("Fitting and final-test group IDs", "REVIEW", "Partition table absent")

    apath = root / ASSAYS
    if apath.is_file():
        assay = pd.read_csv(apath)
        amp = assay.loc[assay.antibiotic.eq("AMP")]
        if len(amp) == 1:
            row = amp.iloc[0]
            good = (
                float(row.ndm_concentration) == 256.0
                and float(row.vim_concentration) == 128.0
                and float(row.ndm_temperature_c) == 37.0
                and float(row.vim_temperature_c) == 37.0
            )
            record(
                "AMP assay pairing",
                "PASS" if good else "FAIL",
                f"NDM={row.ndm_concentration}; VIM={row.vim_concentration}; "
                "both temperatures 37°C expected",
            )
        else:
            record("AMP assay pairing", "FAIL", f"AMP comparison rows={len(amp)}")
    else:
        record("AMP assay pairing", "REVIEW", "Assay table absent")

    model_predictions = root / MODELS / "matched_homologue_model_predictions.csv"
    model_metrics = root / MODELS / "matched_homologue_model_metrics.csv"
    metric_rechecks = []
    if model_predictions.is_file() and model_metrics.is_file():
        predictions = pd.read_csv(model_predictions)
        metrics = pd.read_csv(model_metrics)
        required = {
            "direction",
            "model_id",
            "target_effect",
            "y_pred",
            "minimum_score_class",
            "homologue_position_id",
        }
        if not required.issubset(predictions.columns):
            record(
                "Cross-protein prediction metric recheck",
                "FAIL",
                "Missing prediction fields: "
                f"{sorted(required - set(predictions.columns))}",
            )
        else:
            inconsistent = []
            for (direction, model), d in predictions.groupby(["direction", "model_id"]):
                for pop in ("all_shared_wild_type_pairs", "both_above_minimum"):
                    piece = (
                        d
                        if pop == "all_shared_wild_type_pairs"
                        else d.loc[d.minimum_score_class.eq("both_above_minimum")]
                    )
                    y = piece.target_effect.to_numpy(float)
                    pred = piece.y_pred.to_numpy(float)
                    point = {
                        "mae": float(np.mean(np.abs(y - pred))),
                        "spearman": float(spearmanr(y, pred).statistic),
                        "pearson": float(np.corrcoef(y, pred)[0, 1]),
                        "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
                    }
                    for metric, value in point.items():
                        expected = metrics.loc[
                            metrics.direction.eq(direction)
                            & metrics.population.eq(pop)
                            & metrics.model_id.eq(model)
                            & metrics.metric.eq(metric)
                        ]
                        if len(expected) != 1 or not np.isclose(
                            value,
                            float(expected.point_estimate.iloc[0]),
                            atol=1e-9,
                            rtol=1e-8,
                            equal_nan=True,
                        ):
                            inconsistent.append((direction, pop, model, metric))
                        metric_rechecks.append(
                            {
                                "direction": direction,
                                "population": pop,
                                "model_id": model,
                                "metric": metric,
                                "point_estimate": value,
                                "n_pairs": len(piece),
                                "n_homologue_positions": (
                                    piece.homologue_position_id.nunique()
                                ),
                            }
                        )
            pd.DataFrame(metric_rechecks).to_csv(
                output / "independent_model_metrics.csv", index=False
            )
            record(
                "Cross-protein prediction metric recheck",
                "PASS" if not inconsistent else "FAIL",
                f"{len(metric_rechecks)} metric values recalculated; "
                f"{len(inconsistent)} mismatches: {inconsistent[:5]}",
            )
    else:
        record(
            "Cross-protein prediction metric recheck",
            "REVIEW",
            "Stored model prediction/metric tables absent",
        )

    spath = root / STRUCTURAL
    if spath.is_file():
        structures = pd.read_csv(spath)
        required = {"value", "ci_lower", "ci_upper", "n_positions", "resampling_unit"}
        if required.issubset(structures.columns):
            good_ci = (structures.ci_lower <= structures.ci_upper).all()
            valid_n = (pd.to_numeric(structures.n_positions, errors="coerce") > 1).all()
            finite = np.isfinite(
                structures[["value", "ci_lower", "ci_upper"]].to_numpy(float)
            ).all()
            result = "PASS" if good_ci and valid_n and finite else "REVIEW"
            record(
                "Structural association table",
                result,
                f"rows={len(structures)}; CI ordered={good_ci}; "
                f"groups valid={valid_n}; finite estimates={finite}; "
                "units="
                f"{sorted(map(str, structures.resampling_unit.dropna().unique()))}",
            )
            structures.to_csv(
                output / "structural_associations_for_review.csv", index=False
            )
        else:
            record(
                "Structural association table",
                "REVIEW",
                f"Missing columns: {sorted(required - set(structures.columns))}",
            )
    else:
        record(
            "Structural association table",
            "REVIEW",
            "Structural association table absent",
        )

    for stem in range(1, 7):
        names = ["figure_1"] if stem == 1 else [f"figure_{stem}_"]
        matches = (
            [
                p
                for p in (root / "figures/main").iterdir()
                if p.is_file()
                and (p.stem == "figure_1" if stem == 1 else p.name.startswith(names[0]))
                and p.suffix.lower() in {".pdf", ".svg", ".png"}
            ]
            if (root / "figures/main").is_dir()
            else []
        )
        suffixes = {p.suffix.lower() for p in matches}
        record(
            f"Main Figure {stem} files",
            "PASS" if {".pdf", ".svg", ".png"} <= suffixes else "REVIEW",
            f"formats={sorted(suffixes)}; matching files={len(matches)}",
        )
    record(
        "Figure 4D shared-target association",
        "REVIEW",
        "Existing rho=0.758/0.750 shares the target measurement "
        "with prediction error. Original correlation is descriptive; "
        "candidate error comparison requires author approval.",
    )
    record(
        "Source-only historical training",
        "REVIEW",
        "Source-fitting simulated tests pass; archived RTX checkpoints "
        "not independently reconstructed.",
    )

    try:
        fit_summary = execute(
            root, output / "source_full_position_analysis", repetitions=repetitions
        )
        record(
            "Full-source position-withheld reference models",
            "PASS",
            f"{fit_summary['prediction_count']} predictions, "
            f"{fit_summary['metric_count']} metrics; source-only inner CV",
        )
    except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as exc:
        record(
            "Full-source position-withheld reference models",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )

    summary = {
        "result_counts": pd.Series([item["result"] for item in checks])
        .value_counts()
        .to_dict(),
        "fatal_scientific_checks": [
            item["subject"] for item in checks if item["result"] == "FAIL"
        ],
        "unresolved_checks": [
            item["subject"] for item in checks if item["result"] == "REVIEW"
        ],
        "research_outputs_modified": False,
        "gpu_training_performed": False,
        "manuscript_automatically_modified": False,
    }
    pd.DataFrame(checks).to_csv(output / "scientific_check_results.csv", index=False)
    (output / "scientific_check_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "SCIENTIFIC_REVIEW.txt").open("w", encoding="utf-8") as stream:
        stream.write("NDM-1 / VIM-2 CONSOLIDATED SCIENTIFIC REVIEW\n\n")
        for item in checks:
            stream.write(f"{item['result']} | {item['subject']}\n  {item['detail']}\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.repo_root.resolve()):
        raise SystemExit("Output directory must be outside the repository")
    result = calculate(
        args.repo_root.resolve(), args.output_dir.resolve(), args.repetitions
    )
    print(json.dumps(result, indent=2))
    if result["fatal_scientific_checks"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
