from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


class ConditionConcentrationBiologyError(RuntimeError):
    """Raised when Stage 41 inputs do not satisfy the analysis contract."""


CONDITION_DIR = Path("data/interim/condition_transfer_evaluation")
CONCENTRATION_DIR = Path("data/interim/concentration_response_evaluation")
OUTPUT_DIR = Path("data/interim/condition_concentration_biological_analysis")

CONDITION_FITTED_MODELS = (
    "condition_transfer_esmc_site_delta_ridge",
    "condition_transfer_esm2_site_delta_ridge",
    "condition_transfer_structure_matched_mutation_condition_ridge",
    "condition_transfer_experimental_structure_descriptor_ridge",
)
CONCENTRATION_FITTED_MODELS = (
    "concentration_transfer_esmc_site_delta_ridge",
    "concentration_transfer_esm2_site_delta_ridge",
    "concentration_transfer_structure_matched_mutation_condition_ridge",
    "concentration_transfer_experimental_structure_descriptor_ridge",
)
DIRECT_CONCENTRATION_REFERENCES = (
    "source_concentration_linear_log2",
    "source_concentration_nearest",
)
FULL_POPULATION = "sequence_eligible"
MAE_METRIC = "mae"

EXPECTED_INPUT_HASHES = {
    CONDITION_DIR / "condition_transfer_predictions.csv": (
        "c5464ac73265d08052f04e3f180c6fbc6e20a4d8dd81839cb5d3b95b7ab0bd54"
    ),
    CONDITION_DIR / "condition_transfer_metrics.csv": (
        "3a4168dc4dcc7a56b0e59544fdbe429c7f98b1ae200dc5ed00a326d9e4c586ef"
    ),
    CONDITION_DIR / "condition_transfer_paired_comparisons.csv": (
        "10e8dacfc81f69ff0b565642ec5408e95c1a13b35a1f71d242afe8de56b2ddad"
    ),
    CONDITION_DIR / "condition_transfer_shift_summary.csv": (
        "f8653b048656154dea687bb6e5f185a48731b976f91afb430fa1b7c53eeb350a"
    ),
    CONDITION_DIR / "condition_transfer_summary.json": (
        "dfb10460b6d68d067eb77c77a16e2c8713c78dc5c7d1230a2b8026b147763e67"
    ),
    CONCENTRATION_DIR / "concentration_transfer_predictions.csv": (
        "d38a15859de8bc0f3dff7495487fdfcb123b79aff86cfc2ae1b2c850da5210d4"
    ),
    CONCENTRATION_DIR / "concentration_transfer_metrics.csv": (
        "f305b1e575ca18095c2b549b1531b7e07dee42b8f13b7c7a2cabfd6937b7fe7f"
    ),
    CONCENTRATION_DIR / "concentration_transfer_paired_comparisons.csv": (
        "c7027a8ddeaca5cee5ba303257439a756b6498a72a1026a73ef9b8e20bf81b95"
    ),
    CONCENTRATION_DIR / "concentration_transfer_design_summary.csv": (
        "2c2d8c67007d5ef19826ca00a4c49cc32e7b1d78dece2e39f41d74837c9fb4f3"
    ),
    CONCENTRATION_DIR / "concentration_transfer_summary.json": (
        "232cade2fb52f7275441cd1a5434fba16bbc0716d2f7811710ec2c46c97d5d68"
    ),
}

REQUIRED_PREDICTION_COLUMNS = {
    "scheme_id",
    "population_scope",
    "model_id",
    "observation_id",
    "assay_position",
    "wild_type",
    "mutant",
    "antibiotic",
    "concentration",
    "temperature_c",
    "evaluation_group_id",
    "y_true",
    "y_pred",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ConditionConcentrationBiologyError(
            f"{label} is missing required columns: {missing}"
        )


def review_condition_concentration_inputs(repo_root: Path) -> Mapping[str, object]:
    hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        path = repo_root / relative
        if not path.is_file():
            raise ConditionConcentrationBiologyError(
                f"Missing required input: {relative}"
            )
        observed = _sha256(path)
        if observed != expected:
            raise ConditionConcentrationBiologyError(
                f"Input hash mismatch for {relative}: {observed} != {expected}"
            )
        hashes[str(relative)] = observed

    condition_summary = json.loads(
        (repo_root / CONDITION_DIR / "condition_transfer_summary.json").read_text(
            encoding="utf-8"
        )
    )
    concentration_summary = json.loads(
        (
            repo_root / CONCENTRATION_DIR / "concentration_transfer_summary.json"
        ).read_text(encoding="utf-8")
    )
    if condition_summary.get("status") != "PASS":
        raise ConditionConcentrationBiologyError("Stage 39 summary is not PASS")
    if concentration_summary.get("status") != "PASS":
        raise ConditionConcentrationBiologyError("Stage 40 summary is not PASS")
    if condition_summary.get("test_labels_used_for_model_selection") is not False:
        raise ConditionConcentrationBiologyError(
            "Stage 39 reports held-out labels in model choice"
        )
    if concentration_summary.get("test_labels_used_for_model_selection") is not False:
        raise ConditionConcentrationBiologyError(
            "Stage 40 reports held-out labels in model choice"
        )
    if concentration_summary.get("log2_concentration_used") is not True:
        raise ConditionConcentrationBiologyError(
            "Stage 40 does not report log2 concentration"
        )

    return {
        "status": "PASS",
        "condition_scheme_count": len(condition_summary["scheme_ids"]),
        "concentration_scheme_count": len(concentration_summary["scheme_ids"]),
        "condition_model_selection_uses_test_labels": False,
        "concentration_model_selection_uses_test_labels": False,
        "log2_concentration_used": True,
        "input_sha256": hashes,
        "model_fitting_planned": False,
    }


def _metric_row(
    metrics: pd.DataFrame,
    scheme_id: str,
    population_scope: str,
    model_id: str,
) -> pd.Series:
    rows = metrics[
        metrics["scheme_id"].astype(str).eq(scheme_id)
        & metrics["population_scope"].astype(str).eq(population_scope)
        & metrics["model_id"].astype(str).eq(model_id)
    ]
    if len(rows) != 1:
        raise ConditionConcentrationBiologyError(
            "Expected one metric row for "
            f"{scheme_id}/{population_scope}/{model_id}; found {len(rows)}"
        )
    return rows.iloc[0]


def _mae_comparison_row(
    comparisons: pd.DataFrame,
    scheme_id: str,
    population_scope: str,
    model_id: str,
    reference_model_id: str,
) -> pd.Series:
    rows = comparisons[
        comparisons["scheme_id"].astype(str).eq(scheme_id)
        & comparisons["population_scope"].astype(str).eq(population_scope)
        & comparisons["model_id"].astype(str).eq(model_id)
        & comparisons["reference_model_id"].astype(str).eq(reference_model_id)
        & comparisons["metric"].astype(str).eq(MAE_METRIC)
    ]
    if len(rows) != 1:
        raise ConditionConcentrationBiologyError(
            "Expected one MAE comparison row for "
            f"{scheme_id}/{population_scope}/{model_id}/{reference_model_id}; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def _best_fitted_model(
    metrics: pd.DataFrame,
    scheme_id: str,
    population_scope: str,
    candidates: tuple[str, ...],
) -> pd.Series:
    rows = metrics[
        metrics["scheme_id"].astype(str).eq(scheme_id)
        & metrics["population_scope"].astype(str).eq(population_scope)
        & metrics["model_id"].astype(str).isin(candidates)
    ].copy()
    if rows.empty:
        raise ConditionConcentrationBiologyError(
            f"No fitted representation model for {scheme_id}/{population_scope}"
        )
    rows = rows.sort_values(["mae", "model_id"], kind="mergesort")
    return rows.iloc[0]


def _task_summary(
    condition_metrics: pd.DataFrame,
    condition_comparisons: pd.DataFrame,
    condition_summary: Mapping[str, object],
    concentration_metrics: pd.DataFrame,
    concentration_comparisons: pd.DataFrame,
    concentration_summary: Mapping[str, object],
    concentration_design: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    design_by_scheme = concentration_design.set_index("scheme_id")
    for analysis_type, metrics, comparisons, summary, candidates in [
        (
            "antibiotic_or_temperature",
            condition_metrics,
            condition_comparisons,
            condition_summary,
            CONDITION_FITTED_MODELS,
        ),
        (
            "concentration",
            concentration_metrics,
            concentration_comparisons,
            concentration_summary,
            CONCENTRATION_FITTED_MODELS,
        ),
    ]:
        primary = summary["primary_reference_by_scheme"]
        for scheme_id in summary["scheme_ids"]:
            scheme_metrics = metrics[metrics["scheme_id"].astype(str).eq(scheme_id)]
            for population_scope in sorted(
                scheme_metrics["population_scope"].astype(str).unique()
            ):
                reference_model = str(primary[scheme_id])
                reference = _metric_row(
                    metrics, scheme_id, population_scope, reference_model
                )
                best = _best_fitted_model(
                    metrics, scheme_id, population_scope, candidates
                )
                paired = _mae_comparison_row(
                    comparisons,
                    scheme_id,
                    population_scope,
                    str(best["model_id"]),
                    reference_model,
                )
                test_category = ""
                target_concentration = np.nan
                if analysis_type == "concentration":
                    design = design_by_scheme.loc[scheme_id]
                    test_category = str(design["test_category"])
                    target_concentration = float(design["target_concentration"])

                rows.append(
                    {
                        "analysis_type": analysis_type,
                        "scheme_id": scheme_id,
                        "population_scope": population_scope,
                        "test_category": test_category,
                        "target_concentration": target_concentration,
                        "primary_reference_model_id": reference_model,
                        "reference_mae": float(reference["mae"]),
                        "reference_rmse": float(reference["rmse"]),
                        "reference_spearman": float(reference["spearman"]),
                        "best_fitted_model_id": str(best["model_id"]),
                        "best_fitted_mae": float(best["mae"]),
                        "best_fitted_rmse": float(best["rmse"]),
                        "best_fitted_spearman": float(best["spearman"]),
                        "best_fitted_mae_improvement": float(
                            paired["paired_difference"]
                        ),
                        "mae_ci_lower": float(paired["ci_lower"]),
                        "mae_ci_upper": float(paired["ci_upper"]),
                        "mae_interval_excludes_zero": bool(
                            paired["interval_excludes_zero"]
                        ),
                        "n_observations": int(reference["n_observations"]),
                        "n_groups": int(reference["n_groups"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["analysis_type", "scheme_id", "population_scope"], kind="mergesort"
    )


def _concentration_reference_summary(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    summary: Mapping[str, object],
    design: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    design_by_scheme = design.set_index("scheme_id")
    primary = summary["primary_reference_by_scheme"]
    secondary = summary["secondary_reference_by_scheme"]

    for scheme_id in summary["scheme_ids"]:
        scheme_metrics = metrics[metrics["scheme_id"].astype(str).eq(scheme_id)]
        for population_scope in sorted(
            scheme_metrics["population_scope"].astype(str).unique()
        ):
            primary_id = str(primary[scheme_id])
            secondary_id = str(secondary[scheme_id])
            primary_row = _metric_row(metrics, scheme_id, population_scope, primary_id)
            secondary_row = _metric_row(
                metrics, scheme_id, population_scope, secondary_id
            )
            paired = _mae_comparison_row(
                comparisons,
                scheme_id,
                population_scope,
                secondary_id,
                primary_id,
            )
            design_row = design_by_scheme.loc[scheme_id]
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "population_scope": population_scope,
                    "target_concentration": float(design_row["target_concentration"]),
                    "test_category": str(design_row["test_category"]),
                    "primary_reference_model_id": primary_id,
                    "primary_mae": float(primary_row["mae"]),
                    "primary_spearman": float(primary_row["spearman"]),
                    "secondary_reference_model_id": secondary_id,
                    "secondary_mae": float(secondary_row["mae"]),
                    "secondary_spearman": float(secondary_row["spearman"]),
                    "secondary_mae_improvement": float(paired["paired_difference"]),
                    "mae_ci_lower": float(paired["ci_lower"]),
                    "mae_ci_upper": float(paired["ci_upper"]),
                    "mae_interval_excludes_zero": bool(
                        paired["interval_excludes_zero"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target_concentration", "population_scope"], kind="mergesort"
    )


def _unique_concentration_outcomes(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "Stage 40 predictions")
    rows = predictions[
        predictions["population_scope"].astype(str).eq(FULL_POPULATION)
        & predictions["model_id"].astype(str).eq("source_concentration_nearest")
    ].copy()
    if rows.empty:
        raise ConditionConcentrationBiologyError(
            "No sequence-eligible nearest-concentration rows found"
        )
    identity = ["observation_id"]
    if rows.duplicated(identity).any():
        raise ConditionConcentrationBiologyError(
            "Concentration target observations are duplicated across "
            "leave-one-dose schemes"
        )
    expected = {4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0}
    observed = set(rows["concentration"].astype(float).unique())
    if observed != expected:
        raise ConditionConcentrationBiologyError(
            f"Unexpected concentration series: {sorted(observed)}"
        )
    return rows


def _mutation_trajectory_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    outcomes = _unique_concentration_outcomes(predictions)
    rows: list[dict[str, object]] = []
    grouped = outcomes.groupby("evaluation_group_id", sort=True)

    for mutation_group_id, group in grouped:
        identity = group[["assay_position", "wild_type", "mutant"]].drop_duplicates()
        if len(identity) != 1:
            raise ConditionConcentrationBiologyError(
                f"Mutation group {mutation_group_id} has inconsistent mutation identity"
            )
        group = group.sort_values("concentration", kind="mergesort")
        x = np.log2(group["concentration"].to_numpy(dtype=float))
        y = group["y_true"].to_numpy(dtype=float)
        slope = np.nan if len(np.unique(x)) < 2 else float(np.polyfit(x, y, deg=1)[0])

        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            spearman = np.nan
        else:
            x_rank = pd.Series(x).rank(method="average")
            y_rank = pd.Series(y).rank(method="average")
            spearman = float(x_rank.corr(y_rank))

        dose_to_score = dict(
            zip(
                group["concentration"].astype(float),
                group["y_true"].astype(float),
                strict=True,
            )
        )
        endpoint_change = np.nan
        if 4.0 in dose_to_score and 256.0 in dose_to_score:
            endpoint_change = float(dose_to_score[256.0] - dose_to_score[4.0])

        adjacent = group[["concentration", "y_true"]].copy()
        adjacent["absolute_step"] = adjacent["y_true"].astype(float).diff().abs()
        if adjacent["absolute_step"].notna().any():
            max_index = adjacent["absolute_step"].idxmax()
            max_position = adjacent.index.get_loc(max_index)
            high_row = adjacent.loc[max_index]
            low_row = adjacent.iloc[max_position - 1]
            strongest_interval = (
                f"{int(float(low_row['concentration']))}-"
                f"{int(float(high_row['concentration']))}"
            )
            max_step = float(high_row["absolute_step"])
        else:
            strongest_interval = ""
            max_step = np.nan

        row_identity = identity.iloc[0]
        rows.append(
            {
                "mutation_group_id": str(mutation_group_id),
                "assay_position": int(row_identity["assay_position"]),
                "wild_type": str(row_identity["wild_type"]),
                "mutant": str(row_identity["mutant"]),
                "n_concentrations": int(len(group)),
                "minimum_concentration": float(group["concentration"].min()),
                "maximum_concentration": float(group["concentration"].max()),
                "mean_score": float(np.mean(y)),
                "score_sd": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
                "score_range": float(np.max(y) - np.min(y)),
                "linear_slope_per_log2_concentration": slope,
                "spearman_with_log2_concentration": spearman,
                "score_at_4": dose_to_score.get(4.0, np.nan),
                "score_at_256": dose_to_score.get(256.0, np.nan),
                "endpoint_change_256_minus_4": endpoint_change,
                "absolute_endpoint_change": (
                    abs(endpoint_change) if np.isfinite(endpoint_change) else np.nan
                ),
                "largest_adjacent_absolute_change": max_step,
                "largest_change_interval": strongest_interval,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["assay_position", "wild_type", "mutant"], kind="mergesort"
    )


def _position_trajectory_summary(mutations: pd.DataFrame) -> pd.DataFrame:
    grouped = mutations.groupby("assay_position", sort=True)
    rows: list[dict[str, object]] = []
    for assay_position, group in grouped:
        slopes = group["linear_slope_per_log2_concentration"].astype(float)
        endpoints = group["endpoint_change_256_minus_4"].astype(float)
        rows.append(
            {
                "assay_position": int(assay_position),
                "n_mutations": int(len(group)),
                "median_score_range": float(group["score_range"].median()),
                "mean_score_range": float(group["score_range"].mean()),
                "median_absolute_endpoint_change": float(
                    group["absolute_endpoint_change"].median()
                ),
                "mean_absolute_endpoint_change": float(
                    group["absolute_endpoint_change"].mean()
                ),
                "median_endpoint_change_256_minus_4": float(endpoints.median()),
                "median_linear_slope_per_log2_concentration": float(slopes.median()),
                "fraction_positive_endpoint_change": float((endpoints > 0).mean()),
                "fraction_negative_endpoint_change": float((endpoints < 0).mean()),
                "median_spearman_with_log2_concentration": float(
                    group["spearman_with_log2_concentration"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("assay_position", kind="mergesort")


def _condition_position_summary(
    predictions: pd.DataFrame,
    summary: Mapping[str, object],
) -> pd.DataFrame:
    _require_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "Stage 39 predictions")
    primary = summary["primary_reference_by_scheme"]
    rows: list[dict[str, object]] = []

    for scheme_id in summary["scheme_ids"]:
        reference_id = str(primary[scheme_id])
        frame = predictions[
            predictions["scheme_id"].astype(str).eq(scheme_id)
            & predictions["population_scope"].astype(str).eq(FULL_POPULATION)
            & predictions["model_id"].astype(str).eq(reference_id)
        ].copy()
        if frame.empty:
            raise ConditionConcentrationBiologyError(
                f"No primary-reference rows for {scheme_id}"
            )
        frame["signed_shift"] = frame["y_true"].astype(float) - frame["y_pred"].astype(
            float
        )
        frame["absolute_shift"] = frame["signed_shift"].abs()
        for (antibiotic, assay_position), group in frame.groupby(
            ["antibiotic", "assay_position"], sort=True
        ):
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "antibiotic": str(antibiotic),
                    "assay_position": int(assay_position),
                    "reference_model_id": reference_id,
                    "n_observations": int(len(group)),
                    "n_groups": int(group["evaluation_group_id"].nunique()),
                    "mean_signed_target_minus_reference": float(
                        group["signed_shift"].mean()
                    ),
                    "median_signed_target_minus_reference": float(
                        group["signed_shift"].median()
                    ),
                    "mean_absolute_target_minus_reference": float(
                        group["absolute_shift"].mean()
                    ),
                    "median_absolute_target_minus_reference": float(
                        group["absolute_shift"].median()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scheme_id", "antibiotic", "assay_position"], kind="mergesort"
    )


def prepare_condition_concentration_biological_analysis(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_condition_concentration_inputs(repo_root)

    condition_predictions = pd.read_csv(
        repo_root / CONDITION_DIR / "condition_transfer_predictions.csv",
        dtype={"observation_id": str, "evaluation_iteration": str},
        low_memory=False,
    )
    condition_metrics = pd.read_csv(
        repo_root / CONDITION_DIR / "condition_transfer_metrics.csv", low_memory=False
    )
    condition_comparisons = pd.read_csv(
        repo_root / CONDITION_DIR / "condition_transfer_paired_comparisons.csv",
        low_memory=False,
    )
    condition_summary = json.loads(
        (repo_root / CONDITION_DIR / "condition_transfer_summary.json").read_text(
            encoding="utf-8"
        )
    )

    concentration_predictions = pd.read_csv(
        repo_root / CONCENTRATION_DIR / "concentration_transfer_predictions.csv",
        dtype={"observation_id": str, "evaluation_iteration": str},
        low_memory=False,
    )
    concentration_metrics = pd.read_csv(
        repo_root / CONCENTRATION_DIR / "concentration_transfer_metrics.csv",
        low_memory=False,
    )
    concentration_comparisons = pd.read_csv(
        repo_root / CONCENTRATION_DIR / "concentration_transfer_paired_comparisons.csv",
        low_memory=False,
    )
    concentration_design = pd.read_csv(
        repo_root / CONCENTRATION_DIR / "concentration_transfer_design_summary.csv",
        low_memory=False,
    )
    concentration_summary = json.loads(
        (
            repo_root / CONCENTRATION_DIR / "concentration_transfer_summary.json"
        ).read_text(encoding="utf-8")
    )

    task_summary = _task_summary(
        condition_metrics,
        condition_comparisons,
        condition_summary,
        concentration_metrics,
        concentration_comparisons,
        concentration_summary,
        concentration_design,
    )
    reference_summary = _concentration_reference_summary(
        concentration_metrics,
        concentration_comparisons,
        concentration_summary,
        concentration_design,
    )
    mutation_summary = _mutation_trajectory_summary(concentration_predictions)
    position_summary = _position_trajectory_summary(mutation_summary)
    condition_position = _condition_position_summary(
        condition_predictions, condition_summary
    )
    return (
        task_summary,
        reference_summary,
        mutation_summary,
        position_summary,
        condition_position,
    )


def run_condition_concentration_biological_analysis(
    repo_root: Path,
) -> Mapping[str, object]:
    (
        task_summary,
        reference_summary,
        mutation_summary,
        position_summary,
        condition_position,
    ) = prepare_condition_concentration_biological_analysis(repo_root)

    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "condition_concentration_task_summary.csv": task_summary,
        "concentration_reference_summary.csv": reference_summary,
        "concentration_mutation_summary.csv": mutation_summary,
        "concentration_position_summary.csv": position_summary,
        "condition_position_summary.csv": condition_position,
    }
    for name, frame in outputs.items():
        _write_csv(frame, output_dir / name)

    review = review_condition_concentration_inputs(repo_root)
    task_supported_improvements = int(
        (
            task_summary["mae_interval_excludes_zero"].astype(bool)
            & (task_summary["best_fitted_mae_improvement"].astype(float) > 0)
        ).sum()
    )
    task_supported_losses = int(
        (
            task_summary["mae_interval_excludes_zero"].astype(bool)
            & (task_summary["best_fitted_mae_improvement"].astype(float) < 0)
        ).sum()
    )
    summary = {
        "status": "PASS",
        "model_fitted": False,
        "new_bootstrap_run": False,
        "existing_paired_intervals_reused": True,
        "condition_scheme_count": int(
            task_summary["analysis_type"]
            .eq("antibiotic_or_temperature")
            .groupby(task_summary["scheme_id"])
            .any()
            .sum()
        ),
        "concentration_scheme_count": int(
            task_summary["analysis_type"]
            .eq("concentration")
            .groupby(task_summary["scheme_id"])
            .any()
            .sum()
        ),
        "task_summary_row_count": int(len(task_summary)),
        "concentration_reference_row_count": int(len(reference_summary)),
        "concentration_mutation_count": int(len(mutation_summary)),
        "concentration_position_count": int(len(position_summary)),
        "condition_position_row_count": int(len(condition_position)),
        "best_fitted_model_supported_mae_improvement_count": (
            task_supported_improvements
        ),
        "best_fitted_model_supported_mae_loss_count": task_supported_losses,
        "concentration_scale": "log2",
        "concentration_series": [4, 8, 16, 32, 64, 128, 256],
        "concentration_design_note": (
            "The study uses a twofold dose series. Log2 concentration converts the "
            "seven measured doses to equally spaced values 2 through 8."
        ),
        "position_analysis_note": (
            "Position summaries are descriptive summaries of mutation trajectories; "
            "position-level quantities are not treated as independent observation rows."
        ),
        "input_sha256": review["input_sha256"],
        "outputs": [*outputs, "condition_concentration_summary.json"],
    }
    (output_dir / "condition_concentration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
