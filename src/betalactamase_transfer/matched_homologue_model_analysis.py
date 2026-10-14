from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_matched_homologue_model_analysis"
DIRECTIONS = (
    "NDM-1_to_VIM-2",
    "VIM-2_to_NDM-1",
)
METRICS = ("spearman", "pearson", "mae", "rmse")
POPULATIONS = (
    "all_shared_wild_type_pairs",
    "both_above_minimum",
)
REFERENCE_MODELS = (
    "direct_homologue_source_effect",
    "substitution_mean",
)
MODEL_SOURCES = {
    "esmc": (
        "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv",
        ("esmc_site_delta_ridge",),
    ),
    "esm2": (
        "data/interim/sequence_model_sensitivity/esm2_predictions.csv",
        ("esm2_site_delta_ridge",),
    ),
    "structure": (
        "data/interim/experimental_structure_evaluation/experimental_structure_predictions.csv",
        (
            "experimental_structure_descriptor_ridge",
            "structure_matched_mutation_condition_ridge",
        ),
    ),
    "graph": (
        "data/interim/residue_graph_neural_evaluation/residue_graph_neural_predictions.csv",
        (
            "centre_residue_neural_regressor",
            "residue_graph_neural_regressor",
        ),
    ),
    "geometry": (
        "data/interim/residue_geometry_neural_evaluation/residue_geometry_neural_predictions.csv",
        ("egnn_residue_regressor",),
    ),
    "joint": (
        "data/interim/sequence_residue_graph_neural_evaluation/sequence_residue_graph_predictions.csv",
        ("sequence_residue_graph_neural_regressor",),
    ),
}
ADVANCED_MODELS = tuple(
    model_id for _, model_ids in MODEL_SOURCES.values() for model_id in model_ids
)
SCHEME_BY_DIRECTION = {
    "NDM-1_to_VIM-2": "ndm1_to_vim2_amp_source_cv5",
    "VIM-2_to_NDM-1": "vim2_to_ndm1_amp_source_cv5",
}


class MatchedHomologueModelAnalysisError(RuntimeError):
    pass


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**32)


def _correlation_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2:
        return float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise MatchedHomologueModelAnalysisError(
            "Correlation inputs must contain only finite values"
        )
    if np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        raise MatchedHomologueModelAnalysisError(
            "Cannot calculate metrics on an empty population"
        )
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise MatchedHomologueModelAnalysisError(
            "Metric inputs must contain only finite values"
        )
    error = y_pred - y_true
    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return {
        "spearman": _correlation_or_nan(true_rank, pred_rank),
        "pearson": _correlation_or_nan(y_true, y_pred),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _metric_difference(
    metric: str, model_value: float, reference_value: float
) -> float:
    if not np.isfinite(model_value) or not np.isfinite(reference_value):
        return float("nan")
    if metric in {"spearman", "pearson"}:
        return model_value - reference_value
    if metric in {"mae", "rmse"}:
        return reference_value - model_value
    raise MatchedHomologueModelAnalysisError(f"Unsupported metric: {metric}")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MatchedHomologueModelAnalysisError(
            f"{label} is missing required columns: {missing}"
        )


def _load_inputs(
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    baseline_path = (
        repo_root / "data/interim/baseline_evaluation/homologue_analysis/"
        "matched_homologue_predictions.csv"
    )
    pair_path = (
        repo_root
        / "data/interim/task_definitions/aligned_shared_wild_type_mutation_pairs.csv"
    )
    reported_path = repo_root / "data/processed/reported_cross_protein_comparisons.csv"

    required_paths = [baseline_path, pair_path, reported_path]
    required_paths.extend(repo_root / value[0] for value in MODEL_SOURCES.values())
    missing = [
        str(path.relative_to(repo_root))
        for path in required_paths
        if not path.is_file()
    ]
    if missing:
        raise MatchedHomologueModelAnalysisError(
            f"Required matched-homologue model analysis inputs are missing: {missing}"
        )

    input_hashes = {
        str(path.relative_to(repo_root)): _sha256(path) for path in required_paths
    }

    pairs = pd.read_csv(
        pair_path,
        dtype={"ndm_observation_id": str, "vim_observation_id": str},
        low_memory=False,
    )
    _require_columns(
        pairs,
        {
            "homologue_position_id",
            "wild_type",
            "mutant",
            "ndm_observation_id",
            "vim_observation_id",
            "pairing_uses_outcome_values",
        },
        "Matched substitution table",
    )
    if len(pairs) != 1513 or pairs["homologue_position_id"].nunique() != 81:
        raise MatchedHomologueModelAnalysisError(
            "Matched-homologue model analysis requires 1,513 substitutions "
            "across 81 shared-WT positions"
        )
    outcome_use = pairs["pairing_uses_outcome_values"].astype(str).str.lower()
    if outcome_use.isin({"true", "1"}).any():
        raise MatchedHomologueModelAnalysisError(
            "Matched substitution population must not use outcome values"
        )

    baseline = pd.read_csv(
        baseline_path,
        dtype={"source_observation_id": str, "target_observation_id": str},
        low_memory=False,
    )
    _require_columns(
        baseline,
        {
            "direction",
            "homologue_position_id",
            "wild_type",
            "mutant",
            "source_observation_id",
            "target_observation_id",
            "source_effect",
            "target_effect",
            "predicted_target_effect",
            "substitution_mean_prediction",
            "minimum_score_class",
            "background_difference_analysis",
            "absolute_background_difference_analysis",
        },
        "Existing matched homologue analysis",
    )
    if len(baseline) != 3026:
        raise MatchedHomologueModelAnalysisError(
            f"Expected 3,026 directional matched rows; found {len(baseline)}"
        )
    if set(baseline["direction"].astype(str)) != set(DIRECTIONS):
        raise MatchedHomologueModelAnalysisError("Unexpected transfer directions")
    for direction in DIRECTIONS:
        subset = baseline.loc[baseline["direction"].eq(direction)]
        if len(subset) != 1513 or subset["homologue_position_id"].nunique() != 81:
            raise MatchedHomologueModelAnalysisError(
                f"Unexpected matched population for {direction}"
            )
        if subset["target_observation_id"].duplicated().any():
            raise MatchedHomologueModelAnalysisError(
                f"Duplicate target observations for {direction}"
            )

    reported = pd.read_csv(reported_path, low_memory=False)
    _require_columns(reported, {"independence_status"}, "Reported comparison table")
    if len(reported) != 132:
        raise MatchedHomologueModelAnalysisError(
            f"Expected 132 supporting comparison rows; found {len(reported)}"
        )
    if not reported["independence_status"].eq("not_independent_comparison_table").all():
        raise MatchedHomologueModelAnalysisError(
            "Supporting comparison rows have unexpected independence status"
        )

    prediction_frames: dict[str, pd.DataFrame] = {}
    required_prediction_columns = {
        "scheme_id",
        "model_id",
        "evaluation_iteration",
        "observation_id",
        "y_true",
        "y_pred",
    }
    for source_name, (relative, expected_models) in MODEL_SOURCES.items():
        frame = pd.read_csv(
            repo_root / relative,
            dtype={"evaluation_iteration": str, "observation_id": str},
            low_memory=False,
        )
        _require_columns(
            frame, required_prediction_columns, f"{source_name} predictions"
        )
        prediction_frames[source_name] = frame
        observed = set(frame["model_id"].astype(str))
        missing_models = sorted(set(expected_models).difference(observed))
        if missing_models:
            raise MatchedHomologueModelAnalysisError(
                f"{source_name} predictions are missing expected models: "
                f"{missing_models}"
            )

    return baseline, prediction_frames, input_hashes


def _assemble_predictions(
    baseline: pd.DataFrame,
    prediction_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_columns = [
        "direction",
        "homologue_position_id",
        "wild_type",
        "mutant",
        "source_observation_id",
        "target_observation_id",
        "source_effect",
        "target_effect",
        "minimum_score_class",
        "background_difference_analysis",
        "absolute_background_difference_analysis",
    ]

    for direction in DIRECTIONS:
        base = baseline.loc[baseline["direction"].eq(direction), base_columns].copy()
        scheme_id = SCHEME_BY_DIRECTION[direction]

        references = {
            "direct_homologue_source_effect": baseline.loc[
                baseline["direction"].eq(direction), "predicted_target_effect"
            ].to_numpy(dtype=float),
            "substitution_mean": baseline.loc[
                baseline["direction"].eq(direction), "substitution_mean_prediction"
            ].to_numpy(dtype=float),
        }
        for model_id, prediction in references.items():
            part = base.copy()
            part["scheme_id"] = scheme_id
            part["model_source"] = "matched_homologue_baseline"
            part["model_id"] = model_id
            part["y_pred"] = prediction
            rows.append(part)

        for source_name, frame in prediction_frames.items():
            expected_models = MODEL_SOURCES[source_name][1]
            selected = frame.loc[
                frame["scheme_id"].astype(str).eq(scheme_id)
                & frame["evaluation_iteration"].astype(str).eq("target")
                & frame["model_id"].astype(str).isin(expected_models),
                ["observation_id", "model_id", "y_true", "y_pred"],
            ].copy()
            if selected.duplicated(["model_id", "observation_id"]).any():
                raise MatchedHomologueModelAnalysisError(
                    f"Duplicate target predictions in {source_name}/{direction}"
                )

            for model_id in expected_models:
                model = selected.loc[selected["model_id"].eq(model_id)].copy()
                merged = base.merge(
                    model,
                    left_on="target_observation_id",
                    right_on="observation_id",
                    how="left",
                    validate="one_to_one",
                )
                if merged["y_pred"].isna().any():
                    raise MatchedHomologueModelAnalysisError(
                        f"{model_id}/{direction} does not cover every "
                        "matched substitution"
                    )
                if not np.allclose(
                    merged["y_true"].to_numpy(dtype=float),
                    merged["target_effect"].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise MatchedHomologueModelAnalysisError(
                        f"Stored y_true disagrees with matched targets for "
                        f"{model_id}/{direction}"
                    )
                part = merged[base_columns + ["y_pred"]].copy()
                part["scheme_id"] = scheme_id
                part["model_source"] = source_name
                part["model_id"] = model_id
                rows.append(part)

    result = pd.concat(rows, ignore_index=True)
    result["signed_error"] = result["y_pred"] - result["target_effect"]
    result["absolute_error"] = result["signed_error"].abs()
    ordered = [
        "direction",
        "scheme_id",
        "homologue_position_id",
        "wild_type",
        "mutant",
        "source_observation_id",
        "target_observation_id",
        "source_effect",
        "target_effect",
        "minimum_score_class",
        "background_difference_analysis",
        "absolute_background_difference_analysis",
        "model_source",
        "model_id",
        "y_pred",
        "signed_error",
        "absolute_error",
    ]
    return (
        result[ordered]
        .sort_values(
            ["direction", "homologue_position_id", "wild_type", "mutant", "model_id"]
        )
        .reset_index(drop=True)
    )


def _population_subset(frame: pd.DataFrame, population: str) -> pd.DataFrame:
    if population == "all_shared_wild_type_pairs":
        return frame
    if population == "both_above_minimum":
        return frame.loc[frame["minimum_score_class"].eq("both_above_minimum")]
    raise MatchedHomologueModelAnalysisError(f"Unknown population: {population}")


def _metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        for population in POPULATIONS:
            direction_frame = predictions.loc[predictions["direction"].eq(direction)]
            subset = _population_subset(direction_frame, population)
            expected_pairs = 1513 if population == "all_shared_wild_type_pairs" else 478
            expected_positions = (
                81 if population == "all_shared_wild_type_pairs" else 63
            )
            for model_id, model in subset.groupby("model_id", sort=True):
                if len(model) != expected_pairs:
                    raise MatchedHomologueModelAnalysisError(
                        f"{direction}/{population}/{model_id}: expected "
                        f"{expected_pairs} rows; "
                        f"found {len(model)}"
                    )
                if model["homologue_position_id"].nunique() != expected_positions:
                    raise MatchedHomologueModelAnalysisError(
                        f"{direction}/{population}/{model_id}: unexpected "
                        "position count"
                    )
                metrics = _metric_values(
                    model["target_effect"].to_numpy(dtype=float),
                    model["y_pred"].to_numpy(dtype=float),
                )
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "direction": direction,
                            "population": population,
                            "model_id": model_id,
                            "metric": metric,
                            "point_estimate": value,
                            "n_pairs": len(model),
                            "n_homologue_positions": expected_positions,
                        }
                    )
    return pd.DataFrame(rows)


def _bootstrap_group_tables(
    frame: pd.DataFrame,
) -> tuple[list[object], dict[object, pd.DataFrame]]:
    groups = sorted(frame["homologue_position_id"].unique())
    grouped = {
        group: frame.loc[frame["homologue_position_id"].eq(group)].copy()
        for group in groups
    }
    return groups, grouped


def _paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for direction in DIRECTIONS:
        for population in POPULATIONS:
            wide_source = predictions.loc[predictions["direction"].eq(direction)].copy()
            wide_source = _population_subset(wide_source, population)
            key_columns = [
                "homologue_position_id",
                "target_observation_id",
                "target_effect",
            ]
            wide = wide_source.pivot(
                index=key_columns,
                columns="model_id",
                values="y_pred",
            ).reset_index()
            required_models = set(ADVANCED_MODELS).union(REFERENCE_MODELS)
            if not required_models.issubset(wide.columns):
                missing = sorted(required_models.difference(wide.columns))
                raise MatchedHomologueModelAnalysisError(
                    f"Missing models in paired comparison table: {missing}"
                )

            point_metrics = {
                model_id: _metric_values(
                    wide["target_effect"].to_numpy(dtype=float),
                    wide[model_id].to_numpy(dtype=float),
                )
                for model_id in required_models
            }
            groups, grouped = _bootstrap_group_tables(wide)
            rng = np.random.default_rng(
                _numeric_seed(f"{BOOTSTRAP_SEED}|{direction}|{population}|metrics")
            )
            distributions = {
                (model_id, reference_id, metric): []
                for model_id in ADVANCED_MODELS
                for reference_id in REFERENCE_MODELS
                for metric in METRICS
            }

            for _ in range(BOOTSTRAP_REPLICATES):
                sampled = rng.choice(groups, size=len(groups), replace=True)
                boot = pd.concat(
                    [grouped[group] for group in sampled], ignore_index=True
                )
                boot_true = boot["target_effect"].to_numpy(dtype=float)
                boot_metrics = {
                    model_id: _metric_values(
                        boot_true,
                        boot[model_id].to_numpy(dtype=float),
                    )
                    for model_id in required_models
                }
                for model_id in ADVANCED_MODELS:
                    for reference_id in REFERENCE_MODELS:
                        for metric in METRICS:
                            difference = _metric_difference(
                                metric,
                                boot_metrics[model_id][metric],
                                boot_metrics[reference_id][metric],
                            )
                            if np.isfinite(difference):
                                distributions[(model_id, reference_id, metric)].append(
                                    difference
                                )

            for model_id in ADVANCED_MODELS:
                for reference_id in REFERENCE_MODELS:
                    for metric in METRICS:
                        values = np.asarray(
                            distributions[(model_id, reference_id, metric)], dtype=float
                        )
                        difference = _metric_difference(
                            metric,
                            point_metrics[model_id][metric],
                            point_metrics[reference_id][metric],
                        )
                        ci_lower = (
                            float(np.quantile(values, 0.025))
                            if len(values)
                            else float("nan")
                        )
                        ci_upper = (
                            float(np.quantile(values, 0.975))
                            if len(values)
                            else float("nan")
                        )
                        rows.append(
                            {
                                "direction": direction,
                                "population": population,
                                "model_id": model_id,
                                "reference_model_id": reference_id,
                                "metric": metric,
                                "model_point_estimate": point_metrics[model_id][metric],
                                "reference_point_estimate": point_metrics[reference_id][
                                    metric
                                ],
                                "paired_difference": difference,
                                "ci_lower": ci_lower,
                                "ci_upper": ci_upper,
                                "interval_excludes_zero": bool(
                                    np.isfinite(ci_lower)
                                    and np.isfinite(ci_upper)
                                    and (ci_lower > 0.0 or ci_upper < 0.0)
                                ),
                                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                                "bootstrap_valid_replicates": len(values),
                                "resampling_unit": "homologue_position_id",
                                "difference_interpretation": "positive_favours_model",
                            }
                        )

    return pd.DataFrame(rows)


def _background_associations(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    association_specs = {
        "absolute_error_vs_absolute_background_difference": (
            "absolute_error",
            "absolute_background_difference_analysis",
        ),
        "signed_error_vs_background_difference": (
            "signed_error",
            "background_difference_analysis",
        ),
    }

    for direction in DIRECTIONS:
        for population in POPULATIONS:
            source = predictions.loc[
                predictions["direction"].eq(direction)
                & predictions["model_id"].isin(ADVANCED_MODELS)
            ].copy()
            source = _population_subset(source, population)
            for model_id, model in source.groupby("model_id", sort=True):
                groups, grouped = _bootstrap_group_tables(model)
                for association_id, (
                    left_column,
                    right_column,
                ) in association_specs.items():
                    left = model[left_column].to_numpy(dtype=float)
                    right = model[right_column].to_numpy(dtype=float)
                    point = _correlation_or_nan(
                        pd.Series(left).rank(method="average").to_numpy(dtype=float),
                        pd.Series(right).rank(method="average").to_numpy(dtype=float),
                    )
                    rng = np.random.default_rng(
                        _numeric_seed(
                            f"{BOOTSTRAP_SEED}|{direction}|{population}|{model_id}|"
                            f"{association_id}"
                        )
                    )
                    values: list[float] = []
                    for _ in range(BOOTSTRAP_REPLICATES):
                        sampled = rng.choice(groups, size=len(groups), replace=True)
                        boot = pd.concat(
                            [grouped[group] for group in sampled], ignore_index=True
                        )
                        boot_left = boot[left_column].to_numpy(dtype=float)
                        boot_right = boot[right_column].to_numpy(dtype=float)
                        corr = _correlation_or_nan(
                            pd.Series(boot_left)
                            .rank(method="average")
                            .to_numpy(dtype=float),
                            pd.Series(boot_right)
                            .rank(method="average")
                            .to_numpy(dtype=float),
                        )
                        if np.isfinite(corr):
                            values.append(corr)
                    array = np.asarray(values, dtype=float)
                    ci_lower = (
                        float(np.quantile(array, 0.025)) if len(array) else float("nan")
                    )
                    ci_upper = (
                        float(np.quantile(array, 0.975)) if len(array) else float("nan")
                    )
                    rows.append(
                        {
                            "direction": direction,
                            "population": population,
                            "model_id": model_id,
                            "association_id": association_id,
                            "spearman": point,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "interval_excludes_zero": bool(
                                np.isfinite(ci_lower)
                                and np.isfinite(ci_upper)
                                and (ci_lower > 0.0 or ci_upper < 0.0)
                            ),
                            "n_pairs": len(model),
                            "n_homologue_positions": model[
                                "homologue_position_id"
                            ].nunique(),
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                            "bootstrap_valid_replicates": len(array),
                            "resampling_unit": "homologue_position_id",
                        }
                    )
    return pd.DataFrame(rows)


def _pair_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    base = predictions.loc[
        predictions["direction"].eq("NDM-1_to_VIM-2")
        & predictions["model_id"].eq("direct_homologue_source_effect")
    ].copy()
    if len(base) != 1513:
        raise MatchedHomologueModelAnalysisError(
            "Could not recover the 1,513 unique matched substitutions"
        )
    result = base[
        [
            "homologue_position_id",
            "wild_type",
            "mutant",
            "source_observation_id",
            "target_observation_id",
            "source_effect",
            "target_effect",
            "minimum_score_class",
            "background_difference_analysis",
            "absolute_background_difference_analysis",
        ]
    ].rename(
        columns={
            "source_observation_id": "ndm_observation_id",
            "target_observation_id": "vim_observation_id",
            "source_effect": "ndm_effect",
            "target_effect": "vim_effect",
            "background_difference_analysis": "vim_minus_ndm_effect",
            "absolute_background_difference_analysis": (
                "absolute_ndm_vim_background_difference"
            ),
        }
    )

    advanced = predictions.loc[predictions["model_id"].isin(ADVANCED_MODELS)].copy()
    for direction, suffix in (
        ("NDM-1_to_VIM-2", "ndm_to_vim"),
        ("VIM-2_to_NDM-1", "vim_to_ndm"),
    ):
        direction_errors = (
            advanced.loc[advanced["direction"].eq(direction)]
            .groupby(
                ["homologue_position_id", "wild_type", "mutant"],
                as_index=False,
            )
            .agg(
                **{
                    f"median_model_absolute_error_{suffix}": (
                        "absolute_error",
                        "median",
                    ),
                    f"min_model_absolute_error_{suffix}": (
                        "absolute_error",
                        "min",
                    ),
                    f"max_model_absolute_error_{suffix}": (
                        "absolute_error",
                        "max",
                    ),
                }
            )
        )
        result = result.merge(
            direction_errors,
            on=["homologue_position_id", "wild_type", "mutant"],
            how="left",
            validate="one_to_one",
        )

    return result.sort_values(
        ["absolute_ndm_vim_background_difference", "homologue_position_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _position_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for direction in DIRECTIONS:
        base = predictions.loc[
            predictions["direction"].eq(direction)
            & predictions["model_id"].eq("direct_homologue_source_effect")
        ].copy()
        base_summary = base.groupby("homologue_position_id", as_index=False).agg(
            n_substitutions=("target_observation_id", "size"),
            median_absolute_background_difference=(
                "absolute_background_difference_analysis",
                "median",
            ),
            mean_absolute_background_difference=(
                "absolute_background_difference_analysis",
                "mean",
            ),
            max_absolute_background_difference=(
                "absolute_background_difference_analysis",
                "max",
            ),
            fraction_both_above_minimum=(
                "minimum_score_class",
                lambda values: float(
                    np.mean(pd.Series(values).eq("both_above_minimum"))
                ),
            ),
            fraction_discordant_minimum_score=(
                "minimum_score_class",
                lambda values: float(
                    np.mean(
                        pd.Series(values).isin(
                            {"source_at_minimum", "target_at_minimum"}
                        )
                    )
                ),
            ),
        )
        advanced = predictions.loc[
            predictions["direction"].eq(direction)
            & predictions["model_id"].isin(ADVANCED_MODELS)
        ].copy()
        model_position = (
            advanced.groupby(
                ["homologue_position_id", "model_id"],
                as_index=False,
            )
            .agg(median_absolute_error=("absolute_error", "median"))
            .groupby("homologue_position_id", as_index=False)
            .agg(
                median_model_absolute_error=("median_absolute_error", "median"),
                min_model_absolute_error=("median_absolute_error", "min"),
                max_model_absolute_error=("median_absolute_error", "max"),
            )
        )
        summary = base_summary.merge(
            model_position,
            on="homologue_position_id",
            how="left",
            validate="one_to_one",
        )
        summary.insert(0, "direction", direction)
        rows.append(summary)

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            [
                "direction",
                "median_absolute_background_difference",
                "homologue_position_id",
            ],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )


def _validate_output_contract(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    associations: pd.DataFrame,
    pair_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
) -> None:
    expected_models = set(ADVANCED_MODELS).union(REFERENCE_MODELS)
    if set(predictions["model_id"]) != expected_models:
        raise MatchedHomologueModelAnalysisError("Unexpected model set in predictions")
    expected_prediction_rows = 3026 * len(expected_models)
    if len(predictions) != expected_prediction_rows:
        raise MatchedHomologueModelAnalysisError(
            f"Expected {expected_prediction_rows} long-format prediction rows; "
            f"found {len(predictions)}"
        )
    if len(metrics) != len(DIRECTIONS) * len(POPULATIONS) * len(expected_models) * 4:
        raise MatchedHomologueModelAnalysisError("Unexpected metric row count")
    if len(comparisons) != (
        len(DIRECTIONS)
        * len(POPULATIONS)
        * len(ADVANCED_MODELS)
        * len(REFERENCE_MODELS)
        * 4
    ):
        raise MatchedHomologueModelAnalysisError("Unexpected comparison row count")
    if len(associations) != (
        len(DIRECTIONS) * len(POPULATIONS) * len(ADVANCED_MODELS) * 2
    ):
        raise MatchedHomologueModelAnalysisError("Unexpected association row count")
    if len(pair_summary) != 1513:
        raise MatchedHomologueModelAnalysisError("Expected 1,513 pair-summary rows")
    if len(position_summary) != 162:
        raise MatchedHomologueModelAnalysisError(
            "Expected 162 direction-specific position rows"
        )


def run_matched_homologue_model_analysis(
    repo_root: Path,
    *,
    validate_inputs: bool = False,
) -> Mapping[str, object]:
    baseline, prediction_frames, input_hashes = _load_inputs(repo_root)
    predictions = _assemble_predictions(baseline, prediction_frames)

    input_review = {
        "status": "PASS",
        "validate_inputs": validate_inputs,
        "analysis_is_post_hoc_and_does_not_refit_models": True,
        "pair_count_per_direction": 1513,
        "homologue_position_count": 81,
        "above_minimum_pair_count_per_direction": 478,
        "above_minimum_homologue_position_count": 63,
        "advanced_model_count": len(ADVANCED_MODELS),
        "reference_models": list(REFERENCE_MODELS),
        "supporting_reported_comparison_rows": 132,
        "supporting_rows_used_for_population_or_model_choice": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "homologue_position_id",
        "input_sha256": input_hashes,
    }
    if validate_inputs:
        return input_review

    metrics = _metric_table(predictions)
    comparisons = _paired_comparisons(predictions)
    associations = _background_associations(predictions)
    pair_summary = _pair_summary(predictions)
    position_summary = _position_summary(predictions)
    _validate_output_contract(
        predictions,
        metrics,
        comparisons,
        associations,
        pair_summary,
        position_summary,
    )

    output_dir = repo_root / "data/interim/matched_homologue_model_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "matched_homologue_model_predictions.csv": predictions,
        "matched_homologue_model_metrics.csv": metrics,
        "matched_homologue_model_comparisons.csv": comparisons,
        "matched_homologue_background_associations.csv": associations,
        "matched_homologue_pair_summary.csv": pair_summary,
        "matched_homologue_position_summary.csv": position_summary,
    }
    for name, frame in outputs.items():
        _write_csv(frame, output_dir / name)

    supported_vs_direct = comparisons.loc[
        comparisons["reference_model_id"].eq("direct_homologue_source_effect")
        & comparisons["interval_excludes_zero"]
    ]
    supported_positive_vs_direct = int(
        supported_vs_direct["paired_difference"].gt(0).sum()
    )
    supported_negative_vs_direct = int(
        supported_vs_direct["paired_difference"].lt(0).sum()
    )
    background_supported = associations.loc[associations["interval_excludes_zero"]]

    summary: dict[str, object] = {
        **input_review,
        "validate_inputs": False,
        "prediction_row_count": len(predictions),
        "metric_row_count": len(metrics),
        "comparison_row_count": len(comparisons),
        "background_association_row_count": len(associations),
        "pair_summary_row_count": len(pair_summary),
        "position_summary_row_count": len(position_summary),
        "supported_model_improvements_vs_direct_reference": (
            supported_positive_vs_direct
        ),
        "supported_model_losses_vs_direct_reference": supported_negative_vs_direct,
        "supported_background_association_count": len(background_supported),
        "model_selection_performed": False,
        "test_results_used_to_choose_model_settings": False,
        "reported_wt_reversion_table_role": "supporting_evidence_only_not_used_here",
        "outputs": [*outputs, "matched_homologue_model_summary.json"],
    }
    (output_dir / "matched_homologue_model_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
