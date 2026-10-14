from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_homologue_baseline_analysis"
PRIMARY_COMPARATOR = "substitution_mean"
COMPARATOR_BASELINES = (
    "substitution_mean",
    "ridge_mutation_identity",
    "mutant_mean",
    "training_mean",
)
METRICS = ("spearman", "pearson", "mae", "rmse")


class HomologueBaselineAnalysisError(RuntimeError):
    pass


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _correlation_or_nan(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    if len(left) < 2:
        return float("nan")

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise HomologueBaselineAnalysisError("Correlation inputs must be finite")

    if np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")

    return float(np.corrcoef(left, right)[0, 1])


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Mapping[str, float]:
    if len(y_true) == 0:
        raise HomologueBaselineAnalysisError(
            "Cannot calculate metrics on an empty population"
        )

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise HomologueBaselineAnalysisError(
            "Metric inputs must contain only finite values"
        )

    error = y_pred - y_true

    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)

    return {
        "spearman": _correlation_or_nan(
            true_rank,
            pred_rank,
        ),
        "pearson": _correlation_or_nan(
            y_true,
            y_pred,
        ),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _direction_scheme(direction: str) -> str:
    mapping = {
        "NDM-1_to_VIM-2": "ndm1_to_vim2_amp_source_cv5",
        "VIM-2_to_NDM-1": "vim2_to_ndm1_amp_source_cv5",
    }
    try:
        return mapping[direction]
    except KeyError as exc:
        raise HomologueBaselineAnalysisError(
            f"Unknown transfer direction: {direction}"
        ) from exc


def _minimum_score_class(source: pd.Series, target: pd.Series) -> pd.Series:
    source_floor = source.eq(-4.0)
    target_floor = target.eq(-4.0)
    result = pd.Series("both_above_minimum", index=source.index, dtype="string")
    result.loc[source_floor & target_floor] = "both_at_minimum"
    result.loc[source_floor & ~target_floor] = "source_at_minimum"
    result.loc[~source_floor & target_floor] = "target_at_minimum"
    return result


def _prepare_direction_table(
    direction: str,
    direct: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    table = direct.loc[direct["direction"].eq(direction)].copy()
    if len(table) != 1513:
        raise HomologueBaselineAnalysisError(
            f"Expected 1513 direct homologue rows for {direction}; found {len(table)}"
        )
    if table["homologue_position_id"].nunique() != 81:
        raise HomologueBaselineAnalysisError(
            f"Expected 81 homologue positions for {direction}"
        )

    scheme_id = _direction_scheme(direction)
    selected_predictions = baseline_predictions.loc[
        baseline_predictions["scheme_id"].eq(scheme_id)
        & baseline_predictions["baseline_id"].isin(COMPARATOR_BASELINES)
    ].copy()

    for baseline_id in COMPARATOR_BASELINES:
        values = selected_predictions.loc[
            selected_predictions["baseline_id"].eq(baseline_id),
            ["observation_id", "y_true", "y_pred"],
        ].copy()
        if values["observation_id"].duplicated().any():
            raise HomologueBaselineAnalysisError(
                f"Duplicate target prediction rows for {scheme_id}/{baseline_id}"
            )
        values = values.rename(
            columns={
                "observation_id": "target_observation_id",
                "y_true": f"{baseline_id}_y_true",
                "y_pred": f"{baseline_id}_prediction",
            }
        )
        table = table.merge(
            values,
            on="target_observation_id",
            how="left",
            validate="one_to_one",
        )
        if table[f"{baseline_id}_prediction"].isna().any():
            raise HomologueBaselineAnalysisError(
                "Matched population is missing required "
                f"{baseline_id} predictions for {direction}"
            )
        if not np.allclose(
            table[f"{baseline_id}_y_true"].to_numpy(float),
            table["target_effect"].to_numpy(float),
        ):
            raise HomologueBaselineAnalysisError(
                "Stored target values disagree with direct homologue targets "
                f"for {direction}"
            )

    source_obs = observations[
        ["observation_id", "score_original", "score_analysis"]
    ].rename(
        columns={
            "observation_id": "source_observation_id",
            "score_original": "source_score_original",
            "score_analysis": "source_score_analysis",
        }
    )
    target_obs = observations[
        ["observation_id", "score_original", "score_analysis"]
    ].rename(
        columns={
            "observation_id": "target_observation_id",
            "score_original": "target_score_original",
            "score_analysis": "target_score_analysis",
        }
    )
    table = table.merge(
        source_obs,
        on="source_observation_id",
        how="left",
        validate="many_to_one",
    )
    table = table.merge(
        target_obs,
        on="target_observation_id",
        how="left",
        validate="many_to_one",
    )
    required = [
        "source_score_original",
        "source_score_analysis",
        "target_score_original",
        "target_score_analysis",
    ]
    if table[required].isna().any().any():
        raise HomologueBaselineAnalysisError(
            "Missing source or target score during sensitivity join"
        )
    if not np.allclose(table["source_score_analysis"], table["source_effect"]):
        raise HomologueBaselineAnalysisError(
            "Direct source effect disagrees with score_analysis"
        )
    if not np.allclose(table["target_score_analysis"], table["target_effect"]):
        raise HomologueBaselineAnalysisError(
            "Direct target effect disagrees with score_analysis"
        )

    table["minimum_score_class"] = _minimum_score_class(
        table["source_score_analysis"], table["target_score_analysis"]
    )
    table["background_difference_analysis"] = (
        table["target_score_analysis"] - table["source_score_analysis"]
    )
    table["absolute_background_difference_analysis"] = table[
        "background_difference_analysis"
    ].abs()
    table["background_difference_original"] = (
        table["target_score_original"] - table["source_score_original"]
    )
    table["absolute_background_difference_original"] = table[
        "background_difference_original"
    ].abs()
    return table


def _matched_metric_rows(table: pd.DataFrame) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    predictors = {
        "direct_homologue_source_effect": "predicted_target_effect",
        **{
            baseline_id: f"{baseline_id}_prediction"
            for baseline_id in COMPARATOR_BASELINES
        },
    }
    for baseline_id, column in predictors.items():
        metrics = _metric_values(
            table["target_score_analysis"].to_numpy(float),
            table[column].to_numpy(float),
        )
        for metric, value in metrics.items():
            rows.append(
                {
                    "direction": table["direction"].iloc[0],
                    "population": "aligned_shared_wild_type_identical_substitutions",
                    "baseline_id": baseline_id,
                    "metric": metric,
                    "point_estimate": value,
                    "n_pairs": len(table),
                    "n_homologue_positions": table["homologue_position_id"].nunique(),
                }
            )
    return rows


def _bootstrap_seed(direction: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{direction}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _paired_bootstrap_rows(table: pd.DataFrame) -> list[Mapping[str, object]]:
    groups = sorted(table["homologue_position_id"].unique())
    grouped = {
        group: table.loc[table["homologue_position_id"].eq(group)] for group in groups
    }
    rng = np.random.default_rng(_bootstrap_seed(str(table["direction"].iloc[0])))
    comparison_ids = COMPARATOR_BASELINES
    distributions = {
        (baseline_id, metric): []
        for baseline_id in comparison_ids
        for metric in METRICS
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([grouped[group] for group in sampled], ignore_index=True)
        direct_metrics = _metric_values(
            boot["target_score_analysis"].to_numpy(float),
            boot["predicted_target_effect"].to_numpy(float),
        )
        for baseline_id in comparison_ids:
            comparator_metrics = _metric_values(
                boot["target_score_analysis"].to_numpy(float),
                boot[f"{baseline_id}_prediction"].to_numpy(float),
            )
            for metric in METRICS:
                if metric in {"mae", "rmse"}:
                    delta = comparator_metrics[metric] - direct_metrics[metric]
                else:
                    delta = direct_metrics[metric] - comparator_metrics[metric]
                if np.isfinite(delta):
                    distributions[(baseline_id, metric)].append(delta)

    rows: list[Mapping[str, object]] = []
    direct_point = _metric_values(
        table["target_score_analysis"].to_numpy(float),
        table["predicted_target_effect"].to_numpy(float),
    )
    for baseline_id in comparison_ids:
        comparator_point = _metric_values(
            table["target_score_analysis"].to_numpy(float),
            table[f"{baseline_id}_prediction"].to_numpy(float),
        )
        for metric in METRICS:
            values = np.asarray(distributions[(baseline_id, metric)], dtype=float)
            if metric in {"mae", "rmse"}:
                point = comparator_point[metric] - direct_point[metric]
                interpretation = "positive_favours_direct_homologue"
            else:
                point = direct_point[metric] - comparator_point[metric]
                interpretation = "positive_favours_direct_homologue"
            rows.append(
                {
                    "direction": table["direction"].iloc[0],
                    "direct_baseline_id": "direct_homologue_source_effect",
                    "comparator_baseline_id": baseline_id,
                    "primary_comparator": baseline_id == PRIMARY_COMPARATOR,
                    "metric": metric,
                    "paired_difference": point,
                    "ci_lower": (
                        float(np.quantile(values, 0.025))
                        if len(values)
                        else float("nan")
                    ),
                    "ci_upper": (
                        float(np.quantile(values, 0.975))
                        if len(values)
                        else float("nan")
                    ),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_valid_replicates": len(values),
                    "resampling_unit": "homologue_position_id",
                    "difference_interpretation": interpretation,
                }
            )
    return rows


def _sensitivity_rows(table: pd.DataFrame) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    subsets = {
        "all_pairs_analysis_score": table,
        "both_above_minimum_analysis_score": table.loc[
            table["minimum_score_class"].eq("both_above_minimum")
        ],
        "all_pairs_original_score": table,
    }
    for analysis_id, subset in subsets.items():
        if analysis_id == "all_pairs_original_score":
            source_column = "source_score_original"
            target_column = "target_score_original"
            score_field = "score_original"
        else:
            source_column = "source_score_analysis"
            target_column = "target_score_analysis"
            score_field = "score_analysis"
        metrics = _metric_values(
            subset[target_column].to_numpy(float),
            subset[source_column].to_numpy(float),
        )
        row: dict[str, object] = {
            "direction": table["direction"].iloc[0],
            "analysis_id": analysis_id,
            "score_field": score_field,
            "n_pairs": len(subset),
            "n_homologue_positions": subset["homologue_position_id"].nunique(),
        }
        row.update(metrics)
        rows.append(row)
    return rows


ABOVE_MINIMUM_BOOTSTRAP_SEEDS = {
    "NDM-1_to_VIM-2": 2170354257516925701,
    "VIM-2_to_NDM-1": 6550956855475596474,
}
DIRECTIONS = (
    "NDM-1_to_VIM-2",
    "VIM-2_to_NDM-1",
)


def _numeric_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _metric_difference(
    metric: str,
    direct_value: float,
    comparator_value: float,
) -> float:
    if not np.isfinite(direct_value) or not np.isfinite(comparator_value):
        return float("nan")

    if metric in {"spearman", "pearson"}:
        return direct_value - comparator_value

    if metric in {"mae", "rmse"}:
        return comparator_value - direct_value

    raise HomologueBaselineAnalysisError(f"Unsupported metric: {metric}")


def _resample_positions(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    positions = frame["homologue_position_id"].drop_duplicates().sort_values().tolist()

    sampled = rng.choice(
        positions,
        size=len(positions),
        replace=True,
    )

    blocks = []

    for draw_index, position_id in enumerate(sampled):
        block = frame.loc[frame["homologue_position_id"].eq(position_id)].copy()
        block["_bootstrap_position_draw"] = draw_index
        blocks.append(block)

    if not blocks:
        raise HomologueBaselineAnalysisError("Bootstrap produced no position blocks")

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def _above_minimum_paired_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
) -> pd.DataFrame:
    direction = str(frame["direction"].iloc[0])

    direct_point = _metric_values(
        frame["target_score_analysis"].to_numpy(dtype=float),
        frame["predicted_target_effect"].to_numpy(dtype=float),
    )
    comparator_point = _metric_values(
        frame["target_score_analysis"].to_numpy(dtype=float),
        frame["substitution_mean_prediction"].to_numpy(dtype=float),
    )

    rng = np.random.default_rng(ABOVE_MINIMUM_BOOTSTRAP_SEEDS[direction])

    replicate_values = {metric: [] for metric in METRICS}

    for _ in range(replicates):
        sample = _resample_positions(frame, rng)

        direct_metrics = _metric_values(
            sample["target_score_analysis"].to_numpy(dtype=float),
            sample["predicted_target_effect"].to_numpy(dtype=float),
        )
        comparator_metrics = _metric_values(
            sample["target_score_analysis"].to_numpy(dtype=float),
            sample["substitution_mean_prediction"].to_numpy(dtype=float),
        )

        for metric in METRICS:
            difference = _metric_difference(
                metric,
                direct_metrics[metric],
                comparator_metrics[metric],
            )

            if np.isfinite(difference):
                replicate_values[metric].append(difference)

    rows = []

    for metric in METRICS:
        values = np.asarray(
            replicate_values[metric],
            dtype=float,
        )

        rows.append(
            {
                "direction": direction,
                "population": "both_above_minimum",
                "direct_baseline_id": "direct_homologue_source_effect",
                "comparator_baseline_id": PRIMARY_COMPARATOR,
                "metric": metric,
                "direct_point_estimate": direct_point[metric],
                "comparator_point_estimate": comparator_point[metric],
                "paired_difference": _metric_difference(
                    metric,
                    direct_point[metric],
                    comparator_point[metric],
                ),
                "ci_lower": (
                    float(np.quantile(values, 0.025)) if len(values) else float("nan")
                ),
                "ci_upper": (
                    float(np.quantile(values, 0.975)) if len(values) else float("nan")
                ),
                "bootstrap_replicates": replicates,
                "bootstrap_valid_replicates": len(values),
                "resampling_unit": "homologue_position_id",
                "difference_interpretation": ("positive_favours_direct_homologue"),
            }
        )

    return pd.DataFrame(rows)


def _prepare_above_minimum_analysis(
    matched_pairs: pd.DataFrame,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, object],
]:
    if len(matched_pairs) != 3026:
        raise HomologueBaselineAnalysisError(
            f"Expected 3026 directional matched-pair rows; found {len(matched_pairs)}"
        )

    above_minimum = matched_pairs.loc[
        matched_pairs["minimum_score_class"].eq("both_above_minimum")
    ].copy()

    metric_rows = []
    comparison_tables = []

    for direction in DIRECTIONS:
        group = above_minimum.loc[above_minimum["direction"].eq(direction)].copy()

        if len(group) != 478:
            raise HomologueBaselineAnalysisError(
                f"{direction}: expected 478 above-minimum pairs; found {len(group)}"
            )

        position_count = group["homologue_position_id"].nunique()

        if position_count != 63:
            raise HomologueBaselineAnalysisError(
                f"{direction}: expected 63 above-minimum homologue "
                f"positions; found {position_count}"
            )

        predictors = {
            "direct_homologue_source_effect": ("predicted_target_effect"),
            PRIMARY_COMPARATOR: (f"{PRIMARY_COMPARATOR}_prediction"),
        }

        for baseline_id, column in predictors.items():
            values = _metric_values(
                group["target_score_analysis"].to_numpy(dtype=float),
                group[column].to_numpy(dtype=float),
            )

            for metric, value in values.items():
                metric_rows.append(
                    {
                        "direction": direction,
                        "population": "both_above_minimum",
                        "baseline_id": baseline_id,
                        "metric": metric,
                        "point_estimate": value,
                        "n_pairs": len(group),
                        "n_homologue_positions": position_count,
                    }
                )

        comparison_tables.append(
            _above_minimum_paired_bootstrap(
                group,
                replicates=bootstrap_replicates,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.concat(
        comparison_tables,
        ignore_index=True,
    )

    unique_pairs = matched_pairs.loc[
        matched_pairs["direction"].eq("NDM-1_to_VIM-2")
    ].copy()

    discordant = unique_pairs.loc[
        unique_pairs["minimum_score_class"].isin(
            {
                "source_at_minimum",
                "target_at_minimum",
            }
        )
    ].copy()

    if len(discordant) != 306:
        raise HomologueBaselineAnalysisError(
            f"Expected 306 discordant minimum-score pairs; found {len(discordant)}"
        )

    discordant["minimum_score_switch_direction"] = np.where(
        discordant["minimum_score_class"].eq("source_at_minimum"),
        "NDM_at_minimum_VIM_above_minimum",
        "NDM_above_minimum_VIM_at_minimum",
    )

    discordant_summary = (
        discordant.groupby(
            "minimum_score_switch_direction",
            as_index=False,
        )
        .agg(
            n_pairs=("homologue_position_id", "size"),
            n_homologue_positions=(
                "homologue_position_id",
                "nunique",
            ),
        )
        .sort_values("minimum_score_switch_direction")
        .reset_index(drop=True)
    )

    switch_counts = dict(
        zip(
            discordant_summary["minimum_score_switch_direction"],
            discordant_summary["n_pairs"],
        )
    )

    if switch_counts.get("NDM_at_minimum_VIM_above_minimum") != 118:
        raise HomologueBaselineAnalysisError(
            "Expected 118 NDM-at-minimum/VIM-above-minimum substitutions"
        )

    if switch_counts.get("NDM_above_minimum_VIM_at_minimum") != 188:
        raise HomologueBaselineAnalysisError(
            "Expected 188 NDM-above-minimum/VIM-at-minimum substitutions"
        )

    summary = {
        "above_minimum_population": "both_above_minimum",
        "above_minimum_pair_count_per_direction": 478,
        "above_minimum_homologue_position_count": 63,
        "above_minimum_reference_comparator": PRIMARY_COMPARATOR,
        "above_minimum_bootstrap_replicates": bootstrap_replicates,
        "above_minimum_bootstrap_seeds": ABOVE_MINIMUM_BOOTSTRAP_SEEDS,
        "above_minimum_resampling_unit": "homologue_position_id",
        "discordant_minimum_score_pair_count": len(discordant),
        "ndm_at_minimum_vim_above_minimum_pair_count": 118,
        "ndm_above_minimum_vim_at_minimum_pair_count": 188,
    }

    return (
        metrics,
        comparisons,
        discordant,
        discordant_summary,
        summary,
    )


def run_homologue_baseline_analysis(
    repo_root: Path,
) -> Mapping[str, object]:
    baseline_dir = repo_root / "data/interim/baseline_evaluation"

    direct = pd.read_csv(baseline_dir / "homologue_direct_transfer_predictions.csv")
    baseline_predictions = pd.read_csv(
        baseline_dir / "baseline_predictions.csv",
        dtype={"evaluation_iteration": "string"},
    )
    observations = pd.read_csv(repo_root / "data/processed/missense_observations.csv")

    direction_tables = [
        _prepare_direction_table(
            direction,
            direct,
            baseline_predictions,
            observations,
        )
        for direction in DIRECTIONS
    ]

    matched_pairs = pd.concat(
        direction_tables,
        ignore_index=True,
    )

    matched_metrics = pd.DataFrame(
        [row for table in direction_tables for row in _matched_metric_rows(table)]
    )

    paired_comparisons = pd.DataFrame(
        [row for table in direction_tables for row in _paired_bootstrap_rows(table)]
    )

    sensitivity = pd.DataFrame(
        [row for table in direction_tables for row in _sensitivity_rows(table)]
    )

    floor_counts = (
        matched_pairs.groupby(
            ["direction", "minimum_score_class"],
            observed=True,
        )
        .agg(
            n_pairs=("target_observation_id", "size"),
            n_homologue_positions=(
                "homologue_position_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    (
        above_minimum_metrics,
        above_minimum_comparisons,
        discordant,
        discordant_summary,
        above_minimum_summary,
    ) = _prepare_above_minimum_analysis(matched_pairs)

    output_dir = baseline_dir / "homologue_analysis"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_csv(
        matched_pairs,
        output_dir / "matched_homologue_predictions.csv",
    )
    _write_csv(
        matched_metrics,
        output_dir / "matched_homologue_metrics.csv",
    )
    _write_csv(
        paired_comparisons,
        output_dir / "paired_baseline_comparisons.csv",
    )
    _write_csv(
        sensitivity,
        output_dir / "homologue_minimum_score_sensitivity.csv",
    )
    _write_csv(
        floor_counts,
        output_dir / "homologue_minimum_score_counts.csv",
    )
    _write_csv(
        above_minimum_metrics,
        output_dir / "above_minimum_matched_metrics.csv",
    )
    _write_csv(
        above_minimum_comparisons,
        output_dir / "above_minimum_paired_comparisons.csv",
    )
    _write_csv(
        discordant,
        output_dir / "discordant_minimum_score_pairs.csv",
    )
    _write_csv(
        discordant_summary,
        output_dir / "discordant_minimum_score_summary.csv",
    )

    summary = {
        "status": "PASS",
        "analysis_is_post_hoc_and_does_not_refit_models": True,
        "primary_comparator": PRIMARY_COMPARATOR,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_unit": "homologue_position_id",
        "pair_count_per_direction": 1513,
        "homologue_position_count": 81,
        "score_analysis_remains_primary": True,
        "score_original_role": "sensitivity_analysis_only",
        **above_minimum_summary,
        "outputs": [
            "matched_homologue_predictions.csv",
            "matched_homologue_metrics.csv",
            "paired_baseline_comparisons.csv",
            "homologue_minimum_score_sensitivity.csv",
            "homologue_minimum_score_counts.csv",
            "above_minimum_matched_metrics.csv",
            "above_minimum_paired_comparisons.csv",
            "discordant_minimum_score_pairs.csv",
            "discordant_minimum_score_summary.csv",
            "homologue_analysis_summary.json",
        ],
    }

    (output_dir / "homologue_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    return summary
