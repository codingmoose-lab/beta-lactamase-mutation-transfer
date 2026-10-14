from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

OUTPUT_DIR_NAME = "model_performance_summary"

CORE_SCHEME_IDS = (
    "ndm1_position_nested5",
    "vim2_2020_position_nested5",
    "ndm1_to_vim2_amp_source_cv5",
    "vim2_to_ndm1_amp_source_cv5",
    "vim2_2022_mutation_nested5",
)

CORE_SCHEME_LABELS = {
    "ndm1_position_nested5": "NDM-1 unseen-position regression",
    "vim2_2020_position_nested5": "VIM-2 2020 unseen-position regression",
    "ndm1_to_vim2_amp_source_cv5": "NDM-1 to VIM-2 ampicillin transfer",
    "vim2_to_ndm1_amp_source_cv5": "VIM-2 to NDM-1 ampicillin transfer",
    "vim2_2022_mutation_nested5": "VIM-2 2022 unseen-mutation regression",
}

METRIC_COLUMNS = (
    "spearman",
    "pearson",
    "mae",
    "rmse",
)

MODEL_METRIC_SOURCES = (
    (
        "baseline",
        "data/interim/baseline_evaluation/baseline_metrics.csv",
    ),
    (
        "sequence_neighbourhood",
        "data/interim/sequence_neighbourhood_evaluation/"
        "sequence_neighbourhood_metrics.csv",
    ),
    (
        "sequence_plm",
        "data/interim/sequence_plm_evaluation/sequence_plm_metrics.csv",
    ),
    (
        "experimental_structure",
        "data/interim/experimental_structure_evaluation/"
        "experimental_structure_metrics.csv",
    ),
    (
        "residue_neighbourhood",
        "data/interim/residue_neighbourhood_evaluation/"
        "residue_neighbourhood_metrics.csv",
    ),
    (
        "residue_graph",
        "data/interim/residue_graph_neural_evaluation/residue_graph_neural_metrics.csv",
    ),
    (
        "residue_geometry",
        "data/interim/residue_geometry_neural_evaluation/"
        "residue_geometry_neural_metrics.csv",
    ),
    (
        "sequence_residue_graph",
        "data/interim/sequence_residue_graph_neural_evaluation/"
        "sequence_residue_graph_metrics.csv",
    ),
    (
        "sequence_structure",
        "data/interim/sequence_structure_complementarity/"
        "sequence_structure_metrics.csv",
    ),
    (
        "atomic_local_structure",
        "data/interim/atomic_local_voxel_evaluation/atomic_local_voxel_metrics.csv",
    ),
)

PAIRED_COMPARISON_SOURCES = (
    (
        "sequence_models",
        "data/interim/sequence_model_comparison/sequence_model_paired_comparisons.csv",
    ),
    (
        "experimental_structure",
        "data/interim/experimental_structure_evaluation/"
        "experimental_structure_paired_comparisons.csv",
    ),
    (
        "residue_neighbourhood",
        "data/interim/residue_neighbourhood_evaluation/"
        "residue_neighbourhood_paired_comparisons.csv",
    ),
    (
        "residue_graph",
        "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_paired_comparisons.csv",
    ),
    (
        "residue_geometry",
        "data/interim/residue_geometry_neural_evaluation/"
        "residue_geometry_neural_paired_comparisons.csv",
    ),
    (
        "sequence_residue_graph",
        "data/interim/sequence_residue_graph_neural_evaluation/"
        "sequence_residue_graph_paired_comparisons.csv",
    ),
    (
        "sequence_structure",
        "data/interim/sequence_structure_complementarity/"
        "sequence_structure_paired_comparisons.csv",
    ),
    (
        "atomic_local_structure",
        "data/interim/atomic_local_voxel_evaluation/"
        "atomic_local_voxel_paired_comparisons.csv",
    ),
    (
        "sequence_model_sensitivity",
        "data/interim/sequence_model_sensitivity/esm2_esmc_paired_comparisons.csv",
    ),
)


class ModelPerformanceSummaryError(RuntimeError):
    pass


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    frame.to_csv(
        path,
        index=False,
    )


def _write_json(
    value: Mapping[str, object],
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_csv(
    repo_root: Path,
    relative: str,
) -> pd.DataFrame:
    path = repo_root / relative

    if not path.is_file():
        raise ModelPerformanceSummaryError(
            f"Required completed-result file is absent: {relative}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def _select_final_metric_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "scheme_id",
        "model_id",
        *METRIC_COLUMNS,
    }

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ModelPerformanceSummaryError(
            f"Metric table lacks required columns: {missing}"
        )

    selected = frame[frame["scheme_id"].astype(str).isin(CORE_SCHEME_IDS)].copy()

    if "evaluation_iteration" in selected.columns:
        iteration = selected["evaluation_iteration"].astype(str)

        selected = selected[
            iteration.isin(
                (
                    "pooled",
                    "target",
                    "final",
                )
            )
        ].copy()

    if selected.empty:
        raise ModelPerformanceSummaryError(
            "No final metric rows were found for the central evaluation schemes"
        )

    return selected


def _combine_metrics(
    repo_root: Path,
) -> pd.DataFrame:
    frames = []

    for source_name, relative in MODEL_METRIC_SOURCES:
        frame = _read_csv(
            repo_root,
            relative,
        )

        if source_name == "baseline":
            if "baseline_id" not in frame.columns or "model_id" in frame.columns:
                raise ModelPerformanceSummaryError(
                    "Baseline metric identifier contract changed"
                )

            frame = frame.rename(
                columns={
                    "baseline_id": "model_id",
                }
            )

        frame = _select_final_metric_rows(frame)

        keep = [
            "scheme_id",
            "model_id",
            *METRIC_COLUMNS,
        ]

        for optional in (
            "evaluation_iteration",
            "n_observations",
            "n_groups",
        ):
            if optional in frame.columns:
                keep.append(optional)

        frame = frame[keep].copy()
        frame["source_result"] = source_name
        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    for metric in METRIC_COLUMNS:
        combined[metric] = pd.to_numeric(
            combined[metric],
            errors="coerce",
        )

    combined["scheme_label"] = combined["scheme_id"].map(CORE_SCHEME_LABELS)

    population_pairs = {
        "ndm1_position_nested5": {
            (45369, 270): "full_evaluation_population",
            (40248, 238): "experimental_structure_matched_population",
        },
        "ndm1_to_vim2_amp_source_cv5": {
            (5023, 267): "full_evaluation_population",
            (4324, 230): "experimental_structure_matched_population",
        },
        "vim2_2020_position_nested5": {
            (45222, 267): "full_evaluation_population",
            (38928, 230): "experimental_structure_matched_population",
        },
        "vim2_2022_mutation_nested5": {
            (35090, 5015): "full_evaluation_population",
            (30232, 4321): "experimental_structure_matched_population",
        },
        "vim2_to_ndm1_amp_source_cv5": {
            (5041, 270): "full_evaluation_population",
            (4472, 238): "experimental_structure_matched_population",
        },
    }

    population_labels = []

    for row in combined.itertuples(index=False):
        scheme_id = str(row.scheme_id)

        try:
            n_observations = int(row.n_observations)
            n_groups = int(row.n_groups)
        except (TypeError, ValueError):
            raise ModelPerformanceSummaryError(
                "Model-performance rows require observation and evaluation-group counts"
            ) from None

        label = population_pairs.get(
            scheme_id,
            {},
        ).get(
            (
                n_observations,
                n_groups,
            )
        )

        if label is None:
            raise ModelPerformanceSummaryError(
                "Unrecognized evaluation population for "
                f"{scheme_id}: "
                f"{n_observations} observations, "
                f"{n_groups} groups"
            )

        population_labels.append(label)

    combined["evaluation_population"] = population_labels

    return combined


def _deduplicate_metrics(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "scheme_id",
        "evaluation_population",
        "model_id",
        "n_observations",
        "n_groups",
    ]

    for _, group in combined.groupby(
        keys,
        dropna=False,
    ):
        for metric in METRIC_COLUMNS:
            values = group[metric].dropna().to_numpy(dtype=float)

            if len(values) > 1 and float(np.max(values) - np.min(values)) > 5e-6:
                raise ModelPerformanceSummaryError(
                    "Repeated model metrics disagree within "
                    "the same evaluation population"
                )

    source_order = {name: index for index, (name, _) in enumerate(MODEL_METRIC_SOURCES)}

    ranked = combined.copy()
    ranked["_source_order"] = ranked["source_result"].map(source_order)

    return (
        ranked.sort_values(
            [
                "scheme_id",
                "model_id",
                "_source_order",
            ],
            kind="stable",
        )
        .drop_duplicates(
            keys,
            keep="first",
        )
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )


def _rank_models(
    final_metrics: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    required = {
        "scheme_id",
        "scheme_label",
        "evaluation_population",
        "model_id",
        "n_observations",
        "n_groups",
        "source_result",
        *METRIC_COLUMNS,
    }

    missing = sorted(required.difference(final_metrics.columns))

    if missing:
        raise ModelPerformanceSummaryError(
            f"Final model-performance table lacks ranking columns: {missing}"
        )

    group_columns = [
        "scheme_id",
        "evaluation_population",
        "n_observations",
        "n_groups",
    ]

    for group_key, population in final_metrics.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        (
            scheme_id,
            evaluation_population,
            n_observations,
            n_groups,
        ) = group_key

        if population.empty:
            raise ModelPerformanceSummaryError(
                f"No model metrics were found for {scheme_id} / {evaluation_population}"
            )

        for metric in METRIC_COLUMNS:
            ascending = metric in (
                "mae",
                "rmse",
            )

            ranked = population.sort_values(
                metric,
                ascending=ascending,
                na_position="last",
                kind="stable",
            ).copy()

            ranked["metric"] = metric
            ranked["metric_value"] = ranked[metric]
            ranked["rank"] = np.arange(
                1,
                len(ranked) + 1,
                dtype=int,
            )

            records.append(
                ranked[
                    [
                        "scheme_id",
                        "scheme_label",
                        "evaluation_population",
                        "n_observations",
                        "n_groups",
                        "model_id",
                        "metric",
                        "metric_value",
                        "rank",
                        "source_result",
                    ]
                ]
            )

    if not records:
        raise ModelPerformanceSummaryError(
            "No population-specific model rankings were produced"
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


def _paired_summary(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []

    for source_name, relative in PAIRED_COMPARISON_SOURCES:
        frame = _read_csv(
            repo_root,
            relative,
        ).copy()

        required = {
            "scheme_id",
            "metric",
            "paired_difference",
            "ci_lower",
            "ci_upper",
        }

        missing = sorted(required.difference(frame.columns))

        if missing:
            raise ModelPerformanceSummaryError(
                f"Paired-comparison table lacks required columns: {missing}"
            )

        lower = pd.to_numeric(
            frame["ci_lower"],
            errors="coerce",
        )

        upper = pd.to_numeric(
            frame["ci_upper"],
            errors="coerce",
        )

        frame["interval_excludes_zero"] = (lower > 0) | (upper < 0)

        frame["comparison_source"] = source_name

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if len(combined) != 460:
        raise ModelPerformanceSummaryError(
            f"Expected 460 paired-comparison rows, found {len(combined)}"
        )

    excluded = int(combined["interval_excludes_zero"].sum())

    if excluded != 299:
        raise ModelPerformanceSummaryError(
            f"Expected 299 intervals excluding zero, found {excluded}"
        )

    summary = combined.groupby(
        [
            "comparison_source",
            "metric",
        ],
        as_index=False,
    ).agg(
        comparison_count=(
            "metric",
            "size",
        ),
        intervals_excluding_zero=(
            "interval_excludes_zero",
            "sum",
        ),
    )

    summary["intervals_including_zero"] = (
        summary["comparison_count"] - summary["intervals_excluding_zero"]
    )

    return (
        combined,
        summary,
    )


def _biological_counts(
    repo_root: Path,
) -> pd.DataFrame:
    structural = _read_csv(
        repo_root,
        "data/interim/"
        "biological_model_difference_analysis/"
        "biological_model_difference_"
        "structural_associations.csv",
    )

    categories = _read_csv(
        repo_root,
        "data/interim/"
        "biological_model_difference_analysis/"
        "biological_model_difference_"
        "category_summaries.csv",
    )

    concordance = _read_csv(
        repo_root,
        "data/interim/"
        "biological_model_difference_analysis/"
        "biological_model_difference_"
        "position_concordance.csv",
    )

    primary = structural[structural["evidence_role"].astype(str).eq("primary")].copy()

    def interval_excludes_zero(
        frame: pd.DataFrame,
        lower_column: str,
        upper_column: str,
    ) -> pd.Series:
        lower = pd.to_numeric(
            frame[lower_column],
            errors="coerce",
        )

        upper = pd.to_numeric(
            frame[upper_column],
            errors="coerce",
        )

        return (lower > 0) | (upper < 0)

    structural_supported = interval_excludes_zero(
        structural,
        "ci_lower",
        "ci_upper",
    )

    primary_supported = interval_excludes_zero(
        primary,
        "ci_lower",
        "ci_upper",
    )

    category_supported = interval_excludes_zero(
        categories,
        "ci_lower",
        "ci_upper",
    )

    concordance_supported = interval_excludes_zero(
        concordance,
        "spearman_ci_lower",
        "spearman_ci_upper",
    )

    frame = pd.DataFrame(
        [
            {
                "result": "structural_associations",
                "row_count": int(len(structural)),
                "intervals_excluding_zero": int(structural_supported.sum()),
            },
            {
                "result": "primary_structural_associations",
                "row_count": int(len(primary)),
                "intervals_excluding_zero": int(primary_supported.sum()),
            },
            {
                "result": "biological_category_summaries",
                "row_count": int(len(categories)),
                "intervals_excluding_zero": int(category_supported.sum()),
            },
            {
                "result": "position_concordance",
                "row_count": int(len(concordance)),
                "intervals_excluding_zero": int(concordance_supported.sum()),
            },
        ]
    )

    frame["interpretation_scope"] = "descriptive_association"

    expected = {
        "structural_associations": 125,
        "primary_structural_associations": 75,
        "biological_category_summaries": 225,
        "position_concordance": 20,
    }

    for name, expected_rows in expected.items():
        observed = int(
            frame.loc[
                frame["result"].eq(name),
                "row_count",
            ].iloc[0]
        )

        if observed != expected_rows:
            raise ModelPerformanceSummaryError(f"{name} row count changed: {observed}")

    return frame


def _seed_direction_change(
    repo_root: Path,
) -> pd.DataFrame:
    frame = _read_csv(
        repo_root,
        "data/interim/training_seed_stability/training_seed_comparison_summary.csv",
    )

    changed = frame[
        frame["mean_direction_changed"].astype(str).str.lower().eq("true")
    ].copy()

    if len(frame) != 60 or len(changed) != 1:
        raise ModelPerformanceSummaryError(
            "Training-seed comparison summary no longer matches the reviewed result"
        )

    return changed.reset_index(drop=True)


def inspect_model_performance_summary_inputs(
    repo_root: Path,
) -> Mapping[str, object]:
    final_metrics = _deduplicate_metrics(_combine_metrics(repo_root))

    _, paired = _paired_summary(repo_root)

    biology = _biological_counts(repo_root)

    seed_changes = _seed_direction_change(repo_root)

    scheme_counts = (
        final_metrics.groupby("scheme_id").size().astype(int).sort_index().to_dict()
    )

    if set(scheme_counts) != set(CORE_SCHEME_IDS):
        raise ModelPerformanceSummaryError("Central evaluation scheme coverage changed")

    expected_model_sources = {
        "experimental_structure_descriptor_ridge": "experimental_structure",
        "residue_graph_neural_regressor": "residue_graph",
        "egnn_residue_regressor": "residue_geometry",
        "atomic_local_voxel_cnn": "atomic_local_structure",
    }

    for model_id, expected_source in expected_model_sources.items():
        observed_sources = set(
            final_metrics.loc[
                final_metrics["model_id"].astype(str).eq(model_id),
                "source_result",
            ].astype(str)
        )
        if observed_sources and observed_sources != {expected_source}:
            raise ModelPerformanceSummaryError(
                f"{model_id} source-result assignment changed: "
                f"{sorted(observed_sources)}"
            )

    return {
        "status": "PASS",
        "analysis_scope": "completed_model_performance_summary",
        "analysis_role": "post_evaluation_result_summary",
        "model_fitting_performed": False,
        "parameter_selection_performed": False,
        "prediction_values_modified": False,
        "final_labels_used_for_model_selection": False,
        "core_scheme_count": len(CORE_SCHEME_IDS),
        "final_model_scheme_rows": int(len(final_metrics)),
        "model_count": int(final_metrics["model_id"].nunique()),
        "evaluation_population_count": int(
            final_metrics["evaluation_population"].nunique()
        ),
        "model_source_assignments_checked": True,
        "evaluation_populations": sorted(
            final_metrics["evaluation_population"].astype(str).unique().tolist()
        ),
        "scheme_model_counts": scheme_counts,
        "paired_comparison_count": int(paired["comparison_count"].sum()),
        "paired_intervals_excluding_zero": int(
            paired["intervals_excluding_zero"].sum()
        ),
        "paired_intervals_including_zero": int(
            paired["intervals_including_zero"].sum()
        ),
        "biological_summary_row_count": int(len(biology)),
        "training_seed_direction_change_count": int(len(seed_changes)),
        "biological_interpretation_scope": "descriptive_association_only",
    }


def run_model_performance_summary(
    repo_root: Path,
) -> Mapping[str, object]:
    audit = inspect_model_performance_summary_inputs(repo_root)

    final_metrics = _deduplicate_metrics(_combine_metrics(repo_root))

    rankings = _rank_models(final_metrics)

    paired_details, paired = _paired_summary(repo_root)

    biology = _biological_counts(repo_root)

    seed_changes = _seed_direction_change(repo_root)

    output_dir = repo_root / "data" / "interim" / OUTPUT_DIR_NAME

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_csv(
        final_metrics,
        output_dir / "model_performance_metrics.csv",
    )

    _write_csv(
        rankings,
        output_dir / "model_performance_rankings.csv",
    )

    _write_csv(
        paired,
        output_dir / "model_pair_uncertainty_summary.csv",
    )

    _write_csv(
        biology,
        output_dir / "biological_result_counts.csv",
    )

    _write_csv(
        seed_changes,
        output_dir / "training_seed_direction_change.csv",
    )

    summary = {
        **audit,
        "status": "PASS",
        "ranking_row_count": int(len(rankings)),
        "paired_detail_rows_reviewed": int(len(paired_details)),
        "outputs": [
            "model_performance_metrics.csv",
            "model_performance_rankings.csv",
            "model_pair_uncertainty_summary.csv",
            "biological_result_counts.csv",
            "training_seed_direction_change.csv",
            "model_performance_summary.json",
        ],
    }

    _write_json(
        summary,
        output_dir / "model_performance_summary.json",
    )

    return summary
