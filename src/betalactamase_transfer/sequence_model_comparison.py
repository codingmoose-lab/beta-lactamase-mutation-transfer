from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

MODEL_IDS = (
    "mutation_condition_ridge",
    "sequence_neighbourhood_ridge",
    "esmc_site_delta_ridge",
)

MODEL_PAIRS = (
    ("sequence_neighbourhood_ridge", "mutation_condition_ridge"),
    ("esmc_site_delta_ridge", "mutation_condition_ridge"),
    ("esmc_site_delta_ridge", "sequence_neighbourhood_ridge"),
)

METRICS = ("spearman", "pearson", "mae", "rmse")
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_sequence_model_comparison_bootstrap"
BOOTSTRAP_QUANTILES = (0.025, 0.975)

EXPECTED_SCHEME_OBSERVATION_COUNTS = {
    "ndm1_position_nested5": 45369,
    "ndm1_to_vim2_amp_source_cv5": 5023,
    "vim2_2020_position_nested5": 45222,
    "vim2_2022_mutation_nested5": 35090,
    "vim2_to_ndm1_amp_source_cv5": 5041,
}
EXPECTED_SCHEME_ROLES = {
    "ndm1_position_nested5": "outer_test",
    "ndm1_to_vim2_amp_source_cv5": "target_test",
    "vim2_2020_position_nested5": "outer_test",
    "vim2_2022_mutation_nested5": "outer_test",
    "vim2_to_ndm1_amp_source_cv5": "target_test",
}
EXPECTED_MATCHED_OBSERVATION_COUNT = sum(EXPECTED_SCHEME_OBSERVATION_COUNTS.values())

COMPARISON_KEY = (
    "scheme_id",
    "evaluation_iteration",
    "evaluation_role",
    "observation_id",
)

MATCHED_FIELDS = (
    "task_id",
    "evaluation_group_id",
    "protein_id",
    "assay_position",
    "wild_type",
    "mutant",
    "antibiotic",
    "concentration",
    "temperature_c",
    "y_true",
)


class SequenceModelComparisonError(RuntimeError):
    """Raised when matched sequence-model comparison cannot be completed safely."""


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise SequenceModelComparisonError("Metrics require equal non-empty arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise SequenceModelComparisonError(
            "Metrics require finite truth and prediction values"
        )

    error = y_pred - y_true
    true_constant = np.allclose(y_true, y_true[0], rtol=0.0, atol=1e-12)
    pred_constant = np.allclose(y_pred, y_pred[0], rtol=0.0, atol=1e-12)

    if true_constant or pred_constant:
        spearman = float("nan")
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
        pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])

    return {
        "spearman": spearman,
        "pearson": pearson,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _metric_difference(
    metric: str,
    model_a_value: float,
    model_b_value: float,
) -> float:
    if metric in {"spearman", "pearson"}:
        return model_a_value - model_b_value
    if metric in {"mae", "rmse"}:
        return model_b_value - model_a_value
    raise SequenceModelComparisonError(f"Unsupported metric: {metric}")


def _load_predictions(
    repo_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    sequence_path = (
        repo_root
        / "data/interim/sequence_neighbourhood_evaluation"
        / "sequence_neighbourhood_predictions.csv"
    )
    esmc_path = (
        repo_root
        / "data/interim/sequence_plm_evaluation"
        / "sequence_plm_predictions.csv"
    )

    for path in (sequence_path, esmc_path):
        if not path.is_file():
            raise SequenceModelComparisonError(
                f"Required prediction file is missing: {path}"
            )

    sequence = pd.read_csv(
        sequence_path,
        dtype={"evaluation_iteration": "string"},
        low_memory=False,
    )
    esmc = pd.read_csv(
        esmc_path,
        dtype={"evaluation_iteration": "string"},
        low_memory=False,
    )

    if set(sequence["model_id"].astype(str)) != {
        "mutation_condition_ridge",
        "sequence_neighbourhood_ridge",
    }:
        raise SequenceModelComparisonError(
            "Unexpected models in sequence-neighbourhood predictions"
        )
    if set(esmc["model_id"].astype(str)) != {"esmc_site_delta_ridge"}:
        raise SequenceModelComparisonError("Unexpected models in ESMC predictions")

    frames = {
        "mutation_condition_ridge": sequence[
            sequence["model_id"].eq("mutation_condition_ridge")
        ].copy(),
        "sequence_neighbourhood_ridge": sequence[
            sequence["model_id"].eq("sequence_neighbourhood_ridge")
        ].copy(),
        "esmc_site_delta_ridge": esmc.copy(),
    }
    digests = {
        "sequence_neighbourhood_predictions_sha256": _sha256_file(sequence_path),
        "sequence_plm_predictions_sha256": _sha256_file(esmc_path),
    }
    return frames, digests


def _validate_prediction_frame(model_id: str, frame: pd.DataFrame) -> None:
    required = set(COMPARISON_KEY) | set(MATCHED_FIELDS) | {"model_id", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise SequenceModelComparisonError(
            f"{model_id} predictions are missing columns: {sorted(missing)}"
        )
    if frame.empty:
        raise SequenceModelComparisonError(f"{model_id} prediction table is empty")
    if set(frame["model_id"].astype(str)) != {model_id}:
        raise SequenceModelComparisonError(
            f"Unexpected model identifier in {model_id} predictions"
        )
    if frame.duplicated(list(COMPARISON_KEY)).any():
        raise SequenceModelComparisonError(
            f"Duplicate comparison keys in {model_id} predictions"
        )
    if frame["y_true"].isna().any() or frame["y_pred"].isna().any():
        raise SequenceModelComparisonError(
            f"Missing truth or prediction values for {model_id}"
        )
    if not np.isfinite(frame["y_true"].to_numpy(dtype=float)).all():
        raise SequenceModelComparisonError(f"Non-finite truth values for {model_id}")
    if not np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all():
        raise SequenceModelComparisonError(f"Non-finite predictions for {model_id}")


def _matched_predictions(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    for model_id in MODEL_IDS:
        if model_id not in frames:
            raise SequenceModelComparisonError(
                f"Missing prediction table for {model_id}"
            )
        _validate_prediction_frame(model_id, frames[model_id])

    reference = frames[MODEL_IDS[0]].copy()
    base_columns = list(COMPARISON_KEY) + list(MATCHED_FIELDS)
    matched = reference[base_columns].copy()
    matched[MODEL_IDS[0]] = reference["y_pred"].to_numpy(dtype=float)
    expected_count = len(reference)

    for model_id in MODEL_IDS[1:]:
        candidate = frames[model_id]
        candidate_subset = candidate[
            list(COMPARISON_KEY) + list(MATCHED_FIELDS) + ["y_pred"]
        ].copy()
        rename_map = {field: f"{field}__candidate" for field in MATCHED_FIELDS}
        rename_map["y_pred"] = model_id
        candidate_subset = candidate_subset.rename(columns=rename_map)

        matched = matched.merge(
            candidate_subset,
            on=list(COMPARISON_KEY),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if len(matched) != expected_count:
            raise SequenceModelComparisonError(
                f"Prediction population mismatch for {model_id}"
            )
        if not matched["_merge"].eq("both").all():
            raise SequenceModelComparisonError(
                f"Comparison-key mismatch for {model_id}"
            )
        matched = matched.drop(columns="_merge")

        for field in MATCHED_FIELDS:
            candidate_field = f"{field}__candidate"
            left = matched[field]
            right = matched[candidate_field]
            if pd.api.types.is_numeric_dtype(left):
                same = bool(
                    np.allclose(
                        left.to_numpy(dtype=float),
                        right.to_numpy(dtype=float),
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                )
            else:
                same = left.reset_index(drop=True).equals(right.reset_index(drop=True))
            if not same:
                raise SequenceModelComparisonError(
                    f"Matched field disagrees for {model_id}: {field}"
                )
            matched = matched.drop(columns=candidate_field)

    if matched.duplicated(list(COMPARISON_KEY)).any():
        raise SequenceModelComparisonError(
            "Matched predictions contain duplicate comparison keys"
        )
    return matched


def _validate_matched_population(matched: pd.DataFrame) -> None:
    if len(matched) != EXPECTED_MATCHED_OBSERVATION_COUNT:
        raise SequenceModelComparisonError(
            "Matched prediction count does not match the expected comparison population"
        )

    counts = matched.groupby("scheme_id", sort=True).size().to_dict()
    if counts != EXPECTED_SCHEME_OBSERVATION_COUNTS:
        raise SequenceModelComparisonError(
            "Per-scheme prediction counts do not match the expected "
            "comparison population"
        )

    if matched["evaluation_group_id"].isna().any():
        raise SequenceModelComparisonError(
            "Matched comparison contains missing evaluation-group identifiers"
        )

    roles = matched.groupby("scheme_id", sort=True)["evaluation_role"].unique()
    for scheme_id, expected_role in EXPECTED_SCHEME_ROLES.items():
        observed = sorted(str(value) for value in roles.loc[scheme_id].tolist())
        if observed != [expected_role]:
            raise SequenceModelComparisonError(
                f"Unexpected evaluation role for {scheme_id}: {observed}"
            )

    group_iterations = matched.groupby(
        ["scheme_id", "evaluation_group_id"],
        sort=False,
    )["evaluation_iteration"].nunique(dropna=False)
    if (group_iterations != 1).any():
        raise SequenceModelComparisonError(
            "An evaluation group appears in multiple test iterations"
        )


def _model_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    for scheme_id, group in matched.groupby("scheme_id", sort=True):
        for model_id in MODEL_IDS:
            values = _metric_values(
                group["y_true"].to_numpy(dtype=float),
                group[model_id].to_numpy(dtype=float),
            )
            for metric in METRICS:
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_id": model_id,
                        "metric": metric,
                        "value": values[metric],
                        "n_observations": len(group),
                        "n_groups": group["evaluation_group_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def _paired_comparisons(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    for scheme_id, group in matched.groupby("scheme_id", sort=True):
        group_ids = sorted(group["evaluation_group_id"].unique().tolist())
        if not group_ids:
            raise SequenceModelComparisonError(f"No evaluation groups for {scheme_id}")
        blocks = {
            group_id: group[group["evaluation_group_id"].eq(group_id)].copy()
            for group_id in group_ids
        }

        for model_a, model_b in MODEL_PAIRS:
            point_a = _metric_values(
                group["y_true"].to_numpy(dtype=float),
                group[model_a].to_numpy(dtype=float),
            )
            point_b = _metric_values(
                group["y_true"].to_numpy(dtype=float),
                group[model_b].to_numpy(dtype=float),
            )
            distributions = {metric: [] for metric in METRICS}
            rng = np.random.default_rng(
                _numeric_seed(BOOTSTRAP_SEED, scheme_id, model_a, model_b)
            )

            for _ in range(BOOTSTRAP_REPLICATES):
                sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
                sampled = pd.concat(
                    [blocks[group_id] for group_id in sampled_ids],
                    ignore_index=True,
                )
                metrics_a = _metric_values(
                    sampled["y_true"].to_numpy(dtype=float),
                    sampled[model_a].to_numpy(dtype=float),
                )
                metrics_b = _metric_values(
                    sampled["y_true"].to_numpy(dtype=float),
                    sampled[model_b].to_numpy(dtype=float),
                )
                for metric in METRICS:
                    difference = _metric_difference(
                        metric,
                        metrics_a[metric],
                        metrics_b[metric],
                    )
                    if np.isfinite(difference):
                        distributions[metric].append(difference)

            for metric in METRICS:
                point = _metric_difference(metric, point_a[metric], point_b[metric])
                values = np.asarray(distributions[metric], dtype=float)
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "model_a_value": point_a[metric],
                        "model_b_value": point_b[metric],
                        "paired_difference": point,
                        "ci_lower": (
                            float(np.quantile(values, BOOTSTRAP_QUANTILES[0]))
                            if len(values)
                            else float("nan")
                        ),
                        "ci_upper": (
                            float(np.quantile(values, BOOTSTRAP_QUANTILES[1]))
                            if len(values)
                            else float("nan")
                        ),
                        "n_observations": len(group),
                        "n_groups": len(group_ids),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_valid_replicates": len(values),
                        "resampling_unit": "evaluation_group_id",
                        "interval_method": "percentile",
                        "difference_interpretation": "positive_favours_model_a",
                    }
                )
    return pd.DataFrame(rows)


def prepare_sequence_model_comparison(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    frames, input_digests = _load_predictions(repo_root)
    matched = _matched_predictions(frames)
    _validate_matched_population(matched)
    metrics = _model_metrics(matched)
    comparisons = _paired_comparisons(matched)

    summary = {
        "model_ids": list(MODEL_IDS),
        "model_pairs": [
            {"model_a": model_a, "model_b": model_b} for model_a, model_b in MODEL_PAIRS
        ],
        "scheme_ids": sorted(matched["scheme_id"].astype(str).unique().tolist()),
        "matched_observation_count": int(len(matched)),
        "metric_row_count": int(len(metrics)),
        "paired_comparison_row_count": int(len(comparisons)),
        "metrics": list(METRICS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "bootstrap_interval_method": "percentile",
        "bootstrap_quantiles": list(BOOTSTRAP_QUANTILES),
        "comparison_key": list(COMPARISON_KEY),
        "matched_fields": list(MATCHED_FIELDS),
        "difference_definition": {
            "spearman": "model_a_minus_model_b",
            "pearson": "model_a_minus_model_b",
            "mae": "model_b_minus_model_a",
            "rmse": "model_b_minus_model_a",
        },
        "difference_interpretation": "positive_favours_model_a",
        "model_refitting_performed": False,
        "model_selection_performed": False,
        "test_labels_used_for_model_selection": False,
        **input_digests,
    }
    return metrics, comparisons, summary


def write_sequence_model_comparison(repo_root: Path) -> Mapping[str, object]:
    metrics, comparisons, summary = prepare_sequence_model_comparison(repo_root)
    output_dir = repo_root / "data/interim/sequence_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics, output_dir / "sequence_model_metrics.csv")
    _write_csv(
        comparisons,
        output_dir / "sequence_model_paired_comparisons.csv",
    )
    (output_dir / "sequence_model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
