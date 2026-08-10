from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class BaselineEvaluationError(RuntimeError):
    pass


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
BASELINE_IDS = (
    "training_mean",
    "mutant_mean",
    "substitution_mean",
    "ridge_mutation_identity",
)
CONCENTRATION_BASELINE_ID = "ridge_mutation_log2_concentration"
ANTIBIOTIC_DIRECT_BASELINE_ID = "source_antibiotic_mean"
TEMPERATURE_DIRECT_BASELINE_ID = "source_temperature_matched"
CONCENTRATION_NEAREST_BASELINE_ID = "source_concentration_nearest"
CONCENTRATION_LINEAR_BASELINE_ID = "source_concentration_linear_log2"
TARGET_COLUMN = "score_analysis"
FORBIDDEN_PREDICTOR_COLUMNS = {
    "score_original",
    "score_analysis",
    "score_uncertainty",
    "protein_id",
    "assay_position",
    "observation_id",
    "mutation_group_id",
    "position_group_id",
    "experiment_group_id",
}


@dataclass(frozen=True)
class RidgeTransform:
    include_log2_concentration: bool
    concentration_mean: float = 0.0
    concentration_sd: float = 1.0


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _validate_observations(frame: pd.DataFrame) -> None:
    required = {
        "observation_id",
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
        TARGET_COLUMN,
    }
    missing = required - set(frame.columns)
    if missing:
        raise BaselineEvaluationError(f"Missing observation columns: {sorted(missing)}")
    if frame[TARGET_COLUMN].isna().any():
        raise BaselineEvaluationError("Baseline target contains missing values")
    residues = set(frame["wild_type"]) | set(frame["mutant"])
    unexpected = residues - set(AMINO_ACIDS)
    if unexpected:
        raise BaselineEvaluationError(
            "Unexpected amino-acid symbols in baseline population: "
            f"{sorted(unexpected)}"
        )


def _lookup_prediction(
    train: pd.DataFrame, test: pd.DataFrame, key_columns: Sequence[str]
) -> np.ndarray:
    global_mean = float(train[TARGET_COLUMN].mean())
    means = train.groupby(list(key_columns), sort=True)[TARGET_COLUMN].mean()
    if len(key_columns) == 1:
        keys: Iterable[object] = test[key_columns[0]].tolist()
    else:
        keys = list(test[list(key_columns)].itertuples(index=False, name=None))
    return np.asarray([float(means.get(key, global_mean)) for key in keys], dtype=float)


def _fit_ridge_transform(
    train: pd.DataFrame, *, include_log2_concentration: bool
) -> RidgeTransform:
    if not include_log2_concentration:
        return RidgeTransform(include_log2_concentration=False)
    if train["concentration"].isna().any() or (train["concentration"] <= 0).any():
        raise BaselineEvaluationError(
            "Positive concentration is required for the concentration baseline"
        )
    values = np.log2(train["concentration"].to_numpy(dtype=float))
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if not math.isfinite(sd) or sd == 0.0:
        sd = 1.0
    return RidgeTransform(
        include_log2_concentration=True,
        concentration_mean=mean,
        concentration_sd=sd,
    )


def _ridge_matrix(frame: pd.DataFrame, transform: RidgeTransform) -> np.ndarray:
    columns = [np.ones(len(frame), dtype=float)]
    wild = frame["wild_type"].to_numpy()
    mutant = frame["mutant"].to_numpy()
    for amino_acid in AMINO_ACIDS:
        columns.append((wild == amino_acid).astype(float))
    for amino_acid in AMINO_ACIDS:
        columns.append((mutant == amino_acid).astype(float))
    if transform.include_log2_concentration:
        if frame["concentration"].isna().any() or (frame["concentration"] <= 0).any():
            raise BaselineEvaluationError(
                "Positive concentration is required for the concentration baseline"
            )
        values = np.log2(frame["concentration"].to_numpy(dtype=float))
        columns.append(
            (values - transform.concentration_mean) / transform.concentration_sd
        )
    return np.column_stack(columns)


def _ridge_sufficient_statistics(
    train: pd.DataFrame,
    *,
    include_log2_concentration: bool,
) -> tuple[RidgeTransform, np.ndarray, np.ndarray]:
    transform = _fit_ridge_transform(
        train, include_log2_concentration=include_log2_concentration
    )
    amino_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    wild_index = np.asarray(
        [amino_index[str(value)] for value in train["wild_type"]], dtype=int
    )
    mutant_index = np.asarray(
        [amino_index[str(value)] for value in train["mutant"]], dtype=int
    )
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    amino_count = len(AMINO_ACIDS)
    feature_count = 1 + 2 * amino_count + int(include_log2_concentration)
    wild_offset = 1
    mutant_offset = 1 + amino_count
    gram = np.zeros((feature_count, feature_count), dtype=float)
    target_product = np.zeros(feature_count, dtype=float)

    wild_counts = np.bincount(wild_index, minlength=amino_count).astype(float)
    mutant_counts = np.bincount(mutant_index, minlength=amino_count).astype(float)
    wild_target_sums = np.bincount(wild_index, weights=y, minlength=amino_count).astype(
        float
    )
    mutant_target_sums = np.bincount(
        mutant_index, weights=y, minlength=amino_count
    ).astype(float)
    pair_codes = wild_index * amino_count + mutant_index
    pair_counts = np.bincount(pair_codes, minlength=amino_count * amino_count).reshape(
        amino_count, amino_count
    )

    gram[0, 0] = float(len(train))
    target_product[0] = float(y.sum())
    wild_slice = slice(wild_offset, wild_offset + amino_count)
    mutant_slice = slice(mutant_offset, mutant_offset + amino_count)
    gram[0, wild_slice] = wild_counts
    gram[wild_slice, 0] = wild_counts
    gram[0, mutant_slice] = mutant_counts
    gram[mutant_slice, 0] = mutant_counts
    gram[wild_slice, wild_slice] = np.diag(wild_counts)
    gram[mutant_slice, mutant_slice] = np.diag(mutant_counts)
    gram[wild_slice, mutant_slice] = pair_counts
    gram[mutant_slice, wild_slice] = pair_counts.T
    target_product[wild_slice] = wild_target_sums
    target_product[mutant_slice] = mutant_target_sums

    if include_log2_concentration:
        concentration_feature = feature_count - 1
        concentration = np.log2(train["concentration"].to_numpy(dtype=float))
        z = (concentration - transform.concentration_mean) / transform.concentration_sd
        wild_z_sums = np.bincount(wild_index, weights=z, minlength=amino_count).astype(
            float
        )
        mutant_z_sums = np.bincount(
            mutant_index, weights=z, minlength=amino_count
        ).astype(float)
        gram[0, concentration_feature] = float(z.sum())
        gram[concentration_feature, 0] = gram[0, concentration_feature]
        gram[concentration_feature, concentration_feature] = float(np.sum(z * z))
        gram[wild_slice, concentration_feature] = wild_z_sums
        gram[concentration_feature, wild_slice] = wild_z_sums
        gram[mutant_slice, concentration_feature] = mutant_z_sums
        gram[concentration_feature, mutant_slice] = mutant_z_sums
        target_product[concentration_feature] = float(np.sum(z * y))

    if not np.isfinite(gram).all() or not np.isfinite(target_product).all():
        raise BaselineEvaluationError("Ridge sufficient statistics are non-finite")
    return transform, gram, target_product


def _ridge_coefficients(
    gram: np.ndarray,
    target_product: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise BaselineEvaluationError("Ridge alpha must be positive and finite")
    penalty = np.eye(len(target_product), dtype=float)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + float(alpha) * penalty, target_product)
    if not np.isfinite(coefficients).all():
        raise BaselineEvaluationError("Ridge produced non-finite coefficients")
    return coefficients


def _ridge_predict_from_coefficients(
    frame: pd.DataFrame,
    coefficients: np.ndarray,
    transform: RidgeTransform,
) -> np.ndarray:
    amino_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    wild_index = np.asarray(
        [amino_index[str(value)] for value in frame["wild_type"]], dtype=int
    )
    mutant_index = np.asarray(
        [amino_index[str(value)] for value in frame["mutant"]], dtype=int
    )
    wild_offset = 1
    mutant_offset = 1 + len(AMINO_ACIDS)
    predictions = (
        coefficients[0]
        + coefficients[wild_offset + wild_index]
        + coefficients[mutant_offset + mutant_index]
    )
    if transform.include_log2_concentration:
        concentration = np.log2(frame["concentration"].to_numpy(dtype=float))
        z = (concentration - transform.concentration_mean) / transform.concentration_sd
        predictions = predictions + z * coefficients[-1]
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise BaselineEvaluationError("Ridge produced non-finite predictions")
    return predictions


def _ridge_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float,
    include_log2_concentration: bool,
) -> np.ndarray:
    transform, gram, target_product = _ridge_sufficient_statistics(
        train, include_log2_concentration=include_log2_concentration
    )
    coefficients = _ridge_coefficients(gram, target_product, alpha=alpha)
    return _ridge_predict_from_coefficients(test, coefficients, transform)


def _predict_baseline(
    baseline_id: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    if baseline_id == "training_mean":
        return np.full(len(test), float(train[TARGET_COLUMN].mean()), dtype=float)
    if baseline_id == "mutant_mean":
        return _lookup_prediction(train, test, ["mutant"])
    if baseline_id == "substitution_mean":
        return _lookup_prediction(train, test, ["wild_type", "mutant"])
    if baseline_id == "ridge_mutation_identity":
        if alpha is None:
            raise BaselineEvaluationError("Ridge alpha is required")
        return _ridge_fit_predict(
            train,
            test,
            alpha=alpha,
            include_log2_concentration=False,
        )
    if baseline_id == CONCENTRATION_BASELINE_ID:
        if alpha is None:
            raise BaselineEvaluationError("Ridge alpha is required")
        return _ridge_fit_predict(
            train,
            test,
            alpha=alpha,
            include_log2_concentration=True,
        )
    raise BaselineEvaluationError(f"Unknown baseline: {baseline_id}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise BaselineEvaluationError("Metrics require equal non-empty arrays")
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    true_constant = np.allclose(y_true, y_true[0], rtol=0.0, atol=1e-12)
    pred_constant = np.allclose(y_pred, y_pred[0], rtol=0.0, atol=1e-12)
    if true_constant or pred_constant:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
        pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return {"spearman": spearman, "pearson": pearson, "mae": mae, "rmse": rmse}


def _validation_mae(
    baseline_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    alpha: float,
) -> float:
    predictions = _predict_baseline(
        baseline_id,
        train,
        validation,
        alpha=alpha,
    )
    return float(
        np.mean(np.abs(validation[TARGET_COLUMN].to_numpy(dtype=float) - predictions))
    )


def _select_alpha_from_folds(
    baseline_id: str,
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[float, float, list[Mapping[str, object]]]:
    include_log2_concentration = baseline_id == CONCENTRATION_BASELINE_ID
    prepared = []
    for train, validation in folds:
        transform, gram, target_product = _ridge_sufficient_statistics(
            train, include_log2_concentration=include_log2_concentration
        )
        prepared.append((transform, gram, target_product, validation))

    rows: list[Mapping[str, object]] = []
    candidates: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        fold_maes = []
        for transform, gram, target_product, validation in prepared:
            coefficients = _ridge_coefficients(gram, target_product, alpha=alpha)
            predictions = _ridge_predict_from_coefficients(
                validation, coefficients, transform
            )
            y_validation = validation[TARGET_COLUMN].to_numpy(dtype=float)
            fold_mae = float(np.mean(np.abs(y_validation - predictions)))
            if not math.isfinite(fold_mae):
                raise BaselineEvaluationError(
                    "Ridge model selection produced a non-finite validation MAE"
                )
            fold_maes.append(fold_mae)
        mean_mae = float(np.mean(fold_maes))
        candidates.append((mean_mae, alpha))
        for fold_index, fold_mae in enumerate(fold_maes):
            rows.append(
                {
                    "alpha": alpha,
                    "cv_fold": fold_index,
                    "validation_mae": fold_mae,
                }
            )
    best_mae, best_alpha = min(candidates, key=lambda item: (item[0], item[1]))
    return best_alpha, best_mae, rows


def _scheme_baselines(scheme: pd.Series) -> tuple[str, ...]:
    baselines = list(BASELINE_IDS)
    if str(scheme["regime_id"]) in {
        "mutation_grouped_vim2_2022_concentration",
        "leave_one_concentration_out_vim2_2022",
    }:
        baselines.append(CONCENTRATION_BASELINE_ID)
    return tuple(baselines)


def _join_scheme_observations(
    membership: pd.DataFrame, observations: pd.DataFrame, scheme_id: str
) -> pd.DataFrame:
    selected = membership[membership["scheme_id"].eq(scheme_id)].copy()
    joined = selected.merge(
        observations,
        on="observation_id",
        how="left",
        validate="one_to_one",
        suffixes=("_membership", ""),
    )
    if joined[TARGET_COLUMN].isna().any():
        raise BaselineEvaluationError(
            f"Missing target after joining scheme {scheme_id}"
        )
    return joined


def _prediction_records(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    scheme: pd.Series,
    baseline_id: str,
    evaluation_iteration: object,
    evaluation_role: str,
    alpha: float | None,
) -> list[Mapping[str, object]]:
    result = pd.DataFrame(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "baseline_id": baseline_id,
            "evaluation_iteration": evaluation_iteration,
            "evaluation_role": evaluation_role,
            "observation_id": test["observation_id"].to_numpy(),
            "protein_id": test["protein_id"].to_numpy(),
            "assay_position": test["assay_position"].to_numpy(dtype=int),
            "wild_type": test["wild_type"].to_numpy(),
            "mutant": test["mutant"].to_numpy(),
            "antibiotic": test["antibiotic"].to_numpy(),
            "concentration": test["concentration"].to_numpy(),
            "temperature_c": test["temperature_c"].to_numpy(),
            "evaluation_group_id": test["group_id"].to_numpy(),
            "y_true": test[TARGET_COLUMN].to_numpy(dtype=float),
            "y_pred": predictions,
            "selected_alpha": alpha if alpha is not None else "",
        }
    )
    return result.to_dict(orient="records")


def _nested_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    scheme_assignments = assignments[assignments["scheme_id"].eq(scheme_id)]
    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
    iterations = sorted(scheme_assignments["iteration"].dropna().astype(int).unique())
    for iteration in iterations:
        iteration_assignments = scheme_assignments[
            scheme_assignments["iteration"].eq(iteration)
        ]
        group_roles = dict(
            zip(iteration_assignments["group_id"], iteration_assignments["split_role"])
        )
        roles = frame["group_id"].map(group_roles)
        train = frame[roles.eq("train")].copy()
        validation = frame[roles.eq("validation")].copy()
        test = frame[roles.eq("test")].copy()
        if train.empty or validation.empty or test.empty:
            raise BaselineEvaluationError(
                f"Incomplete nested split for {scheme_id} iteration {iteration}"
            )
        final_fit = pd.concat([train, validation], ignore_index=True)
        for baseline_id in _scheme_baselines(scheme):
            alpha: float | None = None
            if baseline_id.startswith("ridge_"):
                alpha, validation_mae, detail = _select_alpha_from_folds(
                    baseline_id, [(train, validation)]
                )
                for row in detail:
                    selection_rows.append(
                        {
                            "scheme_id": scheme_id,
                            "baseline_id": baseline_id,
                            "evaluation_iteration": iteration,
                            "selection_scope": "inner_validation",
                            **row,
                            "selected": float(row["alpha"]) == alpha,
                        }
                    )
                selection_rows.append(
                    {
                        "scheme_id": scheme_id,
                        "baseline_id": baseline_id,
                        "evaluation_iteration": iteration,
                        "selection_scope": "selected_setting",
                        "alpha": alpha,
                        "cv_fold": "",
                        "validation_mae": validation_mae,
                        "selected": True,
                    }
                )
            predictions = _predict_baseline(baseline_id, final_fit, test, alpha=alpha)
            prediction_rows.extend(
                _prediction_records(
                    test,
                    predictions,
                    scheme=scheme,
                    baseline_id=baseline_id,
                    evaluation_iteration=iteration,
                    evaluation_role="outer_test",
                    alpha=alpha,
                )
            )
    return prediction_rows, selection_rows


def _linear_log2_trajectory_prediction(
    source: pd.DataFrame, target_concentration: float
) -> float:
    ordered = source.sort_values("concentration")
    x = np.log2(ordered["concentration"].to_numpy(dtype=float))
    y = ordered[TARGET_COLUMN].to_numpy(dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        raise BaselineEvaluationError(
            "Concentration trajectory baseline requires at least two source levels"
        )
    target_x = math.log2(float(target_concentration))
    if target_x <= x[0]:
        left, right = 0, 1
    elif target_x >= x[-1]:
        left, right = len(x) - 2, len(x) - 1
    else:
        right = int(np.searchsorted(x, target_x, side="right"))
        left = right - 1
    slope = (y[right] - y[left]) / (x[right] - x[left])
    return float(y[left] + slope * (target_x - x[left]))


def _direct_condition_predictions(
    scheme: pd.Series, frame: pd.DataFrame
) -> list[Mapping[str, object]]:
    regime = str(scheme["regime_id"])
    source = frame[frame["scope_role"].eq("source_model_selection")].copy()
    target = frame[frame["scope_role"].eq("final_test")].copy()
    if source.empty or target.empty:
        return []

    results: list[Mapping[str, object]] = []
    if regime == "leave_one_antibiotic_out_vim2_2020":
        source_means = source.groupby("mutation_group_id", sort=True)[
            TARGET_COLUMN
        ].mean()
        predictions = target["mutation_group_id"].map(source_means)
        if predictions.isna().any():
            raise BaselineEvaluationError("Missing source-antibiotic mutation evidence")
        results.extend(
            _prediction_records(
                target,
                predictions.to_numpy(dtype=float),
                scheme=scheme,
                baseline_id=ANTIBIOTIC_DIRECT_BASELINE_ID,
                evaluation_iteration="final",
                evaluation_role="final_test",
                alpha=None,
            )
        )

    elif regime == "directional_temperature_transfer_vim2_amp":
        source_scores = source.set_index(["mutation_group_id", "concentration"])[
            TARGET_COLUMN
        ]
        if source_scores.index.duplicated().any():
            raise BaselineEvaluationError(
                "Temperature source contains duplicate mutation/concentration rows"
            )
        keys = list(
            target[["mutation_group_id", "concentration"]].itertuples(
                index=False, name=None
            )
        )
        predictions = np.asarray(
            [float(source_scores.get(key, np.nan)) for key in keys], dtype=float
        )
        if np.isnan(predictions).any():
            raise BaselineEvaluationError(
                "Temperature transfer lacks matched source mutation/concentration"
            )
        results.extend(
            _prediction_records(
                target,
                predictions,
                scheme=scheme,
                baseline_id=TEMPERATURE_DIRECT_BASELINE_ID,
                evaluation_iteration="final",
                evaluation_role="final_test",
                alpha=None,
            )
        )

    elif regime == "leave_one_concentration_out_vim2_2022":
        source_by_mutation = {}
        for key, subset in source.groupby("mutation_group_id", sort=True):
            ordered = subset.sort_values("concentration")
            source_by_mutation[key] = (
                np.log2(ordered["concentration"].to_numpy(dtype=float)),
                ordered[TARGET_COLUMN].to_numpy(dtype=float),
            )
        nearest_predictions = []
        linear_predictions = []
        for row in target.itertuples(index=False):
            trajectory = source_by_mutation.get(row.mutation_group_id)
            if trajectory is None:
                raise BaselineEvaluationError(
                    "Concentration transfer lacks source mutation trajectory"
                )
            source_x, source_y = trajectory
            if len(source_x) < 2:
                raise BaselineEvaluationError(
                    "Concentration trajectory requires at least two source levels"
                )
            target_x = math.log2(float(row.concentration))
            nearest_index = int(np.argmin(np.abs(source_x - target_x)))
            nearest_predictions.append(float(source_y[nearest_index]))
            if target_x <= source_x[0]:
                left, right = 0, 1
            elif target_x >= source_x[-1]:
                left, right = len(source_x) - 2, len(source_x) - 1
            else:
                right = int(np.searchsorted(source_x, target_x, side="right"))
                left = right - 1
            slope = (source_y[right] - source_y[left]) / (
                source_x[right] - source_x[left]
            )
            linear_predictions.append(
                float(source_y[left] + slope * (target_x - source_x[left]))
            )
        for baseline_id, predictions in (
            (CONCENTRATION_NEAREST_BASELINE_ID, nearest_predictions),
            (CONCENTRATION_LINEAR_BASELINE_ID, linear_predictions),
        ):
            results.extend(
                _prediction_records(
                    target,
                    np.asarray(predictions, dtype=float),
                    scheme=scheme,
                    baseline_id=baseline_id,
                    evaluation_iteration="final",
                    evaluation_role="final_test",
                    alpha=None,
                )
            )
    return results


def _source_cv_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    scheme_assignments = assignments[assignments["scheme_id"].eq(scheme_id)]
    source_assignments = scheme_assignments[
        scheme_assignments["split_role"].eq("source_model_selection")
    ]
    target_assignments = scheme_assignments[
        scheme_assignments["split_role"].eq("final_test")
    ]
    fold_by_group = dict(
        zip(source_assignments["group_id"], source_assignments["cv_fold"])
    )
    source = frame[frame["scope_role"].eq("source_model_selection")].copy()
    target = frame[frame["scope_role"].eq("final_test")].copy()
    if source.empty or target.empty:
        raise BaselineEvaluationError(f"Missing source/final-test rows for {scheme_id}")
    source["source_cv_fold"] = source["group_id"].map(fold_by_group)
    if source["source_cv_fold"].isna().any():
        raise BaselineEvaluationError(f"Unassigned source CV rows for {scheme_id}")
    target_groups = set(target_assignments["group_id"])
    if set(target["group_id"]) != target_groups:
        raise BaselineEvaluationError(f"Final-test group mismatch for {scheme_id}")

    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
    for baseline_id in _scheme_baselines(scheme):
        alpha: float | None = None
        if baseline_id.startswith("ridge_"):
            folds = []
            for fold in sorted(source["source_cv_fold"].astype(int).unique()):
                validation = source[source["source_cv_fold"].eq(fold)].copy()
                train = source[~source["source_cv_fold"].eq(fold)].copy()
                folds.append((train, validation))
            alpha, validation_mae, detail = _select_alpha_from_folds(baseline_id, folds)
            for row in detail:
                selection_rows.append(
                    {
                        "scheme_id": scheme_id,
                        "baseline_id": baseline_id,
                        "evaluation_iteration": "final",
                        "selection_scope": "source_cv",
                        **row,
                        "selected": float(row["alpha"]) == alpha,
                    }
                )
            selection_rows.append(
                {
                    "scheme_id": scheme_id,
                    "baseline_id": baseline_id,
                    "evaluation_iteration": "final",
                    "selection_scope": "selected_setting",
                    "alpha": alpha,
                    "cv_fold": "",
                    "validation_mae": validation_mae,
                    "selected": True,
                }
            )
        predictions = _predict_baseline(baseline_id, source, target, alpha=alpha)
        prediction_rows.extend(
            _prediction_records(
                target,
                predictions,
                scheme=scheme,
                baseline_id=baseline_id,
                evaluation_iteration="final",
                evaluation_role="final_test",
                alpha=alpha,
            )
        )
    prediction_rows.extend(_direct_condition_predictions(scheme, frame))
    return prediction_rows, selection_rows


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    for (scheme_id, baseline_id), group in predictions.groupby(
        ["scheme_id", "baseline_id"], sort=True
    ):
        for evaluation_iteration, subset in list(
            group.groupby("evaluation_iteration", sort=True)
        ) + [("pooled", group)]:
            values = _metrics(
                subset["y_true"].to_numpy(dtype=float),
                subset["y_pred"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "baseline_id": baseline_id,
                    "evaluation_iteration": evaluation_iteration,
                    "n_observations": len(subset),
                    "n_groups": subset["evaluation_group_id"].nunique(),
                    **values,
                }
            )
    return pd.DataFrame(rows)


def _direct_homologue_transfer(
    repo_root: Path, observations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(
        repo_root
        / "data/interim/task_definitions/aligned_shared_wild_type_mutation_pairs.csv"
    )
    if len(pairs) != 1513 or pairs["pairing_uses_outcome_values"].any():
        raise BaselineEvaluationError(
            "Aligned-pair contract does not contain the defined 1,513 pairs"
        )
    targets = observations.set_index("observation_id")[TARGET_COLUMN]
    rows: list[Mapping[str, object]] = []
    for direction, source_column, target_column in (
        ("NDM-1_to_VIM-2", "ndm_observation_id", "vim_observation_id"),
        ("VIM-2_to_NDM-1", "vim_observation_id", "ndm_observation_id"),
    ):
        for pair in pairs.itertuples(index=False):
            source_id = getattr(pair, source_column)
            target_id = getattr(pair, target_column)
            rows.append(
                {
                    "direction": direction,
                    "homologue_position_id": pair.homologue_position_id,
                    "wild_type": pair.wild_type,
                    "mutant": pair.mutant,
                    "source_observation_id": source_id,
                    "target_observation_id": target_id,
                    "source_effect": float(targets[source_id]),
                    "target_effect": float(targets[target_id]),
                    "predicted_target_effect": float(targets[source_id]),
                }
            )
    predictions = pd.DataFrame(rows)
    metric_rows = []
    for direction, group in predictions.groupby("direction", sort=True):
        point = _metrics(
            group["target_effect"].to_numpy(dtype=float),
            group["predicted_target_effect"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "direction": direction,
                "n_pairs": len(group),
                "n_homologue_positions": group["homologue_position_id"].nunique(),
                **point,
            }
        )
    return predictions, pd.DataFrame(metric_rows)


def prepare_baseline_predictions(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(repo_root / "data/processed/missense_observations.csv")
    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv"
    )
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv"
    )
    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv"
    )
    _validate_observations(observations)
    if TARGET_COLUMN in membership.columns or "score_original" in membership.columns:
        raise BaselineEvaluationError("Outcome leaked into modelling membership")

    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
    for _, scheme in schemes.sort_values("scheme_id").iterrows():
        if scheme["status"] != "ready":
            continue
        frame = _join_scheme_observations(
            membership, observations, str(scheme["scheme_id"])
        )
        family = scheme["assignment_family"]
        if family == "five_outer_folds_with_grouped_inner_validation":
            scheme_predictions, scheme_selection = _nested_evaluation(
                scheme, frame, assignments
            )
        elif family == "source_only_grouped_cv_then_final_test":
            scheme_predictions, scheme_selection = _source_cv_evaluation(
                scheme, frame, assignments
            )
        else:
            raise BaselineEvaluationError(f"Unsupported assignment family: {family}")
        prediction_rows.extend(scheme_predictions)
        selection_rows.extend(scheme_selection)

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise BaselineEvaluationError("No baseline predictions were generated")
    selection = pd.DataFrame(selection_rows)
    direct_predictions, direct_metrics = _direct_homologue_transfer(
        repo_root, observations
    )
    return predictions, selection, direct_predictions, direct_metrics


def run_baseline_evaluation(repo_root: Path) -> Mapping[str, object]:
    predictions, selection, direct_predictions, direct_metrics = (
        prepare_baseline_predictions(repo_root)
    )
    metrics = _metric_rows(predictions)

    output_dir = repo_root / "data/interim/baseline_evaluation"
    _write_csv(predictions, output_dir / "baseline_predictions.csv")
    _write_csv(metrics, output_dir / "baseline_metrics.csv")
    _write_csv(selection, output_dir / "baseline_model_selection.csv")
    _write_csv(
        direct_predictions, output_dir / "homologue_direct_transfer_predictions.csv"
    )
    _write_csv(direct_metrics, output_dir / "homologue_direct_transfer_metrics.csv")

    summary = {
        "status": "PASS",
        "target_column": TARGET_COLUMN,
        "score_original_used_as_predictor": False,
        "score_uncertainty_used_as_predictor": False,
        "protein_id_used_as_direct_predictor": False,
        "assay_position_used_as_direct_predictor": False,
        "baseline_ids": list(BASELINE_IDS)
        + [
            CONCENTRATION_BASELINE_ID,
            ANTIBIOTIC_DIRECT_BASELINE_ID,
            TEMPERATURE_DIRECT_BASELINE_ID,
            CONCENTRATION_NEAREST_BASELINE_ID,
            CONCENTRATION_LINEAR_BASELINE_ID,
        ],
        "concentration_baseline_scope": (
            "VIM-2 Chen-2022 concentration-response schemes only"
        ),
        "direct_condition_transfer_policy": (
            "condition-transfer schemes additionally test source-only conservation: "
            "source-antibiotic mutation mean, matched source-temperature effect, and "
            "nearest or linear log2 concentration trajectory prediction as applicable"
        ),
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "ridge_selection_metric": "validation_mae",
        "source_target_policy": (
            "cross-protein and condition-transfer settings are selected from source "
            "model-selection rows only; held-out target labels are used only after "
            "settings are fixed"
        ),
        "uncertainty_status": (
            "grouped bootstrap intervals are calculated by the separate baseline "
            "uncertainty command after prediction outputs have been generated"
        ),
        "scheme_count": int(predictions["scheme_id"].nunique()),
        "prediction_row_count": len(predictions),
        "metric_row_count": len(metrics),
        "model_selection_row_count": len(selection),
        "direct_homologue_pair_count_per_direction": 1513,
        "direct_homologue_position_count": 81,
        "direct_homologue_transfer_uses_target_labels_for_prediction": False,
        "outputs": [
            "baseline_predictions.csv",
            "baseline_metrics.csv",
            "baseline_model_selection.csv",
            "homologue_direct_transfer_predictions.csv",
            "homologue_direct_transfer_metrics.csv",
            "baseline_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
