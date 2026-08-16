from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


class ExperimentalStructureEvaluationError(RuntimeError):
    pass


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
SUBSTITUTIONS = tuple(
    (wild_type, mutant)
    for wild_type in AMINO_ACIDS
    for mutant in AMINO_ACIDS
    if mutant != wild_type
)
SUBSTITUTION_INDEX = {value: index for index, value in enumerate(SUBSTITUTIONS)}

RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
TARGET_COLUMN = "score_analysis"
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_experimental_structure_bootstrap"

MUTATION_CONDITION_MODEL = "structure_matched_mutation_condition_ridge"
EXPERIMENTAL_STRUCTURE_MODEL = "experimental_structure_descriptor_ridge"
MODEL_IDS = (MUTATION_CONDITION_MODEL, EXPERIMENTAL_STRUCTURE_MODEL)

CORE_SCHEME_IDS = (
    "ndm1_position_nested5",
    "vim2_2020_position_nested5",
    "ndm1_to_vim2_amp_source_cv5",
    "vim2_to_ndm1_amp_source_cv5",
    "vim2_2022_mutation_nested5",
)

CONDITION_MODE_BY_SCHEME = {
    "ndm1_position_nested5": "categorical_condition",
    "vim2_2020_position_nested5": "categorical_condition",
    "ndm1_to_vim2_amp_source_cv5": "none",
    "vim2_to_ndm1_amp_source_cv5": "none",
    "vim2_2022_mutation_nested5": "log2_concentration",
}

STRUCTURAL_NUMERIC_COLUMNS = (
    "sasa_angstrom2",
    "relative_sasa_wilke",
    "ca_neighbour_count_8_angstrom",
    "ca_neighbour_count_10_angstrom",
    "ca_neighbour_count_12_angstrom",
    "heavy_atom_contact_count_4_5_angstrom",
)
SECONDARY_STRUCTURE_COLUMN = "secondary_structure_3state"
SECONDARY_STRUCTURE_STATES = ("helix", "strand", "coil")


@dataclass(frozen=True)
class FeatureTransform:
    include_structure: bool
    condition_mode: str
    condition_categories: tuple[str, ...] = ()
    concentration_mean: float = 0.0
    concentration_sd: float = 1.0
    structural_means: tuple[float, ...] = ()
    structural_sds: tuple[float, ...] = ()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _condition_token(frame: pd.DataFrame) -> pd.Series:
    concentration = frame["concentration"].map(
        lambda value: format(float(value), ".12g")
    )
    temperature = frame["temperature_c"].map(lambda value: format(float(value), ".12g"))
    return frame["antibiotic"].astype(str) + "|" + concentration + "|" + temperature


def _normalise_secondary_structure(value: object) -> str:
    text = str(value).strip().lower()
    mapping = {
        "h": "helix",
        "helix": "helix",
        "e": "strand",
        "strand": "strand",
        "c": "coil",
        "coil": "coil",
    }
    if text not in mapping:
        raise ExperimentalStructureEvaluationError(
            f"Unsupported secondary-structure state: {value}"
        )
    return mapping[text]


def _validate_structure_features(residues: pd.DataFrame) -> None:
    required = {
        "protein_id",
        "assay_position",
        *STRUCTURAL_NUMERIC_COLUMNS,
        SECONDARY_STRUCTURE_COLUMN,
    }
    missing = required - set(residues.columns)
    if missing:
        raise ExperimentalStructureEvaluationError(
            f"Missing residue-feature columns: {sorted(missing)}"
        )
    if residues.duplicated(["protein_id", "assay_position"]).any():
        raise ExperimentalStructureEvaluationError(
            "Residue features contain duplicate protein/position rows"
        )


def _fit_transform(
    train: pd.DataFrame,
    *,
    include_structure: bool,
    condition_mode: str,
) -> FeatureTransform:
    categories: tuple[str, ...] = ()
    concentration_mean = 0.0
    concentration_sd = 1.0

    if condition_mode == "categorical_condition":
        categories = tuple(sorted(_condition_token(train).unique()))
    elif condition_mode == "log2_concentration":
        concentration = train["concentration"].to_numpy(dtype=float)
        if not np.isfinite(concentration).all() or (concentration <= 0).any():
            raise ExperimentalStructureEvaluationError(
                "Positive finite concentration is required"
            )
        values = np.log2(concentration)
        concentration_sd = float(values.std(ddof=0))
        if not math.isfinite(concentration_sd) or concentration_sd == 0.0:
            concentration_sd = 1.0
        concentration_mean = float(values.mean())
    elif condition_mode != "none":
        raise ExperimentalStructureEvaluationError(
            f"Unknown condition mode: {condition_mode}"
        )

    structural_means: tuple[float, ...] = ()
    structural_sds: tuple[float, ...] = ()
    if include_structure:
        values = train[list(STRUCTURAL_NUMERIC_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ExperimentalStructureEvaluationError(
                "Structural fitting features must be finite"
            )
        means = values.mean(axis=0)
        sds = values.std(axis=0, ddof=0)
        sds[~np.isfinite(sds) | (sds == 0.0)] = 1.0
        structural_means = tuple(float(value) for value in means)
        structural_sds = tuple(float(value) for value in sds)

    return FeatureTransform(
        include_structure=include_structure,
        condition_mode=condition_mode,
        condition_categories=categories,
        concentration_mean=concentration_mean,
        concentration_sd=concentration_sd,
        structural_means=structural_means,
        structural_sds=structural_sds,
    )


def _feature_count(transform: FeatureTransform) -> int:
    count = 1 + len(SUBSTITUTIONS)
    if transform.condition_mode == "categorical_condition":
        count += len(transform.condition_categories)
    elif transform.condition_mode == "log2_concentration":
        count += 1
    if transform.include_structure:
        count += len(STRUCTURAL_NUMERIC_COLUMNS)
        count += len(SECONDARY_STRUCTURE_STATES)
    return count


def _design_matrix(
    frame: pd.DataFrame,
    transform: FeatureTransform,
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    substitution_offset = 1
    condition_offset = substitution_offset + len(SUBSTITUTIONS)
    structure_offset = condition_offset

    condition_index = {
        value: index for index, value in enumerate(transform.condition_categories)
    }
    if transform.condition_mode == "categorical_condition":
        structure_offset += len(transform.condition_categories)
        condition_tokens = _condition_token(frame).tolist()
    elif transform.condition_mode == "log2_concentration":
        structure_offset += 1
        condition_tokens = None
    else:
        condition_tokens = None

    secondary_offset = structure_offset + (
        len(STRUCTURAL_NUMERIC_COLUMNS) if transform.include_structure else 0
    )
    secondary_index = {
        state: index for index, state in enumerate(SECONDARY_STRUCTURE_STATES)
    }

    structural_values = None
    if transform.include_structure:
        structural_values = frame[list(STRUCTURAL_NUMERIC_COLUMNS)].to_numpy(
            dtype=float
        )
        if not np.isfinite(structural_values).all():
            raise ExperimentalStructureEvaluationError(
                "Structural prediction features must be finite"
            )
        means = np.asarray(transform.structural_means, dtype=float)
        sds = np.asarray(transform.structural_sds, dtype=float)
        structural_values = (structural_values - means) / sds

    for row_number, row in enumerate(frame.itertuples(index=False)):
        row_indices.append(row_number)
        column_indices.append(0)
        values.append(1.0)

        substitution = (str(row.wild_type), str(row.mutant))
        if substitution not in SUBSTITUTION_INDEX:
            raise ExperimentalStructureEvaluationError(
                f"Invalid missense substitution: {substitution}"
            )
        row_indices.append(row_number)
        column_indices.append(substitution_offset + SUBSTITUTION_INDEX[substitution])
        values.append(1.0)

        if transform.condition_mode == "categorical_condition":
            token = condition_tokens[row_number]
            if token in condition_index:
                row_indices.append(row_number)
                column_indices.append(condition_offset + condition_index[token])
                values.append(1.0)
        elif transform.condition_mode == "log2_concentration":
            concentration = float(row.concentration)
            if not math.isfinite(concentration) or concentration <= 0:
                raise ExperimentalStructureEvaluationError(
                    "Positive finite concentration is required"
                )
            z = (
                math.log2(concentration) - transform.concentration_mean
            ) / transform.concentration_sd
            row_indices.append(row_number)
            column_indices.append(condition_offset)
            values.append(float(z))

        if transform.include_structure:
            for feature_index, feature_value in enumerate(
                structural_values[row_number]
            ):
                row_indices.append(row_number)
                column_indices.append(structure_offset + feature_index)
                values.append(float(feature_value))

            state = _normalise_secondary_structure(
                getattr(row, SECONDARY_STRUCTURE_COLUMN)
            )
            row_indices.append(row_number)
            column_indices.append(secondary_offset + secondary_index[state])
            values.append(1.0)

    matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(frame), _feature_count(transform)),
        dtype=float,
    )
    if not np.isfinite(matrix.data).all():
        raise ExperimentalStructureEvaluationError("Non-finite design matrix")
    return matrix


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    include_structure: bool,
    condition_mode: str,
    alpha: float,
) -> np.ndarray:
    transform = _fit_transform(
        train,
        include_structure=include_structure,
        condition_mode=condition_mode,
    )
    x_train = _design_matrix(train, transform)
    x_test = _design_matrix(test, transform)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)

    feature_count = x_train.shape[1]
    penalty = sparse.diags(
        [0.0] + [math.sqrt(alpha)] * (feature_count - 1),
        offsets=0,
        format="csr",
    )
    augmented_x = sparse.vstack([x_train, penalty], format="csr")
    augmented_y = np.concatenate([y_train, np.zeros(feature_count, dtype=float)])
    solution = lsqr(
        augmented_x,
        augmented_y,
        atol=1e-10,
        btol=1e-10,
        iter_lim=5000,
    )
    if int(solution[1]) not in {1, 2}:
        raise ExperimentalStructureEvaluationError(
            f"Sparse Ridge did not converge cleanly; lsqr istop={solution[1]}"
        )
    coefficients = np.asarray(solution[0], dtype=float)
    predictions = np.asarray(x_test @ coefficients, dtype=float)
    if not np.isfinite(predictions).all():
        raise ExperimentalStructureEvaluationError("Non-finite predictions")
    return predictions


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ExperimentalStructureEvaluationError(
            "Metrics require equal non-empty arrays"
        )
    error = y_pred - y_true
    true_constant = np.allclose(y_true, y_true[0], rtol=0.0, atol=1e-12)
    pred_constant = np.allclose(y_pred, y_pred[0], rtol=0.0, atol=1e-12)
    if true_constant or pred_constant:
        spearman = float("nan")
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        true_rank = pd.Series(y_true).rank(method="average").to_numpy(float)
        pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(float)
        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return {
        "spearman": spearman,
        "pearson": pearson,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _candidate_grid(model_id: str) -> tuple[float, ...]:
    if model_id not in MODEL_IDS:
        raise ExperimentalStructureEvaluationError(f"Unknown model: {model_id}")
    return RIDGE_ALPHAS


def _select_setting(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    model_id: str,
    condition_mode: str,
) -> tuple[float, float, list[Mapping[str, object]]]:
    include_structure = model_id == EXPERIMENTAL_STRUCTURE_MODEL
    detail: list[Mapping[str, object]] = []
    candidates: list[tuple[float, float]] = []

    for alpha in _candidate_grid(model_id):
        fold_maes = []
        for fold_index, (train, validation) in enumerate(folds):
            predictions = _fit_predict(
                train,
                validation,
                include_structure=include_structure,
                condition_mode=condition_mode,
                alpha=alpha,
            )
            mae = float(
                np.mean(
                    np.abs(
                        validation[TARGET_COLUMN].to_numpy(dtype=float) - predictions
                    )
                )
            )
            fold_maes.append(mae)
            detail.append(
                {
                    "model_id": model_id,
                    "alpha": alpha,
                    "cv_fold": fold_index,
                    "validation_mae": mae,
                }
            )
        candidates.append((float(np.mean(fold_maes)), alpha))

    best_mae, best_alpha = min(candidates, key=lambda item: (item[0], item[1]))
    return best_alpha, best_mae, detail


def _join_scheme_observations(
    membership: pd.DataFrame,
    observations: pd.DataFrame,
    residues: pd.DataFrame,
    scheme_id: str,
) -> pd.DataFrame:
    selected = membership[
        membership["scheme_id"].eq(scheme_id)
        & membership["experimental_structure_task_eligible"].astype(bool)
    ].copy()
    joined = selected.merge(
        observations,
        on="observation_id",
        how="left",
        validate="one_to_one",
        suffixes=("_membership", ""),
    )
    if joined[TARGET_COLUMN].isna().any():
        raise ExperimentalStructureEvaluationError(
            f"Missing target after joining {scheme_id}"
        )

    residue_columns = [
        "protein_id",
        "assay_position",
        *STRUCTURAL_NUMERIC_COLUMNS,
        SECONDARY_STRUCTURE_COLUMN,
    ]
    joined = joined.merge(
        residues[residue_columns],
        on=["protein_id", "assay_position"],
        how="left",
        validate="many_to_one",
    )

    if joined[list(STRUCTURAL_NUMERIC_COLUMNS)].isna().any().any():
        raise ExperimentalStructureEvaluationError(
            f"Missing structural descriptor after joining {scheme_id}"
        )
    if joined[SECONDARY_STRUCTURE_COLUMN].isna().any():
        raise ExperimentalStructureEvaluationError(
            f"Missing secondary structure after joining {scheme_id}"
        )
    return joined


def _prediction_records(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    scheme: pd.Series,
    model_id: str,
    evaluation_iteration: object,
    evaluation_role: str,
    alpha: float,
    condition_mode: str,
) -> list[Mapping[str, object]]:
    result = pd.DataFrame(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "model_id": model_id,
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
            "selected_alpha": alpha,
            "condition_mode": condition_mode,
        }
    )
    return result.to_dict(orient="records")


def _nested_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    condition_mode = CONDITION_MODE_BY_SCHEME[scheme_id]
    scheme_assignments = assignments[assignments["scheme_id"].eq(scheme_id)]
    predictions: list[Mapping[str, object]] = []
    selections: list[Mapping[str, object]] = []

    for iteration in sorted(
        scheme_assignments["iteration"].dropna().astype(int).unique()
    ):
        current = scheme_assignments[scheme_assignments["iteration"].eq(iteration)]
        roles = frame["group_id"].map(
            dict(zip(current["group_id"], current["split_role"]))
        )
        train = frame[roles.eq("train")].copy()
        validation = frame[roles.eq("validation")].copy()
        test = frame[roles.eq("test")].copy()
        if train.empty or validation.empty or test.empty:
            raise ExperimentalStructureEvaluationError(
                f"Incomplete structure-matched split for {scheme_id}/{iteration}"
            )
        refit_data = pd.concat([train, validation], ignore_index=True)

        for model_id in MODEL_IDS:
            alpha, validation_mae, detail = _select_setting(
                [(train, validation)],
                model_id=model_id,
                condition_mode=condition_mode,
            )
            for row in detail:
                selections.append(
                    {
                        "scheme_id": scheme_id,
                        "task_id": str(scheme["task_id"]),
                        "evaluation_iteration": iteration,
                        "selection_scope": "inner_validation",
                        **row,
                        "selected": row["alpha"] == alpha,
                    }
                )
            selections.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": str(scheme["task_id"]),
                    "evaluation_iteration": iteration,
                    "selection_scope": "selected_setting",
                    "model_id": model_id,
                    "alpha": alpha,
                    "cv_fold": "",
                    "validation_mae": validation_mae,
                    "selected": True,
                }
            )
            y_pred = _fit_predict(
                refit_data,
                test,
                include_structure=(model_id == EXPERIMENTAL_STRUCTURE_MODEL),
                condition_mode=condition_mode,
                alpha=alpha,
            )
            predictions.extend(
                _prediction_records(
                    test,
                    y_pred,
                    scheme=scheme,
                    model_id=model_id,
                    evaluation_iteration=iteration,
                    evaluation_role="outer_test",
                    alpha=alpha,
                    condition_mode=condition_mode,
                )
            )
    return predictions, selections


def _source_cv_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    condition_mode = CONDITION_MODE_BY_SCHEME[scheme_id]
    scheme_assignments = assignments[assignments["scheme_id"].eq(scheme_id)]
    source_assignments = scheme_assignments[
        scheme_assignments["split_role"].eq("source_model_selection")
    ]
    target_assignments = scheme_assignments[
        scheme_assignments["split_role"].eq("target_test")
    ]
    source = frame[frame["scope_role"].eq("source_model_selection")].copy()
    target = frame[frame["scope_role"].eq("target_test")].copy()
    source["source_cv_fold"] = source["group_id"].map(
        dict(zip(source_assignments["group_id"], source_assignments["cv_fold"]))
    )
    if source["source_cv_fold"].isna().any():
        raise ExperimentalStructureEvaluationError(
            f"Unassigned source CV rows for {scheme_id}"
        )
    if set(target["group_id"]) != set(target_assignments["group_id"]):
        expected_groups = set(target_assignments["group_id"])
        observed_groups = set(target["group_id"])
        if not observed_groups.issubset(expected_groups):
            raise ExperimentalStructureEvaluationError(
                f"Target-test group mismatch for {scheme_id}"
            )

    folds = []
    for fold in sorted(source["source_cv_fold"].astype(int).unique()):
        folds.append(
            (
                source[~source["source_cv_fold"].eq(fold)].copy(),
                source[source["source_cv_fold"].eq(fold)].copy(),
            )
        )

    predictions: list[Mapping[str, object]] = []
    selections: list[Mapping[str, object]] = []
    for model_id in MODEL_IDS:
        alpha, validation_mae, detail = _select_setting(
            folds,
            model_id=model_id,
            condition_mode=condition_mode,
        )
        for row in detail:
            selections.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": str(scheme["task_id"]),
                    "evaluation_iteration": "target",
                    "selection_scope": "source_cv",
                    **row,
                    "selected": row["alpha"] == alpha,
                }
            )
        selections.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": "target",
                "selection_scope": "selected_setting",
                "model_id": model_id,
                "alpha": alpha,
                "cv_fold": "",
                "validation_mae": validation_mae,
                "selected": True,
            }
        )
        y_pred = _fit_predict(
            source,
            target,
            include_structure=(model_id == EXPERIMENTAL_STRUCTURE_MODEL),
            condition_mode=condition_mode,
            alpha=alpha,
        )
        predictions.extend(
            _prediction_records(
                target,
                y_pred,
                scheme=scheme,
                model_id=model_id,
                evaluation_iteration="target",
                evaluation_role="target_test",
                alpha=alpha,
                condition_mode=condition_mode,
            )
        )
    return predictions, selections


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme_id, model_id), group in predictions.groupby(
        ["scheme_id", "model_id"], sort=True
    ):
        for evaluation_iteration, subset in list(
            group.groupby("evaluation_iteration", sort=True)
        ) + [("pooled", group)]:
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_id": model_id,
                    "evaluation_iteration": evaluation_iteration,
                    "n_observations": len(subset),
                    "n_groups": subset["evaluation_group_id"].nunique(),
                    **_metric_values(
                        subset["y_true"].to_numpy(float),
                        subset["y_pred"].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _numeric_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme_id, group in predictions.groupby("scheme_id", sort=True):
        structure = group[group["model_id"].eq(EXPERIMENTAL_STRUCTURE_MODEL)].copy()
        comparator = group[group["model_id"].eq(MUTATION_CONDITION_MODEL)][
            ["observation_id", "y_true", "y_pred"]
        ].rename(
            columns={
                "y_true": "comparator_y_true",
                "y_pred": "comparator_y_pred",
            }
        )
        paired = structure.merge(
            comparator,
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(structure) or not np.allclose(
            paired["y_true"], paired["comparator_y_true"]
        ):
            raise ExperimentalStructureEvaluationError(
                f"Paired model population mismatch for {scheme_id}"
            )

        groups = sorted(paired["evaluation_group_id"].unique())
        blocks = {
            group_id: paired[paired["evaluation_group_id"].eq(group_id)].copy()
            for group_id in groups
        }
        rng = np.random.default_rng(_numeric_seed(f"{BOOTSTRAP_SEED}|{scheme_id}"))
        distributions = {
            metric: [] for metric in ("spearman", "pearson", "mae", "rmse")
        }
        structure_point = _metric_values(
            paired["y_true"].to_numpy(float),
            paired["y_pred"].to_numpy(float),
        )
        comparator_point = _metric_values(
            paired["y_true"].to_numpy(float),
            paired["comparator_y_pred"].to_numpy(float),
        )

        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            boot = pd.concat([blocks[value] for value in sampled], ignore_index=True)
            structure_metrics = _metric_values(
                boot["y_true"].to_numpy(float),
                boot["y_pred"].to_numpy(float),
            )
            comparator_metrics = _metric_values(
                boot["y_true"].to_numpy(float),
                boot["comparator_y_pred"].to_numpy(float),
            )
            for metric in distributions:
                difference = (
                    structure_metrics[metric] - comparator_metrics[metric]
                    if metric in {"spearman", "pearson"}
                    else comparator_metrics[metric] - structure_metrics[metric]
                )
                if np.isfinite(difference):
                    distributions[metric].append(difference)

        for metric, values_list in distributions.items():
            point = (
                structure_point[metric] - comparator_point[metric]
                if metric in {"spearman", "pearson"}
                else comparator_point[metric] - structure_point[metric]
            )
            values = np.asarray(values_list, dtype=float)
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_a": EXPERIMENTAL_STRUCTURE_MODEL,
                    "model_b": MUTATION_CONDITION_MODEL,
                    "metric": metric,
                    "model_a_value": structure_point[metric],
                    "model_b_value": comparator_point[metric],
                    "paired_difference": point,
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_valid_replicates": len(values),
                    "resampling_unit": "evaluation_group_id",
                    "difference_interpretation": (
                        "positive_favours_experimental_structure"
                    ),
                }
            )
    return pd.DataFrame(rows)


def prepare_experimental_structure_evaluation(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(repo_root / "data/processed/missense_observations.csv")
    residues = pd.read_csv(
        repo_root / "data/processed/residue_features.csv",
        low_memory=False,
    )
    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv"
    )
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv"
    )
    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv"
    )

    _validate_structure_features(residues)
    if TARGET_COLUMN in membership.columns or "score_original" in membership.columns:
        raise ExperimentalStructureEvaluationError(
            "Outcome leaked into modelling membership"
        )
    if "experimental_structure_task_eligible" not in membership.columns:
        raise ExperimentalStructureEvaluationError(
            "Experimental-structure eligibility is missing from modelling membership"
        )

    selected_schemes = schemes[schemes["scheme_id"].isin(CORE_SCHEME_IDS)].copy()
    if set(selected_schemes["scheme_id"]) != set(CORE_SCHEME_IDS):
        raise ExperimentalStructureEvaluationError("Core scheme contract is incomplete")
    if not selected_schemes["status"].eq("ready").all():
        raise ExperimentalStructureEvaluationError("Core schemes must all be ready")

    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
    for _, scheme in selected_schemes.sort_values("scheme_id").iterrows():
        frame = _join_scheme_observations(
            membership,
            observations,
            residues,
            str(scheme["scheme_id"]),
        )
        family = str(scheme["assignment_family"])
        if family == "five_outer_folds_with_grouped_inner_validation":
            scheme_predictions, scheme_selection = _nested_evaluation(
                scheme,
                frame,
                assignments,
            )
        elif family == "source_only_grouped_cv_then_target_test":
            scheme_predictions, scheme_selection = _source_cv_evaluation(
                scheme,
                frame,
                assignments,
            )
        else:
            raise ExperimentalStructureEvaluationError(
                f"Unsupported assignment family: {family}"
            )
        prediction_rows.extend(scheme_predictions)
        selection_rows.extend(scheme_selection)

    predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows)
    metrics = _metric_rows(predictions)

    counts = predictions.groupby(["scheme_id", "model_id"]).size().unstack()
    if set(counts.columns) != set(MODEL_IDS):
        raise ExperimentalStructureEvaluationError(
            "Not all structural-evaluation models produced predictions"
        )
    if not (
        counts[MUTATION_CONDITION_MODEL] == counts[EXPERIMENTAL_STRUCTURE_MODEL]
    ).all():
        raise ExperimentalStructureEvaluationError(
            "Structure model and matched comparator populations differ"
        )
    return predictions, metrics, selection


def run_experimental_structure_evaluation(repo_root: Path) -> Mapping[str, object]:
    predictions, metrics, selection = prepare_experimental_structure_evaluation(
        repo_root
    )
    paired = _paired_bootstrap(predictions)

    output_dir = repo_root / "data/interim/experimental_structure_evaluation"
    _write_csv(
        predictions,
        output_dir / "experimental_structure_predictions.csv",
    )
    _write_csv(
        metrics,
        output_dir / "experimental_structure_metrics.csv",
    )
    _write_csv(
        selection,
        output_dir / "experimental_structure_model_selection.csv",
    )
    _write_csv(
        paired,
        output_dir / "experimental_structure_paired_comparisons.csv",
    )

    counts = predictions.groupby("model_id").size().to_dict()
    summary = {
        "status": "PASS",
        "analysis_scope": "core_experimental_structure_descriptor_evaluation",
        "core_scheme_ids": list(CORE_SCHEME_IDS),
        "model_ids": list(MODEL_IDS),
        "primary_structure_model": EXPERIMENTAL_STRUCTURE_MODEL,
        "matched_comparator": MUTATION_CONDITION_MODEL,
        "coordinate_source": "selected_experimental_pdb",
        "eligibility_column": "experimental_structure_task_eligible",
        "structural_numeric_inputs": list(STRUCTURAL_NUMERIC_COLUMNS),
        "secondary_structure_input": SECONDARY_STRUCTURE_COLUMN,
        "secondary_structure_states": list(SECONDARY_STRUCTURE_STATES),
        "condition_modes": CONDITION_MODE_BY_SCHEME,
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "selection_metric": "validation_mae",
        "selection_tiebreak": "lower_validation_mae_then_smaller_alpha",
        "substitution_vocabulary_size": len(SUBSTITUTIONS),
        "substitution_vocabulary_is_predefined": True,
        "structural_scaling": (
            "training-data mean and population standard deviation within each "
            "model-selection or refit partition"
        ),
        "mechanistic_reference_inputs_used": False,
        "catalytic_zinc_distance_inputs_used": False,
        "reported_homologue_pair_inputs_used": False,
        "raw_coordinates_used_as_direct_predictors": False,
        "protein_id_used_as_direct_predictor": False,
        "assay_position_used_as_direct_predictor": False,
        "score_original_used_as_predictor": False,
        "score_uncertainty_used_as_predictor": False,
        "target_column": TARGET_COLUMN,
        "target_labels_used_for_model_selection": False,
        "prediction_count_by_model": counts,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "outputs": [
            "experimental_structure_predictions.csv",
            "experimental_structure_metrics.csv",
            "experimental_structure_model_selection.csv",
            "experimental_structure_paired_comparisons.csv",
            "experimental_structure_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experimental_structure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
