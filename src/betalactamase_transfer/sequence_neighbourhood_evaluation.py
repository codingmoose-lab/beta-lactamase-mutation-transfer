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


class SequenceNeighbourhoodEvaluationError(RuntimeError):
    pass


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
NEIGHBOURHOOD_STATES = AMINO_ACIDS + ("PAD",)
SUBSTITUTIONS = tuple(
    (wild_type, mutant)
    for wild_type in AMINO_ACIDS
    for mutant in AMINO_ACIDS
    if mutant != wild_type
)
SUBSTITUTION_INDEX = {value: index for index, value in enumerate(SUBSTITUTIONS)}
NEIGHBOURHOOD_RADII = (4, 8, 12, 16)
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
TARGET_COLUMN = "score_analysis"
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_sequence_neighbourhood_bootstrap"

MUTATION_CONDITION_MODEL = "mutation_condition_ridge"
SEQUENCE_NEIGHBOURHOOD_MODEL = "sequence_neighbourhood_ridge"
MODEL_IDS = (MUTATION_CONDITION_MODEL, SEQUENCE_NEIGHBOURHOOD_MODEL)

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


@dataclass(frozen=True)
class FeatureTransform:
    radius: int | None
    condition_mode: str
    condition_categories: tuple[str, ...] = ()
    concentration_mean: float = 0.0
    concentration_sd: float = 1.0


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _load_assay_sequences(repo_root: Path) -> Mapping[str, str]:
    frame = pd.read_csv(
        repo_root / "data/interim/sequence_mapping/sequence_sources.csv"
    )
    selected = frame.loc[
        frame["source_type"].eq("genbank_assay_construct")
        & frame["selection_status"].eq("genbank_assay_construct_feature_match")
    ].copy()
    if len(selected) != 2 or selected["protein_id"].duplicated().any():
        raise SequenceNeighbourhoodEvaluationError(
            "Expected one assay construct for NDM-1 and VIM-2"
        )
    sequences = dict(zip(selected["protein_id"], selected["sequence"]))
    if set(sequences) != {"NDM-1", "VIM-2"}:
        raise SequenceNeighbourhoodEvaluationError(
            "Assay-sequence contract does not contain NDM-1 and VIM-2"
        )
    return {protein: str(sequence) for protein, sequence in sequences.items()}


def _validate_observations(
    observations: pd.DataFrame,
    assay_sequences: Mapping[str, str],
) -> None:
    required = {
        "observation_id",
        "protein_id",
        "assay_position",
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
        TARGET_COLUMN,
    }
    missing = required - set(observations.columns)
    if missing:
        raise SequenceNeighbourhoodEvaluationError(
            f"Missing observation columns: {sorted(missing)}"
        )
    if observations[TARGET_COLUMN].isna().any():
        raise SequenceNeighbourhoodEvaluationError("Missing modelling targets")

    positions = observations[
        ["protein_id", "assay_position", "wild_type"]
    ].drop_duplicates()
    for row in positions.itertuples(index=False):
        sequence = assay_sequences[str(row.protein_id)]
        position = int(row.assay_position)
        if position < 1 or position > len(sequence):
            raise SequenceNeighbourhoodEvaluationError(
                f"Assay position outside sequence: {row.protein_id}/{position}"
            )
        if sequence[position - 1] != str(row.wild_type):
            raise SequenceNeighbourhoodEvaluationError(
                f"WT mismatch at {row.protein_id}/{position}: "
                f"{row.wild_type} != {sequence[position - 1]}"
            )


def _condition_token(frame: pd.DataFrame) -> pd.Series:
    concentration = frame["concentration"].map(
        lambda value: format(float(value), ".12g")
    )
    temperature = frame["temperature_c"].map(lambda value: format(float(value), ".12g"))
    return frame["antibiotic"].astype(str) + "|" + concentration + "|" + temperature


def _fit_transform(
    train: pd.DataFrame,
    *,
    radius: int | None,
    condition_mode: str,
) -> FeatureTransform:
    if radius is not None and radius not in NEIGHBOURHOOD_RADII:
        raise SequenceNeighbourhoodEvaluationError(
            f"Unsupported neighbourhood radius: {radius}"
        )
    if condition_mode == "none":
        return FeatureTransform(radius=radius, condition_mode=condition_mode)
    if condition_mode == "categorical_condition":
        return FeatureTransform(
            radius=radius,
            condition_mode=condition_mode,
            condition_categories=tuple(sorted(_condition_token(train).unique())),
        )
    if condition_mode == "log2_concentration":
        concentration = train["concentration"].to_numpy(dtype=float)
        if not np.isfinite(concentration).all() or (concentration <= 0).any():
            raise SequenceNeighbourhoodEvaluationError(
                "Positive finite concentration is required"
            )
        values = np.log2(concentration)
        sd = float(values.std(ddof=0))
        if not math.isfinite(sd) or sd == 0.0:
            sd = 1.0
        return FeatureTransform(
            radius=radius,
            condition_mode=condition_mode,
            concentration_mean=float(values.mean()),
            concentration_sd=sd,
        )
    raise SequenceNeighbourhoodEvaluationError(
        f"Unknown condition mode: {condition_mode}"
    )


def _neighbourhood_states(
    sequence: str,
    position: int,
    radius: int,
) -> tuple[str, ...]:
    states = []
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        index = position - 1 + offset
        states.append("PAD" if index < 0 or index >= len(sequence) else sequence[index])
    return tuple(states)


def _feature_count(transform: FeatureTransform) -> int:
    count = 1 + len(SUBSTITUTIONS)
    if transform.radius is not None:
        count += 2 * transform.radius * len(NEIGHBOURHOOD_STATES)
    if transform.condition_mode == "categorical_condition":
        count += len(transform.condition_categories)
    elif transform.condition_mode == "log2_concentration":
        count += 1
    return count


def _design_matrix(
    frame: pd.DataFrame,
    transform: FeatureTransform,
    assay_sequences: Mapping[str, str],
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    substitution_offset = 1
    neighbourhood_offset = substitution_offset + len(SUBSTITUTIONS)
    condition_offset = neighbourhood_offset
    if transform.radius is not None:
        condition_offset += 2 * transform.radius * len(NEIGHBOURHOOD_STATES)

    state_index = {value: index for index, value in enumerate(NEIGHBOURHOOD_STATES)}
    condition_index = {
        value: index for index, value in enumerate(transform.condition_categories)
    }
    condition_tokens = (
        _condition_token(frame).tolist()
        if transform.condition_mode == "categorical_condition"
        else None
    )

    for row_number, row in enumerate(frame.itertuples(index=False)):
        row_indices.append(row_number)
        column_indices.append(0)
        values.append(1.0)

        substitution = (str(row.wild_type), str(row.mutant))
        if substitution not in SUBSTITUTION_INDEX:
            raise SequenceNeighbourhoodEvaluationError(
                f"Invalid missense substitution: {substitution}"
            )
        row_indices.append(row_number)
        column_indices.append(substitution_offset + SUBSTITUTION_INDEX[substitution])
        values.append(1.0)

        if transform.radius is not None:
            sequence = assay_sequences[str(row.protein_id)]
            states = _neighbourhood_states(
                sequence, int(row.assay_position), transform.radius
            )
            for relative_index, state in enumerate(states):
                row_indices.append(row_number)
                column_indices.append(
                    neighbourhood_offset
                    + relative_index * len(NEIGHBOURHOOD_STATES)
                    + state_index[state]
                )
                values.append(1.0)

        if transform.condition_mode == "categorical_condition":
            token = condition_tokens[row_number]
            if token in condition_index:
                row_indices.append(row_number)
                column_indices.append(condition_offset + condition_index[token])
                values.append(1.0)
        elif transform.condition_mode == "log2_concentration":
            concentration = float(row.concentration)
            z = (
                math.log2(concentration) - transform.concentration_mean
            ) / transform.concentration_sd
            row_indices.append(row_number)
            column_indices.append(condition_offset)
            values.append(float(z))

    matrix = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(frame), _feature_count(transform)),
        dtype=float,
    )
    if not np.isfinite(matrix.data).all():
        raise SequenceNeighbourhoodEvaluationError("Non-finite design matrix")
    return matrix


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    assay_sequences: Mapping[str, str],
    radius: int | None,
    condition_mode: str,
    alpha: float,
) -> np.ndarray:
    transform = _fit_transform(train, radius=radius, condition_mode=condition_mode)
    x_train = _design_matrix(train, transform, assay_sequences)
    x_test = _design_matrix(test, transform, assay_sequences)
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
        raise SequenceNeighbourhoodEvaluationError(
            f"Sparse Ridge did not converge cleanly; lsqr istop={solution[1]}"
        )
    coefficients = np.asarray(solution[0], dtype=float)
    predictions = np.asarray(x_test @ coefficients, dtype=float)
    if not np.isfinite(predictions).all():
        raise SequenceNeighbourhoodEvaluationError("Non-finite predictions")
    return predictions


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise SequenceNeighbourhoodEvaluationError(
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


def _candidate_grid(model_id: str) -> tuple[tuple[int | None, float], ...]:
    if model_id == MUTATION_CONDITION_MODEL:
        return tuple((None, alpha) for alpha in RIDGE_ALPHAS)
    if model_id == SEQUENCE_NEIGHBOURHOOD_MODEL:
        return tuple(
            (radius, alpha) for radius in NEIGHBOURHOOD_RADII for alpha in RIDGE_ALPHAS
        )
    raise SequenceNeighbourhoodEvaluationError(f"Unknown model: {model_id}")


def _select_setting(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    model_id: str,
    assay_sequences: Mapping[str, str],
    condition_mode: str,
) -> tuple[int | None, float, float, list[Mapping[str, object]]]:
    detail: list[Mapping[str, object]] = []
    candidates = []
    for radius, alpha in _candidate_grid(model_id):
        fold_maes = []
        for fold_index, (train, validation) in enumerate(folds):
            predictions = _fit_predict(
                train,
                validation,
                assay_sequences=assay_sequences,
                radius=radius,
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
                    "radius": "" if radius is None else radius,
                    "alpha": alpha,
                    "cv_fold": fold_index,
                    "validation_mae": mae,
                }
            )
        candidates.append(
            (
                float(np.mean(fold_maes)),
                -1 if radius is None else radius,
                alpha,
                radius,
            )
        )
    best_mae, _, best_alpha, best_radius = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return best_radius, best_alpha, best_mae, detail


def _join_scheme_observations(
    membership: pd.DataFrame,
    observations: pd.DataFrame,
    scheme_id: str,
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
        raise SequenceNeighbourhoodEvaluationError(
            f"Missing target after joining {scheme_id}"
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
    radius: int | None,
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
            "selected_radius": "" if radius is None else radius,
            "selected_alpha": alpha,
            "condition_mode": condition_mode,
        }
    )
    return result.to_dict(orient="records")


def _nested_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    assay_sequences: Mapping[str, str],
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
            raise SequenceNeighbourhoodEvaluationError(
                f"Incomplete nested split for {scheme_id}/{iteration}"
            )
        refit_data = pd.concat([train, validation], ignore_index=True)

        for model_id in MODEL_IDS:
            radius, alpha, validation_mae, detail = _select_setting(
                [(train, validation)],
                model_id=model_id,
                assay_sequences=assay_sequences,
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
                        "selected": (
                            row["alpha"] == alpha
                            and (
                                (row["radius"] == "" and radius is None)
                                or row["radius"] == radius
                            )
                        ),
                    }
                )
            selections.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": str(scheme["task_id"]),
                    "evaluation_iteration": iteration,
                    "selection_scope": "selected_setting",
                    "model_id": model_id,
                    "radius": "" if radius is None else radius,
                    "alpha": alpha,
                    "cv_fold": "",
                    "validation_mae": validation_mae,
                    "selected": True,
                }
            )
            y_pred = _fit_predict(
                refit_data,
                test,
                assay_sequences=assay_sequences,
                radius=radius,
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
                    radius=radius,
                    alpha=alpha,
                    condition_mode=condition_mode,
                )
            )
    return predictions, selections


def _source_cv_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    assay_sequences: Mapping[str, str],
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
        raise SequenceNeighbourhoodEvaluationError(
            f"Unassigned source CV rows for {scheme_id}"
        )
    if set(target["group_id"]) != set(target_assignments["group_id"]):
        raise SequenceNeighbourhoodEvaluationError(
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
        radius, alpha, validation_mae, detail = _select_setting(
            folds,
            model_id=model_id,
            assay_sequences=assay_sequences,
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
                    "selected": (
                        row["alpha"] == alpha
                        and (
                            (row["radius"] == "" and radius is None)
                            or row["radius"] == radius
                        )
                    ),
                }
            )
        selections.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": "target",
                "selection_scope": "selected_setting",
                "model_id": model_id,
                "radius": "" if radius is None else radius,
                "alpha": alpha,
                "cv_fold": "",
                "validation_mae": validation_mae,
                "selected": True,
            }
        )
        y_pred = _fit_predict(
            source,
            target,
            assay_sequences=assay_sequences,
            radius=radius,
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
                radius=radius,
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
        sequence = group[group["model_id"].eq(SEQUENCE_NEIGHBOURHOOD_MODEL)].copy()
        comparator = group[group["model_id"].eq(MUTATION_CONDITION_MODEL)][
            ["observation_id", "y_true", "y_pred"]
        ].rename(
            columns={
                "y_true": "comparator_y_true",
                "y_pred": "comparator_y_pred",
            }
        )
        paired = sequence.merge(
            comparator,
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(sequence) or not np.allclose(
            paired["y_true"], paired["comparator_y_true"]
        ):
            raise SequenceNeighbourhoodEvaluationError(
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
        sequence_point = _metric_values(
            paired["y_true"].to_numpy(float), paired["y_pred"].to_numpy(float)
        )
        comparator_point = _metric_values(
            paired["y_true"].to_numpy(float),
            paired["comparator_y_pred"].to_numpy(float),
        )
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            boot = pd.concat([blocks[value] for value in sampled], ignore_index=True)
            seq_metrics = _metric_values(
                boot["y_true"].to_numpy(float), boot["y_pred"].to_numpy(float)
            )
            cmp_metrics = _metric_values(
                boot["y_true"].to_numpy(float),
                boot["comparator_y_pred"].to_numpy(float),
            )
            for metric in distributions:
                difference = (
                    seq_metrics[metric] - cmp_metrics[metric]
                    if metric in {"spearman", "pearson"}
                    else cmp_metrics[metric] - seq_metrics[metric]
                )
                if np.isfinite(difference):
                    distributions[metric].append(difference)
        for metric, values_list in distributions.items():
            point = (
                sequence_point[metric] - comparator_point[metric]
                if metric in {"spearman", "pearson"}
                else comparator_point[metric] - sequence_point[metric]
            )
            values = np.asarray(values_list, dtype=float)
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "metric": metric,
                    "paired_difference": point,
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_valid_replicates": len(values),
                    "resampling_unit": "evaluation_group_id",
                    "difference_interpretation": (
                        "positive_favours_sequence_neighbourhood"
                    ),
                }
            )
    return pd.DataFrame(rows)


def prepare_sequence_neighbourhood_evaluation(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    assay_sequences = _load_assay_sequences(repo_root)
    _validate_observations(observations, assay_sequences)
    if TARGET_COLUMN in membership.columns or "score_original" in membership.columns:
        raise SequenceNeighbourhoodEvaluationError(
            "Outcome leaked into modelling membership"
        )

    selected_schemes = schemes[schemes["scheme_id"].isin(CORE_SCHEME_IDS)].copy()
    if set(selected_schemes["scheme_id"]) != set(CORE_SCHEME_IDS):
        raise SequenceNeighbourhoodEvaluationError("Core scheme contract is incomplete")
    if not selected_schemes["status"].eq("ready").all():
        raise SequenceNeighbourhoodEvaluationError("Core schemes must all be ready")

    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
    for _, scheme in selected_schemes.sort_values("scheme_id").iterrows():
        frame = _join_scheme_observations(
            membership, observations, str(scheme["scheme_id"])
        )
        family = str(scheme["assignment_family"])
        if family == "five_outer_folds_with_grouped_inner_validation":
            scheme_predictions, scheme_selection = _nested_evaluation(
                scheme, frame, assignments, assay_sequences
            )
        elif family == "source_only_grouped_cv_then_target_test":
            scheme_predictions, scheme_selection = _source_cv_evaluation(
                scheme, frame, assignments, assay_sequences
            )
        else:
            raise SequenceNeighbourhoodEvaluationError(
                f"Unsupported assignment family: {family}"
            )
        prediction_rows.extend(scheme_predictions)
        selection_rows.extend(scheme_selection)

    predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows)
    metrics = _metric_rows(predictions)
    return predictions, metrics, selection


def run_sequence_neighbourhood_evaluation(repo_root: Path) -> Mapping[str, object]:
    predictions, metrics, selection = prepare_sequence_neighbourhood_evaluation(
        repo_root
    )
    paired = _paired_bootstrap(predictions)
    output_dir = repo_root / "data/interim/sequence_neighbourhood_evaluation"
    _write_csv(predictions, output_dir / "sequence_neighbourhood_predictions.csv")
    _write_csv(metrics, output_dir / "sequence_neighbourhood_metrics.csv")
    _write_csv(selection, output_dir / "sequence_neighbourhood_model_selection.csv")
    _write_csv(paired, output_dir / "sequence_neighbourhood_paired_comparisons.csv")

    summary = {
        "status": "PASS",
        "analysis_scope": "core_sequence_neighbourhood_evaluation",
        "core_scheme_ids": list(CORE_SCHEME_IDS),
        "model_ids": list(MODEL_IDS),
        "primary_sequence_model": SEQUENCE_NEIGHBOURHOOD_MODEL,
        "primary_ablation": MUTATION_CONDITION_MODEL,
        "assay_sequence_source": "genbank_assay_construct",
        "neighbourhood_radii": list(NEIGHBOURHOOD_RADII),
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "selection_metric": "validation_mae",
        "selection_tiebreak": (
            "lower_validation_mae_then_smaller_radius_then_smaller_alpha"
        ),
        "substitution_vocabulary_size": len(SUBSTITUTIONS),
        "substitution_vocabulary_is_predefined": True,
        "padding_state": "PAD",
        "centre_residue_neighbourhood_encoding": "excluded",
        "condition_modes": CONDITION_MODE_BY_SCHEME,
        "cross_protein_condition_predictors": "excluded",
        "target_column": TARGET_COLUMN,
        "target_labels_used_for_model_selection": False,
        "protein_id_used_as_direct_predictor": False,
        "assay_position_used_as_direct_predictor": False,
        "score_original_used_as_predictor": False,
        "score_uncertainty_used_as_predictor": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "outputs": [
            "sequence_neighbourhood_predictions.csv",
            "sequence_neighbourhood_metrics.csv",
            "sequence_neighbourhood_model_selection.csv",
            "sequence_neighbourhood_paired_comparisons.csv",
            "sequence_neighbourhood_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sequence_neighbourhood_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
