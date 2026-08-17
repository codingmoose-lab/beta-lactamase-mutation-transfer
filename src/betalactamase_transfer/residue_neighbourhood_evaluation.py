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

from . import experimental_structure_evaluation as structure_eval


class ResidueNeighbourhoodEvaluationError(RuntimeError):
    """Raised when residue-neighbourhood evaluation cannot be completed safely."""


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
SECONDARY_STRUCTURE_STATES = ("helix", "strand", "coil")
RIDGE_ALPHAS = structure_eval.RIDGE_ALPHAS
TARGET_COLUMN = structure_eval.TARGET_COLUMN
CORE_SCHEME_IDS = structure_eval.CORE_SCHEME_IDS
CONDITION_MODE_BY_SCHEME = structure_eval.CONDITION_MODE_BY_SCHEME

MUTATION_CONDITION_MODEL = structure_eval.MUTATION_CONDITION_MODEL
EXPERIMENTAL_STRUCTURE_MODEL = structure_eval.EXPERIMENTAL_STRUCTURE_MODEL
RESIDUE_NEIGHBOURHOOD_MODEL = "residue_neighbourhood_descriptor_ridge"
MODEL_IDS = (
    MUTATION_CONDITION_MODEL,
    EXPERIMENTAL_STRUCTURE_MODEL,
    RESIDUE_NEIGHBOURHOOD_MODEL,
)
COMPARATOR_MODELS = (
    EXPERIMENTAL_STRUCTURE_MODEL,
    MUTATION_CONDITION_MODEL,
)

EXPECTED_PREDICTIONS_PER_MODEL = 118_204
EXPECTED_FEATURE_ROW_COUNT = 469
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_residue_neighbourhood_bootstrap"
BOOTSTRAP_QUANTILES = (0.025, 0.975)
METRICS = ("spearman", "pearson", "mae", "rmse")

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

NEIGHBOUR_AMINO_ACID_COLUMNS = tuple(
    f"neighbour_amino_acid_fraction_{amino_acid}" for amino_acid in AMINO_ACIDS
)
NEIGHBOUR_SECONDARY_STRUCTURE_COLUMNS = tuple(
    f"neighbour_secondary_structure_fraction_{state}"
    for state in SECONDARY_STRUCTURE_STATES
)
NEIGHBOUR_CONTINUOUS_COLUMNS = (
    "neighbour_count",
    *NEIGHBOUR_AMINO_ACID_COLUMNS,
    *NEIGHBOUR_SECONDARY_STRUCTURE_COLUMNS,
    "neighbour_heavy_atom_count_mean",
    "neighbour_heavy_atom_count_sd",
    "neighbour_sasa_angstrom2_mean",
    "neighbour_sasa_angstrom2_sd",
    "neighbour_relative_sasa_wilke_mean",
    "neighbour_relative_sasa_wilke_sd",
    "neighbour_ca_distance_angstrom_mean",
    "neighbour_ca_distance_angstrom_sd",
    "neighbour_minimum_heavy_atom_distance_angstrom_mean",
    "neighbour_minimum_heavy_atom_distance_angstrom_sd",
    "neighbour_peptide_adjacent_fraction",
    "neighbour_heavy_atom_contact_4_5_angstrom_fraction",
)
if len(NEIGHBOUR_CONTINUOUS_COLUMNS) != 36:
    raise RuntimeError("Residue-neighbourhood descriptor count must equal 36")

MODEL_CONTINUOUS_COLUMNS = (
    *structure_eval.STRUCTURAL_NUMERIC_COLUMNS,
    *NEIGHBOUR_CONTINUOUS_COLUMNS,
)
FEATURE_OUTPUT_COLUMNS = (
    "protein_id",
    "assay_position",
    *NEIGHBOUR_CONTINUOUS_COLUMNS,
)


@dataclass(frozen=True)
class FeatureTransform:
    condition_mode: str
    condition_categories: tuple[str, ...] = ()
    concentration_mean: float = 0.0
    concentration_sd: float = 1.0
    continuous_means: tuple[float, ...] = ()
    continuous_sds: tuple[float, ...] = ()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "nan", "none"}:
        return False
    raise ResidueNeighbourhoodEvaluationError(
        f"Cannot interpret boolean value: {value}"
    )


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    table_name: str,
) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ResidueNeighbourhoodEvaluationError(
            f"{table_name} is missing required columns: {sorted(missing)}"
        )


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ResidueNeighbourhoodEvaluationError(
            "Neighbourhood summary requires non-empty finite values"
        )
    return float(array.mean()), float(array.std(ddof=0))


def build_residue_neighbourhood_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise direct 8-angstrom graph neighbours for every residue node."""
    _require_columns(
        nodes,
        (
            "protein_id",
            "node_index",
            "assay_position",
            "assay_residue",
            "heavy_atom_count",
            "sasa_angstrom2",
            "relative_sasa_wilke",
            "secondary_structure_3state",
        ),
        table_name="residue graph nodes",
    )
    _require_columns(
        edges,
        (
            "protein_id",
            "source_node_index",
            "target_node_index",
            "ca_distance_angstrom",
            "minimum_heavy_atom_distance_angstrom",
            "peptide_adjacent",
            "heavy_atom_contact_4_5_angstrom",
        ),
        table_name="residue graph edges",
    )
    if nodes.duplicated(["protein_id", "node_index"]).any():
        raise ResidueNeighbourhoodEvaluationError(
            "Residue graph node identifiers are not unique"
        )

    node_lookup = {
        (str(row.protein_id), int(row.node_index)): row
        for row in nodes.itertuples(index=False)
    }
    adjacency: dict[tuple[str, int], list[tuple[int, object]]] = {
        key: [] for key in node_lookup
    }
    for edge in edges.itertuples(index=False):
        protein_id = str(edge.protein_id)
        source = int(edge.source_node_index)
        target = int(edge.target_node_index)
        source_key = (protein_id, source)
        target_key = (protein_id, target)
        if source_key not in node_lookup or target_key not in node_lookup:
            raise ResidueNeighbourhoodEvaluationError(
                "Residue graph edge references an unknown node"
            )
        if source == target:
            raise ResidueNeighbourhoodEvaluationError(
                "Residue graph contains a self-loop"
            )
        adjacency[source_key].append((target, edge))
        adjacency[target_key].append((source, edge))

    rows: list[Mapping[str, object]] = []
    for key in sorted(
        node_lookup,
        key=lambda value: (value[0], int(node_lookup[value].assay_position)),
    ):
        centre = node_lookup[key]
        linked = adjacency[key]
        if not linked:
            raise ResidueNeighbourhoodEvaluationError(
                f"Residue node has no direct neighbours: {key}"
            )
        neighbour_nodes = [node_lookup[(key[0], index)] for index, _ in linked]
        neighbour_count = len(neighbour_nodes)
        amino_acids = [str(node.assay_residue) for node in neighbour_nodes]
        unexpected = sorted(set(amino_acids).difference(AMINO_ACIDS))
        if unexpected:
            raise ResidueNeighbourhoodEvaluationError(
                f"Unsupported neighbour amino acids: {unexpected}"
            )
        secondary = [
            structure_eval._normalise_secondary_structure(
                node.secondary_structure_3state
            )
            for node in neighbour_nodes
        ]
        heavy_mean, heavy_sd = _mean_sd(
            [float(node.heavy_atom_count) for node in neighbour_nodes]
        )
        sasa_mean, sasa_sd = _mean_sd(
            [float(node.sasa_angstrom2) for node in neighbour_nodes]
        )
        relative_sasa_mean, relative_sasa_sd = _mean_sd(
            [float(node.relative_sasa_wilke) for node in neighbour_nodes]
        )
        ca_mean, ca_sd = _mean_sd(
            [float(edge.ca_distance_angstrom) for _, edge in linked]
        )
        heavy_distance_mean, heavy_distance_sd = _mean_sd(
            [float(edge.minimum_heavy_atom_distance_angstrom) for _, edge in linked]
        )

        row: dict[str, object] = {
            "protein_id": str(centre.protein_id),
            "assay_position": int(centre.assay_position),
            "neighbour_count": int(neighbour_count),
        }
        for amino_acid, column in zip(
            AMINO_ACIDS,
            NEIGHBOUR_AMINO_ACID_COLUMNS,
            strict=True,
        ):
            row[column] = float(amino_acids.count(amino_acid) / neighbour_count)
        for state, column in zip(
            SECONDARY_STRUCTURE_STATES,
            NEIGHBOUR_SECONDARY_STRUCTURE_COLUMNS,
            strict=True,
        ):
            row[column] = float(secondary.count(state) / neighbour_count)
        row.update(
            {
                "neighbour_heavy_atom_count_mean": heavy_mean,
                "neighbour_heavy_atom_count_sd": heavy_sd,
                "neighbour_sasa_angstrom2_mean": sasa_mean,
                "neighbour_sasa_angstrom2_sd": sasa_sd,
                "neighbour_relative_sasa_wilke_mean": relative_sasa_mean,
                "neighbour_relative_sasa_wilke_sd": relative_sasa_sd,
                "neighbour_ca_distance_angstrom_mean": ca_mean,
                "neighbour_ca_distance_angstrom_sd": ca_sd,
                "neighbour_minimum_heavy_atom_distance_angstrom_mean": (
                    heavy_distance_mean
                ),
                "neighbour_minimum_heavy_atom_distance_angstrom_sd": (
                    heavy_distance_sd
                ),
                "neighbour_peptide_adjacent_fraction": float(
                    np.mean([_as_bool(edge.peptide_adjacent) for _, edge in linked])
                ),
                "neighbour_heavy_atom_contact_4_5_angstrom_fraction": float(
                    np.mean(
                        [
                            _as_bool(edge.heavy_atom_contact_4_5_angstrom)
                            for _, edge in linked
                        ]
                    )
                ),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if list(result.columns) != list(FEATURE_OUTPUT_COLUMNS):
        raise ResidueNeighbourhoodEvaluationError(
            f"Unexpected neighbourhood feature columns: {list(result.columns)}"
        )
    if result.duplicated(["protein_id", "assay_position"]).any():
        raise ResidueNeighbourhoodEvaluationError(
            "Residue-neighbourhood feature keys are not unique"
        )
    values = result[list(NEIGHBOUR_CONTINUOUS_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ResidueNeighbourhoodEvaluationError(
            "Residue-neighbourhood features contain non-finite values"
        )
    return result


def _fit_transform(train: pd.DataFrame, *, condition_mode: str) -> FeatureTransform:
    categories: tuple[str, ...] = ()
    concentration_mean = 0.0
    concentration_sd = 1.0
    if condition_mode == "categorical_condition":
        categories = tuple(sorted(structure_eval._condition_token(train).unique()))
    elif condition_mode == "log2_concentration":
        concentration = train["concentration"].to_numpy(dtype=float)
        if not np.isfinite(concentration).all() or (concentration <= 0).any():
            raise ResidueNeighbourhoodEvaluationError(
                "Positive finite concentration is required"
            )
        values = np.log2(concentration)
        concentration_mean = float(values.mean())
        concentration_sd = float(values.std(ddof=0))
        if not math.isfinite(concentration_sd) or concentration_sd == 0.0:
            concentration_sd = 1.0
    elif condition_mode != "none":
        raise ResidueNeighbourhoodEvaluationError(
            f"Unknown condition mode: {condition_mode}"
        )

    continuous = train[list(MODEL_CONTINUOUS_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(continuous).all():
        raise ResidueNeighbourhoodEvaluationError("Model fitting inputs must be finite")
    means = continuous.mean(axis=0)
    sds = continuous.std(axis=0, ddof=0)
    sds[~np.isfinite(sds) | (sds == 0.0)] = 1.0
    return FeatureTransform(
        condition_mode=condition_mode,
        condition_categories=categories,
        concentration_mean=concentration_mean,
        concentration_sd=concentration_sd,
        continuous_means=tuple(float(value) for value in means),
        continuous_sds=tuple(float(value) for value in sds),
    )


def _feature_count(transform: FeatureTransform) -> int:
    count = 1 + len(structure_eval.SUBSTITUTIONS)
    if transform.condition_mode == "categorical_condition":
        count += len(transform.condition_categories)
    elif transform.condition_mode == "log2_concentration":
        count += 1
    return count + len(MODEL_CONTINUOUS_COLUMNS) + len(SECONDARY_STRUCTURE_STATES)


def _design_matrix(
    frame: pd.DataFrame,
    transform: FeatureTransform,
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    substitution_offset = 1
    condition_offset = substitution_offset + len(structure_eval.SUBSTITUTIONS)
    continuous_offset = condition_offset
    condition_index = {
        value: index for index, value in enumerate(transform.condition_categories)
    }
    if transform.condition_mode == "categorical_condition":
        continuous_offset += len(transform.condition_categories)
        condition_tokens = structure_eval._condition_token(frame).tolist()
    elif transform.condition_mode == "log2_concentration":
        continuous_offset += 1
        condition_tokens = None
    else:
        condition_tokens = None
    secondary_offset = continuous_offset + len(MODEL_CONTINUOUS_COLUMNS)
    secondary_index = {
        state: index for index, state in enumerate(SECONDARY_STRUCTURE_STATES)
    }
    continuous = frame[list(MODEL_CONTINUOUS_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(continuous).all():
        raise ResidueNeighbourhoodEvaluationError(
            "Model application inputs must be finite"
        )
    continuous = (
        continuous - np.asarray(transform.continuous_means, dtype=float)
    ) / np.asarray(transform.continuous_sds, dtype=float)

    for row_number, row in enumerate(frame.itertuples(index=False)):
        row_indices.append(row_number)
        column_indices.append(0)
        values.append(1.0)
        substitution = (str(row.wild_type), str(row.mutant))
        if substitution not in structure_eval.SUBSTITUTION_INDEX:
            raise ResidueNeighbourhoodEvaluationError(
                f"Invalid missense substitution: {substitution}"
            )
        row_indices.append(row_number)
        column_indices.append(
            substitution_offset + structure_eval.SUBSTITUTION_INDEX[substitution]
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
            if not math.isfinite(concentration) or concentration <= 0:
                raise ResidueNeighbourhoodEvaluationError(
                    "Positive finite concentration is required"
                )
            standardised = (
                math.log2(concentration) - transform.concentration_mean
            ) / transform.concentration_sd
            row_indices.append(row_number)
            column_indices.append(condition_offset)
            values.append(float(standardised))
        for feature_index, feature_value in enumerate(continuous[row_number]):
            row_indices.append(row_number)
            column_indices.append(continuous_offset + feature_index)
            values.append(float(feature_value))
        state = structure_eval._normalise_secondary_structure(
            getattr(row, structure_eval.SECONDARY_STRUCTURE_COLUMN)
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
        raise ResidueNeighbourhoodEvaluationError(
            "Residue-neighbourhood design matrix contains non-finite values"
        )
    return matrix


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    condition_mode: str,
    alpha: float,
) -> np.ndarray:
    transform = _fit_transform(train, condition_mode=condition_mode)
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
        raise ResidueNeighbourhoodEvaluationError(
            f"Sparse Ridge did not converge cleanly; lsqr istop={solution[1]}"
        )
    predictions = np.asarray(x_test @ np.asarray(solution[0], dtype=float), dtype=float)
    if not np.isfinite(predictions).all():
        raise ResidueNeighbourhoodEvaluationError(
            "Residue-neighbourhood predictions contain non-finite values"
        )
    return predictions


def _select_alpha(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    condition_mode: str,
) -> tuple[float, float, list[Mapping[str, object]]]:
    detail: list[Mapping[str, object]] = []
    evaluated: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        fold_maes: list[float] = []
        for fold_index, (train, validation) in enumerate(folds):
            predictions = _fit_predict(
                train,
                validation,
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
                    "model_id": RESIDUE_NEIGHBOURHOOD_MODEL,
                    "alpha": alpha,
                    "cv_fold": fold_index,
                    "validation_mae": mae,
                }
            )
        evaluated.append((float(np.mean(fold_maes)), alpha))
    best_mae, best_alpha = min(evaluated, key=lambda item: (item[0], item[1]))
    return best_alpha, best_mae, detail


def _join_scheme_observations(
    membership: pd.DataFrame,
    observations: pd.DataFrame,
    residues: pd.DataFrame,
    neighbourhood: pd.DataFrame,
    scheme_id: str,
) -> pd.DataFrame:
    joined = structure_eval._join_scheme_observations(
        membership,
        observations,
        residues,
        scheme_id,
    )
    joined = joined.merge(
        neighbourhood,
        on=["protein_id", "assay_position"],
        how="left",
        validate="many_to_one",
    )
    if joined[list(NEIGHBOUR_CONTINUOUS_COLUMNS)].isna().any().any():
        raise ResidueNeighbourhoodEvaluationError(
            f"Missing residue-neighbourhood descriptor after joining {scheme_id}"
        )
    return joined


def _prediction_records(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    scheme: pd.Series,
    evaluation_iteration: object,
    evaluation_role: str,
    alpha: float,
    condition_mode: str,
) -> list[Mapping[str, object]]:
    result = pd.DataFrame(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "model_id": RESIDUE_NEIGHBOURHOOD_MODEL,
            "evaluation_iteration": str(evaluation_iteration),
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
    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
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
            raise ResidueNeighbourhoodEvaluationError(
                f"Incomplete residue-neighbourhood split for {scheme_id}/{iteration}"
            )
        alpha, validation_mae, detail = _select_alpha(
            [(train, validation)],
            condition_mode=condition_mode,
        )
        for row in detail:
            selection_rows.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": str(scheme["task_id"]),
                    "evaluation_iteration": str(iteration),
                    "selection_scope": "inner_validation",
                    **row,
                    "selected": row["alpha"] == alpha,
                }
            )
        selection_rows.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": str(iteration),
                "selection_scope": "selected_alpha",
                "model_id": RESIDUE_NEIGHBOURHOOD_MODEL,
                "alpha": alpha,
                "cv_fold": "",
                "validation_mae": validation_mae,
                "selected": True,
            }
        )
        refit_data = pd.concat([train, validation], ignore_index=True)
        y_pred = _fit_predict(
            refit_data,
            test,
            condition_mode=condition_mode,
            alpha=alpha,
        )
        prediction_rows.extend(
            _prediction_records(
                test,
                y_pred,
                scheme=scheme,
                evaluation_iteration=iteration,
                evaluation_role="outer_test",
                alpha=alpha,
                condition_mode=condition_mode,
            )
        )
    return prediction_rows, selection_rows


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
        raise ResidueNeighbourhoodEvaluationError(
            f"Unassigned source CV rows for {scheme_id}"
        )
    if not set(target["group_id"]).issubset(set(target_assignments["group_id"])):
        raise ResidueNeighbourhoodEvaluationError(
            f"Target-test group mismatch for {scheme_id}"
        )
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in sorted(source["source_cv_fold"].astype(int).unique()):
        folds.append(
            (
                source[~source["source_cv_fold"].eq(fold)].copy(),
                source[source["source_cv_fold"].eq(fold)].copy(),
            )
        )
    alpha, validation_mae, detail = _select_alpha(
        folds,
        condition_mode=condition_mode,
    )
    selection_rows: list[Mapping[str, object]] = []
    for row in detail:
        selection_rows.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": "target",
                "selection_scope": "source_cv",
                **row,
                "selected": row["alpha"] == alpha,
            }
        )
    selection_rows.append(
        {
            "scheme_id": scheme_id,
            "task_id": str(scheme["task_id"]),
            "evaluation_iteration": "target",
            "selection_scope": "selected_alpha",
            "model_id": RESIDUE_NEIGHBOURHOOD_MODEL,
            "alpha": alpha,
            "cv_fold": "",
            "validation_mae": validation_mae,
            "selected": True,
        }
    )
    y_pred = _fit_predict(
        source,
        target,
        condition_mode=condition_mode,
        alpha=alpha,
    )
    return (
        _prediction_records(
            target,
            y_pred,
            scheme=scheme,
            evaluation_iteration="target",
            evaluation_role="target_test",
            alpha=alpha,
            condition_mode=condition_mode,
        ),
        selection_rows,
    )


def _validate_graph_summary(repo_root: Path) -> Mapping[str, object]:
    path = repo_root / "data/interim/residue_graphs/residue_graph_summary.json"
    if not path.is_file():
        raise ResidueNeighbourhoodEvaluationError(
            f"Missing residue graph summary: {path}"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ResidueNeighbourhoodEvaluationError(
            "Residue graph summary does not report PASS"
        )
    if float(summary.get("primary_ca_distance_cutoff_angstrom", float("nan"))) != 8.0:
        raise ResidueNeighbourhoodEvaluationError(
            "Residue graph does not use the reviewed 8-angstrom C-alpha definition"
        )
    for field in (
        "outcome_values_used_for_graph_construction",
        "evaluation_assignments_used_for_graph_construction",
        "mechanistic_reference_fields_used_for_graph_construction",
        "reported_homologue_fields_used_for_graph_construction",
    ):
        if summary.get(field) is not False:
            raise ResidueNeighbourhoodEvaluationError(
                f"Residue graph summary does not establish {field}=False"
            )
    if summary.get("all_core_tested_positions_have_graph_nodes") is not True:
        raise ResidueNeighbourhoodEvaluationError(
            "Residue graph does not cover every core structure-supported test position"
        )
    return summary


def _load_existing_structure_predictions(repo_root: Path) -> pd.DataFrame:
    path = (
        repo_root
        / "data/interim/experimental_structure_evaluation"
        / "experimental_structure_predictions.csv"
    )
    if not path.is_file():
        raise ResidueNeighbourhoodEvaluationError(
            f"Missing Script-27 predictions: {path}"
        )
    predictions = pd.read_csv(
        path,
        dtype={"evaluation_iteration": "string"},
        low_memory=False,
    )
    expected_models = {MUTATION_CONDITION_MODEL, EXPERIMENTAL_STRUCTURE_MODEL}
    if set(predictions["model_id"].astype(str)) != expected_models:
        raise ResidueNeighbourhoodEvaluationError(
            "Script-27 predictions do not contain the two expected models"
        )
    counts = predictions.groupby("model_id").size().to_dict()
    if any(
        int(counts.get(model_id, -1)) != EXPECTED_PREDICTIONS_PER_MODEL
        for model_id in expected_models
    ):
        raise ResidueNeighbourhoodEvaluationError(
            f"Unexpected Script-27 prediction counts: {counts}"
        )
    if predictions.duplicated([*COMPARISON_KEY, "model_id"]).any():
        raise ResidueNeighbourhoodEvaluationError(
            "Script-27 predictions contain duplicate comparison keys"
        )
    return predictions


def _validate_matched_predictions(
    existing: pd.DataFrame,
    neighbourhood: pd.DataFrame,
) -> pd.DataFrame:
    reference = existing[existing["model_id"].eq(EXPERIMENTAL_STRUCTURE_MODEL)].copy()
    if len(neighbourhood) != EXPECTED_PREDICTIONS_PER_MODEL:
        raise ResidueNeighbourhoodEvaluationError(
            f"Unexpected residue-neighbourhood prediction count: {len(neighbourhood)}"
        )
    merged = reference[list(COMPARISON_KEY)].merge(
        neighbourhood[list(COMPARISON_KEY)],
        on=list(COMPARISON_KEY),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ResidueNeighbourhoodEvaluationError(
            "Residue-neighbourhood and Script-27 test populations disagree"
        )
    compare = reference[[*COMPARISON_KEY, *MATCHED_FIELDS]].merge(
        neighbourhood[[*COMPARISON_KEY, *MATCHED_FIELDS]],
        on=list(COMPARISON_KEY),
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_new"),
    )
    for field in MATCHED_FIELDS:
        left = compare[f"{field}_reference"]
        right = compare[f"{field}_new"]
        if pd.api.types.is_numeric_dtype(left):
            absolute_tolerance = 1e-12 if field == "y_true" else 0.0
            same = bool(
                np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=absolute_tolerance,
                    equal_nan=True,
                )
            )
        else:
            same = left.reset_index(drop=True).equals(right.reset_index(drop=True))
        if not same:
            raise ResidueNeighbourhoodEvaluationError(
                f"Matched field disagrees with Script-27 predictions: {field}"
            )
    combined = pd.concat([existing, neighbourhood], ignore_index=True)
    combined["evaluation_iteration"] = combined["evaluation_iteration"].astype(str)
    return combined.sort_values(
        [*COMPARISON_KEY, "model_id"],
        kind="stable",
    ).reset_index(drop=True)


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    for (scheme_id, model_id), group in predictions.groupby(
        ["scheme_id", "model_id"],
        sort=True,
    ):
        grouped = list(group.groupby("evaluation_iteration", sort=True))
        grouped.append(("pooled", group))
        for evaluation_iteration, subset in grouped:
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_id": model_id,
                    "evaluation_iteration": evaluation_iteration,
                    "n_observations": len(subset),
                    "n_groups": subset["evaluation_group_id"].nunique(),
                    **structure_eval._metric_values(
                        subset["y_true"].to_numpy(dtype=float),
                        subset["y_pred"].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _numeric_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _metric_difference(
    metric: str,
    model_a_value: float,
    model_b_value: float,
) -> float:
    if metric in {"spearman", "pearson"}:
        return model_a_value - model_b_value
    if metric in {"mae", "rmse"}:
        return model_b_value - model_a_value
    raise ResidueNeighbourhoodEvaluationError(f"Unsupported metric: {metric}")


def _paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    for scheme_id, scheme_rows in predictions.groupby("scheme_id", sort=True):
        model_a = scheme_rows[
            scheme_rows["model_id"].eq(RESIDUE_NEIGHBOURHOOD_MODEL)
        ].copy()
        for model_b_id in COMPARATOR_MODELS:
            model_b = scheme_rows[scheme_rows["model_id"].eq(model_b_id)].copy()
            paired = model_a[
                [*COMPARISON_KEY, "evaluation_group_id", "y_true", "y_pred"]
            ].merge(
                model_b[[*COMPARISON_KEY, "y_true", "y_pred"]],
                on=list(COMPARISON_KEY),
                how="inner",
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            if len(paired) != len(model_a) or len(paired) != len(model_b):
                raise ResidueNeighbourhoodEvaluationError(
                    f"Paired population mismatch for {scheme_id}/{model_b_id}"
                )
            if not np.allclose(
                paired["y_true_a"].to_numpy(dtype=float),
                paired["y_true_b"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ResidueNeighbourhoodEvaluationError(
                    f"Paired target values disagree for {scheme_id}/{model_b_id}"
                )
            group_ids = sorted(paired["evaluation_group_id"].unique())
            blocks = {
                group_id: paired[paired["evaluation_group_id"].eq(group_id)].copy()
                for group_id in group_ids
            }
            rng = np.random.default_rng(
                _numeric_seed(BOOTSTRAP_SEED, scheme_id, model_b_id)
            )
            point_a = structure_eval._metric_values(
                paired["y_true_a"].to_numpy(dtype=float),
                paired["y_pred_a"].to_numpy(dtype=float),
            )
            point_b = structure_eval._metric_values(
                paired["y_true_a"].to_numpy(dtype=float),
                paired["y_pred_b"].to_numpy(dtype=float),
            )
            distributions = {metric: [] for metric in METRICS}
            for _ in range(BOOTSTRAP_REPLICATES):
                sampled_groups = rng.choice(
                    group_ids,
                    size=len(group_ids),
                    replace=True,
                )
                boot = pd.concat(
                    [blocks[group_id] for group_id in sampled_groups],
                    ignore_index=True,
                )
                boot_a = structure_eval._metric_values(
                    boot["y_true_a"].to_numpy(dtype=float),
                    boot["y_pred_a"].to_numpy(dtype=float),
                )
                boot_b = structure_eval._metric_values(
                    boot["y_true_a"].to_numpy(dtype=float),
                    boot["y_pred_b"].to_numpy(dtype=float),
                )
                for metric in METRICS:
                    difference = _metric_difference(
                        metric,
                        boot_a[metric],
                        boot_b[metric],
                    )
                    if np.isfinite(difference):
                        distributions[metric].append(difference)
            for metric in METRICS:
                distribution = np.asarray(distributions[metric], dtype=float)
                if len(distribution) == 0:
                    raise ResidueNeighbourhoodEvaluationError(
                        "No valid bootstrap values for "
                        f"{scheme_id}/{model_b_id}/{metric}"
                    )
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_a": RESIDUE_NEIGHBOURHOOD_MODEL,
                        "model_b": model_b_id,
                        "metric": metric,
                        "model_a_value": point_a[metric],
                        "model_b_value": point_b[metric],
                        "paired_difference": _metric_difference(
                            metric,
                            point_a[metric],
                            point_b[metric],
                        ),
                        "ci_lower": float(
                            np.quantile(distribution, BOOTSTRAP_QUANTILES[0])
                        ),
                        "ci_upper": float(
                            np.quantile(distribution, BOOTSTRAP_QUANTILES[1])
                        ),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_valid_replicates": len(distribution),
                        "resampling_unit": "evaluation_group_id",
                        "difference_interpretation": "positive_favours_model_a",
                    }
                )
    return pd.DataFrame(rows)


def prepare_residue_neighbourhood_evaluation(
    repo_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, object],
]:
    repo_root = repo_root.resolve()
    graph_summary = _validate_graph_summary(repo_root)
    observations = pd.read_csv(
        repo_root / "data/processed/missense_observations.csv",
        low_memory=False,
    )
    residues = pd.read_csv(
        repo_root / "data/processed/residue_features.csv",
        low_memory=False,
    )
    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv",
        low_memory=False,
    )
    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv",
        low_memory=False,
    )
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv",
        low_memory=False,
    )
    nodes = pd.read_csv(
        repo_root / "data/interim/residue_graphs/residue_graph_nodes.csv",
        low_memory=False,
    )
    edges = pd.read_csv(
        repo_root / "data/interim/residue_graphs/residue_graph_edges.csv",
        low_memory=False,
    )
    structure_eval._validate_structure_features(residues)
    neighbourhood_features = build_residue_neighbourhood_features(nodes, edges)
    if len(neighbourhood_features) != EXPECTED_FEATURE_ROW_COUNT:
        raise ResidueNeighbourhoodEvaluationError(
            "Unexpected residue-neighbourhood feature row count: "
            f"{len(neighbourhood_features)}"
        )
    if TARGET_COLUMN in membership.columns or "score_original" in membership.columns:
        raise ResidueNeighbourhoodEvaluationError(
            "Outcome columns entered modelling membership"
        )
    selected_schemes = schemes[schemes["scheme_id"].isin(CORE_SCHEME_IDS)].copy()
    if set(selected_schemes["scheme_id"]) != set(CORE_SCHEME_IDS):
        raise ResidueNeighbourhoodEvaluationError(
            "Core evaluation scheme contract is incomplete"
        )
    if not selected_schemes["status"].eq("ready").all():
        raise ResidueNeighbourhoodEvaluationError(
            "Core evaluation schemes are not all ready"
        )

    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
    for _, scheme in selected_schemes.sort_values("scheme_id").iterrows():
        scheme_id = str(scheme["scheme_id"])
        frame = _join_scheme_observations(
            membership,
            observations,
            residues,
            neighbourhood_features,
            scheme_id,
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
            raise ResidueNeighbourhoodEvaluationError(
                f"Unsupported assignment family: {family}"
            )
        prediction_rows.extend(scheme_predictions)
        selection_rows.extend(scheme_selection)

    neighbourhood_predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows)
    existing_predictions = _load_existing_structure_predictions(repo_root)
    combined_predictions = _validate_matched_predictions(
        existing_predictions,
        neighbourhood_predictions,
    )
    metrics = _metric_rows(combined_predictions)
    paired = _paired_bootstrap(combined_predictions)

    output_dir = repo_root / "data/interim/residue_neighbourhood_evaluation"
    feature_path = output_dir / "residue_neighbourhood_features.csv"
    prediction_path = output_dir / "residue_neighbourhood_predictions.csv"
    metric_path = output_dir / "residue_neighbourhood_metrics.csv"
    selection_path = output_dir / "residue_neighbourhood_model_selection.csv"
    paired_path = output_dir / "residue_neighbourhood_paired_comparisons.csv"
    summary_path = output_dir / "residue_neighbourhood_summary.json"
    _write_csv(neighbourhood_features, feature_path)
    _write_csv(combined_predictions, prediction_path)
    _write_csv(metrics, metric_path)
    _write_csv(selection, selection_path)
    _write_csv(paired, paired_path)

    counts = combined_predictions.groupby("model_id").size().to_dict()
    selected_alpha_rows = selection[selection["selection_scope"].eq("selected_alpha")]
    if len(selected_alpha_rows) != 17:
        raise ResidueNeighbourhoodEvaluationError(
            f"Unexpected selected-alpha row count: {len(selected_alpha_rows)}"
        )
    summary: dict[str, object] = {
        "status": "PASS",
        "analysis_scope": "experimental_structure_residue_neighbourhood_evaluation",
        "core_scheme_ids": list(CORE_SCHEME_IDS),
        "model_ids": list(MODEL_IDS),
        "primary_model": RESIDUE_NEIGHBOURHOOD_MODEL,
        "primary_comparator": EXPERIMENTAL_STRUCTURE_MODEL,
        "secondary_comparator": MUTATION_CONDITION_MODEL,
        "coordinate_source": graph_summary["coordinate_source"],
        "direct_neighbour_definition": (
            "residues connected by the reviewed C-alpha distance <= 8 angstrom graph"
        ),
        "graph_ca_distance_cutoff_angstrom": 8.0,
        "neighbour_descriptor_count": len(NEIGHBOUR_CONTINUOUS_COLUMNS),
        "neighbour_descriptors": list(NEIGHBOUR_CONTINUOUS_COLUMNS),
        "centre_structural_numeric_inputs": list(
            structure_eval.STRUCTURAL_NUMERIC_COLUMNS
        ),
        "centre_secondary_structure_input": structure_eval.SECONDARY_STRUCTURE_COLUMN,
        "centre_secondary_structure_states": list(SECONDARY_STRUCTURE_STATES),
        "absolute_coordinates_used_as_direct_model_inputs": False,
        "mechanistic_reference_inputs_used": False,
        "catalytic_zinc_distance_inputs_used": False,
        "reported_homologue_pair_inputs_used": False,
        "outcome_values_used_for_neighbourhood_construction": False,
        "evaluation_assignments_used_for_neighbourhood_construction": False,
        "condition_modes": dict(CONDITION_MODE_BY_SCHEME),
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "selection_metric": "validation_mae",
        "selection_tiebreak": "lower_validation_mae_then_lower_alpha",
        "continuous_input_scaling": (
            "fitting-data mean and population standard deviation within each "
            "validation or source-CV fit"
        ),
        "target_column": TARGET_COLUMN,
        "target_test_labels_used_for_alpha_selection": False,
        "prediction_count_by_model": {
            str(key): int(value) for key, value in counts.items()
        },
        "feature_row_count": int(len(neighbourhood_features)),
        "paired_model_comparisons": [
            {"model_a": RESIDUE_NEIGHBOURHOOD_MODEL, "model_b": model_b}
            for model_b in COMPARATOR_MODELS
        ],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resampling_unit": "evaluation_group_id",
        "bootstrap_interval_method": "percentile",
        "bootstrap_quantiles": list(BOOTSTRAP_QUANTILES),
        "difference_definition": {
            "spearman": "model_a_minus_model_b",
            "pearson": "model_a_minus_model_b",
            "mae": "model_b_minus_model_a",
            "rmse": "model_b_minus_model_a",
        },
        "difference_interpretation": "positive_favours_model_a",
        "script_27_predictions_reused": True,
        "outputs": [
            feature_path.name,
            prediction_path.name,
            metric_path.name,
            selection_path.name,
            paired_path.name,
            summary_path.name,
        ],
        "input_sha256": {
            "missense_observations": _sha256_file(
                repo_root / "data/processed/missense_observations.csv"
            ),
            "residue_features": _sha256_file(
                repo_root / "data/processed/residue_features.csv"
            ),
            "scheme_observation_membership": _sha256_file(
                repo_root
                / "data/interim/modelling_views/scheme_observation_membership.csv"
            ),
            "partition_schemes": _sha256_file(
                repo_root / "data/interim/evaluation_partitions/partition_schemes.csv"
            ),
            "group_assignments": _sha256_file(
                repo_root / "data/interim/evaluation_partitions/group_assignments.csv"
            ),
            "residue_graph_nodes": _sha256_file(
                repo_root / "data/interim/residue_graphs/residue_graph_nodes.csv"
            ),
            "residue_graph_edges": _sha256_file(
                repo_root / "data/interim/residue_graphs/residue_graph_edges.csv"
            ),
            "experimental_structure_predictions": _sha256_file(
                repo_root
                / "data/interim/experimental_structure_evaluation"
                / "experimental_structure_predictions.csv"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        neighbourhood_features,
        combined_predictions,
        metrics,
        selection,
        paired,
        summary,
    )


def run_residue_neighbourhood_evaluation(repo_root: Path) -> Mapping[str, object]:
    """Run and save the residue-neighbourhood structural evaluation."""
    return prepare_residue_neighbourhood_evaluation(repo_root)[-1]
