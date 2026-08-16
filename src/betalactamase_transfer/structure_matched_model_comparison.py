from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from . import experimental_structure_evaluation as structure_eval
from . import sequence_neighbourhood_evaluation as sequence_eval
from . import sequence_plm_evaluation as plm_eval


class StructureMatchedModelComparisonError(RuntimeError):
    """Raised when the structure-matched model comparison cannot be completed safely."""


STRUCTURE_BASELINE_MODEL = structure_eval.MUTATION_CONDITION_MODEL
EXPERIMENTAL_STRUCTURE_MODEL = structure_eval.EXPERIMENTAL_STRUCTURE_MODEL
MATCHED_SEQUENCE_MODEL = "structure_matched_sequence_neighbourhood_ridge"
MATCHED_ESMC_MODEL = "structure_matched_esmc_site_delta_ridge"

MODEL_IDS = (
    STRUCTURE_BASELINE_MODEL,
    MATCHED_SEQUENCE_MODEL,
    MATCHED_ESMC_MODEL,
    EXPERIMENTAL_STRUCTURE_MODEL,
)

MODEL_PAIRS = (
    (MATCHED_SEQUENCE_MODEL, STRUCTURE_BASELINE_MODEL),
    (MATCHED_ESMC_MODEL, STRUCTURE_BASELINE_MODEL),
    (EXPERIMENTAL_STRUCTURE_MODEL, STRUCTURE_BASELINE_MODEL),
    (MATCHED_ESMC_MODEL, MATCHED_SEQUENCE_MODEL),
    (EXPERIMENTAL_STRUCTURE_MODEL, MATCHED_SEQUENCE_MODEL),
    (MATCHED_ESMC_MODEL, EXPERIMENTAL_STRUCTURE_MODEL),
)

METRICS = ("spearman", "pearson", "mae", "rmse")
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_structure_matched_model_bootstrap"
BOOTSTRAP_QUANTILES = (0.025, 0.975)

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

EXPECTED_PREDICTIONS_PER_MODEL = 118_204
EXPECTED_SCHEME_COUNTS = {
    "ndm1_position_nested5": 40_248,
    "ndm1_to_vim2_amp_source_cv5": 4_324,
    "vim2_2020_position_nested5": 38_928,
    "vim2_2022_mutation_nested5": 30_232,
    "vim2_to_ndm1_amp_source_cv5": 4_472,
}


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


def _metric_difference(
    metric: str,
    model_a_value: float,
    model_b_value: float,
) -> float:
    if metric in {"spearman", "pearson"}:
        return model_a_value - model_b_value
    if metric in {"mae", "rmse"}:
        return model_b_value - model_a_value
    raise StructureMatchedModelComparisonError(f"Unsupported metric: {metric}")


def _load_matched_inputs(
    repo_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    Mapping[str, str],
]:
    (
        observations,
        schemes,
        assignments,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = plm_eval._load_inputs(repo_root)

    residues = pd.read_csv(
        repo_root / "data/processed/residue_features.csv",
        low_memory=False,
    )

    structure_eval._validate_structure_features(residues)
    plm_eval._validate_inputs(
        observations,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    plm_eval._validate_embedding_provenance(
        repo_root,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )

    if "experimental_structure_task_eligible" not in membership.columns:
        raise StructureMatchedModelComparisonError(
            "Experimental-structure eligibility is missing from modelling membership"
        )

    matched_membership = membership[
        membership["experimental_structure_task_eligible"].astype(bool)
    ].copy()

    assay_sequences = sequence_eval._load_assay_sequences(repo_root)
    return (
        observations,
        residues,
        schemes,
        assignments,
        matched_membership,
        mutant_embeddings,
        wild_type_embeddings,
        assay_sequences,
    )


def _matched_scheme_frame(
    *,
    scheme_id: str,
    membership: pd.DataFrame,
    observations: pd.DataFrame,
    residues: pd.DataFrame,
    embedding_index: pd.DataFrame,
) -> pd.DataFrame:
    structure_frame = structure_eval._join_scheme_observations(
        membership,
        observations,
        residues,
        scheme_id,
    )
    esmc_frame = plm_eval._join_scheme_observations(
        membership,
        observations,
        embedding_index,
        scheme_id,
    )

    left_ids = set(structure_frame["observation_id"].astype(str))
    right_ids = set(esmc_frame["observation_id"].astype(str))
    if left_ids != right_ids:
        raise StructureMatchedModelComparisonError(
            f"Structure and ESMC matched populations disagree for {scheme_id}"
        )

    embedding_rows = esmc_frame[["observation_id", "embedding_row"]].copy()
    if embedding_rows["observation_id"].duplicated().any():
        raise StructureMatchedModelComparisonError(
            f"Duplicate ESMC observation identifiers for {scheme_id}"
        )

    frame = structure_frame.merge(
        embedding_rows,
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if frame["embedding_row"].isna().any():
        raise StructureMatchedModelComparisonError(
            f"Missing ESMC embedding rows after matched join for {scheme_id}"
        )
    frame["embedding_row"] = frame["embedding_row"].astype(int)
    return frame


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
    radius: int | None = None,
    hidden_state_index: int | None = None,
) -> list[Mapping[str, object]]:
    result = pd.DataFrame(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "model_id": model_id,
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
            "y_true": test[structure_eval.TARGET_COLUMN].to_numpy(dtype=float),
            "y_pred": predictions,
            "selected_radius": "" if radius is None else radius,
            "selected_hidden_state_index": (
                "" if hidden_state_index is None else hidden_state_index
            ),
            "selected_alpha": alpha,
            "condition_mode": condition_mode,
        }
    )
    return result.to_dict(orient="records")


def _selection_rows(
    *,
    scheme_id: str,
    task_id: str,
    evaluation_iteration: object,
    selection_scope: str,
    model_id: str,
    detail: list[Mapping[str, object]],
    alpha: float,
    validation_mae: float,
    radius: int | None = None,
    hidden_state_index: int | None = None,
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for row in detail:
        row_radius = row.get("radius", "")
        row_hidden = row.get("hidden_state_index", "")
        selected = row["alpha"] == alpha
        if radius is not None:
            selected = selected and row_radius == radius
        if hidden_state_index is not None:
            selected = selected and row_hidden == hidden_state_index
        rows.append(
            {
                "scheme_id": scheme_id,
                "task_id": task_id,
                "evaluation_iteration": str(evaluation_iteration),
                "selection_scope": selection_scope,
                "model_id": model_id,
                "radius": row_radius,
                "hidden_state_index": row_hidden,
                "alpha": row["alpha"],
                "cv_fold": row["cv_fold"],
                "validation_mae": row["validation_mae"],
                "selected": selected,
            }
        )

    rows.append(
        {
            "scheme_id": scheme_id,
            "task_id": task_id,
            "evaluation_iteration": str(evaluation_iteration),
            "selection_scope": "selected_setting",
            "model_id": model_id,
            "radius": "" if radius is None else radius,
            "hidden_state_index": (
                "" if hidden_state_index is None else hidden_state_index
            ),
            "alpha": alpha,
            "cv_fold": "",
            "validation_mae": validation_mae,
            "selected": True,
        }
    )
    return rows


def _evaluate_sequence_model(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    refit_data: pd.DataFrame,
    test: pd.DataFrame,
    *,
    scheme: pd.Series,
    evaluation_iteration: object,
    evaluation_role: str,
    selection_scope: str,
    assay_sequences: Mapping[str, str],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    condition_mode = sequence_eval.CONDITION_MODE_BY_SCHEME[scheme_id]
    radius, alpha, validation_mae, detail = sequence_eval._select_setting(
        folds,
        model_id=sequence_eval.SEQUENCE_NEIGHBOURHOOD_MODEL,
        assay_sequences=assay_sequences,
        condition_mode=condition_mode,
    )
    predictions = sequence_eval._fit_predict(
        refit_data,
        test,
        assay_sequences=assay_sequences,
        radius=radius,
        condition_mode=condition_mode,
        alpha=alpha,
    )
    prediction_rows = _prediction_records(
        test,
        predictions,
        scheme=scheme,
        model_id=MATCHED_SEQUENCE_MODEL,
        evaluation_iteration=evaluation_iteration,
        evaluation_role=evaluation_role,
        radius=radius,
        alpha=alpha,
        condition_mode=condition_mode,
    )
    selection_rows = _selection_rows(
        scheme_id=scheme_id,
        task_id=str(scheme["task_id"]),
        evaluation_iteration=evaluation_iteration,
        selection_scope=selection_scope,
        model_id=MATCHED_SEQUENCE_MODEL,
        detail=detail,
        radius=radius,
        alpha=alpha,
        validation_mae=validation_mae,
    )
    return prediction_rows, selection_rows


def _evaluate_esmc_model(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    refit_data: pd.DataFrame,
    test: pd.DataFrame,
    *,
    scheme: pd.Series,
    evaluation_iteration: object,
    evaluation_role: str,
    selection_scope: str,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    scheme_id = str(scheme["scheme_id"])
    condition_mode = sequence_eval.CONDITION_MODE_BY_SCHEME[scheme_id]
    hidden_state_index, alpha, validation_mae, detail = plm_eval._select_model_setting(
        folds,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
    )
    predictions = plm_eval._fit_predict(
        refit_data,
        test,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
        hidden_state_index=hidden_state_index,
        alpha=alpha,
    )
    prediction_rows = _prediction_records(
        test,
        predictions,
        scheme=scheme,
        model_id=MATCHED_ESMC_MODEL,
        evaluation_iteration=evaluation_iteration,
        evaluation_role=evaluation_role,
        hidden_state_index=hidden_state_index,
        alpha=alpha,
        condition_mode=condition_mode,
    )
    selection_rows = _selection_rows(
        scheme_id=scheme_id,
        task_id=str(scheme["task_id"]),
        evaluation_iteration=evaluation_iteration,
        selection_scope=selection_scope,
        model_id=MATCHED_ESMC_MODEL,
        detail=detail,
        hidden_state_index=hidden_state_index,
        alpha=alpha,
        validation_mae=validation_mae,
    )
    return prediction_rows, selection_rows


def _evaluate_new_matched_models(
    schemes: pd.DataFrame,
    assignments: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    assay_sequences: Mapping[str, str],
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []

    for _, scheme in schemes.sort_values("scheme_id").iterrows():
        scheme_id = str(scheme["scheme_id"])
        frame = frames[scheme_id]
        family = str(scheme["assignment_family"])
        scheme_assignments = assignments[assignments["scheme_id"].eq(scheme_id)]

        if family == "five_outer_folds_with_grouped_inner_validation":
            for iteration in sorted(
                scheme_assignments["iteration"].dropna().astype(int).unique()
            ):
                current = scheme_assignments[
                    scheme_assignments["iteration"].eq(iteration)
                ]
                roles = frame["group_id"].map(
                    dict(zip(current["group_id"], current["split_role"]))
                )
                train = frame[roles.eq("train")].copy()
                validation = frame[roles.eq("validation")].copy()
                test = frame[roles.eq("test")].copy()
                if train.empty or validation.empty or test.empty:
                    raise StructureMatchedModelComparisonError(
                        f"Incomplete matched split for {scheme_id}/{iteration}"
                    )
                folds = [(train, validation)]
                refit_data = pd.concat([train, validation], ignore_index=True)

                seq_predictions, seq_selection = _evaluate_sequence_model(
                    folds,
                    refit_data,
                    test,
                    scheme=scheme,
                    evaluation_iteration=iteration,
                    evaluation_role="outer_test",
                    selection_scope="inner_validation",
                    assay_sequences=assay_sequences,
                )
                esmc_predictions, esmc_selection = _evaluate_esmc_model(
                    folds,
                    refit_data,
                    test,
                    scheme=scheme,
                    evaluation_iteration=iteration,
                    evaluation_role="outer_test",
                    selection_scope="inner_validation",
                    mutant_embeddings=mutant_embeddings,
                    wild_type_embeddings=wild_type_embeddings,
                )
                prediction_rows.extend(seq_predictions)
                prediction_rows.extend(esmc_predictions)
                selection_rows.extend(seq_selection)
                selection_rows.extend(esmc_selection)

        elif family == "source_only_grouped_cv_then_target_test":
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
                raise StructureMatchedModelComparisonError(
                    f"Unassigned matched source CV rows for {scheme_id}"
                )
            target_groups = set(target["group_id"])
            expected_target_groups = set(target_assignments["group_id"])
            if not target_groups.issubset(expected_target_groups):
                raise StructureMatchedModelComparisonError(
                    f"Matched target-test group mismatch for {scheme_id}"
                )

            folds = []
            for fold in sorted(source["source_cv_fold"].astype(int).unique()):
                folds.append(
                    (
                        source[~source["source_cv_fold"].eq(fold)].copy(),
                        source[source["source_cv_fold"].eq(fold)].copy(),
                    )
                )

            seq_predictions, seq_selection = _evaluate_sequence_model(
                folds,
                source,
                target,
                scheme=scheme,
                evaluation_iteration="target",
                evaluation_role="target_test",
                selection_scope="source_cv",
                assay_sequences=assay_sequences,
            )
            esmc_predictions, esmc_selection = _evaluate_esmc_model(
                folds,
                source,
                target,
                scheme=scheme,
                evaluation_iteration="target",
                evaluation_role="target_test",
                selection_scope="source_cv",
                mutant_embeddings=mutant_embeddings,
                wild_type_embeddings=wild_type_embeddings,
            )
            prediction_rows.extend(seq_predictions)
            prediction_rows.extend(esmc_predictions)
            selection_rows.extend(seq_selection)
            selection_rows.extend(esmc_selection)

        else:
            raise StructureMatchedModelComparisonError(
                f"Unsupported assignment family: {family}"
            )

    return pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def _load_existing_structure_results(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object], Mapping[str, str]]:
    output_dir = repo_root / "data/interim/experimental_structure_evaluation"
    prediction_path = output_dir / "experimental_structure_predictions.csv"
    selection_path = output_dir / "experimental_structure_model_selection.csv"
    summary_path = output_dir / "experimental_structure_summary.json"
    for path in (prediction_path, selection_path, summary_path):
        if not path.is_file():
            raise StructureMatchedModelComparisonError(
                f"Required Script-27 output is missing: {path}"
            )

    predictions = pd.read_csv(
        prediction_path,
        dtype={"evaluation_iteration": "string"},
        low_memory=False,
    )
    selection = pd.read_csv(
        selection_path,
        dtype={"evaluation_iteration": "string"},
        low_memory=False,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_models = {STRUCTURE_BASELINE_MODEL, EXPERIMENTAL_STRUCTURE_MODEL}
    if set(predictions["model_id"].astype(str)) != expected_models:
        raise StructureMatchedModelComparisonError(
            "Unexpected models in Script-27 structural predictions"
        )
    if summary.get("target_labels_used_for_model_selection") is not False:
        raise StructureMatchedModelComparisonError(
            "Script-27 summary does not establish target-label separation"
        )

    counts = predictions.groupby("model_id").size().to_dict()
    if any(
        int(counts.get(model_id, -1)) != EXPECTED_PREDICTIONS_PER_MODEL
        for model_id in expected_models
    ):
        raise StructureMatchedModelComparisonError(
            f"Unexpected Script-27 prediction counts: {counts}"
        )

    hashes = {
        "experimental_structure_predictions_sha256": _sha256_file(prediction_path),
        "experimental_structure_model_selection_sha256": _sha256_file(selection_path),
    }
    return predictions, selection, summary, hashes


def _normalise_existing_structure_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["evaluation_iteration"] = result["evaluation_iteration"].astype(str)
    result["selected_radius"] = ""
    result["selected_hidden_state_index"] = ""
    columns = [
        "scheme_id",
        "task_id",
        "model_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
        "protein_id",
        "assay_position",
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
        "evaluation_group_id",
        "y_true",
        "y_pred",
        "selected_radius",
        "selected_hidden_state_index",
        "selected_alpha",
        "condition_mode",
    ]
    missing = set(columns).difference(result.columns)
    if missing:
        raise StructureMatchedModelComparisonError(
            f"Script-27 predictions are missing columns: {sorted(missing)}"
        )
    return result[columns].copy()


def _normalise_existing_structure_selection(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["evaluation_iteration"] = result["evaluation_iteration"].astype(str)
    result["radius"] = ""
    result["hidden_state_index"] = ""
    columns = [
        "scheme_id",
        "task_id",
        "evaluation_iteration",
        "selection_scope",
        "model_id",
        "radius",
        "hidden_state_index",
        "alpha",
        "cv_fold",
        "validation_mae",
        "selected",
    ]
    missing = set(columns).difference(result.columns)
    if missing:
        raise StructureMatchedModelComparisonError(
            f"Script-27 model selection is missing columns: {sorted(missing)}"
        )
    return result[columns].copy()


def _validate_combined_predictions(predictions: pd.DataFrame) -> None:
    required = (
        set(COMPARISON_KEY)
        | set(MATCHED_FIELDS)
        | {
            "model_id",
            "y_pred",
            "selected_alpha",
        }
    )
    missing = required.difference(predictions.columns)
    if missing:
        raise StructureMatchedModelComparisonError(
            f"Combined predictions are missing columns: {sorted(missing)}"
        )
    if set(predictions["model_id"].astype(str)) != set(MODEL_IDS):
        raise StructureMatchedModelComparisonError(
            "Combined prediction table does not contain the four declared models"
        )
    if predictions.duplicated([*COMPARISON_KEY, "model_id"]).any():
        raise StructureMatchedModelComparisonError(
            "Combined predictions contain duplicate model/comparison keys"
        )
    if predictions["y_true"].isna().any() or predictions["y_pred"].isna().any():
        raise StructureMatchedModelComparisonError(
            "Combined predictions contain missing truth or prediction values"
        )
    if not np.isfinite(predictions["y_true"].to_numpy(dtype=float)).all():
        raise StructureMatchedModelComparisonError("Non-finite matched truth values")
    if not np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all():
        raise StructureMatchedModelComparisonError("Non-finite matched predictions")

    counts = predictions.groupby("model_id").size().to_dict()
    if any(
        int(counts.get(model_id, -1)) != EXPECTED_PREDICTIONS_PER_MODEL
        for model_id in MODEL_IDS
    ):
        raise StructureMatchedModelComparisonError(
            f"Unexpected matched prediction counts by model: {counts}"
        )


def _matched_prediction_table(predictions: pd.DataFrame) -> pd.DataFrame:
    _validate_combined_predictions(predictions)
    reference = predictions[predictions["model_id"].eq(STRUCTURE_BASELINE_MODEL)].copy()
    base_columns = [*COMPARISON_KEY, *MATCHED_FIELDS]
    matched = reference[base_columns].copy()
    matched[STRUCTURE_BASELINE_MODEL] = reference["y_pred"].to_numpy(dtype=float)

    for model_id in MODEL_IDS:
        if model_id == STRUCTURE_BASELINE_MODEL:
            continue
        candidate = predictions[predictions["model_id"].eq(model_id)].copy()
        candidate_columns = [*COMPARISON_KEY, *MATCHED_FIELDS, "y_pred"]
        rename = {field: f"{field}__candidate" for field in MATCHED_FIELDS}
        rename["y_pred"] = model_id
        candidate = candidate[candidate_columns].rename(columns=rename)
        matched = matched.merge(
            candidate,
            on=list(COMPARISON_KEY),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not matched["_merge"].eq("both").all():
            raise StructureMatchedModelComparisonError(
                f"Matched comparison-key mismatch for {model_id}"
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
                raise StructureMatchedModelComparisonError(
                    f"Matched field disagrees for {model_id}: {field}"
                )
            matched = matched.drop(columns=candidate_field)

    if len(matched) != EXPECTED_PREDICTIONS_PER_MODEL:
        raise StructureMatchedModelComparisonError(
            f"Unexpected matched observation count: {len(matched)}"
        )
    counts = matched.groupby("scheme_id", sort=True).size().to_dict()
    if counts != EXPECTED_SCHEME_COUNTS:
        raise StructureMatchedModelComparisonError(
            f"Unexpected matched scheme counts: {counts}"
        )
    group_iterations = matched.groupby(
        ["scheme_id", "evaluation_group_id"],
        sort=False,
    )["evaluation_iteration"].nunique(dropna=False)
    if (group_iterations != 1).any():
        raise StructureMatchedModelComparisonError(
            "An evaluation group appears in multiple matched test iterations"
        )
    return matched


def _model_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []
    for scheme_id, group in matched.groupby("scheme_id", sort=True):
        for model_id in MODEL_IDS:
            values = structure_eval._metric_values(
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
            raise StructureMatchedModelComparisonError(
                f"No matched evaluation groups for {scheme_id}"
            )
        row_indices = {
            group_id: np.flatnonzero(
                group["evaluation_group_id"].to_numpy() == group_id
            )
            for group_id in group_ids
        }
        y_true = group["y_true"].to_numpy(dtype=float)
        model_predictions = {
            model_id: group[model_id].to_numpy(dtype=float) for model_id in MODEL_IDS
        }
        point_metrics = {
            model_id: structure_eval._metric_values(y_true, model_predictions[model_id])
            for model_id in MODEL_IDS
        }
        distributions = {
            (model_a, model_b, metric): []
            for model_a, model_b in MODEL_PAIRS
            for metric in METRICS
        }
        rng = np.random.default_rng(_numeric_seed(BOOTSTRAP_SEED, scheme_id))

        for _ in range(BOOTSTRAP_REPLICATES):
            sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
            sampled_indices = np.concatenate(
                [row_indices[group_id] for group_id in sampled_groups]
            )
            sampled_true = y_true[sampled_indices]
            sampled_metrics = {
                model_id: structure_eval._metric_values(
                    sampled_true,
                    model_predictions[model_id][sampled_indices],
                )
                for model_id in MODEL_IDS
            }
            for model_a, model_b in MODEL_PAIRS:
                for metric in METRICS:
                    difference = _metric_difference(
                        metric,
                        sampled_metrics[model_a][metric],
                        sampled_metrics[model_b][metric],
                    )
                    if np.isfinite(difference):
                        distributions[(model_a, model_b, metric)].append(difference)

        for model_a, model_b in MODEL_PAIRS:
            for metric in METRICS:
                point = _metric_difference(
                    metric,
                    point_metrics[model_a][metric],
                    point_metrics[model_b][metric],
                )
                values = np.asarray(
                    distributions[(model_a, model_b, metric)],
                    dtype=float,
                )
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "model_a_value": point_metrics[model_a][metric],
                        "model_b_value": point_metrics[model_b][metric],
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


def prepare_structure_matched_model_comparison(
    repo_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, object],
]:
    repo_root = repo_root.resolve()
    (
        observations,
        residues,
        schemes,
        assignments,
        matched_membership,
        mutant_embeddings,
        wild_type_embeddings,
        assay_sequences,
    ) = _load_matched_inputs(repo_root)

    embedding_index = pd.read_csv(
        repo_root / "data/interim/sequence_plm_embeddings/mutation_embedding_index.csv",
        low_memory=False,
    )

    selected_schemes = schemes[
        schemes["scheme_id"].isin(sequence_eval.CORE_SCHEME_IDS)
    ].copy()
    if set(selected_schemes["scheme_id"]) != set(sequence_eval.CORE_SCHEME_IDS):
        raise StructureMatchedModelComparisonError(
            "Core structure-matched-model schemes are incomplete"
        )
    if not selected_schemes["status"].eq("ready").all():
        raise StructureMatchedModelComparisonError(
            "Core structure-matched-model schemes must all be ready"
        )

    frames = {
        scheme_id: _matched_scheme_frame(
            scheme_id=scheme_id,
            membership=matched_membership,
            observations=observations,
            residues=residues,
            embedding_index=embedding_index,
        )
        for scheme_id in sorted(sequence_eval.CORE_SCHEME_IDS)
    }

    structure_predictions, structure_selection, structure_summary, structure_hashes = (
        _load_existing_structure_results(repo_root)
    )
    structure_predictions = _normalise_existing_structure_predictions(
        structure_predictions
    )
    structure_selection = _normalise_existing_structure_selection(structure_selection)

    new_predictions, new_selection = _evaluate_new_matched_models(
        selected_schemes,
        assignments,
        frames,
        assay_sequences,
        mutant_embeddings,
        wild_type_embeddings,
    )

    predictions = pd.concat(
        [structure_predictions, new_predictions],
        ignore_index=True,
        sort=False,
    )
    predictions["evaluation_iteration"] = predictions["evaluation_iteration"].astype(
        str
    )
    predictions = predictions.sort_values(
        [
            "scheme_id",
            "evaluation_iteration",
            "model_id",
            "observation_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    selection = pd.concat(
        [structure_selection, new_selection],
        ignore_index=True,
        sort=False,
    )
    selection["evaluation_iteration"] = selection["evaluation_iteration"].astype(str)
    selection = selection.sort_values(
        [
            "scheme_id",
            "evaluation_iteration",
            "model_id",
            "selection_scope",
        ],
        kind="stable",
    ).reset_index(drop=True)

    matched = _matched_prediction_table(predictions)
    metrics = _model_metrics(matched)
    comparisons = _paired_comparisons(matched)

    summary: Mapping[str, object] = {
        "status": "PASS",
        "analysis_scope": "structure_matched_model_comparison",
        "model_ids": list(MODEL_IDS),
        "model_pairs": [
            {"model_a": model_a, "model_b": model_b} for model_a, model_b in MODEL_PAIRS
        ],
        "scheme_ids": sorted(matched["scheme_id"].astype(str).unique().tolist()),
        "eligibility_column": "experimental_structure_task_eligible",
        "matched_observation_count_per_model": int(len(matched)),
        "combined_prediction_row_count": int(len(predictions)),
        "metric_row_count": int(len(metrics)),
        "paired_comparison_row_count": int(len(comparisons)),
        "comparison_key": list(COMPARISON_KEY),
        "matched_fields": list(MATCHED_FIELDS),
        "metrics": list(METRICS),
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
        "existing_structure_predictions_reused": True,
        "existing_structure_models_refit": False,
        "matched_sequence_model_refit": True,
        "matched_esmc_model_refit": True,
        "matched_sequence_model_selection": (
            "validation/source-CV restricted to the experimental-structure-eligible "
            "training population"
        ),
        "matched_esmc_model_selection": (
            "hidden-state index and ridge alpha selected from validation/"
            "source-CV rows restricted to the experimental-structure-eligible "
            "training population"
        ),
        "target_labels_used_for_model_selection": False,
        "mechanistic_reference_inputs_used": False,
        "catalytic_zinc_distance_inputs_used": False,
        "reported_homologue_pair_inputs_used": False,
        "raw_coordinates_used_as_direct_predictors": False,
        "experimental_structure_summary_status": structure_summary.get("status"),
        **structure_hashes,
    }
    return predictions, metrics, selection, comparisons, summary


def write_structure_matched_model_comparison(repo_root: Path) -> Mapping[str, object]:
    predictions, metrics, selection, comparisons, summary = (
        prepare_structure_matched_model_comparison(repo_root)
    )
    output_dir = repo_root / "data/interim/structure_matched_model_comparison"
    _write_csv(predictions, output_dir / "structure_matched_predictions.csv")
    _write_csv(metrics, output_dir / "structure_matched_metrics.csv")
    _write_csv(selection, output_dir / "structure_matched_model_selection.csv")
    _write_csv(
        comparisons,
        output_dir / "structure_matched_paired_comparisons.csv",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structure_matched_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
