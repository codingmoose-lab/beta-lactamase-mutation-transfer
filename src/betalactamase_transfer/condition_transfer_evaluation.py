from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class ConditionTransferEvaluationError(RuntimeError):
    """Raised when the condition-transfer analysis cannot be completed safely."""


CONDITION_SCHEME_IDS = (
    "vim2_2020_loo_antibiotic_amp",
    "vim2_2020_loo_antibiotic_ctx",
    "vim2_2020_loo_antibiotic_mem",
    "vim2_2020_temperature_25_to_37",
    "vim2_2020_temperature_37_to_25",
)

CONDITION_MODE_BY_SCHEME = {
    "vim2_2020_loo_antibiotic_amp": "none",
    "vim2_2020_loo_antibiotic_ctx": "none",
    "vim2_2020_loo_antibiotic_mem": "none",
    "vim2_2020_temperature_25_to_37": "log2_concentration",
    "vim2_2020_temperature_37_to_25": "log2_concentration",
}

PRIMARY_REFERENCE_BY_SCHEME = {
    "vim2_2020_loo_antibiotic_amp": "source_antibiotic_mean",
    "vim2_2020_loo_antibiotic_ctx": "source_antibiotic_mean",
    "vim2_2020_loo_antibiotic_mem": "source_antibiotic_mean",
    "vim2_2020_temperature_25_to_37": "source_temperature_matched",
    "vim2_2020_temperature_37_to_25": "source_temperature_matched",
}

SEQUENCE_MODEL_IDS = (
    "condition_transfer_esmc_site_delta_ridge",
    "condition_transfer_esm2_site_delta_ridge",
)
STRUCTURE_MODEL_IDS = (
    "condition_transfer_structure_matched_mutation_condition_ridge",
    "condition_transfer_experimental_structure_descriptor_ridge",
)
BASELINE_MODEL_IDS = (
    "ridge_mutation_identity",
    "source_antibiotic_mean",
    "source_temperature_matched",
)

FULL_POPULATION = "sequence_eligible"
STRUCTURE_POPULATION = "experimental_structure_matched"
METRICS = ("spearman", "pearson", "mae", "rmse")
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_condition_transfer_bootstrap"
TARGET_COLUMN = "score_analysis"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_seed(*parts: str) -> int:
    text = "|".join((BOOTSTRAP_SEED, *parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ConditionTransferEvaluationError("Metrics require equal non-empty arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ConditionTransferEvaluationError("Metrics require finite values")

    error = y_pred - y_true
    if np.allclose(y_true, y_true[0], rtol=0.0, atol=1e-12) or np.allclose(
        y_pred, y_pred[0], rtol=0.0, atol=1e-12
    ):
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
    metric: str, model_value: float, reference_value: float
) -> float:
    if metric in {"spearman", "pearson"}:
        return float(model_value - reference_value)
    if metric in {"mae", "rmse"}:
        return float(reference_value - model_value)
    raise ConditionTransferEvaluationError(f"Unknown metric: {metric}")


def _source_target(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    scheme_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[pd.DataFrame, pd.DataFrame]]]:
    scheme_assignments = assignments[assignments["scheme_id"].astype(str).eq(scheme_id)]
    source_assignments = scheme_assignments[
        scheme_assignments["split_role"].astype(str).eq("source_model_selection")
    ]
    target_assignments = scheme_assignments[
        scheme_assignments["split_role"].astype(str).eq("target_test")
    ]
    if source_assignments.empty or target_assignments.empty:
        raise ConditionTransferEvaluationError(
            f"Missing source/test assignments for {scheme_id}"
        )

    source = frame[frame["scope_role"].astype(str).eq("source_model_selection")].copy()
    target = frame[frame["scope_role"].astype(str).eq("target_test")].copy()
    if source.empty or target.empty:
        raise ConditionTransferEvaluationError(
            f"Missing source/test rows for {scheme_id}"
        )

    fold_by_group = dict(
        zip(source_assignments["group_id"], source_assignments["cv_fold"], strict=False)
    )
    source["source_cv_fold"] = source["group_id"].map(fold_by_group)
    if source["source_cv_fold"].isna().any():
        raise ConditionTransferEvaluationError(
            f"Unassigned source CV rows for {scheme_id}"
        )

    expected_target_groups = set(target_assignments["group_id"].astype(str))
    observed_target_groups = set(target["group_id"].astype(str))
    if not observed_target_groups.issubset(expected_target_groups):
        raise ConditionTransferEvaluationError(
            f"Target-test group mismatch for {scheme_id}"
        )

    folds = []
    for fold in sorted(source["source_cv_fold"].astype(int).unique()):
        train = source[~source["source_cv_fold"].astype(int).eq(fold)].copy()
        validation = source[source["source_cv_fold"].astype(int).eq(fold)].copy()
        if train.empty or validation.empty:
            raise ConditionTransferEvaluationError(
                f"Empty source CV fold for {scheme_id}/{fold}"
            )
        folds.append((train, validation))
    if len(folds) != 5:
        raise ConditionTransferEvaluationError(
            f"Expected five source CV folds for {scheme_id}, observed {len(folds)}"
        )
    return source, target, folds


def _sequence_prediction_records(
    target: pd.DataFrame,
    predictions: np.ndarray,
    *,
    scheme: pd.Series,
    model_id: str,
    hidden_state_index: int,
    alpha: float,
    condition_mode: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "population_scope": FULL_POPULATION,
            "model_id": model_id,
            "evaluation_iteration": "target",
            "evaluation_role": "target_test",
            "observation_id": target["observation_id"].astype(str).to_numpy(),
            "protein_id": target["protein_id"].astype(str).to_numpy(),
            "assay_position": target["assay_position"].to_numpy(dtype=int),
            "wild_type": target["wild_type"].astype(str).to_numpy(),
            "mutant": target["mutant"].astype(str).to_numpy(),
            "antibiotic": target["antibiotic"].astype(str).to_numpy(),
            "concentration": target["concentration"].to_numpy(dtype=float),
            "temperature_c": target["temperature_c"].to_numpy(dtype=float),
            "evaluation_group_id": target["group_id"].astype(str).to_numpy(),
            "y_true": target[TARGET_COLUMN].to_numpy(dtype=float),
            "y_pred": np.asarray(predictions, dtype=float),
            "condition_mode": condition_mode,
            "hidden_state_index": int(hidden_state_index),
            "selected_alpha": float(alpha),
        }
    )


def _selection_records(
    detail: Sequence[Mapping[str, object]],
    *,
    scheme: pd.Series,
    model_id: str,
    selected_hidden_state: int | None,
    selected_alpha: float,
    validation_mae: float,
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for record in detail:
        hidden_state = record.get("hidden_state_index", "")
        record_alpha = float(record["alpha"])
        selected = record_alpha == float(selected_alpha)
        if selected_hidden_state is not None:
            selected = selected and int(hidden_state) == int(selected_hidden_state)
        rows.append(
            {
                "scheme_id": str(scheme["scheme_id"]),
                "task_id": str(scheme["task_id"]),
                "model_id": model_id,
                "selection_scope": "source_cv",
                "hidden_state_index": hidden_state,
                "alpha": record_alpha,
                "cv_fold": record.get("cv_fold", ""),
                "validation_mae": float(record["validation_mae"]),
                "selected": bool(selected),
            }
        )
    rows.append(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "model_id": model_id,
            "selection_scope": "selected_setting",
            "hidden_state_index": ""
            if selected_hidden_state is None
            else int(selected_hidden_state),
            "alpha": float(selected_alpha),
            "cv_fold": "",
            "validation_mae": float(validation_mae),
            "selected": True,
        }
    )
    return rows


def _prepare_esmc(
    repo_root: Path,
    schemes: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Mapping[str, object]]]:
    from . import sequence_plm_evaluation as esmc_stage

    (
        observations,
        _,
        _,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = esmc_stage._load_inputs(repo_root)
    esmc_stage._validate_inputs(
        observations,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    prediction_frames = []
    selection_rows: list[Mapping[str, object]] = []
    for index, scheme_id in enumerate(CONDITION_SCHEME_IDS, start=1):
        print(
            f"ESMC condition transfer {index}/{len(CONDITION_SCHEME_IDS)}: {scheme_id}",
            flush=True,
        )
        scheme = schemes[schemes["scheme_id"].astype(str).eq(scheme_id)].iloc[0]
        frame = esmc_stage._join_scheme_observations(
            membership, observations, embedding_index, scheme_id
        )
        source, target, folds = _source_target(frame, assignments, scheme_id)
        mode = CONDITION_MODE_BY_SCHEME[scheme_id]
        hidden_state, alpha, validation_mae, detail = esmc_stage._select_model_setting(
            folds,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=mode,
        )
        predictions = esmc_stage._fit_predict(
            source,
            target,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=mode,
            hidden_state_index=hidden_state,
            alpha=alpha,
        )
        prediction_frames.append(
            _sequence_prediction_records(
                target,
                predictions,
                scheme=scheme,
                model_id=SEQUENCE_MODEL_IDS[0],
                hidden_state_index=hidden_state,
                alpha=alpha,
                condition_mode=mode,
            )
        )
        selection_rows.extend(
            _selection_records(
                detail,
                scheme=scheme,
                model_id=SEQUENCE_MODEL_IDS[0],
                selected_hidden_state=hidden_state,
                selected_alpha=alpha,
                validation_mae=validation_mae,
            )
        )
    return pd.concat(prediction_frames, ignore_index=True), selection_rows


def _prepare_esm2(
    repo_root: Path,
    schemes: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Mapping[str, object]]]:
    from . import sequence_model_sensitivity as esm2_stage

    (
        observations,
        _,
        _,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = esm2_stage._load_inputs(repo_root)
    esm2_stage._validate_inputs(
        observations,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    prediction_frames = []
    selection_rows: list[Mapping[str, object]] = []
    for index, scheme_id in enumerate(CONDITION_SCHEME_IDS, start=1):
        print(
            f"ESM-2 condition transfer {index}/"
            f"{len(CONDITION_SCHEME_IDS)}: {scheme_id}",
            flush=True,
        )
        scheme = schemes[schemes["scheme_id"].astype(str).eq(scheme_id)].iloc[0]
        frame = esm2_stage._join_scheme_observations(
            membership, observations, embedding_index, scheme_id
        )
        source, target, folds = _source_target(frame, assignments, scheme_id)
        mode = CONDITION_MODE_BY_SCHEME[scheme_id]
        hidden_state, alpha, validation_mae, detail = esm2_stage._select_model_setting(
            folds,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=mode,
        )
        predictions = esm2_stage._fit_predict(
            source,
            target,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=mode,
            hidden_state_index=hidden_state,
            alpha=alpha,
        )
        prediction_frames.append(
            _sequence_prediction_records(
                target,
                predictions,
                scheme=scheme,
                model_id=SEQUENCE_MODEL_IDS[1],
                hidden_state_index=hidden_state,
                alpha=alpha,
                condition_mode=mode,
            )
        )
        selection_rows.extend(
            _selection_records(
                detail,
                scheme=scheme,
                model_id=SEQUENCE_MODEL_IDS[1],
                selected_hidden_state=hidden_state,
                selected_alpha=alpha,
                validation_mae=validation_mae,
            )
        )
    return pd.concat(prediction_frames, ignore_index=True), selection_rows


def _prepare_structure(
    repo_root: Path,
    schemes: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Mapping[str, object]]]:
    from . import experimental_structure_evaluation as structure_stage

    observations = pd.read_csv(
        repo_root / "data/processed/missense_observations.csv", low_memory=False
    )
    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv",
        low_memory=False,
    )
    residues = pd.read_csv(
        repo_root / "data/processed/residue_features.csv", low_memory=False
    )
    structure_stage._validate_structure_features(residues)

    prediction_frames = []
    selection_rows: list[Mapping[str, object]] = []
    model_map = {
        structure_stage.MUTATION_CONDITION_MODEL: STRUCTURE_MODEL_IDS[0],
        structure_stage.EXPERIMENTAL_STRUCTURE_MODEL: STRUCTURE_MODEL_IDS[1],
    }
    for index, scheme_id in enumerate(CONDITION_SCHEME_IDS, start=1):
        print(
            f"Structure condition transfer {index}/"
            f"{len(CONDITION_SCHEME_IDS)}: {scheme_id}",
            flush=True,
        )
        scheme = schemes[schemes["scheme_id"].astype(str).eq(scheme_id)].iloc[0]
        frame = structure_stage._join_scheme_observations(
            membership, observations, residues, scheme_id
        )
        source, target, folds = _source_target(frame, assignments, scheme_id)
        mode = CONDITION_MODE_BY_SCHEME[scheme_id]
        for original_model, output_model in model_map.items():
            alpha, validation_mae, detail = structure_stage._select_setting(
                folds,
                model_id=original_model,
                condition_mode=mode,
            )
            predictions = structure_stage._fit_predict(
                source,
                target,
                include_structure=(
                    original_model == structure_stage.EXPERIMENTAL_STRUCTURE_MODEL
                ),
                condition_mode=mode,
                alpha=alpha,
            )
            records = structure_stage._prediction_records(
                target,
                predictions,
                scheme=scheme,
                model_id=output_model,
                evaluation_iteration="target",
                evaluation_role="target_test",
                alpha=alpha,
                condition_mode=mode,
            )
            frame_out = pd.DataFrame(records)
            frame_out["population_scope"] = STRUCTURE_POPULATION
            frame_out["hidden_state_index"] = ""
            prediction_frames.append(frame_out)
            selection_rows.extend(
                _selection_records(
                    detail,
                    scheme=scheme,
                    model_id=output_model,
                    selected_hidden_state=None,
                    selected_alpha=alpha,
                    validation_mae=validation_mae,
                )
            )
    return pd.concat(prediction_frames, ignore_index=True), selection_rows


def _baseline_predictions(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data/interim/baseline_evaluation/baseline_predictions.csv"
    frame = pd.read_csv(
        path,
        dtype={"evaluation_iteration": str, "observation_id": str},
        low_memory=False,
    )
    frame = frame[
        frame["scheme_id"].astype(str).isin(CONDITION_SCHEME_IDS)
        & frame["evaluation_role"].astype(str).eq("final_test")
        & frame["evaluation_iteration"].astype(str).eq("final")
    ].copy()
    allowed = {
        "ridge_mutation_identity",
        "source_antibiotic_mean",
        "source_temperature_matched",
    }
    frame = frame[frame["baseline_id"].astype(str).isin(allowed)].copy()
    frame = frame.rename(columns={"baseline_id": "model_id"})
    frame["population_scope"] = FULL_POPULATION
    frame["condition_mode"] = "source_measurement_reference"
    frame["hidden_state_index"] = ""
    required = []
    for scheme_id in CONDITION_SCHEME_IDS:
        direct = PRIMARY_REFERENCE_BY_SCHEME[scheme_id]
        for model_id in ("ridge_mutation_identity", direct):
            count = len(
                frame[
                    frame["scheme_id"].astype(str).eq(scheme_id)
                    & frame["model_id"].astype(str).eq(model_id)
                ]
            )
            if count == 0:
                required.append((scheme_id, model_id))
    if required:
        raise ConditionTransferEvaluationError(
            f"Missing baseline prediction records: {required}"
        )
    return frame


def _structure_matched_copy(
    full_predictions: pd.DataFrame,
    structure_predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = structure_predictions[["scheme_id", "observation_id"]].drop_duplicates()
    if keys.duplicated(["scheme_id", "observation_id"]).any():
        raise ConditionTransferEvaluationError("Duplicate structure target identities")
    matched = full_predictions.merge(
        keys,
        on=["scheme_id", "observation_id"],
        how="inner",
        validate="many_to_one",
    )
    matched["population_scope"] = STRUCTURE_POPULATION
    return matched


def _validate_prediction_contract(predictions: pd.DataFrame) -> None:
    identity = ["scheme_id", "population_scope", "model_id", "observation_id"]
    if predictions.duplicated(identity).any():
        raise ConditionTransferEvaluationError("Duplicate prediction identities")
    if not np.isfinite(predictions["y_true"].to_numpy(dtype=float)).all():
        raise ConditionTransferEvaluationError("Non-finite targets in predictions")
    if not np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all():
        raise ConditionTransferEvaluationError("Non-finite model predictions")
    for scheme_id in CONDITION_SCHEME_IDS:
        direct = PRIMARY_REFERENCE_BY_SCHEME[scheme_id]
        for population in (FULL_POPULATION, STRUCTURE_POPULATION):
            subset = predictions[
                predictions["scheme_id"].astype(str).eq(scheme_id)
                & predictions["population_scope"].astype(str).eq(population)
            ]
            if subset.empty:
                raise ConditionTransferEvaluationError(
                    f"Missing prediction population for {scheme_id}/{population}"
                )
            reference_ids = set(
                subset[subset["model_id"].astype(str).eq(direct)][
                    "observation_id"
                ].astype(str)
            )
            if not reference_ids:
                raise ConditionTransferEvaluationError(
                    f"Missing direct reference for {scheme_id}/{population}"
                )
            for model_id, group in subset.groupby("model_id", sort=True):
                ids = set(group["observation_id"].astype(str))
                if ids != reference_ids:
                    raise ConditionTransferEvaluationError(
                        f"Population mismatch for {scheme_id}/{population}/{model_id}"
                    )


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme_id, population, model_id), group in predictions.groupby(
        ["scheme_id", "population_scope", "model_id"], sort=True
    ):
        rows.append(
            {
                "scheme_id": scheme_id,
                "population_scope": population,
                "model_id": model_id,
                "n_observations": len(group),
                "n_groups": group["evaluation_group_id"].nunique(),
                **_metric_values(
                    group["y_true"].to_numpy(dtype=float),
                    group["y_pred"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = list(predictions.groupby(["scheme_id", "population_scope"], sort=True))
    for index, ((scheme_id, population), group) in enumerate(grouped, start=1):
        print(
            f"Condition-transfer uncertainty {index}/{len(grouped)}: "
            f"{scheme_id} / {population}",
            flush=True,
        )
        reference_id = PRIMARY_REFERENCE_BY_SCHEME[str(scheme_id)]
        reference = group[group["model_id"].astype(str).eq(reference_id)].copy()
        key = ["observation_id", "evaluation_group_id", "y_true"]
        reference = reference[key + ["y_pred"]].rename(
            columns={"y_pred": "reference_prediction"}
        )
        for model_id in sorted(set(group["model_id"].astype(str)) - {reference_id}):
            model = group[group["model_id"].astype(str).eq(model_id)][
                key + ["y_pred"]
            ].rename(columns={"y_pred": "model_prediction"})
            paired = reference.merge(model, on=key, how="inner", validate="one_to_one")
            if len(paired) != len(reference):
                raise ConditionTransferEvaluationError(
                    f"Incomplete paired comparison for "
                    f"{scheme_id}/{population}/{model_id}"
                )
            groups = sorted(paired["evaluation_group_id"].astype(str).unique())
            rows_by_group = {
                value: paired.index[
                    paired["evaluation_group_id"].astype(str).eq(value)
                ].to_numpy()
                for value in groups
            }
            for metric in METRICS:
                model_value = _metric_values(
                    paired["y_true"].to_numpy(dtype=float),
                    paired["model_prediction"].to_numpy(dtype=float),
                )[metric]
                reference_value = _metric_values(
                    paired["y_true"].to_numpy(dtype=float),
                    paired["reference_prediction"].to_numpy(dtype=float),
                )[metric]
                point = _metric_difference(metric, model_value, reference_value)
                rng = np.random.default_rng(
                    _bootstrap_seed(str(scheme_id), str(population), model_id, metric)
                )
                draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                for index in range(BOOTSTRAP_REPLICATES):
                    sampled_groups = rng.choice(groups, size=len(groups), replace=True)
                    sampled_index = np.concatenate(
                        [rows_by_group[value] for value in sampled_groups]
                    )
                    sample = paired.loc[sampled_index]
                    model_metric = _metric_values(
                        sample["y_true"].to_numpy(dtype=float),
                        sample["model_prediction"].to_numpy(dtype=float),
                    )[metric]
                    reference_metric = _metric_values(
                        sample["y_true"].to_numpy(dtype=float),
                        sample["reference_prediction"].to_numpy(dtype=float),
                    )[metric]
                    draws[index] = _metric_difference(
                        metric, model_metric, reference_metric
                    )
                finite = draws[np.isfinite(draws)]
                if len(finite) < int(0.95 * BOOTSTRAP_REPLICATES):
                    raise ConditionTransferEvaluationError(
                        f"Too few finite bootstrap draws for "
                        f"{scheme_id}/{population}/{model_id}/{metric}"
                    )
                lower, upper = np.quantile(finite, [0.025, 0.975])
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "population_scope": population,
                        "model_id": model_id,
                        "reference_model_id": reference_id,
                        "metric": metric,
                        "model_value": model_value,
                        "reference_value": reference_value,
                        "paired_difference": point,
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "interval_excludes_zero": bool(lower > 0 or upper < 0),
                        "n_observations": len(paired),
                        "n_groups": len(groups),
                        "difference_positive_means_model_better": True,
                    }
                )
    return pd.DataFrame(rows)


def _shift_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = predictions[predictions["population_scope"].astype(str).eq(FULL_POPULATION)]
    for scheme_id in CONDITION_SCHEME_IDS:
        reference_id = PRIMARY_REFERENCE_BY_SCHEME[scheme_id]
        frame = full[
            full["scheme_id"].astype(str).eq(scheme_id)
            & full["model_id"].astype(str).eq(reference_id)
        ].copy()
        shift = frame["y_true"].to_numpy(dtype=float) - frame["y_pred"].to_numpy(
            dtype=float
        )
        groups = sorted(frame["evaluation_group_id"].astype(str).unique())
        group_shift = (
            frame.assign(shift=shift).groupby("evaluation_group_id")["shift"].mean()
        )
        rng = np.random.default_rng(
            _bootstrap_seed(scheme_id, "mean_target_minus_source")
        )
        draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        values = group_shift.to_numpy(dtype=float)
        for index in range(BOOTSTRAP_REPLICATES):
            draws[index] = float(
                np.mean(rng.choice(values, size=len(values), replace=True))
            )
        lower, upper = np.quantile(draws, [0.025, 0.975])
        metrics = _metric_values(
            frame["y_true"].to_numpy(dtype=float),
            frame["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "scheme_id": scheme_id,
                "reference_model_id": reference_id,
                "n_observations": len(frame),
                "n_groups": len(groups),
                "mean_target_minus_source": float(np.mean(shift)),
                "median_target_minus_source": float(np.median(shift)),
                "mean_shift_ci_lower": float(lower),
                "mean_shift_ci_upper": float(upper),
                "mean_absolute_target_minus_source": float(np.mean(np.abs(shift))),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def review_condition_transfer_inputs(repo_root: Path) -> Mapping[str, object]:
    required = [
        "data/processed/missense_observations.csv",
        "data/processed/residue_features.csv",
        "data/interim/evaluation_partitions/partition_schemes.csv",
        "data/interim/evaluation_partitions/group_assignments.csv",
        "data/interim/modelling_views/scheme_observation_membership.csv",
        "data/interim/baseline_evaluation/baseline_predictions.csv",
        "data/interim/sequence_plm_embeddings/mutation_embedding_index.csv",
        "data/interim/sequence_plm_embeddings/mutant_site_embeddings.npy",
        "data/interim/sequence_plm_embeddings/wild_type_residue_embeddings.npy",
        "data/interim/sequence_model_sensitivity/esm2_embeddings/mutation_embedding_index.csv",
        "data/interim/sequence_model_sensitivity/esm2_embeddings/mutant_site_embeddings.npy",
        "data/interim/sequence_model_sensitivity/esm2_embeddings/wild_type_residue_embeddings.npy",
    ]
    missing = [
        relative for relative in required if not (repo_root / relative).is_file()
    ]
    if missing:
        raise ConditionTransferEvaluationError(f"Missing required inputs: {missing}")

    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv",
        low_memory=False,
    )
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv",
        low_memory=False,
    )
    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv",
        low_memory=False,
    )
    selected = schemes[
        schemes["scheme_id"].astype(str).isin(CONDITION_SCHEME_IDS)
    ].copy()
    if set(selected["scheme_id"].astype(str)) != set(CONDITION_SCHEME_IDS):
        raise ConditionTransferEvaluationError(
            "Condition-transfer scheme set is incomplete"
        )
    if not selected["status"].astype(str).eq("ready").all():
        raise ConditionTransferEvaluationError(
            "A condition-transfer scheme is not ready"
        )
    if selected["test_labels_used_for_model_selection"].astype(bool).any():
        raise ConditionTransferEvaluationError(
            "Condition-transfer test labels enter model selection"
        )
    if selected["assignment_uses_outcome_values"].astype(bool).any():
        raise ConditionTransferEvaluationError(
            "Condition-transfer assignment uses outcome values"
        )

    rows = []
    for scheme_id in CONDITION_SCHEME_IDS:
        part = membership[membership["scheme_id"].astype(str).eq(scheme_id)]
        scheme_assignments = assignments[
            assignments["scheme_id"].astype(str).eq(scheme_id)
        ]
        source_groups = set(
            scheme_assignments[
                scheme_assignments["split_role"]
                .astype(str)
                .eq("source_model_selection")
            ]["group_id"].astype(str)
        )
        target_groups = set(
            scheme_assignments[
                scheme_assignments["split_role"].astype(str).eq("target_test")
            ]["group_id"].astype(str)
        )
        source = part[part["group_id"].astype(str).isin(source_groups)]
        target = part[part["group_id"].astype(str).isin(target_groups)]
        rows.append(
            {
                "scheme_id": scheme_id,
                "condition_mode": CONDITION_MODE_BY_SCHEME[scheme_id],
                "source_rows": len(source),
                "target_rows": len(target),
                "target_structure_rows": int(
                    target["experimental_structure_task_eligible"].astype(bool).sum()
                ),
                "source_cv_folds": int(
                    scheme_assignments["cv_fold"].dropna().nunique()
                ),
                "primary_reference": PRIMARY_REFERENCE_BY_SCHEME[scheme_id],
            }
        )
    return {
        "status": "PASS",
        "scheme_count": len(CONDITION_SCHEME_IDS),
        "schemes": rows,
        "antibiotic_condition_mode": "none",
        "temperature_condition_mode": "log2_concentration",
        "test_labels_used_for_model_selection": False,
        "graph_neural_models_planned_for_condition_eval": False,
        "graph_neural_model_reason": (
            "condition-transfer analysis tests sequence and experimental-structure "
            "ridge representations against direct source-condition evidence. The "
            "held-out antibiotic has no learned condition identity, so an "
            "unseen antibiotic category is not supplied to any fitted model."
        ),
    }


def prepare_condition_transfer_evaluation(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_condition_transfer_inputs(repo_root)
    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv",
        low_memory=False,
    )
    schemes = schemes[
        schemes["scheme_id"].astype(str).isin(CONDITION_SCHEME_IDS)
    ].copy()
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv",
        low_memory=False,
    )

    baseline = _baseline_predictions(repo_root)
    esmc, esmc_selection = _prepare_esmc(repo_root, schemes, assignments)
    esm2, esm2_selection = _prepare_esm2(repo_root, schemes, assignments)
    structure, structure_selection = _prepare_structure(repo_root, schemes, assignments)

    full = pd.concat([baseline, esmc, esm2], ignore_index=True, sort=False)
    matched_full = _structure_matched_copy(full, structure)
    predictions = pd.concat(
        [full, matched_full, structure],
        ignore_index=True,
        sort=False,
    )
    predictions["observation_id"] = predictions["observation_id"].astype(str)
    predictions["evaluation_iteration"] = predictions["evaluation_iteration"].astype(
        str
    )
    predictions["model_id"] = predictions["model_id"].astype(str)
    predictions["population_scope"] = predictions["population_scope"].astype(str)
    _validate_prediction_contract(predictions)

    selection = pd.DataFrame(esmc_selection + esm2_selection + structure_selection)
    metrics = _metric_rows(predictions)
    comparisons = _paired_bootstrap(predictions)
    shifts = _shift_summary(predictions)
    return predictions, selection, metrics, comparisons, shifts


def run_condition_transfer_evaluation(repo_root: Path) -> Mapping[str, object]:
    predictions, selection, metrics, comparisons, shifts = (
        prepare_condition_transfer_evaluation(repo_root)
    )
    output_dir = repo_root / "data/interim/condition_transfer_evaluation"
    _write_csv(predictions, output_dir / "condition_transfer_predictions.csv")
    _write_csv(selection, output_dir / "condition_transfer_model_selection.csv")
    _write_csv(metrics, output_dir / "condition_transfer_metrics.csv")
    _write_csv(comparisons, output_dir / "condition_transfer_paired_comparisons.csv")
    _write_csv(shifts, output_dir / "condition_transfer_shift_summary.csv")

    summary = {
        "status": "PASS",
        "scheme_ids": list(CONDITION_SCHEME_IDS),
        "condition_modes": dict(CONDITION_MODE_BY_SCHEME),
        "primary_reference_by_scheme": dict(PRIMARY_REFERENCE_BY_SCHEME),
        "sequence_model_ids": list(SEQUENCE_MODEL_IDS),
        "structure_model_ids": list(STRUCTURE_MODEL_IDS),
        "population_scopes": [FULL_POPULATION, STRUCTURE_POPULATION],
        "prediction_count": int(len(predictions)),
        "metric_count": int(len(metrics)),
        "paired_comparison_count": int(len(comparisons)),
        "model_selection_record_count": int(len(selection)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "test_labels_used_for_model_selection": False,
        "assignment_uses_outcome_values": False,
        "held_out_antibiotic_identity_used_as_model_input": False,
        "temperature_models_use_log2_concentration": True,
        "graph_neural_models_fitted": False,
        "graph_neural_model_reason": (
            "The held-out antibiotic has no learned condition identity. "
            "condition-transfer analysis therefore tests source-only sequence and "
            "experimental-structure ridge models against direct observed "
            "source-condition transfer without inventing an encoding for "
            "the unseen antibiotic."
        ),
        "input_sha256": {
            relative: _sha256(repo_root / relative)
            for relative in [
                "data/processed/missense_observations.csv",
                "data/processed/residue_features.csv",
                "data/interim/evaluation_partitions/partition_schemes.csv",
                "data/interim/evaluation_partitions/group_assignments.csv",
                "data/interim/modelling_views/scheme_observation_membership.csv",
                "data/interim/baseline_evaluation/baseline_predictions.csv",
                "data/interim/sequence_plm_embeddings/mutant_site_embeddings.npy",
                "data/interim/sequence_model_sensitivity/esm2_embeddings/mutant_site_embeddings.npy",
            ]
        },
        "outputs": [
            "condition_transfer_predictions.csv",
            "condition_transfer_model_selection.csv",
            "condition_transfer_metrics.csv",
            "condition_transfer_paired_comparisons.csv",
            "condition_transfer_shift_summary.csv",
            "condition_transfer_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "condition_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
