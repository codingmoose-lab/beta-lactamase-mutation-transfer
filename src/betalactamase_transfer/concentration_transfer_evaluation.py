from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from . import condition_transfer_evaluation as stage39


class ConcentrationTransferEvaluationError(RuntimeError):
    """Raised when the concentration-transfer analysis cannot be completed safely."""


CONCENTRATION_SCHEME_IDS = (
    "vim2_2022_loo_concentration_4",
    "vim2_2022_loo_concentration_8",
    "vim2_2022_loo_concentration_16",
    "vim2_2022_loo_concentration_32",
    "vim2_2022_loo_concentration_64",
    "vim2_2022_loo_concentration_128",
    "vim2_2022_loo_concentration_256",
)

CONDITION_MODE = "log2_concentration"
FULL_POPULATION = "sequence_eligible"
STRUCTURE_POPULATION = "experimental_structure_matched"
TARGET_COLUMN = "score_analysis"
METRICS = ("spearman", "pearson", "mae", "rmse")
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_concentration_transfer_bootstrap"

SEQUENCE_MODEL_IDS = (
    "concentration_transfer_esmc_site_delta_ridge",
    "concentration_transfer_esm2_site_delta_ridge",
)
STRUCTURE_MODEL_IDS = (
    "concentration_transfer_structure_matched_mutation_condition_ridge",
    "concentration_transfer_experimental_structure_descriptor_ridge",
)
BASELINE_MODEL_IDS = (
    "ridge_mutation_log2_concentration",
    "source_concentration_nearest",
    "source_concentration_linear_log2",
)


def _reference_models(test_category: str) -> tuple[str, str]:
    if test_category == "interpolation":
        return "source_concentration_linear_log2", "source_concentration_nearest"
    if test_category == "edge_extrapolation":
        return "source_concentration_nearest", "source_concentration_linear_log2"
    raise ConcentrationTransferEvaluationError(
        f"Unexpected concentration test category: {test_category}"
    )


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


def _scheme_row(schemes: pd.DataFrame, scheme_id: str) -> pd.Series:
    rows = schemes[schemes["scheme_id"].astype(str).eq(scheme_id)]
    if len(rows) != 1:
        raise ConcentrationTransferEvaluationError(
            f"Expected one scheme definition for {scheme_id}, observed {len(rows)}"
        )
    return rows.iloc[0]


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
    for index, scheme_id in enumerate(CONCENTRATION_SCHEME_IDS, start=1):
        print(
            f"ESMC concentration transfer {index}/{len(CONCENTRATION_SCHEME_IDS)}: "
            f"{scheme_id}",
            flush=True,
        )
        scheme = _scheme_row(schemes, scheme_id)
        frame = esmc_stage._join_scheme_observations(
            membership, observations, embedding_index, scheme_id
        )
        source, target, folds = stage39._source_target(frame, assignments, scheme_id)
        hidden_state, alpha, validation_mae, detail = esmc_stage._select_model_setting(
            folds,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=CONDITION_MODE,
        )
        predictions = esmc_stage._fit_predict(
            source,
            target,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=CONDITION_MODE,
            hidden_state_index=hidden_state,
            alpha=alpha,
        )
        prediction_frames.append(
            stage39._sequence_prediction_records(
                target,
                predictions,
                scheme=scheme,
                model_id=SEQUENCE_MODEL_IDS[0],
                hidden_state_index=hidden_state,
                alpha=alpha,
                condition_mode=CONDITION_MODE,
            )
        )
        selection_rows.extend(
            stage39._selection_records(
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
    for index, scheme_id in enumerate(CONCENTRATION_SCHEME_IDS, start=1):
        print(
            f"ESM-2 concentration transfer {index}/{len(CONCENTRATION_SCHEME_IDS)}: "
            f"{scheme_id}",
            flush=True,
        )
        scheme = _scheme_row(schemes, scheme_id)
        frame = esm2_stage._join_scheme_observations(
            membership, observations, embedding_index, scheme_id
        )
        source, target, folds = stage39._source_target(frame, assignments, scheme_id)
        hidden_state, alpha, validation_mae, detail = esm2_stage._select_model_setting(
            folds,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=CONDITION_MODE,
        )
        predictions = esm2_stage._fit_predict(
            source,
            target,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=CONDITION_MODE,
            hidden_state_index=hidden_state,
            alpha=alpha,
        )
        prediction_frames.append(
            stage39._sequence_prediction_records(
                target,
                predictions,
                scheme=scheme,
                model_id=SEQUENCE_MODEL_IDS[1],
                hidden_state_index=hidden_state,
                alpha=alpha,
                condition_mode=CONDITION_MODE,
            )
        )
        selection_rows.extend(
            stage39._selection_records(
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
    for index, scheme_id in enumerate(CONCENTRATION_SCHEME_IDS, start=1):
        print(
            f"Structure concentration transfer {index}/"
            f"{len(CONCENTRATION_SCHEME_IDS)}: {scheme_id}",
            flush=True,
        )
        scheme = _scheme_row(schemes, scheme_id)
        frame = structure_stage._join_scheme_observations(
            membership, observations, residues, scheme_id
        )
        source, target, folds = stage39._source_target(frame, assignments, scheme_id)
        for original_model, output_model in model_map.items():
            alpha, validation_mae, detail = structure_stage._select_setting(
                folds,
                model_id=original_model,
                condition_mode=CONDITION_MODE,
            )
            predictions = structure_stage._fit_predict(
                source,
                target,
                include_structure=(
                    original_model == structure_stage.EXPERIMENTAL_STRUCTURE_MODEL
                ),
                condition_mode=CONDITION_MODE,
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
                condition_mode=CONDITION_MODE,
            )
            frame_out = pd.DataFrame(records)
            frame_out["population_scope"] = STRUCTURE_POPULATION
            frame_out["hidden_state_index"] = ""
            prediction_frames.append(frame_out)
            selection_rows.extend(
                stage39._selection_records(
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
        frame["scheme_id"].astype(str).isin(CONCENTRATION_SCHEME_IDS)
        & frame["evaluation_role"].astype(str).eq("final_test")
        & frame["evaluation_iteration"].astype(str).eq("final")
        & frame["baseline_id"].astype(str).isin(BASELINE_MODEL_IDS)
    ].copy()
    frame = frame.rename(columns={"baseline_id": "model_id"})
    frame["population_scope"] = FULL_POPULATION
    frame["condition_mode"] = "source_concentration_reference"
    frame["hidden_state_index"] = ""
    for scheme_id in CONCENTRATION_SCHEME_IDS:
        for model_id in BASELINE_MODEL_IDS:
            if not (
                frame["scheme_id"].astype(str).eq(scheme_id)
                & frame["model_id"].astype(str).eq(model_id)
            ).any():
                raise ConcentrationTransferEvaluationError(
                    f"Missing baseline prediction records for {scheme_id}/{model_id}"
                )
    return frame


def _structure_matched_copy(
    full_predictions: pd.DataFrame,
    structure_predictions: pd.DataFrame,
) -> pd.DataFrame:
    return stage39._structure_matched_copy(full_predictions, structure_predictions)


def _primary_reference_by_scheme(schemes: pd.DataFrame) -> dict[str, str]:
    result = {}
    for scheme_id in CONCENTRATION_SCHEME_IDS:
        scheme = _scheme_row(schemes, scheme_id)
        primary, _ = _reference_models(str(scheme["test_category"]))
        result[scheme_id] = primary
    return result


def _secondary_reference_by_scheme(schemes: pd.DataFrame) -> dict[str, str]:
    result = {}
    for scheme_id in CONCENTRATION_SCHEME_IDS:
        scheme = _scheme_row(schemes, scheme_id)
        _, secondary = _reference_models(str(scheme["test_category"]))
        result[scheme_id] = secondary
    return result


def _validate_prediction_contract(
    predictions: pd.DataFrame,
    primary_reference: Mapping[str, str],
) -> None:
    identity = ["scheme_id", "population_scope", "model_id", "observation_id"]
    if predictions.duplicated(identity).any():
        raise ConcentrationTransferEvaluationError("Duplicate prediction identities")
    for column in ("y_true", "y_pred"):
        if not np.isfinite(predictions[column].to_numpy(dtype=float)).all():
            raise ConcentrationTransferEvaluationError(
                f"Non-finite values in prediction column {column}"
            )
    for scheme_id in CONCENTRATION_SCHEME_IDS:
        reference_model = primary_reference[scheme_id]
        for population in (FULL_POPULATION, STRUCTURE_POPULATION):
            subset = predictions[
                predictions["scheme_id"].astype(str).eq(scheme_id)
                & predictions["population_scope"].astype(str).eq(population)
            ]
            reference_ids = set(
                subset[subset["model_id"].astype(str).eq(reference_model)][
                    "observation_id"
                ].astype(str)
            )
            if not reference_ids:
                raise ConcentrationTransferEvaluationError(
                    f"Missing primary reference for {scheme_id}/{population}"
                )
            for model_id, group in subset.groupby("model_id", sort=True):
                ids = set(group["observation_id"].astype(str))
                if ids != reference_ids:
                    raise ConcentrationTransferEvaluationError(
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
                **stage39._metric_values(
                    group["y_true"].to_numpy(dtype=float),
                    group["y_pred"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_bootstrap(
    predictions: pd.DataFrame,
    primary_reference: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    grouped = list(predictions.groupby(["scheme_id", "population_scope"], sort=True))
    for index, ((scheme_id, population), group) in enumerate(grouped, start=1):
        print(
            f"Concentration-transfer uncertainty {index}/{len(grouped)}: "
            f"{scheme_id} / {population}",
            flush=True,
        )
        reference_model = primary_reference[scheme_id]
        reference = group[group["model_id"].astype(str).eq(reference_model)].copy()
        reference = reference.set_index("observation_id", drop=False)
        model_ids = sorted(set(group["model_id"].astype(str)) - {reference_model})
        for model_id in model_ids:
            model = group[group["model_id"].astype(str).eq(model_id)].copy()
            model = model.set_index("observation_id", drop=False)
            paired = reference.join(
                model[["y_pred"]].rename(columns={"y_pred": "model_y_pred"}),
                how="inner",
                validate="one_to_one",
            )
            if len(paired) != len(reference):
                raise ConcentrationTransferEvaluationError(
                    "Incomplete paired comparison for "
                    f"{scheme_id}/{population}/{model_id}"
                )
            group_ids = paired["evaluation_group_id"].astype(str).to_numpy()
            unique_groups = np.unique(group_ids)
            group_indices = {
                value: np.flatnonzero(group_ids == value) for value in unique_groups
            }
            y_true = paired["y_true"].to_numpy(dtype=float)
            reference_pred = paired["y_pred"].to_numpy(dtype=float)
            model_pred = paired["model_y_pred"].to_numpy(dtype=float)
            reference_metrics = stage39._metric_values(y_true, reference_pred)
            model_metrics = stage39._metric_values(y_true, model_pred)
            for metric in METRICS:
                point = stage39._metric_difference(
                    metric, model_metrics[metric], reference_metrics[metric]
                )
                rng = np.random.default_rng(
                    _bootstrap_seed(scheme_id, population, model_id, metric)
                )
                draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                for draw in range(BOOTSTRAP_REPLICATES):
                    sampled_groups = rng.choice(
                        unique_groups, size=len(unique_groups), replace=True
                    )
                    indices = np.concatenate(
                        [group_indices[value] for value in sampled_groups]
                    )
                    ref_draw = stage39._metric_values(
                        y_true[indices], reference_pred[indices]
                    )[metric]
                    model_draw = stage39._metric_values(
                        y_true[indices], model_pred[indices]
                    )[metric]
                    draws[draw] = stage39._metric_difference(
                        metric, model_draw, ref_draw
                    )
                finite = draws[np.isfinite(draws)]
                if len(finite) < int(0.95 * BOOTSTRAP_REPLICATES):
                    raise ConcentrationTransferEvaluationError(
                        "Too few finite bootstrap draws for "
                        f"{scheme_id}/{population}/{model_id}/{metric}"
                    )
                lower, upper = np.quantile(finite, [0.025, 0.975])
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "population_scope": population,
                        "model_id": model_id,
                        "reference_model_id": reference_model,
                        "metric": metric,
                        "model_value": model_metrics[metric],
                        "reference_value": reference_metrics[metric],
                        "paired_difference": point,
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "n_observations": len(paired),
                        "n_groups": len(unique_groups),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "interval_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                    }
                )
    return pd.DataFrame(rows)


def _concentration_design_summary(
    repo_root: Path,
    schemes: pd.DataFrame,
    primary_reference: Mapping[str, str],
    secondary_reference: Mapping[str, str],
) -> pd.DataFrame:
    observations = pd.read_csv(
        repo_root / "data/processed/missense_observations.csv", low_memory=False
    )
    vim = observations[
        observations["assay_id"].astype(str).eq("vim2_2022_amp_concentration_dms")
    ].copy()
    available = sorted(vim["concentration"].astype(float).unique().tolist())
    rows = []
    for scheme_id in CONCENTRATION_SCHEME_IDS:
        scheme = _scheme_row(schemes, scheme_id)
        target = float(scheme["test_value"])
        source = [value for value in available if value != target]
        lower = [value for value in source if value < target]
        upper = [value for value in source if value > target]
        nearest_distance = min(
            abs(np.log2(value) - np.log2(target)) for value in source
        )
        nearest = [
            value
            for value in source
            if abs(abs(np.log2(value) - np.log2(target)) - nearest_distance) < 1e-12
        ]
        target_rows = vim[vim["concentration"].astype(float).eq(target)]
        rows.append(
            {
                "scheme_id": scheme_id,
                "target_concentration": target,
                "target_log2_concentration": float(np.log2(target)),
                "test_category": str(scheme["test_category"]),
                "source_concentrations": ";".join(str(int(value)) for value in source),
                "lower_bracket": "" if not lower else max(lower),
                "upper_bracket": "" if not upper else min(upper),
                "nearest_source_concentrations": ";".join(
                    str(int(value)) for value in nearest
                ),
                "n_target_observations": len(target_rows),
                "n_target_mutations": target_rows["mutation_group_id"].nunique(),
                "primary_reference_model_id": primary_reference[scheme_id],
                "secondary_reference_model_id": secondary_reference[scheme_id],
            }
        )
    return pd.DataFrame(rows)


def review_concentration_transfer_inputs(repo_root: Path) -> Mapping[str, object]:
    required = [
        "data/processed/missense_observations.csv",
        "data/processed/residue_features.csv",
        "data/interim/evaluation_partitions/partition_schemes.csv",
        "data/interim/evaluation_partitions/group_assignments.csv",
        "data/interim/modelling_views/scheme_observation_membership.csv",
        "data/interim/baseline_evaluation/baseline_predictions.csv",
        "data/interim/sequence_plm_embeddings/mutant_site_embeddings.npy",
        (
            "data/interim/sequence_model_sensitivity/esm2_embeddings/"
            "mutant_site_embeddings.npy"
        ),
        "data/interim/condition_transfer_evaluation/condition_transfer_summary.json",
    ]
    missing = [
        relative for relative in required if not (repo_root / relative).is_file()
    ]
    if missing:
        raise ConcentrationTransferEvaluationError(
            f"Missing concentration-transfer inputs: {missing}"
        )

    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv",
        low_memory=False,
    )
    selected = schemes[schemes["scheme_id"].astype(str).isin(CONCENTRATION_SCHEME_IDS)]
    if len(selected) != len(CONCENTRATION_SCHEME_IDS):
        raise ConcentrationTransferEvaluationError("Missing concentration schemes")
    if selected["test_labels_used_for_model_selection"].astype(bool).any():
        raise ConcentrationTransferEvaluationError(
            "Concentration test labels enter model selection"
        )
    if selected["assignment_uses_outcome_values"].astype(bool).any():
        raise ConcentrationTransferEvaluationError(
            "Concentration assignments use outcome values"
        )

    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv",
        low_memory=False,
    )
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv",
        low_memory=False,
    )
    rows = []
    for scheme_id in CONCENTRATION_SCHEME_IDS:
        scheme = _scheme_row(selected, scheme_id)
        part = membership[membership["scheme_id"].astype(str).eq(scheme_id)]
        source = part[part["scope_role"].astype(str).eq("source_model_selection")]
        target = part[part["scope_role"].astype(str).eq("target_test")]
        scheme_assignments = assignments[
            assignments["scheme_id"].astype(str).eq(scheme_id)
        ]
        primary, secondary = _reference_models(str(scheme["test_category"]))
        rows.append(
            {
                "scheme_id": scheme_id,
                "test_concentration": float(scheme["test_value"]),
                "test_category": str(scheme["test_category"]),
                "source_rows": len(source),
                "target_rows": len(target),
                "source_mutations": source["mutation_group_id"].nunique(),
                "target_mutations": target["mutation_group_id"].nunique(),
                "target_structure_rows": int(
                    target["experimental_structure_task_eligible"].astype(bool).sum()
                ),
                "source_cv_folds": int(
                    scheme_assignments["cv_fold"].dropna().nunique()
                ),
                "primary_reference": primary,
                "secondary_reference": secondary,
            }
        )
    return {
        "status": "PASS",
        "scheme_count": len(CONCENTRATION_SCHEME_IDS),
        "schemes": rows,
        "condition_mode": CONDITION_MODE,
        "test_labels_used_for_model_selection": False,
        "assignment_uses_outcome_values": False,
        "boundary_tests": [4.0, 256.0],
        "interpolation_tests": [8.0, 16.0, 32.0, 64.0, 128.0],
        "neural_models_fitted_in_stage40": False,
        "neural_model_reason": (
            "Stage 40 first tests whether fixed sequence and experimental WT-structure "
            "representations add predictive information to a numerical "
            "concentration-transfer problem. The direct dose-trajectory references are "
            "retained as the central comparators. Neural concentration models are not "
            "added before this comparison is "
            "known."
        ),
    }


def prepare_concentration_transfer_evaluation(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_concentration_transfer_inputs(repo_root)
    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv",
        low_memory=False,
    )
    schemes = schemes[
        schemes["scheme_id"].astype(str).isin(CONCENTRATION_SCHEME_IDS)
    ].copy()
    assignments = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/group_assignments.csv",
        low_memory=False,
    )
    primary_reference = _primary_reference_by_scheme(schemes)
    secondary_reference = _secondary_reference_by_scheme(schemes)

    baseline = _baseline_predictions(repo_root)
    esmc, esmc_selection = _prepare_esmc(repo_root, schemes, assignments)
    esm2, esm2_selection = _prepare_esm2(repo_root, schemes, assignments)
    structure, structure_selection = _prepare_structure(repo_root, schemes, assignments)

    full = pd.concat([baseline, esmc, esm2], ignore_index=True, sort=False)
    matched_full = _structure_matched_copy(full, structure)
    predictions = pd.concat(
        [full, matched_full, structure], ignore_index=True, sort=False
    )
    predictions["observation_id"] = predictions["observation_id"].astype(str)
    predictions["evaluation_iteration"] = predictions["evaluation_iteration"].astype(
        str
    )
    predictions["model_id"] = predictions["model_id"].astype(str)
    predictions["population_scope"] = predictions["population_scope"].astype(str)
    _validate_prediction_contract(predictions, primary_reference)

    selection = pd.DataFrame(esmc_selection + esm2_selection + structure_selection)
    metrics = _metric_rows(predictions)
    comparisons = _paired_bootstrap(predictions, primary_reference)
    design = _concentration_design_summary(
        repo_root, schemes, primary_reference, secondary_reference
    )
    return predictions, selection, metrics, comparisons, design


def run_concentration_transfer_evaluation(repo_root: Path) -> Mapping[str, object]:
    predictions, selection, metrics, comparisons, design = (
        prepare_concentration_transfer_evaluation(repo_root)
    )
    output_dir = repo_root / "data/interim/concentration_response_evaluation"
    _write_csv(predictions, output_dir / "concentration_transfer_predictions.csv")
    _write_csv(selection, output_dir / "concentration_transfer_model_selection.csv")
    _write_csv(metrics, output_dir / "concentration_transfer_metrics.csv")
    _write_csv(
        comparisons, output_dir / "concentration_transfer_paired_comparisons.csv"
    )
    _write_csv(design, output_dir / "concentration_transfer_design_summary.csv")

    schemes = pd.read_csv(
        repo_root / "data/interim/evaluation_partitions/partition_schemes.csv",
        low_memory=False,
    )
    schemes = schemes[schemes["scheme_id"].astype(str).isin(CONCENTRATION_SCHEME_IDS)]
    primary_reference = _primary_reference_by_scheme(schemes)
    secondary_reference = _secondary_reference_by_scheme(schemes)
    summary = {
        "status": "PASS",
        "scheme_ids": list(CONCENTRATION_SCHEME_IDS),
        "condition_mode": CONDITION_MODE,
        "primary_reference_by_scheme": primary_reference,
        "secondary_reference_by_scheme": secondary_reference,
        "sequence_model_ids": list(SEQUENCE_MODEL_IDS),
        "structure_model_ids": list(STRUCTURE_MODEL_IDS),
        "baseline_model_ids": list(BASELINE_MODEL_IDS),
        "population_scopes": [FULL_POPULATION, STRUCTURE_POPULATION],
        "prediction_count": int(len(predictions)),
        "metric_count": int(len(metrics)),
        "paired_comparison_count": int(len(comparisons)),
        "model_selection_record_count": int(len(selection)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "test_labels_used_for_model_selection": False,
        "assignment_uses_outcome_values": False,
        "log2_concentration_used": True,
        "boundary_tests": [4.0, 256.0],
        "interpolation_tests": [8.0, 16.0, 32.0, 64.0, 128.0],
        "neural_models_fitted": False,
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
                (
                    "data/interim/sequence_model_sensitivity/esm2_embeddings/"
                    "mutant_site_embeddings.npy"
                ),
                (
                    "data/interim/condition_transfer_evaluation/"
                    "condition_transfer_summary.json"
                ),
            ]
        },
        "outputs": [
            "concentration_transfer_predictions.csv",
            "concentration_transfer_model_selection.csv",
            "concentration_transfer_metrics.csv",
            "concentration_transfer_paired_comparisons.csv",
            "concentration_transfer_design_summary.csv",
            "concentration_transfer_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "concentration_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
