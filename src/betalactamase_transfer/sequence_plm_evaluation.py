from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .sequence_neighbourhood_evaluation import (
    CONDITION_MODE_BY_SCHEME,
    CORE_SCHEME_IDS,
    SUBSTITUTION_INDEX,
    SUBSTITUTIONS,
    TARGET_COLUMN,
    _load_assay_sequences,
)

MODEL_ID = "esmc_site_delta_ridge"

ESMC_SOURCE_MODEL_ID = "biohub/ESMC-600M"
ESMC_SOURCE_MODEL_REVISION = "a7e82012c83126b9eedb055fea9fa84b6c02f094"
ESMC_HIDDEN_SIZE = 1152

EMBEDDING_HIDDEN_STATE_INDICES = (9, 18, 27, 36)
ESMC_RIDGE_ALPHAS = (
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
)

EXPECTED_PROTEINS = ("NDM-1", "VIM-2")

CHUNK_SIZE = 2048


class SequencePLMEvaluationError(RuntimeError):
    """Raised when ESMC mutation-effect evaluation cannot be completed safely."""


@dataclass(frozen=True)
class ESMCTransform:
    hidden_state_index: int
    condition_mode: str
    condition_categories: tuple[str, ...]
    concentration_mean: float
    concentration_sd: float
    embedding_mean: np.ndarray
    embedding_sd: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_embedding_provenance(
    repo_root: Path,
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> None:
    embedding_dir = repo_root / "data/interim/sequence_plm_embeddings"
    index_path = embedding_dir / "mutation_embedding_index.csv"

    embedding_summary = json.loads(
        (embedding_dir / "embedding_summary.json").read_text(encoding="utf-8")
    )
    extraction_summary = json.loads(
        (embedding_dir / "extraction_summary.json").read_text(encoding="utf-8")
    )

    for name, summary in (
        ("embedding summary", embedding_summary),
        ("extraction summary", extraction_summary),
    ):
        if summary.get("model_id") != ESMC_SOURCE_MODEL_ID:
            raise SequencePLMEvaluationError(
                f"{name} has an unexpected ESMC model identifier"
            )
        if summary.get("model_revision") != ESMC_SOURCE_MODEL_REVISION:
            raise SequencePLMEvaluationError(
                f"{name} has an unexpected ESMC model revision"
            )
        if list(summary.get("hidden_state_indices", [])) != list(
            EMBEDDING_HIDDEN_STATE_INDICES
        ):
            raise SequencePLMEvaluationError(
                f"{name} has unexpected ESMC hidden-state indices"
            )
        if int(summary.get("hidden_size", -1)) != ESMC_HIDDEN_SIZE:
            raise SequencePLMEvaluationError(
                f"{name} has an unexpected ESMC hidden size"
            )
        if int(summary.get("mutation_count", -1)) != len(embedding_index):
            raise SequencePLMEvaluationError(
                f"{name} mutation count does not match the embedding index"
            )
        if summary.get("precision") != "float16":
            raise SequencePLMEvaluationError(
                f"{name} has an unexpected embedding precision"
            )
        if summary.get("outcome_columns_loaded") is not False:
            raise SequencePLMEvaluationError(
                f"{name} does not establish outcome-blind extraction"
            )

    actual_index_sha256 = _sha256_file(index_path)
    summary_index_sha256 = str(embedding_summary.get("embedding_index_sha256", ""))
    extraction_index_sha256 = str(extraction_summary.get("embedding_index_sha256", ""))

    if not summary_index_sha256 or not extraction_index_sha256:
        raise SequencePLMEvaluationError(
            "Embedding-index SHA-256 is missing from extraction record"
        )
    if summary_index_sha256 != extraction_index_sha256:
        raise SequencePLMEvaluationError(
            "Embedding-index SHA-256 disagrees between source summaries"
        )
    if actual_index_sha256 != summary_index_sha256:
        raise SequencePLMEvaluationError(
            "Embedding-index SHA-256 does not match extraction record"
        )

    assay_sequences = _load_assay_sequences(repo_root)

    if set(assay_sequences) != set(EXPECTED_PROTEINS):
        raise SequencePLMEvaluationError(
            "Unexpected proteins in the assay-sequence contract"
        )

    recorded_lengths = embedding_summary.get("assay_sequence_lengths")
    recorded_hashes = embedding_summary.get("assay_sequence_sha256")

    if not isinstance(recorded_lengths, dict) or not isinstance(recorded_hashes, dict):
        raise SequencePLMEvaluationError(
            "Embedding summary lacks assay-sequence source record"
        )

    for protein_id in EXPECTED_PROTEINS:
        sequence = assay_sequences[protein_id]
        if int(recorded_lengths.get(protein_id, -1)) != len(sequence):
            raise SequencePLMEvaluationError(
                f"Assay-sequence length disagrees for {protein_id}"
            )
        if recorded_hashes.get(protein_id) != _sha256_text(sequence):
            raise SequencePLMEvaluationError(
                f"Assay-sequence SHA-256 disagrees for {protein_id}"
            )

    expected_mutant_shape = (
        len(embedding_index),
        len(EMBEDDING_HIDDEN_STATE_INDICES),
        ESMC_HIDDEN_SIZE,
    )
    if mutant_embeddings.shape != expected_mutant_shape:
        raise SequencePLMEvaluationError(
            f"Unexpected mutant embedding shape: {mutant_embeddings.shape}"
        )

    expected_wild_type_shape = (
        len(EXPECTED_PROTEINS),
        len(EMBEDDING_HIDDEN_STATE_INDICES),
        max(len(assay_sequences[protein]) for protein in EXPECTED_PROTEINS),
        ESMC_HIDDEN_SIZE,
    )
    if wild_type_embeddings.shape != expected_wild_type_shape:
        raise SequencePLMEvaluationError(
            f"Unexpected wild-type embedding shape: {wild_type_embeddings.shape}"
        )

    if not np.isfinite(mutant_embeddings).all():
        raise SequencePLMEvaluationError(
            "Mutant embedding array contains non-finite values"
        )

    for protein_index, protein_id in enumerate(EXPECTED_PROTEINS):
        length = len(assay_sequences[protein_id])
        used = wild_type_embeddings[protein_index, :, :length, :]
        if not np.isfinite(used).all():
            raise SequencePLMEvaluationError(
                f"Wild-type embedding array contains non-finite values for {protein_id}"
            )


def _condition_token(frame: pd.DataFrame) -> pd.Series:
    concentration = frame["concentration"].map(
        lambda value: "" if pd.isna(value) else format(float(value), ".12g")
    )
    temperature = frame["temperature_c"].map(
        lambda value: "" if pd.isna(value) else format(float(value), ".12g")
    )
    return (
        frame["antibiotic"].fillna("").astype(str)
        + "|"
        + concentration
        + "|"
        + temperature
    )


def _load_inputs(
    repo_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    observations = pd.read_csv(
        repo_root / "data/processed/missense_observations.csv",
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
    membership = pd.read_csv(
        repo_root / "data/interim/modelling_views/scheme_observation_membership.csv",
        low_memory=False,
    )

    embedding_dir = repo_root / "data/interim/sequence_plm_embeddings"

    embedding_index = pd.read_csv(
        embedding_dir / "mutation_embedding_index.csv",
        low_memory=False,
    )
    mutant_embeddings = np.load(
        embedding_dir / "mutant_site_embeddings.npy",
        mmap_mode="r",
    )
    wild_type_embeddings = np.load(
        embedding_dir / "wild_type_residue_embeddings.npy",
        mmap_mode="r",
    )

    return (
        observations,
        schemes,
        assignments,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )


def _validate_inputs(
    observations: pd.DataFrame,
    membership: pd.DataFrame,
    embedding_index: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> None:
    if TARGET_COLUMN not in observations:
        raise SequencePLMEvaluationError(f"Missing outcome column: {TARGET_COLUMN}")
    if observations[TARGET_COLUMN].isna().any():
        raise SequencePLMEvaluationError("Outcome column contains missing values")

    if TARGET_COLUMN in membership.columns or "score_original" in membership.columns:
        raise SequencePLMEvaluationError("Outcome leaked into modelling membership")

    required_index_columns = {
        "embedding_row",
        "protein_id",
        "assay_mutation",
        "assay_position",
        "wild_type",
        "mutant",
        "mutant_sequence_sha256",
    }
    missing = required_index_columns.difference(embedding_index.columns)
    if missing:
        raise SequencePLMEvaluationError(
            f"Embedding index is missing columns: {sorted(missing)}"
        )

    if embedding_index.duplicated(["protein_id", "assay_mutation"]).any():
        raise SequencePLMEvaluationError(
            "Embedding index contains duplicate protein/mutation rows"
        )

    expected_rows = np.arange(len(embedding_index), dtype=int)
    try:
        actual_rows = embedding_index["embedding_row"].to_numpy(dtype=int)
    except (TypeError, ValueError) as exc:
        raise SequencePLMEvaluationError(
            "Embedding rows must be integer-valued"
        ) from exc

    if not np.array_equal(actual_rows, expected_rows):
        raise SequencePLMEvaluationError(
            "Embedding rows must form the contiguous range 0..N-1"
        )

    proteins = set(embedding_index["protein_id"].astype(str))
    if proteins != set(EXPECTED_PROTEINS):
        raise SequencePLMEvaluationError(
            f"Unexpected embedding-index proteins: {sorted(proteins)}"
        )


def _join_scheme_observations(
    membership: pd.DataFrame,
    observations: pd.DataFrame,
    embedding_index: pd.DataFrame,
    scheme_id: str,
) -> pd.DataFrame:
    scheme_membership = membership[membership["scheme_id"].eq(scheme_id)].copy()

    if scheme_membership.empty:
        raise SequencePLMEvaluationError(f"No modelling rows for {scheme_id}")

    observation_columns = [
        "observation_id",
        "protein_id",
        "assay_mutation",
        "assay_position",
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
        TARGET_COLUMN,
    ]

    frame = scheme_membership.merge(
        observations[observation_columns],
        on=["observation_id", "protein_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_observation"),
    )

    if frame[TARGET_COLUMN].isna().any():
        raise SequencePLMEvaluationError(f"Missing outcomes after joining {scheme_id}")

    frame = frame.merge(
        embedding_index[
            [
                "embedding_row",
                "protein_id",
                "assay_mutation",
                "assay_position",
                "wild_type",
                "mutant",
            ]
        ],
        on=[
            "protein_id",
            "assay_mutation",
            "assay_position",
            "wild_type",
            "mutant",
        ],
        how="left",
        validate="many_to_one",
    )

    if frame["embedding_row"].isna().any():
        raise SequencePLMEvaluationError(f"Missing ESMC embeddings for {scheme_id}")

    frame["embedding_row"] = frame["embedding_row"].astype(int)
    return frame


def _hidden_state_array_index(hidden_state_index: int) -> int:
    try:
        return EMBEDDING_HIDDEN_STATE_INDICES.index(int(hidden_state_index))
    except ValueError as exc:
        raise SequencePLMEvaluationError(
            f"Unsupported ESMC hidden-state index: {hidden_state_index}"
        ) from exc


def _site_delta(
    frame: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    hidden_state_index: int,
) -> np.ndarray:
    layer_array_index = _hidden_state_array_index(hidden_state_index)
    embedding_rows = frame["embedding_row"].to_numpy(dtype=int)
    positions = frame["assay_position"].to_numpy(dtype=int) - 1
    proteins = frame["protein_id"].astype(str).to_numpy()

    unexpected_proteins = sorted(set(proteins).difference(EXPECTED_PROTEINS))
    if unexpected_proteins:
        raise SequencePLMEvaluationError(
            f"Unexpected protein identifiers in ESMC rows: {unexpected_proteins}"
        )

    protein_index = {
        protein_id: index for index, protein_id in enumerate(EXPECTED_PROTEINS)
    }
    protein_indices = np.asarray(
        [protein_index[protein_id] for protein_id in proteins],
        dtype=int,
    )

    sequence_lengths = {
        "NDM-1": wild_type_embeddings.shape[2],
        "VIM-2": 267,
    }
    for protein_id in EXPECTED_PROTEINS:
        mask = proteins == protein_id
        if mask.any():
            invalid = (positions[mask] < 0) | (
                positions[mask] >= sequence_lengths[protein_id]
            )
            if invalid.any():
                raise SequencePLMEvaluationError(
                    f"ESMC assay position outside {protein_id} sequence"
                )

    mutant = np.asarray(
        mutant_embeddings[
            embedding_rows,
            layer_array_index,
            :,
        ],
        dtype=np.float32,
    )

    wild_type = np.asarray(
        wild_type_embeddings[
            protein_indices,
            layer_array_index,
            positions,
            :,
        ],
        dtype=np.float32,
    )

    delta = mutant - wild_type

    if not np.isfinite(delta).all():
        raise SequencePLMEvaluationError("Non-finite ESMC site-delta values")

    return delta


def _fit_transform(
    train: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    condition_mode: str,
    hidden_state_index: int,
) -> ESMCTransform:
    delta = _site_delta(
        train,
        mutant_embeddings,
        wild_type_embeddings,
        hidden_state_index=hidden_state_index,
    ).astype(np.float64)

    mean = delta.mean(axis=0)
    sd = delta.std(axis=0, ddof=0)

    bad_sd = ~np.isfinite(sd) | (sd == 0.0)
    sd[bad_sd] = 1.0

    categories: tuple[str, ...] = ()
    concentration_mean = 0.0
    concentration_sd = 1.0

    if condition_mode == "categorical_condition":
        categories = tuple(sorted(_condition_token(train).unique()))

    elif condition_mode == "log2_concentration":
        concentration = train["concentration"].to_numpy(dtype=float)
        if not np.isfinite(concentration).all() or (concentration <= 0).any():
            raise SequencePLMEvaluationError(
                "Positive finite concentration is required"
            )

        values = np.log2(concentration)
        concentration_mean = float(values.mean())
        concentration_sd = float(values.std(ddof=0))

        if not math.isfinite(concentration_sd) or concentration_sd == 0.0:
            concentration_sd = 1.0

    elif condition_mode != "none":
        raise SequencePLMEvaluationError(f"Unknown condition mode: {condition_mode}")

    return ESMCTransform(
        hidden_state_index=hidden_state_index,
        condition_mode=condition_mode,
        condition_categories=categories,
        concentration_mean=concentration_mean,
        concentration_sd=concentration_sd,
        embedding_mean=mean,
        embedding_sd=sd,
    )


def _feature_count(transform: ESMCTransform) -> int:
    count = 1 + len(SUBSTITUTIONS)

    if transform.condition_mode == "categorical_condition":
        count += len(transform.condition_categories)
    elif transform.condition_mode == "log2_concentration":
        count += 1

    count += 1152
    return count


def _design_chunk(
    frame: pd.DataFrame,
    transform: ESMCTransform,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> np.ndarray:
    feature_count = _feature_count(transform)
    matrix = np.zeros(
        (len(frame), feature_count),
        dtype=np.float64,
    )

    matrix[:, 0] = 1.0
    substitution_offset = 1

    for row_number, row in enumerate(frame.itertuples(index=False)):
        substitution = (str(row.wild_type), str(row.mutant))
        if substitution not in SUBSTITUTION_INDEX:
            raise SequencePLMEvaluationError(
                f"Invalid missense substitution: {substitution}"
            )

        matrix[
            row_number,
            substitution_offset + SUBSTITUTION_INDEX[substitution],
        ] = 1.0

    condition_offset = substitution_offset + len(SUBSTITUTIONS)
    embedding_offset = condition_offset

    if transform.condition_mode == "categorical_condition":
        condition_index = {
            value: index for index, value in enumerate(transform.condition_categories)
        }
        tokens = _condition_token(frame).tolist()

        for row_number, token in enumerate(tokens):
            if token in condition_index:
                matrix[
                    row_number,
                    condition_offset + condition_index[token],
                ] = 1.0

        embedding_offset += len(transform.condition_categories)

    elif transform.condition_mode == "log2_concentration":
        concentration = frame["concentration"].to_numpy(dtype=float)
        if not np.isfinite(concentration).all() or (concentration <= 0).any():
            raise SequencePLMEvaluationError(
                "Positive finite concentration is required"
            )

        matrix[:, condition_offset] = (
            np.log2(concentration) - transform.concentration_mean
        ) / transform.concentration_sd

        embedding_offset += 1

    delta = _site_delta(
        frame,
        mutant_embeddings,
        wild_type_embeddings,
        hidden_state_index=transform.hidden_state_index,
    ).astype(np.float64)

    matrix[:, embedding_offset:] = (
        delta - transform.embedding_mean
    ) / transform.embedding_sd

    if not np.isfinite(matrix).all():
        raise SequencePLMEvaluationError("Non-finite ESMC design matrix")

    return matrix


def _ridge_statistics(
    train: pd.DataFrame,
    transform: ESMCTransform,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = _feature_count(transform)
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    target_product = np.zeros(feature_count, dtype=np.float64)

    for start in range(0, len(train), CHUNK_SIZE):
        chunk = train.iloc[start : start + CHUNK_SIZE]
        matrix = _design_chunk(
            chunk,
            transform,
            mutant_embeddings,
            wild_type_embeddings,
        )
        target = chunk[TARGET_COLUMN].to_numpy(dtype=np.float64)

        with np.errstate(
            divide="ignore",
            over="ignore",
            invalid="ignore",
        ):
            gram_update = matrix.T @ matrix
            target_update = matrix.T @ target

        if not np.isfinite(gram_update).all():
            raise SequencePLMEvaluationError("Non-finite ridge Gram-matrix update")
        if not np.isfinite(target_update).all():
            raise SequencePLMEvaluationError("Non-finite ridge target-product update")

        gram += gram_update
        target_product += target_update

    if not np.isfinite(gram).all():
        raise SequencePLMEvaluationError("Non-finite ridge Gram matrix")
    if not np.isfinite(target_product).all():
        raise SequencePLMEvaluationError("Non-finite ridge target product")

    return gram, target_product


def _ridge_coefficients(
    gram: np.ndarray,
    target_product: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise SequencePLMEvaluationError("Ridge alpha must be positive and finite")

    penalty = np.eye(len(target_product), dtype=np.float64)
    penalty[0, 0] = 0.0

    coefficients = np.linalg.solve(
        gram + float(alpha) * penalty,
        target_product,
    )

    if not np.isfinite(coefficients).all():
        raise SequencePLMEvaluationError("Ridge produced non-finite coefficients")

    return coefficients


def _predict(
    frame: pd.DataFrame,
    transform: ESMCTransform,
    coefficients: np.ndarray,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=np.float64)

    for start in range(0, len(frame), CHUNK_SIZE):
        chunk = frame.iloc[start : start + CHUNK_SIZE]
        matrix = _design_chunk(
            chunk,
            transform,
            mutant_embeddings,
            wild_type_embeddings,
        )
        with np.errstate(
            divide="ignore",
            over="ignore",
            invalid="ignore",
        ):
            prediction_chunk = matrix @ coefficients

        if not np.isfinite(prediction_chunk).all():
            raise SequencePLMEvaluationError("Non-finite ESMC prediction")

        predictions[start : start + len(chunk)] = prediction_chunk

    if not np.isfinite(predictions).all():
        raise SequencePLMEvaluationError("Non-finite ESMC predictions")

    return predictions


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    condition_mode: str,
    hidden_state_index: int,
    alpha: float,
) -> np.ndarray:
    transform = _fit_transform(
        train,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
        hidden_state_index=hidden_state_index,
    )

    gram, target_product = _ridge_statistics(
        train,
        transform,
        mutant_embeddings,
        wild_type_embeddings,
    )

    coefficients = _ridge_coefficients(
        gram,
        target_product,
        alpha=alpha,
    )

    return _predict(
        test,
        transform,
        coefficients,
        mutant_embeddings,
        wild_type_embeddings,
    )


def _select_model_setting(
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
    *,
    condition_mode: str,
) -> tuple[int, float, float, list[Mapping[str, object]]]:
    detail: list[Mapping[str, object]] = []
    candidates: list[tuple[float, int, float]] = []

    for hidden_state_index in EMBEDDING_HIDDEN_STATE_INDICES:
        fold_data = []

        for train, validation in folds:
            transform = _fit_transform(
                train,
                mutant_embeddings,
                wild_type_embeddings,
                condition_mode=condition_mode,
                hidden_state_index=hidden_state_index,
            )

            gram, target_product = _ridge_statistics(
                train,
                transform,
                mutant_embeddings,
                wild_type_embeddings,
            )

            fold_data.append(
                (
                    validation,
                    transform,
                    gram,
                    target_product,
                )
            )

        for alpha in ESMC_RIDGE_ALPHAS:
            fold_maes = []

            for fold_index, (
                validation,
                transform,
                gram,
                target_product,
            ) in enumerate(fold_data):
                coefficients = _ridge_coefficients(
                    gram,
                    target_product,
                    alpha=alpha,
                )

                predictions = _predict(
                    validation,
                    transform,
                    coefficients,
                    mutant_embeddings,
                    wild_type_embeddings,
                )

                mae = float(
                    np.mean(
                        np.abs(
                            validation[TARGET_COLUMN].to_numpy(dtype=float)
                            - predictions
                        )
                    )
                )

                fold_maes.append(mae)

                detail.append(
                    {
                        "model_id": MODEL_ID,
                        "hidden_state_index": hidden_state_index,
                        "alpha": alpha,
                        "cv_fold": fold_index,
                        "validation_mae": mae,
                    }
                )

            candidates.append(
                (
                    float(np.mean(fold_maes)),
                    hidden_state_index,
                    alpha,
                )
            )

    best_mae, best_hidden_state_index, best_alpha = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )

    return best_hidden_state_index, best_alpha, best_mae, detail


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Mapping[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise SequencePLMEvaluationError("Metrics require equal non-empty arrays")

    error = y_pred - y_true
    true_constant = np.allclose(
        y_true,
        y_true[0],
        rtol=0.0,
        atol=1e-12,
    )
    pred_constant = np.allclose(
        y_pred,
        y_pred[0],
        rtol=0.0,
        atol=1e-12,
    )

    if true_constant or pred_constant:
        pearson = float("nan")
        spearman = float("nan")
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


def _prediction_records(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    scheme: pd.Series,
    evaluation_iteration: object,
    evaluation_role: str,
    hidden_state_index: int,
    alpha: float,
) -> list[Mapping[str, object]]:
    result = pd.DataFrame(
        {
            "scheme_id": str(scheme["scheme_id"]),
            "task_id": str(scheme["task_id"]),
            "model_id": MODEL_ID,
            "hidden_state_index": hidden_state_index,
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
        }
    )

    return result.to_dict(orient="records")


def _nested_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
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

        role_map = dict(
            zip(
                current["group_id"],
                current["split_role"],
            )
        )
        roles = frame["group_id"].map(role_map)

        train = frame[roles.eq("train")].copy()
        validation = frame[roles.eq("validation")].copy()
        test = frame[roles.eq("test")].copy()

        if train.empty or validation.empty or test.empty:
            raise SequencePLMEvaluationError(
                f"Incomplete nested split for {scheme_id}/{iteration}"
            )

        hidden_state_index, alpha, validation_mae, detail = _select_model_setting(
            [(train, validation)],
            mutant_embeddings,
            wild_type_embeddings,
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
                        row["hidden_state_index"] == hidden_state_index
                        and row["alpha"] == alpha
                    ),
                }
            )

        selections.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": iteration,
                "selection_scope": "selected_setting",
                "model_id": MODEL_ID,
                "hidden_state_index": hidden_state_index,
                "alpha": alpha,
                "cv_fold": "",
                "validation_mae": validation_mae,
                "selected": True,
            }
        )

        refit_data = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        y_pred = _fit_predict(
            refit_data,
            test,
            mutant_embeddings,
            wild_type_embeddings,
            condition_mode=condition_mode,
            hidden_state_index=hidden_state_index,
            alpha=alpha,
        )

        predictions.extend(
            _prediction_records(
                test,
                y_pred,
                scheme=scheme,
                evaluation_iteration=iteration,
                evaluation_role="outer_test",
                hidden_state_index=hidden_state_index,
                alpha=alpha,
            )
        )

    return predictions, selections


def _source_cv_evaluation(
    scheme: pd.Series,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    mutant_embeddings: np.ndarray,
    wild_type_embeddings: np.ndarray,
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
        dict(
            zip(
                source_assignments["group_id"],
                source_assignments["cv_fold"],
            )
        )
    )

    if source["source_cv_fold"].isna().any():
        raise SequencePLMEvaluationError(f"Unassigned source CV rows for {scheme_id}")

    if set(target["group_id"]) != set(target_assignments["group_id"]):
        raise SequencePLMEvaluationError(f"Target-test group mismatch for {scheme_id}")

    folds = []

    for fold in sorted(source["source_cv_fold"].astype(int).unique()):
        folds.append(
            (
                source[~source["source_cv_fold"].eq(fold)].copy(),
                source[source["source_cv_fold"].eq(fold)].copy(),
            )
        )

    hidden_state_index, alpha, validation_mae, detail = _select_model_setting(
        folds,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
    )

    selections: list[Mapping[str, object]] = []

    for row in detail:
        selections.append(
            {
                "scheme_id": scheme_id,
                "task_id": str(scheme["task_id"]),
                "evaluation_iteration": "target",
                "selection_scope": "source_cv",
                **row,
                "selected": (
                    row["hidden_state_index"] == hidden_state_index
                    and row["alpha"] == alpha
                ),
            }
        )

    selections.append(
        {
            "scheme_id": scheme_id,
            "task_id": str(scheme["task_id"]),
            "evaluation_iteration": "target",
            "selection_scope": "selected_setting",
            "model_id": MODEL_ID,
            "hidden_state_index": hidden_state_index,
            "alpha": alpha,
            "cv_fold": "",
            "validation_mae": validation_mae,
            "selected": True,
        }
    )

    y_pred = _fit_predict(
        source,
        target,
        mutant_embeddings,
        wild_type_embeddings,
        condition_mode=condition_mode,
        hidden_state_index=hidden_state_index,
        alpha=alpha,
    )

    predictions = _prediction_records(
        target,
        y_pred,
        scheme=scheme,
        evaluation_iteration="target",
        evaluation_role="target_test",
        hidden_state_index=hidden_state_index,
        alpha=alpha,
    )

    return predictions, selections


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (scheme_id, model_id), group in predictions.groupby(
        ["scheme_id", "model_id"],
        sort=True,
    ):
        subsets = list(
            group.groupby(
                "evaluation_iteration",
                sort=True,
            )
        )
        subsets.append(("pooled", group))

        for evaluation_iteration, subset in subsets:
            hidden_states = sorted(subset["hidden_state_index"].unique().tolist())
            metric_hidden_state: int | str
            if len(hidden_states) == 1:
                metric_hidden_state = int(hidden_states[0])
            else:
                metric_hidden_state = "mixed"

            rows.append(
                {
                    "scheme_id": scheme_id,
                    "model_id": model_id,
                    "hidden_state_index": metric_hidden_state,
                    "evaluation_iteration": evaluation_iteration,
                    "n_observations": len(subset),
                    "n_groups": subset["evaluation_group_id"].nunique(),
                    **_metric_values(
                        subset["y_true"].to_numpy(dtype=float),
                        subset["y_pred"].to_numpy(dtype=float),
                    ),
                }
            )

    return pd.DataFrame(rows)


def prepare_sequence_plm_evaluation(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (
        observations,
        schemes,
        assignments,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    ) = _load_inputs(repo_root)

    _validate_inputs(
        observations,
        membership,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )
    _validate_embedding_provenance(
        repo_root,
        embedding_index,
        mutant_embeddings,
        wild_type_embeddings,
    )

    selected_schemes = schemes[schemes["scheme_id"].isin(CORE_SCHEME_IDS)].copy()

    if set(selected_schemes["scheme_id"]) != set(CORE_SCHEME_IDS):
        raise SequencePLMEvaluationError("Core evaluation schemes are incomplete")

    if not selected_schemes["status"].eq("ready").all():
        raise SequencePLMEvaluationError("Core evaluation schemes must all be ready")

    prediction_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []

    for _, scheme in selected_schemes.sort_values("scheme_id").iterrows():
        frame = _join_scheme_observations(
            membership,
            observations,
            embedding_index,
            str(scheme["scheme_id"]),
        )

        family = str(scheme["assignment_family"])

        if family == "five_outer_folds_with_grouped_inner_validation":
            scheme_predictions, scheme_selection = _nested_evaluation(
                scheme,
                frame,
                assignments,
                mutant_embeddings,
                wild_type_embeddings,
            )

        elif family == "source_only_grouped_cv_then_target_test":
            scheme_predictions, scheme_selection = _source_cv_evaluation(
                scheme,
                frame,
                assignments,
                mutant_embeddings,
                wild_type_embeddings,
            )

        else:
            raise SequencePLMEvaluationError(f"Unsupported assignment family: {family}")

        prediction_rows.extend(scheme_predictions)
        selection_rows.extend(scheme_selection)

    predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows)

    if predictions.empty:
        raise SequencePLMEvaluationError("No ESMC predictions were generated")

    metrics = _metric_rows(predictions)

    return predictions, metrics, selection


def write_sequence_plm_evaluation(
    repo_root: Path,
) -> Mapping[str, object]:
    predictions, metrics, selection = prepare_sequence_plm_evaluation(repo_root)

    output_dir = repo_root / "data/interim/sequence_plm_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        output_dir / "sequence_plm_predictions.csv",
        index=False,
    )
    metrics.to_csv(
        output_dir / "sequence_plm_metrics.csv",
        index=False,
    )
    selection.to_csv(
        output_dir / "sequence_plm_model_selection.csv",
        index=False,
    )

    summary = {
        "model_id": MODEL_ID,
        "candidate_hidden_state_indices": list(EMBEDDING_HIDDEN_STATE_INDICES),
        "candidate_alphas": list(ESMC_RIDGE_ALPHAS),
        "model_selection_metric": "mean_validation_mae",
        "model_selection_tie_break": (
            "lower hidden-state index, then lower alpha for exact mean-MAE ties"
        ),
        "scheme_ids": sorted(predictions["scheme_id"].unique().tolist()),
        "prediction_count": int(len(predictions)),
        "metric_row_count": int(len(metrics)),
        "model_selection_row_count": int(len(selection)),
        "test_labels_used_for_model_selection": False,
        "embedding_scaling": (
            "training-data mean and standard deviation within each "
            "model-selection or refit partition"
        ),
        "site_delta_definition": (
            "mutant-site ESMC hidden state minus the corresponding "
            "wild-type residue ESMC hidden state"
        ),
    }

    (output_dir / "sequence_plm_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary
