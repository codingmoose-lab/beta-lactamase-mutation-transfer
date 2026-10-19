"""Evaluate ESM-C site-difference Ridge models without source test-position scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr

from .position_withheld_homologue_baselines import fold_for_positions
from .sequence_plm_evaluation import (
    _site_delta,
    _validate_embedding_provenance,
)
from .source_full_position_baselines import (
    DIRECTIONS,
    alpha_selection,
    digest,
    load_data,
    predict_reference,
    prepare_direction,
)

EMBEDDINGS = Path("data/interim/sequence_plm_embeddings")
LAYERS = (36,)
RIDGE_PENALTIES = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
)
MODEL = "esmc_site_delta_source_position_ridge"
REFERENCE = "source_assay_identity_ridge"


class PositionWithheldESMCError(ValueError):
    """An assay, residue-group or sequence-input check failed."""


def source_cv_fold(position: str) -> int:
    value = hashlib.sha256(f"source_inner_cv|{position}".encode()).hexdigest()
    return int(value, 16) % 3


def fit_esmc(
    train: np.ndarray,
    labels: np.ndarray,
    test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit using source observations and their own feature statistics only."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise PositionWithheldESMCError("Ridge penalty must be positive")
    source = np.asarray(train, dtype=np.float64)
    target = np.asarray(test, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if (
        source.ndim != 2
        or target.ndim != 2
        or y.ndim != 1
        or not len(source)
        or not len(target)
        or len(source) != len(y)
        or source.shape[1] != target.shape[1]
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or not np.isfinite(y).all()
    ):
        raise PositionWithheldESMCError("Invalid or non-finite ESM-C fitting inputs")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            # Source-only feature statistics are calculated in float64.
            mean = source.mean(axis=0)
            sd = source.std(axis=0)
            sd[sd < 1e-8] = 1.0
            a = (source - mean) / sd
            b = (target - mean) / sd
            intercept = float(y.mean())
    except FloatingPointError as exc:
        raise PositionWithheldESMCError(
            "Non-finite ESM-C feature standardization"
        ) from exc
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(intercept)
    ):
        raise PositionWithheldESMCError(
            "Non-finite ESM-C standardized features or intercept"
        )
    # LSQR may use external BLAS; validate its fitted coefficients explicitly.
    coefficients, stop, *_ = lsqr(
        a,
        y - intercept,
        damp=np.sqrt(alpha),
        atol=1e-6,
        btol=1e-6,
        iter_lim=3000,
    )
    if stop not in (0, 1, 2):
        raise PositionWithheldESMCError(
            f"Ridge fitting did not converge: LSQR stop code {stop}"
        )
    if not np.isfinite(coefficients).all():
        raise PositionWithheldESMCError(
            "Non-finite ESM-C Ridge coefficients or predictions"
        )
    # Compute each row dot product without an external BLAS matmul call.
    predictions = intercept + np.einsum("ij,j->i", b, coefficients, optimize=False)
    if not np.isfinite(predictions).all():
        raise PositionWithheldESMCError(
            "Non-finite ESM-C Ridge coefficients or predictions"
        )
    return predictions


def choose_alpha(
    features: np.ndarray,
    scores: np.ndarray,
    source_positions: np.ndarray,
    alphas: tuple[float, ...] = RIDGE_PENALTIES,
) -> tuple[float, list[dict[str, float]]]:
    """Select Ridge penalty from three source-position-disjoint folds."""
    roles = np.asarray([source_cv_fold(str(p)) for p in source_positions])
    if len(set(roles)) != 3:
        raise PositionWithheldESMCError("Source fitting positions omit a CV fold")
    records = []
    for alpha in alphas:
        error, count = 0.0, 0
        for fold in range(3):
            tr, va = roles != fold, roles == fold
            if not tr.any() or not va.any():
                raise PositionWithheldESMCError("Empty source CV partition")
            estimate = fit_esmc(features[tr], scores[tr], features[va], alpha)
            fold_error = float(np.abs(scores[va] - estimate).sum())
            if not np.isfinite(fold_error):
                raise PositionWithheldESMCError("Non-finite ESM-C source CV error")
            error += fold_error
            count += int(va.sum())
        mean_error = float(error / count)
        if not np.isfinite(mean_error):
            raise PositionWithheldESMCError("Non-finite ESM-C source CV MAE")
        records.append(
            {
                "alpha": float(alpha),
                "source_cv_mae": mean_error,
                "source_cv_rows": count,
            }
        )
    best = min(records, key=lambda item: (item["source_cv_mae"], item["alpha"]))
    return float(best["alpha"]), records


def add_embedding_rows(frame: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Join precomputed sequence inputs without joining mutation-effect values."""
    keys = ["protein_id", "assay_position", "wild_type", "mutant"]
    lookup = index[keys + ["embedding_row"]]
    if lookup.duplicated(keys).any():
        raise PositionWithheldESMCError("Repeated mutation identities in index")
    joined = frame.merge(lookup, how="left", on=keys, validate="many_to_one")
    if joined.embedding_row.isna().any():
        raise PositionWithheldESMCError("ESM-C embedding missing for an assay mutation")
    joined["embedding_row"] = joined.embedding_row.astype(int)
    return joined


def summarize_positions(predictions: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    """Compare model and identity reference on the same held-out positions."""
    output = []
    for direction, subset in predictions.groupby("direction", sort=True):
        for pop in ("all_shared_wild_type_pairs", "both_above_minimum"):
            d = (
                subset
                if pop == "all_shared_wild_type_pairs"
                else subset.loc[subset.minimum_score_class.eq("both_above_minimum")]
            )
            if d.empty:
                raise PositionWithheldESMCError("Empty substitution population")
            y = d.target_effect.to_numpy(float)
            model_error = abs(y - d.model_prediction.to_numpy(float))
            reference_error = abs(y - d.reference_prediction.to_numpy(float))
            delta = reference_error - model_error
            if not np.isfinite(delta).all():
                raise PositionWithheldESMCError("Non-finite paired prediction errors")
            groups = d.homologue_position_id.astype(str).to_numpy()
            indices = [np.flatnonzero(groups == p) for p in sorted(set(groups))]
            rng = np.random.default_rng(
                int(
                    hashlib.sha256(f"{direction}|{pop}|esmc".encode()).hexdigest()[:8],
                    16,
                )
            )
            values = []
            for _ in range(repetitions):
                ids = np.concatenate(
                    [indices[j] for j in rng.integers(len(indices), size=len(indices))]
                )
                values.append(float(delta[ids].mean()))
            lo, hi = np.quantile(values, [0.025, 0.975])
            model_rho = float(spearmanr(y, d.model_prediction).statistic)
            reference_rho = float(spearmanr(y, d.reference_prediction).statistic)
            if not np.isfinite([lo, hi, model_rho, reference_rho]).all():
                raise PositionWithheldESMCError(
                    "Non-finite grouped comparison statistics"
                )
            output.append(
                {
                    "direction": direction,
                    "population": pop,
                    "n_substitutions": len(d),
                    "n_homologue_positions": len(indices),
                    "model_mae": float(model_error.mean()),
                    "reference_mae": float(reference_error.mean()),
                    "model_spearman": model_rho,
                    "reference_spearman": reference_rho,
                    "mae_improvement_model": float(delta.mean()),
                    "mae_improvement_ci_lower": float(lo),
                    "mae_improvement_ci_upper": float(hi),
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(output)


def run(root: Path, output: Path, repetitions: int = 1000) -> dict:
    """Fit source-only ESM-C Ridge on aligned-position five-fold holdouts."""
    if repetitions < 100:
        raise PositionWithheldESMCError("At least 100 grouped bootstrap repetitions")
    if output.resolve().is_relative_to(root.resolve()):
        raise PositionWithheldESMCError(
            "Position-withheld outputs must be written outside the source repository"
        )
    paired, dms, hashes = load_data(root)
    directory = root / EMBEDDINGS
    required = (
        "mutation_embedding_index.csv",
        "mutant_site_embeddings.npy",
        "wild_type_residue_embeddings.npy",
        "embedding_summary.json",
        "extraction_summary.json",
    )
    for name in required:
        if not (directory / name).is_file():
            raise PositionWithheldESMCError(f"Required ESM-C file absent: {name}")
    index = pd.read_csv(directory / required[0])
    mutant = np.load(directory / required[1], mmap_mode="r")
    wild_type = np.load(directory / required[2], mmap_mode="r")
    _validate_embedding_provenance(root, index, mutant, wild_type)
    for name in required:
        hashes[str(EMBEDDINGS / name)] = digest(directory / name)

    predictions, selection, reference_selection = [], [], []
    for direction in DIRECTIONS:
        target, source, assay = prepare_direction(paired, dms, direction)
        source = add_embedding_rows(source, index)
        target_inputs = target[["target_observation_id"]].merge(
            dms[
                [
                    "observation_id",
                    "protein_id",
                    "assay_position",
                    "wild_type",
                    "mutant",
                ]
            ],
            left_on="target_observation_id",
            right_on="observation_id",
            how="left",
            validate="many_to_one",
        )
        input_columns = ["protein_id", "assay_position", "wild_type", "mutant"]
        if target_inputs[input_columns].isna().any().any():
            raise PositionWithheldESMCError("Unresolved target mutation identity")
        target_inputs = add_embedding_rows(target_inputs, index)
        if not target_inputs.target_observation_id.eq(
            target.target_observation_id
        ).all():
            raise PositionWithheldESMCError("Target substitution order differs")
        x_source = _site_delta(source, mutant, wild_type, hidden_state_index=LAYERS[0])
        x_target = _site_delta(
            target_inputs, mutant, wild_type, hidden_state_index=LAYERS[0]
        )
        scores = source.score_analysis.to_numpy(float)
        fold_map = fold_for_positions(target.homologue_position_id)
        role = target.homologue_position_id.astype(str).map(fold_map).to_numpy(int)
        for fold in range(5):
            heldout = role == fold
            excluded = set(target.loc[heldout, "assay_position"])
            fit = ~source.assay_position.isin(excluded).to_numpy()
            if not heldout.any() or not fit.any():
                raise PositionWithheldESMCError("Empty source or target position fold")
            if set(source.loc[fit, "assay_position"]) & excluded:
                raise PositionWithheldESMCError(
                    "Held-out source position entered fitting"
                )
            alpha, cv = choose_alpha(
                x_source[fit],
                scores[fit],
                source.loc[fit, "assay_position"].to_numpy(),
            )
            estimate = fit_esmc(x_source[fit], scores[fit], x_target[heldout], alpha)
            # This reference is independently fitted on identical source rows.
            baseline_alpha, baseline_cv = alpha_selection(
                source.loc[fit], alphas=RIDGE_PENALTIES
            )
            baseline_cv_mae = next(
                item["source_cv_mae"]
                for item in baseline_cv
                if item["alpha"] == baseline_alpha
            )
            for item in baseline_cv:
                reference_selection.append(
                    {
                        "direction": direction,
                        "heldout_fold": fold,
                        "source_assay_id": assay,
                        "source_fit_rows": int(fit.sum()),
                        "selected_alpha": baseline_alpha,
                        **item,
                    }
                )
            ref = predict_reference(
                source.loc[fit],
                target.loc[heldout, ["wild_type", "mutant", "assay_position"]],
                baseline_alpha,
            )[REFERENCE]
            block = target.loc[
                heldout,
                [
                    "direction",
                    "homologue_position_id",
                    "target_observation_id",
                    "target_effect",
                    "minimum_score_class",
                ],
            ].copy()
            block["heldout_fold"] = fold
            block["model_id"] = MODEL
            block["model_prediction"] = estimate
            block["reference_model_id"] = REFERENCE
            block["reference_prediction"] = ref
            predictions.append(block)
            for item in cv:
                selection.append(
                    {
                        "direction": direction,
                        "heldout_fold": fold,
                        "source_assay_id": assay,
                        "hidden_state_index": LAYERS[0],
                        "selected_alpha": alpha,
                        "reference_selected_alpha": baseline_alpha,
                        "reference_source_cv_mae": baseline_cv_mae,
                        "source_fit_rows": int(fit.sum()),
                        "source_fit_positions": int(
                            source.loc[fit, "assay_position"].nunique()
                        ),
                        "excluded_source_positions": len(excluded),
                        **item,
                    }
                )
    frame = pd.concat(predictions, ignore_index=True)
    if (
        len(frame) != 3026
        or frame.duplicated(["direction", "target_observation_id"]).any()
    ):
        raise PositionWithheldESMCError(
            "Expected 1,513 target predictions in each direction"
        )
    if (
        not np.isfinite(frame.model_prediction.to_numpy(float)).all()
        or not np.isfinite(frame.reference_prediction.to_numpy(float)).all()
        or not np.isfinite(frame.target_effect.to_numpy(float)).all()
        or any(
            len(part) != 1513 or part.homologue_position_id.nunique() != 81
            for _, part in frame.groupby("direction")
        )
    ):
        raise PositionWithheldESMCError(
            "Invalid paired prediction population or values"
        )
    summary = summarize_positions(frame, repetitions)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "source_position_withheld_esmc_predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(
        output / "source_position_withheld_esmc_source_cv.csv", index=False
    )
    pd.DataFrame(reference_selection).to_csv(
        output / "source_position_withheld_esmc_reference_source_cv.csv", index=False
    )
    summary.to_csv(
        output / "source_position_withheld_esmc_comparisons.csv", index=False
    )
    result = {
        "method": "source_position_heldout_five_fold_cv",
        "model_id": MODEL,
        "reference_model_id": REFERENCE,
        "esmc_hidden_state_index": LAYERS[0],
        "esmc_penalty_candidates": list(RIDGE_PENALTIES),
        "reference_penalty_candidates": list(RIDGE_PENALTIES),
        "prediction_count": len(frame),
        "source_only_parameter_selection": True,
        "target_effects_used_for_fitting": False,
        "source_effect_at_heldout_homologue_position_used_for_fitting": False,
        "full_source_ampicillin_assay_used": True,
        "source_position_bootstrap_repetitions": repetitions,
        "limitations": (
            "Evaluation restricted to 81 matched shared-wild-type positions. "
            "This is a Ridge regression using previously extracted ESM-C features, "
            "not a repeat of neural residue-graph training. "
            "The above-minus-four population is selected using measured target effects."
        ),
        "input_sha256": hashes,
    }
    (output / "source_position_withheld_esmc_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
