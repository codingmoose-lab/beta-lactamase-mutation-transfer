"""Source-protein DMS fitting with homologous positions withheld.

Target mutation-effect measurements are read only after fitting and predictions.
The source assay, held-out positions, and all tuning folds are fixed without
accessing target outcomes. The Ridge penalty uses source-position CV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from betalactamase_transfer.position_withheld_homologue_baselines import (
    fold_for_positions,
)

MATCHED = (
    "data/interim/baseline_evaluation/homologue_analysis/"
    "matched_homologue_predictions.csv"
)
SOURCE = "data/processed/missense_observations.csv"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIRECTIONS = ("NDM-1_to_VIM-2", "VIM-2_to_NDM-1")
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
SHRINKAGE = 5.0
MODELS = (
    "source_assay_mean",
    "source_assay_mutant_mean",
    "source_assay_substitution_mean",
    "source_assay_identity_ridge",
)


class SourcePositionEvaluationError(ValueError):
    """Invalid source assay or fitting population."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    absent = columns - set(frame.columns)
    if absent:
        raise SourcePositionEvaluationError(
            f"{name} is missing columns: {sorted(absent)}"
        )


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paired_file = root / MATCHED
    source_file = root / SOURCE
    if not paired_file.is_file() or not source_file.is_file():
        raise SourcePositionEvaluationError(
            f"Required tables absent: {paired_file}; {source_file}"
        )
    pairs = pd.read_csv(
        paired_file,
        dtype={"source_observation_id": str, "target_observation_id": str},
        low_memory=False,
    )
    require(
        pairs,
        {
            "direction",
            "homologue_position_id",
            "source_observation_id",
            "target_observation_id",
            "source_effect",
            "target_effect",
            "minimum_score_class",
            "wild_type",
            "mutant",
        },
        "paired observations",
    )
    if len(pairs) != 3026 or set(pairs.direction) != set(DIRECTIONS):
        raise SourcePositionEvaluationError("Expected 1,513 pairs per direction")
    cols = [
        "observation_id",
        "assay_id",
        "protein_id",
        "assay_position",
        "wild_type",
        "mutant",
        "score_analysis",
        "antibiotic",
        "concentration",
        "temperature_c",
    ]
    source = pd.read_csv(
        source_file,
        usecols=cols,
        dtype={"observation_id": str},
        low_memory=False,
    )
    require(source, set(cols), "source missense observations")
    if source.observation_id.duplicated().any():
        raise SourcePositionEvaluationError("Repeated observation IDs in DMS data")
    if not np.isfinite(source.score_analysis.to_numpy(dtype=float)).all():
        raise SourcePositionEvaluationError("Non-finite source DMS scores")
    if not source.wild_type.isin(list(AMINO_ACIDS)).all():
        raise SourcePositionEvaluationError("Unexpected source wild-type residue")
    if not source.mutant.isin(list(AMINO_ACIDS)).all():
        raise SourcePositionEvaluationError("Unexpected source mutant residue")
    return pairs, source, {MATCHED: digest(paired_file), SOURCE: digest(source_file)}


def prepare_direction(
    paired: pd.DataFrame,
    dms: pd.DataFrame,
    direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = paired.loc[paired.direction.eq(direction)].copy()
    if len(rows) != 1513 or rows.homologue_position_id.nunique() != 81:
        raise SourcePositionEvaluationError("Invalid paired substitution population")
    if rows.source_observation_id.duplicated().any():
        raise SourcePositionEvaluationError("Repeated paired source observation")
    if rows.target_observation_id.duplicated().any():
        raise SourcePositionEvaluationError("Repeated paired target observation")
    identities = rows[
        [
            "source_observation_id",
            "homologue_position_id",
            "wild_type",
            "mutant",
            "source_effect",
        ]
    ].merge(
        dms,
        left_on="source_observation_id",
        right_on="observation_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_paired", "_dms"),
    )
    if not identities._merge.eq("both").all():
        raise SourcePositionEvaluationError("Missing paired source observations")
    if not identities.wild_type_paired.eq(identities.wild_type_dms).all():
        raise SourcePositionEvaluationError("Paired wild-type residue differs")
    if not identities.mutant_paired.eq(identities.mutant_dms).all():
        raise SourcePositionEvaluationError("Paired mutant residue differs")
    if not np.allclose(
        identities.source_effect, identities.score_analysis, rtol=0, atol=1e-10
    ):
        raise SourcePositionEvaluationError("Paired source effects differ from DMS")
    assay_ids = identities.assay_id.dropna().unique()
    if len(assay_ids) != 1:
        raise SourcePositionEvaluationError(
            "Paired source observations must identify exactly one assay"
        )
    assay = str(assay_ids[0])
    # An assay identifier may cover multiple antibiotics or concentrations.
    # Paired source observations determine the experimental condition to fit.
    fields = ("protein_id", "antibiotic", "concentration", "temperature_c")
    if identities[list(fields)].isna().any().any():
        raise SourcePositionEvaluationError(
            "Paired source observations have missing experimental conditions"
        )
    for field in fields:
        if identities[field].nunique(dropna=False) != 1:
            raise SourcePositionEvaluationError(
                f"Paired source observations span multiple {field} values"
            )
    expected_protein = identities.protein_id.iloc[0]
    expected_drug = identities.antibiotic.iloc[0]
    expected_concentration = float(identities.concentration.iloc[0])
    expected_temperature = float(identities.temperature_c.iloc[0])
    if expected_drug != "AMP":
        raise SourcePositionEvaluationError(
            "The matched homologue fitting task requires ampicillin"
        )
    expected_concentration_by_direction = {
        "NDM-1_to_VIM-2": 256.0,
        "VIM-2_to_NDM-1": 128.0,
    }
    if not np.isclose(
        expected_concentration,
        expected_concentration_by_direction[direction],
        atol=1e-9,
        rtol=0,
    ) or not np.isclose(expected_temperature, 37.0, atol=1e-9, rtol=0):
        raise SourcePositionEvaluationError(
            "Source ampicillin concentration or temperature differs from the "
            "verified cross-protein comparison"
        )
    assay_all = dms.loc[dms.assay_id.astype(str).eq(assay)].copy()
    if assay_all.empty:
        raise SourcePositionEvaluationError("Source assay has no measurements")
    match = (
        assay_all.protein_id.eq(expected_protein)
        & assay_all.antibiotic.eq(expected_drug)
        & np.isclose(
            pd.to_numeric(assay_all.concentration, errors="coerce"),
            expected_concentration,
            atol=1e-9,
            rtol=0,
        )
        & np.isclose(
            pd.to_numeric(assay_all.temperature_c, errors="coerce"),
            expected_temperature,
            atol=1e-9,
            rtol=0,
        )
    )
    assay_dms = assay_all.loc[match].copy()
    if assay_dms.empty:
        raise SourcePositionEvaluationError(
            "No source observations match the paired ampicillin condition"
        )
    if not set(rows.source_observation_id).issubset(set(assay_dms.observation_id)):
        raise SourcePositionEvaluationError(
            "Paired source observations are absent from the selected condition"
        )
    if assay_dms.duplicated(["assay_position", "wild_type", "mutant"]).any():
        raise SourcePositionEvaluationError(
            "Source condition has repeated position and substitution records"
        )
    position_map = identities.groupby("homologue_position_id").assay_position.nunique()
    if not position_map.eq(1).all():
        raise SourcePositionEvaluationError("Ambiguous paired source position")
    result = identities[["homologue_position_id", "assay_position"]].drop_duplicates()
    rows = rows.merge(result, on="homologue_position_id", validate="many_to_one")
    # Test source residue IDs and target outcomes are NEVER passed to the fit.
    return rows, assay_dms, assay


def matrix(frame: pd.DataFrame) -> np.ndarray:
    wild = frame.wild_type.to_numpy(str)
    mutant = frame.mutant.to_numpy(str)
    return np.column_stack(
        [np.ones(len(frame), dtype=float)]
        + [(wild == aa).astype(float) for aa in AMINO_ACIDS]
        + [(mutant == aa).astype(float) for aa in AMINO_ACIDS]
    )


def ridge_fit(train: pd.DataFrame, alpha: float) -> np.ndarray:
    # A regularized least-squares solve avoids explicitly squaring X's
    # condition number through the normal equations. The intercept is free.
    from scipy.linalg import lstsq

    if not np.isfinite(alpha) or alpha <= 0:
        raise SourcePositionEvaluationError("Ridge penalty must be positive")
    x = matrix(train)
    y = train.score_analysis.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise SourcePositionEvaluationError("Non-finite Ridge fitting inputs")
    penalty = np.eye(x.shape[1], dtype=float) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    coefficients, _, _, _ = lstsq(
        np.vstack((x, penalty)),
        np.concatenate((y, np.zeros(x.shape[1], dtype=float))),
        lapack_driver="gelsy",
        check_finite=True,
    )
    if not np.isfinite(coefficients).all():
        raise SourcePositionEvaluationError("Ridge fitting returned non-finite values")
    return coefficients


def alpha_selection(
    train: pd.DataFrame, alphas: tuple[float, ...] = RIDGE_ALPHAS
) -> tuple[float, list[dict]]:
    # Inner folds group SOURCE residue positions, not source mutation rows.
    sites = sorted(set(train.assay_position.astype(str)))
    if len(sites) < 6:
        raise SourcePositionEvaluationError("Too few source positions for inner CV")
    fold_map = {
        site: int(hashlib.sha256(f"source_inner_cv|{site}".encode()).hexdigest(), 16)
        % 3
        for site in sites
    }
    fold_role = train.assay_position.astype(str).map(fold_map)
    errors = []
    for alpha in alphas:
        total_abs = 0.0
        total_n = 0
        for fold in range(3):
            fitted = train.loc[fold_role.ne(fold)]
            validated = train.loc[fold_role.eq(fold)]
            if fitted.empty or validated.empty:
                raise SourcePositionEvaluationError("Empty source inner-CV fold")
            beta = ridge_fit(fitted, alpha)
            y = validated.score_analysis.to_numpy(float)
            if not np.isfinite(beta).all():
                raise SourcePositionEvaluationError(
                    "Non-finite identity Ridge source CV coefficients"
                )
            estimate = np.einsum("ij,j->i", matrix(validated), beta, optimize=False)
            if not np.isfinite(estimate).all():
                raise SourcePositionEvaluationError(
                    "Non-finite identity Ridge source CV predictions"
                )
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    fold_error = float(np.abs(estimate - y).sum())
            except FloatingPointError as exc:
                raise SourcePositionEvaluationError(
                    "Non-finite identity Ridge source CV error"
                ) from exc
            if not np.isfinite(fold_error):
                raise SourcePositionEvaluationError(
                    "Non-finite identity Ridge source CV error"
                )
            total_abs += fold_error
            total_n += len(y)
        mean_error = total_abs / total_n
        if not np.isfinite(mean_error):
            raise SourcePositionEvaluationError("Non-finite identity Ridge CV MAE")
        errors.append(
            {
                "alpha": alpha,
                "source_cv_mae": mean_error,
                "source_cv_rows": total_n,
            }
        )
    return min(errors, key=lambda item: (item["source_cv_mae"], item["alpha"]))[
        "alpha"
    ], errors


def predict_reference(
    train: pd.DataFrame, test: pd.DataFrame, alpha: float
) -> dict[str, np.ndarray]:
    if train.empty or test.empty:
        raise SourcePositionEvaluationError("Empty source fit or target test")
    if set(train.assay_position).intersection(set(test.assay_position)):
        raise SourcePositionEvaluationError("Held-out source position entered fitting")
    if not np.isfinite(train.score_analysis.to_numpy(float)).all():
        raise SourcePositionEvaluationError("Non-finite source Ridge scores")
    global_mean = float(train.score_analysis.mean())
    mutants = train.groupby("mutant").score_analysis.agg(["sum", "count"])
    mutant_mean = {
        aa: (float(r["sum"]) + SHRINKAGE * global_mean) / (r["count"] + SHRINKAGE)
        for aa, r in mutants.iterrows()
    }
    substitutions = train.groupby(["wild_type", "mutant"]).score_analysis.agg(
        ["sum", "count"]
    )
    predicted_mutant = np.asarray(
        [mutant_mean.get(aa, global_mean) for aa in test.mutant]
    )
    predicted_substitution = np.asarray(
        [
            (
                float(substitutions.loc[(w, m), "sum"])
                + SHRINKAGE * mutant_mean.get(m, global_mean)
            )
            / (float(substitutions.loc[(w, m), "count"]) + SHRINKAGE)
            if (w, m) in substitutions.index
            else mutant_mean.get(m, global_mean)
            for w, m in zip(test.wild_type, test.mutant)
        ]
    )
    beta = ridge_fit(train, alpha)
    if not np.isfinite(beta).all():
        raise SourcePositionEvaluationError("Non-finite source reference prediction")
    ridge_prediction = np.einsum("ij,j->i", matrix(test), beta, optimize=False)
    predictions = dict(
        zip(
            MODELS,
            (
                np.full(len(test), global_mean),
                predicted_mutant,
                predicted_substitution,
                ridge_prediction,
            ),
        )
    )
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise SourcePositionEvaluationError("Non-finite source reference prediction")
    return predictions


def run_prediction(
    paired: pd.DataFrame, dms: pd.DataFrame, repetitions: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if repetitions < 100:
        raise SourcePositionEvaluationError("At least 100 bootstrap samples")
    results = []
    folds = []
    cv_rows = []
    populations = []
    for direction in DIRECTIONS:
        target, full_source, assay_id = prepare_direction(paired, dms, direction)
        fold_map = fold_for_positions(target.homologue_position_id)
        target["heldout_fold"] = target.homologue_position_id.astype(str).map(fold_map)
        for fold in range(5):
            heldout = target.loc[target.heldout_fold.eq(fold)]
            excluded = set(heldout.assay_position)
            # Source assay measurements from all other positions are available.
            fit = full_source.loc[~full_source.assay_position.isin(excluded)].copy()
            if heldout.empty or fit.empty:
                raise SourcePositionEvaluationError("Empty source or held-out fold")
            assert not set(fit.assay_position).intersection(excluded)
            alpha, candidates = alpha_selection(fit)
            for item in candidates:
                cv_rows.append(
                    {
                        "direction": direction,
                        "heldout_fold": fold,
                        "source_assay_id": assay_id,
                        **item,
                        "selected_alpha": alpha,
                    }
                )
            for position in sorted(set(heldout.homologue_position_id.astype(str))):
                folds.append(
                    {
                        "direction": direction,
                        "homologue_position_id": position,
                        "heldout_fold": fold,
                        "source_assay_id": assay_id,
                        "source_fit_rows": len(fit),
                        "source_fit_positions": fit.assay_position.nunique(),
                        "excluded_assay_position_count": len(excluded),
                    }
                )
            # Target observations are only appended after fitting is complete.
            test_inputs = heldout[["wild_type", "mutant", "assay_position"]].copy()
            predictions = predict_reference(fit, test_inputs, alpha)
            for model, scores in predictions.items():
                record = heldout[
                    [
                        "direction",
                        "homologue_position_id",
                        "source_observation_id",
                        "target_observation_id",
                        "target_effect",
                        "minimum_score_class",
                        "heldout_fold",
                    ]
                ].copy()
                record["model_id"] = model
                record["y_pred"] = scores
                results.append(record)
        populations.append(
            {
                "direction": direction,
                "source_assay_id": assay_id,
                "source_assay_rows": len(full_source),
                "source_assay_positions": (full_source.assay_position.nunique()),
            }
        )
    pred = pd.concat(results, ignore_index=True)
    if len(pred) != 2 * 1513 * len(MODELS):
        raise SourcePositionEvaluationError("Incomplete source-full predictions")
    if pred.duplicated(["direction", "target_observation_id", "model_id"]).any():
        raise SourcePositionEvaluationError("Duplicate target predictions")
    if not np.isfinite(pred.y_pred.to_numpy(float)).all():
        raise SourcePositionEvaluationError("Non-finite predicted effects")
    return pred, pd.DataFrame(folds), pd.DataFrame(cv_rows), pd.DataFrame(populations)


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def score(pred: pd.DataFrame, repetitions: int = 1000) -> pd.DataFrame:
    output = []
    for direction in DIRECTIONS:
        for population in ("all_shared_wild_type_pairs", "both_above_minimum"):
            p = pred.loc[pred.direction.eq(direction)]
            if population == "both_above_minimum":
                p = p.loc[p.minimum_score_class.eq("both_above_minimum")]
            position_ids = sorted(p.homologue_position_id.astype(str).unique())
            ref = p.loc[p.model_id.eq("source_assay_mean")]
            baseline = ref.set_index("target_observation_id").y_pred
            for model in MODELS:
                frame = p.loc[p.model_id.eq(model)].copy()
                y = frame.target_effect.to_numpy(float)
                predicted = frame.y_pred.to_numpy(float)
                baseline_pred = frame.target_observation_id.map(baseline).to_numpy(
                    float
                )
                if not np.isfinite(baseline_pred).all():
                    raise SourcePositionEvaluationError("Missing reference prediction")
                mean_improvement = float(
                    np.mean(np.abs(y - baseline_pred) - np.abs(y - predicted))
                )
                by_pos = frame.homologue_position_id.astype(str).to_numpy()
                groups = [
                    np.flatnonzero(by_pos == position) for position in position_ids
                ]
                seed = int(
                    hashlib.sha256(
                        f"{direction}|{population}|{model}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                rng = np.random.default_rng(seed)
                draws = []
                for _ in range(repetitions):
                    ids = np.concatenate(
                        [
                            groups[k]
                            for k in rng.integers(0, len(groups), size=len(groups))
                        ]
                    )
                    draws.append(
                        float(
                            np.mean(
                                np.abs(y[ids] - baseline_pred[ids])
                                - np.abs(y[ids] - predicted[ids])
                            )
                        )
                    )
                low, high = np.quantile(draws, [0.025, 0.975])
                output.append(
                    {
                        "direction": direction,
                        "population": population,
                        "model_id": model,
                        "n_pairs": len(frame),
                        "n_positions": len(groups),
                        "mae": float(np.abs(y - predicted).mean()),
                        "spearman": _spearman(y, predicted),
                        "mae_improvement_over_source_assay_mean": mean_improvement,
                        "mae_improvement_ci_lower": float(low),
                        "mae_improvement_ci_upper": float(high),
                        "bootstrap_replicates": repetitions,
                    }
                )
    return pd.DataFrame(output)


def execute(root: Path, out: Path, repetitions: int = 1000) -> dict:
    pairs, source, hashes = load_data(root)
    predictions, folds, cv, population = run_prediction(pairs, source, repetitions)
    metrics = score(predictions, repetitions)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("source_full_position_predictions.csv", predictions),
        ("source_full_position_metrics.csv", metrics),
        ("source_full_position_fold_assignments.csv", folds),
        ("source_full_position_model_selection.csv", cv),
        ("source_full_position_assays.csv", population),
    ):
        frame.to_csv(out / name, index=False, lineterminator="\n")
    summary = {
        "source_input_hashes": hashes,
        "fit_scope": "all observations from the selected source ampicillin assay",
        "excluded_source_positions": (
            "all observations at held-out homologous positions"
        ),
        "target_measurements_used_in_fitting": False,
        "ridge_alpha_selected_with": "source-only three-fold position CV",
        "same_position_folds_as_prior_matched_only_experiment": True,
        "comparable_to_previous_full_source_neural_results": False,
        "model_count": len(MODELS),
        "prediction_count": len(predictions),
        "metric_count": len(metrics),
        "source_population": population.to_dict(orient="records"),
    }
    (out / "source_full_position_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
