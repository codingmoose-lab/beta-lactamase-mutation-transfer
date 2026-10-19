"""Paired mutation-effect prediction comparisons for NDM-1 and VIM-2.

Existing fitted predictions are read without fitting or model selection. The
experimental source-substitution estimate is a separate reference method.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIRECTIONS = ("NDM-1_to_VIM-2", "VIM-2_to_NDM-1")
MODELS = (
    "sequence_residue_graph_neural_regressor",
    "residue_graph_neural_regressor",
)
POPULATIONS = ("all_shared_wild_type_pairs", "both_above_minimum")

BASELINE_PATH = (
    "data/interim/baseline_evaluation/homologue_analysis/"
    "matched_homologue_predictions.csv"
)
MODEL_PATH = (
    "data/interim/matched_homologue_model_analysis/"
    "matched_homologue_model_predictions.csv"
)


class HomologuePredictionDiagnosticError(ValueError):
    """Required observations or paired prediction values are inconsistent."""


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise HomologuePredictionDiagnosticError(
            f"{label} is missing columns: {missing}"
        )


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise HomologuePredictionDiagnosticError("Correlation array shapes differ")
    if first.ndim != 1 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise HomologuePredictionDiagnosticError(
            "Correlation requires finite 1D values"
        )
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan")
    return float(spearmanr(first, second).statistic)


def load_predictions(repo_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    paths = (repo_root / BASELINE_PATH, repo_root / MODEL_PATH)
    for path in paths:
        if not path.is_file():
            raise HomologuePredictionDiagnosticError(f"Missing input: {path}")
    hashes = {str(p.relative_to(repo_root)): _digest(p) for p in paths}
    baseline = pd.read_csv(
        paths[0],
        dtype={"source_observation_id": str, "target_observation_id": str},
        low_memory=False,
    )
    predictions = pd.read_csv(
        paths[1],
        dtype={"source_observation_id": str, "target_observation_id": str},
        low_memory=False,
    )
    fields = (
        "direction",
        "source_observation_id",
        "target_observation_id",
        "homologue_position_id",
        "source_effect",
        "target_effect",
        "minimum_score_class",
        "predicted_target_effect",
    )
    _require_columns(baseline, fields, "Matched experimental observations")
    _require_columns(
        predictions,
        (
            "direction",
            "source_observation_id",
            "target_observation_id",
            "homologue_position_id",
            "model_id",
            "source_effect",
            "target_effect",
            "minimum_score_class",
            "y_pred",
        ),
        "Matched model predictions",
    )
    if len(baseline) != 3026 or set(baseline.direction) != set(DIRECTIONS):
        raise HomologuePredictionDiagnosticError(
            "Expected 1,513 matched experimental observations in each direction"
        )
    key = ["direction", "source_observation_id", "target_observation_id"]
    if baseline.duplicated(key).any():
        raise HomologuePredictionDiagnosticError("Repeated experimental pair keys")
    if not np.allclose(
        baseline["predicted_target_effect"].to_numpy(dtype=float),
        baseline["source_effect"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise HomologuePredictionDiagnosticError(
            "Direct reference differs from the measured source substitution"
        )
    for direction in DIRECTIONS:
        b = baseline.loc[baseline.direction.eq(direction)]
        if len(b) != 1513 or b.homologue_position_id.nunique() != 81:
            raise HomologuePredictionDiagnosticError(
                f"Unexpected pair/position count for {direction}"
            )
    selected = predictions.loc[predictions.model_id.isin(MODELS)].copy()
    if selected.duplicated(key + ["model_id"]).any():
        raise HomologuePredictionDiagnosticError("Repeated model prediction keys")
    expected_count = len(baseline) * len(MODELS)
    if len(selected) != expected_count:
        raise HomologuePredictionDiagnosticError(
            f"Expected {expected_count} model prediction rows; got {len(selected)}"
        )
    merged = selected.merge(
        baseline[
            key
            + [
                "homologue_position_id",
                "source_effect",
                "target_effect",
                "minimum_score_class",
            ]
        ],
        on=key,
        how="left",
        validate="many_to_one",
        suffixes=("", "_observed"),
        indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise HomologuePredictionDiagnosticError(
            "A model prediction does not have a matched experimental observation"
        )
    for field in ("source_effect", "target_effect"):
        x = merged[field].to_numpy(dtype=float)
        y = merged[field + "_observed"].to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise HomologuePredictionDiagnosticError(f"Non-finite {field}")
        if not np.allclose(x, y, rtol=0, atol=1e-10):
            raise HomologuePredictionDiagnosticError(
                f"{field} differs between model and measured observations"
            )
    if (
        not merged.homologue_position_id.astype(str)
        .eq(merged.homologue_position_id_observed.astype(str))
        .all()
    ):
        raise HomologuePredictionDiagnosticError(
            "Homologous position identifiers differ"
        )
    if not merged.minimum_score_class.eq(merged.minimum_score_class_observed).all():
        raise HomologuePredictionDiagnosticError("Lower-score categories differ")
    if not np.isfinite(merged.y_pred.to_numpy(dtype=float)).all():
        raise HomologuePredictionDiagnosticError("Non-finite model predictions")
    fields_out = key + [
        "homologue_position_id",
        "minimum_score_class",
        "source_effect",
        "target_effect",
        "model_id",
        "y_pred",
    ]
    return merged[fields_out].sort_values(key + ["model_id"]).reset_index(
        drop=True
    ), hashes


def _indices_by_position(frame: pd.DataFrame) -> list[np.ndarray]:
    groups = frame.groupby("homologue_position_id", sort=True).indices
    return [np.asarray(groups[group], dtype=int) for group in sorted(groups, key=str)]


def diagnostic_statistics(
    frame: pd.DataFrame, *, repetitions: int = 1000, seed: int = 31027
) -> dict[str, float | int | str]:
    """Compare model and direct errors; describe the shared-target dependence.

    Randomizing targets within homologous positions is descriptive only. Both
    error quantities still use the same target value after randomization.
    """
    _require_columns(
        frame,
        (
            "homologue_position_id",
            "source_effect",
            "target_effect",
            "y_pred",
        ),
        "Paired prediction population",
    )
    if repetitions < 20:
        raise HomologuePredictionDiagnosticError(
            "At least 20 within-position permutations are required"
        )
    s = frame.source_effect.to_numpy(dtype=float)
    y = frame.target_effect.to_numpy(dtype=float)
    p = frame.y_pred.to_numpy(dtype=float)
    if not np.isfinite(np.concatenate((s, y, p))).all():
        raise HomologuePredictionDiagnosticError(
            "Non-finite measured or predicted scores"
        )
    groups = _indices_by_position(frame)
    if len(groups) < 2:
        raise HomologuePredictionDiagnosticError(
            "At least two homologous positions required"
        )
    direct_error = np.abs(s - y)
    model_error = np.abs(p - y)
    experimental_difference = direct_error.copy()
    rho = rank_correlation(experimental_difference, model_error)
    rng = np.random.default_rng(seed)
    null_rhos = []
    for _ in range(repetitions):
        randomized_y = y.copy()
        for positions in groups:
            if len(positions) > 1:
                randomized_y[positions] = y[rng.permutation(positions)]
        randomized_source_error = np.abs(s - randomized_y)
        randomized_model_error = np.abs(p - randomized_y)
        value = rank_correlation(randomized_source_error, randomized_model_error)
        if np.isfinite(value):
            null_rhos.append(value)
    null = np.asarray(null_rhos, dtype=float)
    if len(null) < 20:
        raise HomologuePredictionDiagnosticError(
            "Insufficient finite within-position permutation correlations"
        )
    bound = np.abs(model_error - direct_error) - np.abs(p - s)
    if np.any(bound > 1e-9):
        raise HomologuePredictionDiagnosticError(
            "Absolute-error triangle inequality failed"
        )
    return {
        "n_pairs": len(frame),
        "n_homologue_positions": len(groups),
        "model_spearman": rank_correlation(y, p),
        "direct_spearman": rank_correlation(y, s),
        "model_mae": float(np.mean(model_error)),
        "direct_mae": float(np.mean(direct_error)),
        "mae_improvement_over_direct": float(np.mean(direct_error - model_error)),
        "experimental_difference_vs_model_error_spearman": rho,
        "within_position_null_lower": float(np.quantile(null, 0.025)),
        "within_position_null_median": float(np.median(null)),
        "within_position_null_upper": float(np.quantile(null, 0.975)),
        "null_correlations_calculated": len(null),
        "observed_fraction_below_null": float(np.mean(null < rho)),
        "median_model_to_source_score_difference": float(np.median(np.abs(p - s))),
        "triangle_inequality_violations": int(np.count_nonzero(bound > 1e-9)),
        "interpretation": (
            "Descriptive association only: measured disagreement and model "
            "error both contain the target measurement. The within-position "
            "permutation is not a biochemical null hypothesis test."
        ),
    }


def grouped_model_comparison(
    frame: pd.DataFrame, *, repetitions: int = 1000, seed: int = 10031
) -> dict[str, float | int]:
    """Resample homologous positions to estimate paired model differences."""
    if repetitions < 20:
        raise HomologuePredictionDiagnosticError(
            "At least 20 grouped bootstrap repetitions are required"
        )
    groups = _indices_by_position(frame)
    if len(groups) < 2:
        raise HomologuePredictionDiagnosticError("At least two positions required")
    y = frame.target_effect.to_numpy(dtype=float)
    s = frame.source_effect.to_numpy(dtype=float)
    p = frame.y_pred.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    mae_diffs, rho_diffs = [], []
    for _ in range(repetitions):
        selected = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[j] for j in selected])
        diff = np.abs(s[idx] - y[idx]) - np.abs(p[idx] - y[idx])
        mae_diffs.append(float(np.mean(diff)))
        rho_model = rank_correlation(y[idx], p[idx])
        rho_direct = rank_correlation(y[idx], s[idx])
        if np.isfinite(rho_model) and np.isfinite(rho_direct):
            rho_diffs.append(rho_model - rho_direct)
    if len(rho_diffs) < 20:
        raise HomologuePredictionDiagnosticError(
            "Insufficient finite grouped Spearman comparisons"
        )
    mae_ci = np.quantile(mae_diffs, [0.025, 0.975])
    rho_ci = np.quantile(rho_diffs, [0.025, 0.975])
    return {
        "mae_improvement_ci_lower": float(mae_ci[0]),
        "mae_improvement_ci_upper": float(mae_ci[1]),
        "spearman_difference": rank_correlation(y, p) - rank_correlation(y, s),
        "spearman_difference_ci_lower": float(rho_ci[0]),
        "spearman_difference_ci_upper": float(rho_ci[1]),
        "group_bootstrap_repetitions": repetitions,
        "valid_spearman_bootstrap_repetitions": len(rho_diffs),
    }


def analyze(
    predictions: pd.DataFrame, *, repetitions: int = 1000, seed: int = 20261016
) -> pd.DataFrame:
    results = []
    for direction in DIRECTIONS:
        for population in POPULATIONS:
            for model in MODELS:
                selection = predictions.loc[
                    predictions.direction.eq(direction) & predictions.model_id.eq(model)
                ]
                if population == "both_above_minimum":
                    selection = selection.loc[
                        selection.minimum_score_class.eq("both_above_minimum")
                    ]
                selection = selection.reset_index(drop=True)
                expected_n, expected_positions = (
                    (1513, 81)
                    if population == "all_shared_wild_type_pairs"
                    else (478, 63)
                )
                if (
                    len(selection) != expected_n
                    or selection.homologue_position_id.nunique() != expected_positions
                ):
                    raise HomologuePredictionDiagnosticError(
                        f"{direction}/{population}/{model}: unexpected population "
                        f"({len(selection)} observations and "
                        f"{selection.homologue_position_id.nunique()} positions)"
                    )
                digest = hashlib.sha256(
                    f"{seed}|{direction}|{population}|{model}".encode()
                ).digest()
                run_seed = int.from_bytes(digest[:4], "big")
                diagnostics = diagnostic_statistics(
                    selection, repetitions=repetitions, seed=run_seed
                )
                intervals = grouped_model_comparison(
                    selection, repetitions=repetitions, seed=run_seed + 1
                )
                results.append(
                    {
                        "direction": direction,
                        "population": population,
                        "model_id": model,
                        **diagnostics,
                        **intervals,
                    }
                )
    return pd.DataFrame(results)
