"""Source-protein reference estimates with homologous positions withheld.

Fitting uses source mutation-effect observations from other homologous positions.
Target experimental values are used only after every prediction is computed.
The experiment uses the 81 shared-wild-type matched positions, not all source DMS rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAIRS_FILE = (
    "data/interim/baseline_evaluation/homologue_analysis/"
    "matched_homologue_predictions.csv"
)
DIRECTIONS = ("NDM-1_to_VIM-2", "VIM-2_to_NDM-1")
MODEL_NAMES = (
    "source_global_mean",
    "source_mutant_mean_shrunk",
    "source_substitution_mean_shrunk",
)
FOLD_COUNT = 5
SHRINKAGE_COUNT = 5.0
SEED = "beta_lactamase_position_withheld_source_reference_v1"


class PositionWithheldEvaluationError(ValueError):
    """The matched substitution population or fitting separation is invalid."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(frame: pd.DataFrame, names: set[str]) -> None:
    missing = names.difference(frame.columns)
    if missing:
        raise PositionWithheldEvaluationError(
            f"Missing source columns: {sorted(missing)}"
        )


def load_matched_data(repo_root: Path) -> tuple[pd.DataFrame, str]:
    path = repo_root / PAIRS_FILE
    if not path.is_file():
        raise PositionWithheldEvaluationError(f"Missing matching table: {path}")
    frame = pd.read_csv(
        path,
        dtype={"source_observation_id": str, "target_observation_id": str},
        low_memory=False,
    )
    required = {
        "direction",
        "homologue_position_id",
        "wild_type",
        "mutant",
        "source_observation_id",
        "target_observation_id",
        "source_effect",
        "target_effect",
        "minimum_score_class",
        "predicted_target_effect",
    }
    _required(frame, required)
    if len(frame) != 3026 or set(frame.direction) != set(DIRECTIONS):
        raise PositionWithheldEvaluationError(
            "Expected 1,513 matched pairs per direction"
        )
    for direction in DIRECTIONS:
        rows = frame.loc[frame.direction.eq(direction)]
        if len(rows) != 1513 or rows.homologue_position_id.nunique() != 81:
            raise PositionWithheldEvaluationError(f"Invalid matching for {direction}")
        if rows["target_observation_id"].duplicated().any():
            raise PositionWithheldEvaluationError(f"Repeated targets: {direction}")
        if rows["source_observation_id"].duplicated().any():
            raise PositionWithheldEvaluationError(f"Repeated sources: {direction}")
        if not np.allclose(
            rows.source_effect.to_numpy(float),
            rows.predicted_target_effect.to_numpy(float),
            rtol=0,
            atol=1e-12,
        ):
            raise PositionWithheldEvaluationError(
                "Direct reference differs from source"
            )
    if not np.isfinite(frame[["source_effect", "target_effect"]].to_numpy(float)).all():
        raise PositionWithheldEvaluationError("Experimental effects must be finite")
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if not frame.wild_type.isin(allowed).all() or not frame.mutant.isin(allowed).all():
        raise PositionWithheldEvaluationError("Unexpected residue identities")
    return frame, sha256(path)


def fold_for_positions(
    positions: pd.Series, *, folds: int = FOLD_COUNT
) -> dict[str, int]:
    if folds < 2:
        raise PositionWithheldEvaluationError("At least two held-out folds required")
    items = sorted(set(positions.astype(str)))
    if len(items) < folds:
        raise PositionWithheldEvaluationError("Too few positions for grouped folds")
    ordered = sorted(
        items,
        key=lambda item: (hashlib.sha256(f"{SEED}|{item}".encode()).hexdigest(), item),
    )
    return {item: index % folds for index, item in enumerate(ordered)}


def _fit_reference_predictions(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Fitting deliberately does not access target_effect or test source_effect."""
    if train.empty or test.empty:
        raise PositionWithheldEvaluationError("Empty fitting or held-out set")
    if set(train.homologue_position_id).intersection(test.homologue_position_id):
        raise PositionWithheldEvaluationError("Held-out position used for fitting")
    global_mean = float(train.source_effect.mean())
    mutant = train.groupby("mutant")["source_effect"].agg(["sum", "count"])
    mutant_prior = {
        name: (float(row["sum"]) + SHRINKAGE_COUNT * global_mean)
        / (float(row["count"]) + SHRINKAGE_COUNT)
        for name, row in mutant.iterrows()
    }
    substitution = train.groupby(["wild_type", "mutant"])["source_effect"].agg(
        ["sum", "count"]
    )
    source_mutant_mean = np.array(
        [mutant_prior.get(residue, global_mean) for residue in test.mutant],
        dtype=float,
    )
    source_substitution_mean = np.array(
        [
            (
                float(substitution.loc[(wild_type, mutant), "sum"])
                + SHRINKAGE_COUNT * mutant_prior.get(mutant, global_mean)
            )
            / (float(substitution.loc[(wild_type, mutant), "count"]) + SHRINKAGE_COUNT)
            if (wild_type, mutant) in substitution.index
            else mutant_prior.get(mutant, global_mean)
            for wild_type, mutant in zip(test.wild_type, test.mutant)
        ],
        dtype=float,
    )
    return {
        MODEL_NAMES[0]: np.full(len(test), global_mean, dtype=float),
        MODEL_NAMES[1]: source_mutant_mean,
        MODEL_NAMES[2]: source_substitution_mean,
    }


def calculate_heldout_predictions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute estimates and folds without target-based model choice."""
    _required(
        frame,
        {
            "direction",
            "homologue_position_id",
            "wild_type",
            "mutant",
            "source_observation_id",
            "target_observation_id",
            "source_effect",
            "target_effect",
            "minimum_score_class",
        },
    )
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        direction_rows = frame.loc[frame.direction.eq(direction)].copy()
        if direction_rows.empty:
            raise PositionWithheldEvaluationError(f"Missing direction: {direction}")
        assignments = fold_for_positions(direction_rows.homologue_position_id)
        direction_rows["heldout_fold"] = direction_rows.homologue_position_id.astype(
            str
        ).map(assignments)
        for fold in range(FOLD_COUNT):
            train = direction_rows.loc[direction_rows.heldout_fold.ne(fold)]
            test = direction_rows.loc[direction_rows.heldout_fold.eq(fold)]
            if test.empty or train.empty:
                raise PositionWithheldEvaluationError(
                    "Empty source or test position fold"
                )
            if set(train.source_observation_id).intersection(
                test.source_observation_id
            ):
                raise PositionWithheldEvaluationError(
                    "Source observation in both roles"
                )
            for position in sorted(set(test.homologue_position_id.astype(str))):
                folds.append(
                    {
                        "direction": direction,
                        "homologue_position_id": position,
                        "heldout_fold": fold,
                        "fit_position_count": train.homologue_position_id.nunique(),
                        "fit_source_observation_count": len(train),
                    }
                )
            predictions = _fit_reference_predictions(train, test)
            for model_id, values in predictions.items():
                result = test[
                    [
                        "direction",
                        "homologue_position_id",
                        "wild_type",
                        "mutant",
                        "source_observation_id",
                        "target_observation_id",
                        "target_effect",
                        "minimum_score_class",
                        "heldout_fold",
                    ]
                ].copy()
                result["model_id"] = model_id
                result["y_pred"] = values
                rows.append(result)
    result = pd.concat(rows, ignore_index=True)
    expected = len(frame) * len(MODEL_NAMES)
    if len(result) != expected or not np.isfinite(result.y_pred.to_numpy(float)).all():
        raise PositionWithheldEvaluationError("Incomplete fitted baseline predictions")
    keys = ["direction", "source_observation_id", "target_observation_id", "model_id"]
    if result.duplicated(keys).any():
        raise PositionWithheldEvaluationError("Repeated held-out prediction keys")
    result = result.sort_values(keys).reset_index(drop=True)
    return result, pd.DataFrame(folds).sort_values(
        ["direction", "homologue_position_id"]
    ).reset_index(drop=True)


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def score_predictions(frame: pd.DataFrame, *, repetitions: int = 1000) -> pd.DataFrame:
    """Position-bootstrap MAE difference against source global-mean prediction."""
    if repetitions < 100:
        raise PositionWithheldEvaluationError("At least 100 resamples are required")
    results: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        for population in ("all_shared_wild_type_pairs", "both_above_minimum"):
            subset = frame.loc[frame.direction.eq(direction)]
            if population == "both_above_minimum":
                subset = subset.loc[subset.minimum_score_class.eq("both_above_minimum")]
            n_pairs = subset.target_observation_id.nunique()
            n_pos = subset.homologue_position_id.nunique()
            if (n_pairs, n_pos) != (
                (1513, 81) if population == "all_shared_wild_type_pairs" else (478, 63)
            ):
                raise PositionWithheldEvaluationError(
                    "Unexpected evaluation population"
                )
            reference = subset.loc[
                subset.model_id.eq("source_global_mean"),
                ["target_observation_id", "target_effect", "y_pred"],
            ].rename(columns={"y_pred": "reference_prediction"})
            for model_id, model in subset.groupby("model_id", sort=True):
                merged = model.merge(
                    reference[["target_observation_id", "reference_prediction"]],
                    on="target_observation_id",
                    how="left",
                    validate="one_to_one",
                )
                if len(merged) != n_pairs or merged.reference_prediction.isna().any():
                    raise PositionWithheldEvaluationError(
                        "Missing reference predictions"
                    )
                y = merged.target_effect.to_numpy(float)
                p = merged.y_pred.to_numpy(float)
                baseline = merged.reference_prediction.to_numpy(float)
                improvement = np.abs(baseline - y) - np.abs(p - y)
                groups = list(
                    merged.groupby("homologue_position_id", sort=True).indices.values()
                )
                if len(groups) != n_pos:
                    raise PositionWithheldEvaluationError("Position grouping mismatch")
                rng = np.random.default_rng(21026)
                simulations = np.empty(repetitions, dtype=float)
                for k in range(repetitions):
                    sample = rng.integers(0, len(groups), size=len(groups))
                    indices = np.concatenate([groups[j] for j in sample])
                    simulations[k] = float(improvement[indices].mean())
                results.append(
                    {
                        "direction": direction,
                        "population": population,
                        "model_id": model_id,
                        "n_pairs": n_pairs,
                        "n_homologue_positions": n_pos,
                        "spearman": _spearman(y, p),
                        "mae": float(np.abs(p - y).mean()),
                        "source_global_mean_mae": float(np.abs(baseline - y).mean()),
                        "mae_improvement_vs_source_global_mean": float(
                            improvement.mean()
                        ),
                        "mae_improvement_ci_lower": float(
                            np.quantile(simulations, 0.025)
                        ),
                        "mae_improvement_ci_upper": float(
                            np.quantile(simulations, 0.975)
                        ),
                        "position_bootstrap_repetitions": repetitions,
                    }
                )
    return (
        pd.DataFrame(results)
        .sort_values(["direction", "population", "model_id"])
        .reset_index(drop=True)
    )


def run_analysis(repo_root: Path, output_dir: Path, *, repetitions: int = 1000) -> dict:
    if output_dir.resolve().is_relative_to(repo_root.resolve()):
        allowed_output = (
            repo_root.resolve()
            / "data"
            / "interim"
            / "position_withheld_homologue_baselines"
        )
        if output_dir.resolve() != allowed_output:
            raise PositionWithheldEvaluationError(
                "Repository output must be saved in "
                "data/interim/position_withheld_homologue_baselines"
            )
    if output_dir.exists():
        raise PositionWithheldEvaluationError("Output directory already exists")
    matched, source_hash = load_matched_data(repo_root)
    predictions, folds = calculate_heldout_predictions(matched)
    metrics = score_predictions(predictions, repetitions=repetitions)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "position_withheld_predictions.csv", index=False)
    folds.to_csv(output_dir / "position_withheld_fold_assignments.csv", index=False)
    metrics.to_csv(output_dir / "position_withheld_metrics.csv", index=False)
    summary = {
        "input_sha256": {PAIRS_FILE: source_hash},
        "method": "source_position_heldout_five_fold_cv",
        "target_measurements_used_for_model_fitting": False,
        "target_measurements_used_for_setting_selection": False,
        "heldout_equivalent_source_effect_used_for_prediction": False,
        "model_settings_selected_on_target": False,
        "source_observations_only_from_matched_81_positions": True,
        "not_directly_comparable_to_full_source_trained_neural_models": True,
        "source_substitution_shrinkage_count": SHRINKAGE_COUNT,
        "source_mutant_shrinkage_count": SHRINKAGE_COUNT,
        "group_count": 81,
        "fold_count": FOLD_COUNT,
        "prediction_count": len(predictions),
        "comparison_count": len(metrics),
        "notes": (
            "Prediction of the held-out target substitution uses source-protein "
            "effects at other homologous positions. No source effect at the held-out "
            "homologous position enters fitting. The direct experimental transfer "
            "estimate is unavailable for these prediction decisions."
        ),
    }
    (output_dir / "position_withheld_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
