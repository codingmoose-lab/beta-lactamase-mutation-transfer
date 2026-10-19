from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

DIRECTIONS = ("NDM-1_to_VIM-2", "VIM-2_to_NDM-1")
REFERENCE_NAMES = {
    "source_assay_mean": "source assay mean",
    "source_assay_mutant_mean": "source mutant mean",
    "source_assay_substitution_mean": "source substitution mean",
}
STRATA = (
    "target_above_minimum",
    "source_above_minimum",
    "both_above_minimum",
    "discordant_minimum",
    "all_shared_wild_type_pairs",
)
COMPARATORS = ("source_assay_substitution_mean", "source_assay_identity_ridge")
KEY = ["direction", "target_observation_id"]


def population_mask(classes: pd.Series, population: str) -> np.ndarray:
    c = classes.astype(str)
    if population == "all_shared_wild_type_pairs":
        return np.ones(len(c), dtype=bool)
    if population == "both_above_minimum":
        return c.eq("both_above_minimum").to_numpy()
    if population == "target_above_minimum":
        return c.isin(("both_above_minimum", "source_at_minimum")).to_numpy()
    if population == "source_above_minimum":
        return c.isin(("both_above_minimum", "target_at_minimum")).to_numpy()
    if population == "discordant_minimum":
        return c.isin(("source_at_minimum", "target_at_minimum")).to_numpy()
    raise ValueError(f"Unknown substitution population: {population}")


def minimum_score_auc(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite numerical values in ROC AUC input")
    at_minimum = np.isclose(y, -4.0, rtol=0, atol=1e-12)
    if len(np.unique(at_minimum)) != 2:
        raise ValueError("ROC AUC needs both minimum and above-minimum observations")
    positive_count = int(at_minimum.sum())
    negative_count = len(at_minimum) - positive_count
    ranks = rankdata(-prediction, method="average")
    rank_sum = float(ranks[at_minimum].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def grouped_interval(
    improvement: np.ndarray,
    positions: np.ndarray,
    *,
    repetitions: int,
    seed_label: str,
) -> tuple[float, float]:
    if repetitions < 100:
        raise ValueError("At least 100 resamples required")
    values = np.asarray(improvement, dtype=float)
    groups = np.asarray(positions, dtype=str)
    if len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid paired errors or position identifiers")
    members = [np.flatnonzero(groups == group) for group in sorted(set(groups))]
    if any(len(part) == 0 for part in members):
        raise ValueError("Empty homologue-position group")
    seed = int(hashlib.sha256(seed_label.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = np.concatenate(
            [members[i] for i in rng.integers(len(members), size=len(members))]
        )
        draws.append(float(values[sampled].mean()))
    lo, hi = np.quantile(draws, (0.025, 0.975))
    return float(lo), float(hi)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _check_numeric(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for col in columns:
        if (
            not pd.api.types.is_numeric_dtype(frame[col])
            or not np.isfinite(frame[col].to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"Non-finite or nonnumerical {label}.{col}")


def combine_predictions(esmc: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    fields = {
        *KEY,
        "homologue_position_id",
        "target_effect",
        "minimum_score_class",
        "heldout_fold",
    }
    _require_columns(
        esmc,
        fields | {"model_prediction", "reference_prediction"},
        "ESM-C predictions",
    )
    _require_columns(full, fields | {"model_id", "y_pred"}, "source-full predictions")
    _check_numeric(
        esmc, ("target_effect", "model_prediction", "reference_prediction"), "ESM-C"
    )
    _check_numeric(full, ("target_effect", "y_pred"), "source-full")
    if esmc.duplicated(KEY).any() or full.duplicated([*KEY, "model_id"]).any():
        raise ValueError("Duplicate model/target observation records")
    if set(esmc.direction) != set(DIRECTIONS):
        raise ValueError("Incorrect transfer directions")
    if set(full.model_id) != {*REFERENCE_NAMES, "source_assay_identity_ridge"}:
        raise ValueError("Unexpected source-full model set")
    if set(full.direction) != set(DIRECTIONS):
        raise ValueError("Incorrect source-full transfer directions")
    classes = {
        "both_at_minimum",
        "source_at_minimum",
        "target_at_minimum",
        "both_above_minimum",
    }
    if not set(esmc.minimum_score_class).issubset(classes):
        raise ValueError("Unexpected minimum-score classification")

    for model in sorted(set(full.model_id)):
        subset = full.loc[full.model_id.eq(model)]
        merged = esmc.merge(
            subset,
            on=KEY,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_esmc", "_full"),
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"Target observation mismatch for {model}")
        if not np.array_equal(
            merged.target_effect_esmc.to_numpy(), merged.target_effect_full.to_numpy()
        ):
            raise ValueError(f"Target measurement difference for {model}")
        for column in ("homologue_position_id", "minimum_score_class", "heldout_fold"):
            if not merged[f"{column}_esmc"].equals(merged[f"{column}_full"]):
                raise ValueError(f"{column} mismatch for {model}")

    result = esmc.copy()
    for model in REFERENCE_NAMES:
        other = full.loc[full.model_id.eq(model), [*KEY, "y_pred"]]
        result = result.merge(
            other.rename(columns={"y_pred": model}), on=KEY, validate="one_to_one"
        )
    for direction in DIRECTIONS:
        subset = result.loc[result.direction.eq(direction)]
        if len(subset) != 1513 or subset.homologue_position_id.nunique() != 81:
            raise ValueError(f"Incorrect matching population for {direction}")
        if subset.heldout_fold.nunique() != 5:
            raise ValueError(f"Not five held-out folds for {direction}")
        if (subset.groupby("homologue_position_id").heldout_fold.nunique() != 1).any():
            raise ValueError(f"One position crosses held-out folds for {direction}")
        y = subset.target_effect.to_numpy(dtype=float)
        at_floor = np.isclose(y, -4, rtol=0, atol=1e-12)
        c = subset.minimum_score_class
        if (
            at_floor != c.isin(("both_at_minimum", "target_at_minimum")).to_numpy()
        ).any():
            raise ValueError(
                f"Target floor class differs from numeric effect for {direction}"
            )
    return result


def summarize(
    esmc: pd.DataFrame,
    full: pd.DataFrame,
    *,
    repetitions: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = combine_predictions(esmc, full)
    auc_rows = []
    strata_rows = []
    for direction in DIRECTIONS:
        df = predictions.loc[predictions.direction.eq(direction)].copy()
        target = df.target_effect.to_numpy(dtype=float)
        methods = [
            ("ESM-C site-difference Ridge", df.model_prediction.to_numpy(dtype=float)),
            *(
                (label, df[model].to_numpy(dtype=float))
                for model, label in REFERENCE_NAMES.items()
            ),
            # Use the identity Ridge selected by the extended ESM-C run (not the
            # independently fitted, four-penalty source-full identity Ridge).
            ("source identity Ridge", df.reference_prediction.to_numpy(dtype=float)),
        ]
        for label, values in methods:
            auc_rows.append(
                {
                    "direction": direction,
                    "method": label,
                    "n_minimum": int(np.isclose(target, -4, rtol=0, atol=1e-12).sum()),
                    "n_above_minimum": int(
                        (~np.isclose(target, -4, rtol=0, atol=1e-12)).sum()
                    ),
                    "roc_auc_minimum": minimum_score_auc(target, values),
                }
            )
        for population in STRATA:
            part = df.loc[population_mask(df.minimum_score_class, population)]
            if part.empty:
                raise ValueError(f"Empty {population} in {direction}")
            y = part.target_effect.to_numpy(dtype=float)
            m = part.model_prediction.to_numpy(dtype=float)
            for comparator in COMPARATORS:
                reference = (
                    part.reference_prediction.to_numpy(dtype=float)
                    if comparator == "source_assay_identity_ridge"
                    else part[comparator].to_numpy(dtype=float)
                )
                difference = np.abs(y - reference) - np.abs(y - m)
                lo, hi = grouped_interval(
                    difference,
                    part.homologue_position_id.to_numpy(),
                    repetitions=repetitions,
                    seed_label=(
                        f"{direction}|{population}|{comparator}|source_position_tables"
                    ),
                )
                erho = float(spearmanr(y, m).statistic)
                rrho = float(spearmanr(y, reference).statistic)
                if not np.isfinite((erho, rrho)).all():
                    raise ValueError(
                        f"Non-finite Spearman result in {direction}, {population}"
                    )
                strata_rows.append(
                    {
                        "direction": direction,
                        "population": population,
                        "comparator": comparator,
                        "n_substitutions": len(part),
                        "n_positions": part.homologue_position_id.nunique(),
                        "esmc_mae": float(np.abs(y - m).mean()),
                        "reference_mae": float(np.abs(y - reference).mean()),
                        "esmc_spearman": erho,
                        "reference_spearman": rrho,
                        "paired_mae_gain": float(difference.mean()),
                        "position_ci_lower": lo,
                        "position_ci_upper": hi,
                    }
                )
    return pd.DataFrame(auc_rows), pd.DataFrame(strata_rows)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
