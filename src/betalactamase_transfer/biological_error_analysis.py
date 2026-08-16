from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


class BiologicalErrorAnalysisError(RuntimeError):
    pass


MODEL_IDS = (
    "structure_matched_mutation_condition_ridge",
    "structure_matched_sequence_neighbourhood_ridge",
    "structure_matched_esmc_site_delta_ridge",
    "experimental_structure_descriptor_ridge",
)

ESMC_MODEL = "structure_matched_esmc_site_delta_ridge"
EXPERIMENTAL_STRUCTURE_MODEL = "experimental_structure_descriptor_ridge"
ESMC_STRUCTURE_DIFFERENCE = "esmc_minus_structure_absolute_error"

CORE_SCHEME_IDS = (
    "ndm1_position_nested5",
    "ndm1_to_vim2_amp_source_cv5",
    "vim2_2020_position_nested5",
    "vim2_2022_mutation_nested5",
    "vim2_to_ndm1_amp_source_cv5",
)

EXPECTED_PREDICTION_ROWS = 472_816
EXPECTED_PREDICTIONS_PER_MODEL = 118_204
EXPECTED_TESTED_RESIDUE_KEYS = 468
EXPECTED_GROUP_COUNTS = {
    "ndm1_position_nested5": 238,
    "ndm1_to_vim2_amp_source_cv5": 230,
    "vim2_2020_position_nested5": 230,
    "vim2_2022_mutation_nested5": 4_321,
    "vim2_to_ndm1_amp_source_cv5": 238,
}

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_biological_error_analysis_bootstrap"
BOOTSTRAP_QUANTILES = (0.025, 0.975)

REFERENCE_CONTINUOUS_COLUMNS = (
    "minimum_atom_distance_to_catalytic_zinc_angstrom",
    "ca_distance_to_catalytic_zinc_centroid_angstrom",
    "minimum_atom_distance_to_nearest_mcsa_reference_correspondence_angstrom",
    "minimum_atom_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
)

STRUCTURAL_INPUT_CONTINUOUS_COLUMNS = (
    "sasa_angstrom2",
    "relative_sasa_wilke",
    "ca_neighbour_count_8_angstrom",
    "ca_neighbour_count_10_angstrom",
    "ca_neighbour_count_12_angstrom",
    "heavy_atom_contact_count_4_5_angstrom",
)

REFERENCE_CATEGORY_COLUMNS = (
    "mcsa_reference_position",
    "mcsa_metal_ligand_position",
    "catalytic_zinc_contact_position",
    "homologue_position_class",
)

STRUCTURAL_INPUT_CATEGORY_COLUMNS = ("secondary_structure_3state",)

GROUP_BIOLOGICAL_COLUMNS = (
    *REFERENCE_CONTINUOUS_COLUMNS,
    *STRUCTURAL_INPUT_CONTINUOUS_COLUMNS,
    *REFERENCE_CATEGORY_COLUMNS,
    *STRUCTURAL_INPUT_CATEGORY_COLUMNS,
    "homologue_position_id",
)

ANALYSIS_TARGETS = (*MODEL_IDS, ESMC_STRUCTURE_DIFFERENCE)


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


def _as_bool(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalised = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }
    converted = normalised.map(mapping)
    if converted.isna().any():
        values = sorted(series[converted.isna()].astype(str).unique().tolist())
        raise BiologicalErrorAnalysisError(
            f"Cannot interpret {column} as boolean: {values}"
        )
    return converted.astype(bool)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.allclose(x, x[0], rtol=0.0, atol=1e-12):
        return float("nan")
    if np.allclose(y, y[0], rtol=0.0, atol=1e-12):
        return float("nan")
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _load_inputs(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, str]]:
    prediction_path = (
        repo_root / "data/interim/structure_matched_model_comparison/"
        "structure_matched_predictions.csv"
    )
    prediction_summary_path = (
        repo_root / "data/interim/structure_matched_model_comparison/"
        "structure_matched_summary.json"
    )
    feature_path = repo_root / "data/processed/residue_features.csv"
    homologue_path = (
        repo_root / "data/interim/residue_evidence/"
        "homologue_position_correspondence.csv"
    )

    required = (
        prediction_path,
        prediction_summary_path,
        feature_path,
        homologue_path,
    )
    missing = [
        str(path.relative_to(repo_root)) for path in required if not path.exists()
    ]
    if missing:
        raise BiologicalErrorAnalysisError(
            f"Required biological-error inputs are missing: {missing}"
        )

    prediction_summary = json.loads(prediction_summary_path.read_text(encoding="utf-8"))
    if prediction_summary.get("status") != "PASS":
        raise BiologicalErrorAnalysisError(
            "Structure-matched model comparison summary is not PASS"
        )
    if prediction_summary.get("target_labels_used_for_model_selection") is not False:
        raise BiologicalErrorAnalysisError(
            "Script 28 does not confirm target-label separation"
        )
    for field in (
        "mechanistic_reference_inputs_used",
        "catalytic_zinc_distance_inputs_used",
        "reported_homologue_pair_inputs_used",
    ):
        if prediction_summary.get(field) is not False:
            raise BiologicalErrorAnalysisError(
                f"Script 28 does not reserve {field} from prediction"
            )

    predictions = pd.read_csv(prediction_path, low_memory=False)
    features = pd.read_csv(feature_path, low_memory=False)
    homologue = pd.read_csv(homologue_path, low_memory=False)

    input_hashes = {
        "structure_matched_predictions_sha256": _sha256_file(prediction_path),
        "structure_matched_summary_sha256": _sha256_file(prediction_summary_path),
        "residue_features_sha256": _sha256_file(feature_path),
        "homologue_position_correspondence_sha256": _sha256_file(homologue_path),
    }
    return predictions, features, homologue, input_hashes


def _validate_inputs(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    homologue: pd.DataFrame,
) -> None:
    prediction_required = {
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
        "evaluation_group_id",
        "y_true",
        "y_pred",
    }
    missing_prediction = prediction_required.difference(predictions.columns)
    if missing_prediction:
        raise BiologicalErrorAnalysisError(
            f"Prediction table is missing columns: {sorted(missing_prediction)}"
        )

    feature_required = {
        "protein_id",
        "assay_position",
        "assay_residue",
        "minimum_atom_distance_to_catalytic_zinc_1_angstrom",
        "minimum_atom_distance_to_catalytic_zinc_2_angstrom",
        "ca_distance_to_catalytic_zinc_centroid_angstrom",
        "minimum_atom_distance_to_nearest_mcsa_reference_correspondence_angstrom",
        "minimum_atom_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
        "mcsa_reference_correspondence_count",
        "mcsa_any_metal_ligand_role_correspondence",
        "mcsa_any_selected_catalytic_zinc_contact",
        *STRUCTURAL_INPUT_CONTINUOUS_COLUMNS,
        "secondary_structure_3state",
    }
    missing_feature = feature_required.difference(features.columns)
    if missing_feature:
        raise BiologicalErrorAnalysisError(
            f"Residue feature table is missing columns: {sorted(missing_feature)}"
        )

    homologue_required = {
        "homologue_position_id",
        "mapping_uses_outcome_values",
        "ndm_assay_position",
        "vim_assay_position",
        "same_wild_type_residue",
        "position_class",
    }
    missing_homologue = homologue_required.difference(homologue.columns)
    if missing_homologue:
        raise BiologicalErrorAnalysisError(
            f"Homologue correspondence is missing columns: {sorted(missing_homologue)}"
        )

    if len(predictions) != EXPECTED_PREDICTION_ROWS:
        raise BiologicalErrorAnalysisError(
            f"Unexpected Script 28 prediction row count: {len(predictions)}"
        )
    model_counts = predictions.groupby("model_id").size().to_dict()
    expected_model_counts = {
        model_id: EXPECTED_PREDICTIONS_PER_MODEL for model_id in MODEL_IDS
    }
    if model_counts != expected_model_counts:
        raise BiologicalErrorAnalysisError(
            f"Unexpected Script 28 prediction counts: {model_counts}"
        )
    if set(predictions["scheme_id"].astype(str)) != set(CORE_SCHEME_IDS):
        raise BiologicalErrorAnalysisError(
            "Script 28 prediction schemes do not match the core scheme set"
        )
    prediction_key = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
        "model_id",
    ]
    if predictions.duplicated(prediction_key).any():
        raise BiologicalErrorAnalysisError(
            "Script 28 predictions contain duplicate model/comparison keys"
        )
    if not np.isfinite(predictions["y_true"].to_numpy(dtype=float)).all():
        raise BiologicalErrorAnalysisError("Script 28 truth values are not finite")
    if not np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all():
        raise BiologicalErrorAnalysisError("Script 28 predictions are not finite")

    if features.duplicated(["protein_id", "assay_position"]).any():
        raise BiologicalErrorAnalysisError(
            "Residue features contain duplicate protein/position keys"
        )

    if homologue["homologue_position_id"].duplicated().any():
        raise BiologicalErrorAnalysisError(
            "Homologue correspondence IDs are not unique"
        )
    if homologue["ndm_assay_position"].duplicated().any():
        raise BiologicalErrorAnalysisError("NDM-1 homologue positions are not unique")
    if homologue["vim_assay_position"].duplicated().any():
        raise BiologicalErrorAnalysisError("VIM-2 homologue positions are not unique")
    mapping_uses_outcomes = _as_bool(
        homologue["mapping_uses_outcome_values"],
        column="mapping_uses_outcome_values",
    )
    if mapping_uses_outcomes.any():
        raise BiologicalErrorAnalysisError(
            "Homologue correspondence uses mutation-effect outcomes"
        )


def _homologue_lookup(homologue: pd.DataFrame) -> pd.DataFrame:
    ndm = homologue[
        [
            "homologue_position_id",
            "ndm_assay_position",
            "same_wild_type_residue",
            "position_class",
        ]
    ].rename(columns={"ndm_assay_position": "assay_position"})
    ndm["protein_id"] = "NDM-1"

    vim = homologue[
        [
            "homologue_position_id",
            "vim_assay_position",
            "same_wild_type_residue",
            "position_class",
        ]
    ].rename(columns={"vim_assay_position": "assay_position"})
    vim["protein_id"] = "VIM-2"

    lookup = pd.concat([ndm, vim], ignore_index=True)
    if lookup.duplicated(["protein_id", "assay_position"]).any():
        raise BiologicalErrorAnalysisError(
            "Homologue lookup contains duplicate protein/position keys"
        )
    return lookup


def _prediction_error_table(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    homologue: pd.DataFrame,
) -> pd.DataFrame:
    joined = predictions.merge(
        features,
        on=["protein_id", "assay_position"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_feature"),
    )
    if joined["assay_residue"].isna().any():
        raise BiologicalErrorAnalysisError(
            "Some Script 28 predictions do not map to residue features"
        )
    wild_type_matches = (
        joined["wild_type"].astype(str).eq(joined["assay_residue"].astype(str))
    )
    if not wild_type_matches.all():
        raise BiologicalErrorAnalysisError(
            "Script 28 wild-type residues do not match residue features"
        )

    joined = joined.merge(
        _homologue_lookup(homologue),
        on=["protein_id", "assay_position"],
        how="left",
        validate="many_to_one",
    )
    joined["homologue_position_class"] = joined["position_class"].fillna("unmapped")
    joined = joined.drop(columns=["position_class"])

    joined["mcsa_reference_position"] = np.where(
        pd.to_numeric(
            joined["mcsa_reference_correspondence_count"], errors="raise"
        ).to_numpy(dtype=float)
        > 0,
        "reference_position",
        "other_position",
    )
    metal_ligand = _as_bool(
        joined["mcsa_any_metal_ligand_role_correspondence"],
        column="mcsa_any_metal_ligand_role_correspondence",
    )
    joined["mcsa_metal_ligand_position"] = np.where(
        metal_ligand,
        "metal_ligand_position",
        "other_position",
    )
    zinc_contact = _as_bool(
        joined["mcsa_any_selected_catalytic_zinc_contact"],
        column="mcsa_any_selected_catalytic_zinc_contact",
    )
    joined["catalytic_zinc_contact_position"] = np.where(
        zinc_contact,
        "zinc_contact_position",
        "other_position",
    )

    joined["minimum_atom_distance_to_catalytic_zinc_angstrom"] = joined[
        [
            "minimum_atom_distance_to_catalytic_zinc_1_angstrom",
            "minimum_atom_distance_to_catalytic_zinc_2_angstrom",
        ]
    ].min(axis=1)

    for column in (
        *REFERENCE_CONTINUOUS_COLUMNS,
        *STRUCTURAL_INPUT_CONTINUOUS_COLUMNS,
        "secondary_structure_3state",
    ):
        if joined[column].isna().any():
            raise BiologicalErrorAnalysisError(
                f"Biological-error input contains missing values: {column}"
            )

    tested_residues = joined[
        ["protein_id", "assay_position", "wild_type"]
    ].drop_duplicates()
    if len(tested_residues) != EXPECTED_TESTED_RESIDUE_KEYS:
        raise BiologicalErrorAnalysisError(
            f"Unexpected tested residue count: {len(tested_residues)}"
        )

    joined["signed_error"] = joined["y_pred"].to_numpy(dtype=float) - joined[
        "y_true"
    ].to_numpy(dtype=float)
    joined["absolute_error"] = np.abs(joined["signed_error"].to_numpy(dtype=float))
    joined["squared_error"] = np.square(joined["signed_error"].to_numpy(dtype=float))

    keep = [
        "scheme_id",
        "task_id",
        "model_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
        "evaluation_group_id",
        "protein_id",
        "assay_position",
        "wild_type",
        "mutant",
        "antibiotic",
        "concentration",
        "temperature_c",
        "y_true",
        "y_pred",
        "signed_error",
        "absolute_error",
        "squared_error",
        *GROUP_BIOLOGICAL_COLUMNS,
    ]
    keep = [column for column in keep if column in joined.columns]
    result = joined[keep].copy()
    result = result.sort_values(
        [
            "scheme_id",
            "evaluation_iteration",
            "model_id",
            "observation_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return result


def _require_group_constant(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
) -> None:
    grouped = frame.groupby(list(group_columns), sort=False, dropna=False)
    for column in value_columns:
        counts = grouped[column].nunique(dropna=False)
        if (counts > 1).any():
            examples = counts[counts > 1].head(5).index.tolist()
            raise BiologicalErrorAnalysisError(
                f"{column} varies within evaluation groups: {examples}"
            )


def _group_error_table(prediction_errors: pd.DataFrame) -> pd.DataFrame:
    group_key = ("scheme_id", "model_id", "evaluation_group_id")
    _require_group_constant(
        prediction_errors,
        group_columns=group_key,
        value_columns=(
            "protein_id",
            "assay_position",
            *GROUP_BIOLOGICAL_COLUMNS,
        ),
    )

    rows: list[Mapping[str, object]] = []
    for (scheme_id, model_id, evaluation_group_id), group in prediction_errors.groupby(
        list(group_key), sort=True
    ):
        row: dict[str, object] = {
            "scheme_id": scheme_id,
            "model_id": model_id,
            "evaluation_group_id": evaluation_group_id,
            "protein_id": group["protein_id"].iloc[0],
            "assay_position": int(group["assay_position"].iloc[0]),
            "n_observations": len(group),
            "mean_signed_error": float(group["signed_error"].mean()),
            "mean_absolute_error": float(group["absolute_error"].mean()),
            "median_absolute_error": float(group["absolute_error"].median()),
            "mean_squared_error": float(group["squared_error"].mean()),
            "rmse": float(np.sqrt(group["squared_error"].mean())),
        }
        for column in GROUP_BIOLOGICAL_COLUMNS:
            row[column] = group[column].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)


def _position_error_table(group_errors: pd.DataFrame) -> pd.DataFrame:
    position_key = ("scheme_id", "model_id", "protein_id", "assay_position")
    _require_group_constant(
        group_errors,
        group_columns=position_key,
        value_columns=GROUP_BIOLOGICAL_COLUMNS,
    )

    rows: list[Mapping[str, object]] = []
    for (
        scheme_id,
        model_id,
        protein_id,
        assay_position,
    ), group in group_errors.groupby(list(position_key), sort=True):
        row: dict[str, object] = {
            "scheme_id": scheme_id,
            "model_id": model_id,
            "protein_id": protein_id,
            "assay_position": int(assay_position),
            "n_evaluation_groups": len(group),
            "n_observations": int(group["n_observations"].sum()),
            "mean_signed_error": float(group["mean_signed_error"].mean()),
            "mean_absolute_error": float(group["mean_absolute_error"].mean()),
            "median_group_absolute_error": float(group["mean_absolute_error"].median()),
            "rmse": float(np.sqrt(group["mean_squared_error"].mean())),
        }
        for column in GROUP_BIOLOGICAL_COLUMNS:
            row[column] = group[column].iloc[0]
        rows.append(row)

    result = pd.DataFrame(rows)
    for scheme_id, scheme in result.groupby("scheme_id", sort=True):
        counts = scheme.groupby("model_id").size().to_dict()
        if len(set(counts.values())) != 1 or set(counts) != set(MODEL_IDS):
            raise BiologicalErrorAnalysisError(
                f"Position populations differ among models for {scheme_id}: {counts}"
            )
    return result


def _esmc_structure_position_differences(position_errors: pd.DataFrame) -> pd.DataFrame:
    key = ["scheme_id", "protein_id", "assay_position"]
    biological_columns = [
        "n_evaluation_groups",
        "n_observations",
        *GROUP_BIOLOGICAL_COLUMNS,
    ]

    esmc = position_errors[position_errors["model_id"].eq(ESMC_MODEL)][
        [*key, *biological_columns, "mean_absolute_error"]
    ].rename(columns={"mean_absolute_error": "esmc_mean_absolute_error"})

    structure = position_errors[
        position_errors["model_id"].eq(EXPERIMENTAL_STRUCTURE_MODEL)
    ][[*key, *biological_columns, "mean_absolute_error"]].rename(
        columns={"mean_absolute_error": "structure_mean_absolute_error"}
    )

    matched = esmc.merge(
        structure,
        on=key,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_structure"),
    )

    for column in biological_columns:
        other = f"{column}_structure"
        if other not in matched.columns:
            continue
        left = matched[column]
        right = matched[other]
        same = left.eq(right) | (left.isna() & right.isna())
        if not same.all():
            raise BiologicalErrorAnalysisError(
                f"ESMC/structure group fields differ for {column}"
            )
        matched = matched.drop(columns=[other])

    matched[ESMC_STRUCTURE_DIFFERENCE] = matched["esmc_mean_absolute_error"].to_numpy(
        dtype=float
    ) - matched["structure_mean_absolute_error"].to_numpy(dtype=float)
    matched["difference_interpretation"] = "positive_favours_experimental_structure"
    return matched.sort_values(key, kind="stable").reset_index(drop=True)


def _analysis_matrix(
    position_errors: pd.DataFrame,
    differences: pd.DataFrame,
    scheme_id: str,
) -> pd.DataFrame:
    scheme_positions = differences[differences["scheme_id"].eq(scheme_id)].copy()
    base_columns = [
        "protein_id",
        "assay_position",
        *GROUP_BIOLOGICAL_COLUMNS,
        ESMC_STRUCTURE_DIFFERENCE,
    ]
    matrix = scheme_positions[base_columns].copy()

    for model_id in MODEL_IDS:
        values = position_errors[
            position_errors["scheme_id"].eq(scheme_id)
            & position_errors["model_id"].eq(model_id)
        ][["protein_id", "assay_position", "mean_absolute_error"]].rename(
            columns={"mean_absolute_error": model_id}
        )
        matrix = matrix.merge(
            values,
            on=["protein_id", "assay_position"],
            how="inner",
            validate="one_to_one",
        )

    expected = scheme_positions[["protein_id", "assay_position"]].drop_duplicates()
    if len(matrix) != len(expected):
        raise BiologicalErrorAnalysisError(
            f"Position matching failed for {scheme_id}: "
            f"{len(matrix)} != {len(expected)}"
        )
    return matrix.sort_values(
        ["protein_id", "assay_position"], kind="stable"
    ).reset_index(drop=True)


def _continuous_associations(
    position_errors: pd.DataFrame,
    differences: pd.DataFrame,
    variables: Sequence[str],
    *,
    evidence_role: str,
) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []

    for scheme_id in CORE_SCHEME_IDS:
        matrix = _analysis_matrix(position_errors, differences, scheme_id)
        position_ids = (
            matrix["protein_id"].astype(str)
            + ":"
            + matrix["assay_position"].astype(str)
        ).to_numpy()
        variable_arrays = {
            variable: matrix[variable].to_numpy(dtype=float) for variable in variables
        }
        target_arrays = {
            target: matrix[target].to_numpy(dtype=float) for target in ANALYSIS_TARGETS
        }
        distributions = {
            (target, variable): []
            for target in ANALYSIS_TARGETS
            for variable in variables
        }

        rng = np.random.default_rng(
            _numeric_seed(BOOTSTRAP_SEED, scheme_id, evidence_role)
        )
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.integers(0, len(position_ids), size=len(position_ids))
            for variable in variables:
                x = variable_arrays[variable][sampled]
                for target in ANALYSIS_TARGETS:
                    value = _spearman(x, target_arrays[target][sampled])
                    if np.isfinite(value):
                        distributions[(target, variable)].append(value)

        for target in ANALYSIS_TARGETS:
            for variable in variables:
                point = _spearman(variable_arrays[variable], target_arrays[target])
                values = np.asarray(distributions[(target, variable)], dtype=float)
                rows.append(
                    {
                        "scheme_id": scheme_id,
                        "analysis_target": target,
                        "variable": variable,
                        "evidence_role": evidence_role,
                        "association": "spearman",
                        "value": point,
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
                        "n_positions": len(matrix),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_valid_replicates": len(values),
                        "resampling_unit": "protein_id_assay_position",
                        "interval_method": "percentile",
                    }
                )

    return pd.DataFrame(rows)


def _category_summaries(
    position_errors: pd.DataFrame,
    differences: pd.DataFrame,
    categories: Sequence[str],
    *,
    evidence_role: str,
) -> pd.DataFrame:
    rows: list[Mapping[str, object]] = []

    for scheme_id in CORE_SCHEME_IDS:
        matrix = _analysis_matrix(position_errors, differences, scheme_id)
        target_arrays = {
            target: matrix[target].to_numpy(dtype=float) for target in ANALYSIS_TARGETS
        }
        category_arrays = {
            category: matrix[category].astype(str).to_numpy() for category in categories
        }
        levels = {
            category: sorted(pd.unique(category_arrays[category]).tolist())
            for category in categories
        }
        distributions = {
            (target, category, level): []
            for target in ANALYSIS_TARGETS
            for category in categories
            for level in levels[category]
        }

        rng = np.random.default_rng(
            _numeric_seed(BOOTSTRAP_SEED, scheme_id, evidence_role, "categories")
        )
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.integers(0, len(matrix), size=len(matrix))
            for category in categories:
                sampled_categories = category_arrays[category][sampled]
                for level in levels[category]:
                    mask = sampled_categories == level
                    if not mask.any():
                        continue
                    for target in ANALYSIS_TARGETS:
                        value = float(np.mean(target_arrays[target][sampled][mask]))
                        if np.isfinite(value):
                            distributions[(target, category, level)].append(value)

        for category in categories:
            category_values = category_arrays[category]
            for level in levels[category]:
                mask = category_values == level
                n_positions = int(mask.sum())
                if n_positions == 0:
                    continue
                for target in ANALYSIS_TARGETS:
                    values_for_level = target_arrays[target][mask]
                    bootstrap_values = np.asarray(
                        distributions[(target, category, level)],
                        dtype=float,
                    )
                    rows.append(
                        {
                            "scheme_id": scheme_id,
                            "analysis_target": target,
                            "category": category,
                            "category_value": level,
                            "evidence_role": evidence_role,
                            "mean_value": float(np.mean(values_for_level)),
                            "median_value": float(np.median(values_for_level)),
                            "ci_lower": (
                                float(
                                    np.quantile(
                                        bootstrap_values,
                                        BOOTSTRAP_QUANTILES[0],
                                    )
                                )
                                if len(bootstrap_values)
                                else float("nan")
                            ),
                            "ci_upper": (
                                float(
                                    np.quantile(
                                        bootstrap_values,
                                        BOOTSTRAP_QUANTILES[1],
                                    )
                                )
                                if len(bootstrap_values)
                                else float("nan")
                            ),
                            "n_positions": n_positions,
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                            "bootstrap_valid_replicates": len(bootstrap_values),
                            "resampling_unit": "protein_id_assay_position",
                            "interval_method": "percentile",
                        }
                    )

    return pd.DataFrame(rows)


def prepare_biological_error_analysis(
    repo_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, object],
]:
    repo_root = repo_root.resolve()
    predictions, features, homologue, input_hashes = _load_inputs(repo_root)
    _validate_inputs(predictions, features, homologue)

    prediction_errors = _prediction_error_table(predictions, features, homologue)
    group_errors = _group_error_table(prediction_errors)
    position_errors = _position_error_table(group_errors)
    differences = _esmc_structure_position_differences(position_errors)

    reference_associations = _continuous_associations(
        position_errors,
        differences,
        REFERENCE_CONTINUOUS_COLUMNS,
        evidence_role="withheld_biological_reference",
    )
    reference_categories = _category_summaries(
        position_errors,
        differences,
        REFERENCE_CATEGORY_COLUMNS,
        evidence_role="withheld_biological_reference",
    )
    structural_associations = _continuous_associations(
        position_errors,
        differences,
        STRUCTURAL_INPUT_CONTINUOUS_COLUMNS,
        evidence_role="structural_model_input_stratification",
    )
    structural_categories = _category_summaries(
        position_errors,
        differences,
        STRUCTURAL_INPUT_CATEGORY_COLUMNS,
        evidence_role="structural_model_input_stratification",
    )

    expected_group_rows = sum(EXPECTED_GROUP_COUNTS.values()) * len(MODEL_IDS)
    if len(group_errors) != expected_group_rows:
        raise BiologicalErrorAnalysisError(
            f"Unexpected group-error row count: {len(group_errors)}"
        )
    for model_id in MODEL_IDS:
        counts = (
            group_errors[group_errors["model_id"].eq(model_id)]
            .groupby("scheme_id")
            .size()
            .to_dict()
        )
        counts = {
            scheme_id: int(counts.get(scheme_id, 0)) for scheme_id in CORE_SCHEME_IDS
        }
        if counts != EXPECTED_GROUP_COUNTS:
            raise BiologicalErrorAnalysisError(
                f"Unexpected group counts for {model_id}: {counts}"
            )
    position_counts = (
        position_errors[position_errors["model_id"].eq(MODEL_IDS[0])]
        .groupby("scheme_id")
        .size()
        .astype(int)
        .to_dict()
    )
    if set(position_counts) != set(CORE_SCHEME_IDS):
        raise BiologicalErrorAnalysisError(
            f"Position summaries do not cover all core schemes: {position_counts}"
        )
    expected_difference_rows = sum(position_counts.values())
    if len(differences) != expected_difference_rows:
        raise BiologicalErrorAnalysisError(
            "Unexpected ESMC/structure position-difference row count: "
            f"{len(differences)}"
        )

    summary: Mapping[str, object] = {
        "status": "PASS",
        "analysis_scope": "biological_error_analysis",
        "source_analysis": "structure_matched_model_comparison",
        "model_ids": list(MODEL_IDS),
        "scheme_ids": list(CORE_SCHEME_IDS),
        "prediction_error_definition": "y_pred_minus_y_true",
        "absolute_error_definition": "absolute_y_pred_minus_y_true",
        "group_error_statistic": "mean_absolute_error_within_evaluation_group",
        "position_error_statistic": (
            "mean_of_evaluation_group_mean_absolute_errors_within_residue_position"
        ),
        "group_counts": dict(EXPECTED_GROUP_COUNTS),
        "position_counts": position_counts,
        "prediction_error_row_count": int(len(prediction_errors)),
        "group_error_row_count": int(len(group_errors)),
        "position_error_row_count": int(len(position_errors)),
        "esmc_structure_position_difference_row_count": int(len(differences)),
        "reference_association_row_count": int(len(reference_associations)),
        "reference_category_row_count": int(len(reference_categories)),
        "structural_input_association_row_count": int(len(structural_associations)),
        "structural_input_category_row_count": int(len(structural_categories)),
        "reference_continuous_variables": list(REFERENCE_CONTINUOUS_COLUMNS),
        "reference_category_variables": list(REFERENCE_CATEGORY_COLUMNS),
        "structural_input_continuous_variables": list(
            STRUCTURAL_INPUT_CONTINUOUS_COLUMNS
        ),
        "structural_input_category_variables": list(STRUCTURAL_INPUT_CATEGORY_COLUMNS),
        "esmc_structure_difference_definition": (
            "esmc_position_mean_absolute_error_minus_"
            "experimental_structure_position_mean_absolute_error"
        ),
        "esmc_structure_difference_interpretation": (
            "positive_favours_experimental_structure"
        ),
        "association_statistic": "spearman",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resampling_unit": "protein_id_assay_position",
        "bootstrap_interval_method": "percentile",
        "bootstrap_quantiles": list(BOOTSTRAP_QUANTILES),
        "target_labels_used_for_model_selection": False,
        "model_refitting_performed": False,
        "model_selection_performed": False,
        "mechanistic_reference_inputs_used_for_prediction": False,
        "catalytic_zinc_distance_inputs_used_for_prediction": False,
        "reported_homologue_pair_inputs_used_for_prediction": False,
        "homologue_mapping_uses_outcome_values": False,
        "structural_input_stratification_is_independent_reference_evidence": False,
        "reference_analysis_is_post_model": True,
        **input_hashes,
        "outputs": [
            "biological_error_predictions.csv",
            "biological_error_group_summary.csv",
            "biological_error_position_summary.csv",
            "esmc_structure_position_error_differences.csv",
            "biological_reference_associations.csv",
            "biological_reference_category_summaries.csv",
            "structural_input_error_associations.csv",
            "structural_input_error_category_summaries.csv",
            "biological_error_analysis_summary.json",
        ],
    }

    return (
        prediction_errors,
        group_errors,
        position_errors,
        differences,
        reference_associations,
        reference_categories,
        structural_associations,
        structural_categories,
        summary,
    )


def write_biological_error_analysis(repo_root: Path) -> Mapping[str, object]:
    (
        prediction_errors,
        group_errors,
        position_errors,
        differences,
        reference_associations,
        reference_categories,
        structural_associations,
        structural_categories,
        summary,
    ) = prepare_biological_error_analysis(repo_root)

    output_dir = repo_root / "data/interim/biological_error_analysis"
    _write_csv(
        prediction_errors,
        output_dir / "biological_error_predictions.csv",
    )
    _write_csv(
        group_errors,
        output_dir / "biological_error_group_summary.csv",
    )
    _write_csv(
        position_errors,
        output_dir / "biological_error_position_summary.csv",
    )
    _write_csv(
        differences,
        output_dir / "esmc_structure_position_error_differences.csv",
    )
    _write_csv(
        reference_associations,
        output_dir / "biological_reference_associations.csv",
    )
    _write_csv(
        reference_categories,
        output_dir / "biological_reference_category_summaries.csv",
    )
    _write_csv(
        structural_associations,
        output_dir / "structural_input_error_associations.csv",
    )
    _write_csv(
        structural_categories,
        output_dir / "structural_input_error_category_summaries.csv",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "biological_error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
