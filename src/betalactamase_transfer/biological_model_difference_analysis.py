from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import residue_geometry_neural_evaluation as geometry_stage
from . import residue_graph_neural_evaluation as graph_stage
from . import sequence_plm_evaluation as sequence_stage
from . import sequence_residue_graph_neural_evaluation as joint_stage
from . import sequence_structure_complementarity as mixture_stage


class BiologicalModelDifferenceError(RuntimeError):
    pass


OUTPUT_DIR_NAME = "biological_model_difference_analysis"
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_biological_model_difference"
MIN_CATEGORY_POSITIONS = 5
TARGET_TOLERANCE = graph_stage.TARGET_ABSOLUTE_TOLERANCE
EXPECTED_ROWS = graph_stage.EXPECTED_PREDICTIONS_PER_MODEL

CENTRE_MODEL = graph_stage.CENTRE_MODEL
GRAPH_MODEL = graph_stage.GRAPH_MODEL
GEOMETRY_MODEL = geometry_stage.MODEL_ID
SEQUENCE_MODEL = sequence_stage.MODEL_ID
JOINT_MODEL = joint_stage.MODEL_ID
MIXTURE_MODEL = mixture_stage.MODEL_ID

MODEL_COLUMNS = {
    CENTRE_MODEL: "centre_prediction",
    GRAPH_MODEL: "graph_prediction",
    GEOMETRY_MODEL: "geometry_prediction",
    SEQUENCE_MODEL: "sequence_prediction",
    JOINT_MODEL: "joint_prediction",
    MIXTURE_MODEL: "validation_selected_prediction",
}

CONTRASTS = {
    "graph_minus_centre_absolute_error": (
        "graph_absolute_error",
        "centre_absolute_error",
    ),
    "geometry_minus_graph_absolute_error": (
        "geometry_absolute_error",
        "graph_absolute_error",
    ),
    "sequence_minus_graph_absolute_error": (
        "sequence_absolute_error",
        "graph_absolute_error",
    ),
    "joint_minus_graph_absolute_error": (
        "joint_absolute_error",
        "graph_absolute_error",
    ),
    "validation_selected_minus_graph_absolute_error": (
        "validation_selected_absolute_error",
        "graph_absolute_error",
    ),
}

PRIMARY_CONTINUOUS_VARIABLES = (
    "relative_sasa_wilke",
    "heavy_atom_contact_count_4_5_angstrom",
    "ca_distance_to_catalytic_zinc_centroid_angstrom",
)
SENSITIVITY_CONTINUOUS_VARIABLES = (
    "ca_neighbour_count_10_angstrom",
    "minimum_atom_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
)
CONTINUOUS_VARIABLES = (
    *PRIMARY_CONTINUOUS_VARIABLES,
    *SENSITIVITY_CONTINUOUS_VARIABLES,
)

PRIMARY_CATEGORY_VARIABLES = (
    "secondary_structure_3state",
    "mcsa_any_metal_ligand_role_correspondence",
    "reported_homologue_pair_position",
)
SENSITIVITY_CATEGORY_VARIABLES = ("mcsa_any_selected_catalytic_zinc_contact",)
CATEGORY_VARIABLES = (
    *PRIMARY_CATEGORY_VARIABLES,
    *SENSITIVITY_CATEGORY_VARIABLES,
)

RESULT_GUARDS = {
    "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv": (
        "a811b884f18db4cb16dae1599e7657051e064db8b3d270d6e8ed2f8b0eef3660"
    ),
    (
        "data/interim/residue_graph_neural_evaluation/"
        "residue_graph_neural_predictions.csv"
    ): "98aa268232824720aecd2a8510d4169e4faba4f4dfd71c31659b92088af0203a",
    (
        "data/interim/residue_geometry_neural_evaluation/"
        "residue_geometry_neural_predictions.csv"
    ): "478008269ed73f584ca7c5677f747cb97c08547db4d5c8925f82a2aac3243ea6",
    (
        "data/interim/sequence_residue_graph_neural_evaluation/"
        "sequence_residue_graph_predictions.csv"
    ): "6d6ddae3f9654b8dd0a79dd750ede8755b2ac35bb3f8454d0e4d3457796fed80",
    "data/processed/missense_observations.csv": (
        "846e4343b31418f7f5b9f0841c524c7da4beff27f2c85cbf1c60d9da4d55f608"
    ),
    "data/processed/residue_features.csv": (
        "44fd50744d7ade9b6236c168a2a74c2dcdbeecd5bbdca6febfcac8397a76f644"
    ),
}


KEY_COLUMNS = list(graph_stage.COMPARISON_KEY)
BASE_FIELDS = list(graph_stage.MATCHED_FIELDS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _read_prediction(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"evaluation_iteration": str, "observation_id": str},
        low_memory=False,
    )


def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3:
        return float("nan")
    if np.allclose(x, x[0], rtol=0.0, atol=1e-12):
        return float("nan")
    if np.allclose(y, y[0], rtol=0.0, atol=1e-12):
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    raise ValueError(method)


def _seed(*parts: object) -> int:
    text = "|".join(str(part) for part in (BOOTSTRAP_SEED, *parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _percentile_interval(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )
    if len(finite) == 0:
        return float("nan"), float("nan"), 0
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper), int(len(finite))


def _bootstrap_position_correlation(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    scheme_id: str,
    contrast: str,
    method: str,
) -> tuple[float, float, int]:
    data = frame[[x_column, y_column]].dropna()
    if len(data) < 3:
        return float("nan"), float("nan"), 0
    x = data[x_column].to_numpy(dtype=float)
    y = data[y_column].to_numpy(dtype=float)
    rng = np.random.default_rng(
        _seed("position-correlation", scheme_id, contrast, x_column, method)
    )
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(data), size=len(data))
        values.append(_safe_correlation(x[indices], y[indices], method))
    return _percentile_interval(values)


def _bootstrap_position_mean(
    values: np.ndarray,
    *,
    scheme_id: str,
    contrast: str,
    category: str,
    category_value: str,
) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_CATEGORY_POSITIONS:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(
        _seed("position-category", scheme_id, contrast, category, category_value)
    )
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = values[rng.integers(0, len(values), size=len(values))]
        samples.append(float(np.mean(draw)))
    return _percentile_interval(samples)


def _bootstrap_group_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    scheme_id: str,
    variable: str,
    level: str,
    contrast: str,
) -> tuple[float, float, int]:
    groups = frame["evaluation_group_id"].astype(str).to_numpy()
    values = frame[value_column].to_numpy(dtype=float)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan"), float("nan"), 0
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(
        _seed("condition", scheme_id, variable, level, contrast)
    )
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in selected])
        samples.append(float(np.mean(values[indices])))
    return _percentile_interval(samples)


def _model_frame(
    frame: pd.DataFrame,
    model_id: str,
    column_name: str,
    *,
    require_exact_population: bool = True,
) -> pd.DataFrame:
    selected = frame[frame["model_id"].astype(str).eq(model_id)].copy()
    if require_exact_population and len(selected) != EXPECTED_ROWS:
        raise BiologicalModelDifferenceError(
            f"Expected {EXPECTED_ROWS} rows for {model_id}, observed {len(selected)}"
        )
    if not require_exact_population and len(selected) < EXPECTED_ROWS:
        raise BiologicalModelDifferenceError(
            f"Prediction table for {model_id} is smaller than the matched population"
        )
    if selected.duplicated(KEY_COLUMNS).any():
        raise BiologicalModelDifferenceError(
            f"Duplicate evaluation identities for {model_id}"
        )
    return selected[[*KEY_COLUMNS, "y_true", "y_pred"]].rename(
        columns={"y_pred": column_name}
    )


def _validate_sequence_structure_summary(repo_root: Path) -> Mapping[str, object]:
    path = (
        repo_root
        / "data/interim/sequence_structure_complementarity"
        / "sequence_structure_summary.json"
    )
    if not path.is_file():
        raise BiologicalModelDifferenceError(
            "Sequence/graph combination summary is absent"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise BiologicalModelDifferenceError(
            "Sequence/graph combination summary is not PASS"
        )
    if summary.get("model_id") != MIXTURE_MODEL:
        raise BiologicalModelDifferenceError(
            "Sequence/graph combination terminology must be current before "
            "biological model-difference analysis"
        )
    if summary.get("final_evaluation_labels_used_for_weight_selection") is not False:
        raise BiologicalModelDifferenceError(
            "Sequence/graph combination summary does not confirm final-label "
            "separation for weight selection"
        )
    if (
        summary.get("cross_protein_target_labels_used_for_weight_selection")
        is not False
    ):
        raise BiologicalModelDifferenceError(
            "Sequence/graph combination summary does not confirm target-label "
            "separation"
        )
    if summary.get("diagnostic_oracle_used_for_model_selection") is not False:
        raise BiologicalModelDifferenceError(
            "Sequence/graph combination diagnostic oracle was used for model selection"
        )
    return summary


def _load_inputs(repo_root: Path) -> Mapping[str, pd.DataFrame]:
    for relative, expected in RESULT_GUARDS.items():
        path = repo_root / relative
        if not path.is_file():
            raise BiologicalModelDifferenceError(f"Required file is absent: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise BiologicalModelDifferenceError(
                f"Established input identity changed for {relative}: {observed}"
            )

    _validate_sequence_structure_summary(repo_root)

    graph = _read_prediction(
        repo_root
        / "data/interim/residue_graph_neural_evaluation"
        / "residue_graph_neural_predictions.csv"
    )
    geometry = _read_prediction(
        repo_root
        / "data/interim/residue_geometry_neural_evaluation"
        / "residue_geometry_neural_predictions.csv"
    )
    sequence = _read_prediction(
        repo_root / "data/interim/sequence_plm_evaluation/sequence_plm_predictions.csv"
    )
    joint = _read_prediction(
        repo_root
        / "data/interim/sequence_residue_graph_neural_evaluation"
        / "sequence_residue_graph_predictions.csv"
    )
    selected = _read_prediction(
        repo_root
        / "data/interim/sequence_structure_complementarity"
        / "sequence_structure_predictions.csv"
    )
    residue = pd.read_csv(
        repo_root / "data/processed/residue_features.csv", low_memory=False
    )
    observations = pd.read_csv(
        repo_root / "data/processed/missense_observations.csv",
        dtype={"observation_id": str},
        low_memory=False,
    )
    return {
        "graph": graph,
        "geometry": geometry,
        "sequence": sequence,
        "joint": joint,
        "selected": selected,
        "residue": residue,
        "observations": observations,
    }


def _validate_residue_contract(residue: pd.DataFrame) -> None:
    required = {
        "protein_id",
        "assay_position",
        "experimental_structure_matches_assay_residue",
        "reported_homologue_pair_position",
        "reported_homologue_partner_assay_position",
        *CONTINUOUS_VARIABLES,
        *CATEGORY_VARIABLES,
    }
    missing = sorted(required.difference(residue.columns))
    if missing:
        raise BiologicalModelDifferenceError(
            f"Residue-feature columns are missing: {missing}"
        )
    if residue.duplicated(["protein_id", "assay_position"]).any():
        raise BiologicalModelDifferenceError(
            "Residue features contain duplicate protein/position identities"
        )


def _build_matched_predictions(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    graph_all = tables["graph"]
    graph = graph_all[graph_all["model_id"].astype(str).eq(GRAPH_MODEL)].copy()
    centre = graph_all[graph_all["model_id"].astype(str).eq(CENTRE_MODEL)].copy()
    if len(graph) != EXPECTED_ROWS or len(centre) != EXPECTED_ROWS:
        raise BiologicalModelDifferenceError(
            "Residue-graph/centre model row counts do not match the expected population"
        )
    if graph.duplicated(KEY_COLUMNS).any() or centre.duplicated(KEY_COLUMNS).any():
        raise BiologicalModelDifferenceError(
            "Residue-graph/centre model identities are duplicated"
        )

    base = graph[[*KEY_COLUMNS, *BASE_FIELDS, "y_pred"]].rename(
        columns={"y_pred": "graph_prediction"}
    )

    sources = (
        (centre, CENTRE_MODEL, "centre_prediction"),
        (tables["geometry"], GEOMETRY_MODEL, "geometry_prediction"),
        (tables["sequence"], SEQUENCE_MODEL, "sequence_prediction"),
        (tables["joint"], JOINT_MODEL, "joint_prediction"),
        (tables["selected"], MIXTURE_MODEL, "validation_selected_prediction"),
    )

    for frame, model_id, column_name in sources:
        model = _model_frame(
            frame,
            model_id,
            column_name,
            require_exact_population=model_id != SEQUENCE_MODEL,
        )
        before = len(base)
        base = base.merge(
            model,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
            suffixes=("", "_other"),
        )
        if len(base) != before:
            raise BiologicalModelDifferenceError(
                f"Population size changed while joining {model_id}"
            )
        if base[column_name].isna().any():
            raise BiologicalModelDifferenceError(
                f"Matched population lacks predictions from {model_id}"
            )
        if "y_true_other" in base.columns:
            difference = np.abs(
                base["y_true"].to_numpy(dtype=float)
                - base["y_true_other"].to_numpy(dtype=float)
            )
            if float(np.max(difference)) > TARGET_TOLERANCE:
                raise BiologicalModelDifferenceError(
                    f"Outcome values disagree while joining {model_id}"
                )
            base = base.drop(columns=["y_true_other"])

    prediction_columns = list(MODEL_COLUMNS.values())
    if not np.isfinite(base[prediction_columns].to_numpy(dtype=float)).all():
        raise BiologicalModelDifferenceError(
            "Matched predictions contain non-finite values"
        )

    for model_name, prediction_column in (
        ("centre", "centre_prediction"),
        ("graph", "graph_prediction"),
        ("geometry", "geometry_prediction"),
        ("sequence", "sequence_prediction"),
        ("joint", "joint_prediction"),
        ("validation_selected", "validation_selected_prediction"),
    ):
        base[f"{model_name}_absolute_error"] = np.abs(
            base["y_true"].to_numpy(dtype=float)
            - base[prediction_column].to_numpy(dtype=float)
        )

    for contrast, (left, right) in CONTRASTS.items():
        base[contrast] = base[left] - base[right]

    return base


def _performance_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scheme_id in graph_stage.CORE_SCHEME_IDS:
        scheme = matched[matched["scheme_id"].astype(str).eq(scheme_id)]
        truth = scheme["y_true"].to_numpy(dtype=float)
        for model_id, prediction_column in MODEL_COLUMNS.items():
            values = graph_stage._metric_values(
                truth, scheme[prediction_column].to_numpy(dtype=float)
            )
            records.append(
                {
                    "scheme_id": scheme_id,
                    "model_id": model_id,
                    "n_observations": int(len(scheme)),
                    "n_evaluation_groups": int(scheme["evaluation_group_id"].nunique()),
                    **values,
                }
            )
    return pd.DataFrame(records)


def _position_summary(matched: pd.DataFrame, residue: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_observations": ("observation_id", "size"),
        "n_evaluation_groups": ("evaluation_group_id", "nunique"),
    }
    for contrast in CONTRASTS:
        aggregations[f"mean_{contrast}"] = (contrast, "mean")
        aggregations[f"median_{contrast}"] = (contrast, "median")

    positions = (
        matched.groupby(["scheme_id", "protein_id", "assay_position"], as_index=False)
        .agg(**aggregations)
        .merge(
            residue,
            on=["protein_id", "assay_position"],
            how="left",
            validate="many_to_one",
        )
    )
    if positions[list(CONTINUOUS_VARIABLES)].isna().all(axis=1).any():
        raise BiologicalModelDifferenceError(
            "At least one evaluated position lacks all structural variables"
        )
    structure_match = positions["experimental_structure_matches_assay_residue"]
    structure_match = (
        structure_match.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if structure_match.isna().any() or not structure_match.all():
        raise BiologicalModelDifferenceError(
            "Biological model-difference positions are not fully covered by matching "
            "experimental structures"
        )
    return positions


def _structural_associations(position_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scheme_id in graph_stage.CORE_SCHEME_IDS:
        scheme = position_summary[
            position_summary["scheme_id"].astype(str).eq(scheme_id)
        ]
        for contrast in CONTRASTS:
            y_column = f"mean_{contrast}"
            for variable in CONTINUOUS_VARIABLES:
                data = scheme[[variable, y_column]].dropna()
                x = data[variable].to_numpy(dtype=float)
                y = data[y_column].to_numpy(dtype=float)
                observed = _safe_correlation(x, y, "spearman")
                lower, upper, valid = _bootstrap_position_correlation(
                    data,
                    variable,
                    y_column,
                    scheme_id=scheme_id,
                    contrast=contrast,
                    method="spearman",
                )
                records.append(
                    {
                        "scheme_id": scheme_id,
                        "contrast": contrast,
                        "variable": variable,
                        "evidence_role": (
                            "primary"
                            if variable in PRIMARY_CONTINUOUS_VARIABLES
                            else "sensitivity"
                        ),
                        "association": "spearman",
                        "value": observed,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n_positions": int(len(data)),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_valid_replicates": valid,
                        "resampling_unit": "protein_position",
                    }
                )
    return pd.DataFrame(records)


def _category_summaries(position_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scheme_id in graph_stage.CORE_SCHEME_IDS:
        scheme = position_summary[
            position_summary["scheme_id"].astype(str).eq(scheme_id)
        ]
        for contrast in CONTRASTS:
            value_column = f"mean_{contrast}"
            for category in CATEGORY_VARIABLES:
                values = scheme[category].dropna().unique().tolist()
                for category_value in sorted(values, key=lambda value: str(value)):
                    subset = scheme[scheme[category].eq(category_value)]
                    observed = subset[value_column].to_numpy(dtype=float)
                    lower, upper, valid = _bootstrap_position_mean(
                        observed,
                        scheme_id=scheme_id,
                        contrast=contrast,
                        category=category,
                        category_value=str(category_value),
                    )
                    records.append(
                        {
                            "scheme_id": scheme_id,
                            "contrast": contrast,
                            "category": category,
                            "category_value": str(category_value),
                            "evidence_role": (
                                "primary"
                                if category in PRIMARY_CATEGORY_VARIABLES
                                else "sensitivity"
                            ),
                            "mean_difference": float(np.mean(observed)),
                            "median_difference": float(np.median(observed)),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "n_positions": int(len(subset)),
                            "minimum_category_positions": MIN_CATEGORY_POSITIONS,
                            "minimum_size_met": bool(
                                len(subset) >= MIN_CATEGORY_POSITIONS
                            ),
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                            "bootstrap_valid_replicates": valid,
                            "resampling_unit": "protein_position",
                        }
                    )
    return pd.DataFrame(records)


def _condition_summaries(matched: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scheme_id in graph_stage.CORE_SCHEME_IDS:
        scheme = matched[matched["scheme_id"].astype(str).eq(scheme_id)]
        for variable in ("antibiotic", "concentration", "temperature_c"):
            nonmissing = scheme[variable].dropna()
            if nonmissing.nunique() <= 1:
                continue
            levels = sorted(nonmissing.unique().tolist(), key=lambda value: str(value))
            for level in levels:
                subset = scheme[scheme[variable].eq(level)]
                for contrast in CONTRASTS:
                    values = subset[contrast].to_numpy(dtype=float)
                    lower, upper, valid = _bootstrap_group_mean(
                        subset,
                        contrast,
                        scheme_id=scheme_id,
                        variable=variable,
                        level=str(level),
                        contrast=contrast,
                    )
                    records.append(
                        {
                            "scheme_id": scheme_id,
                            "contrast": contrast,
                            "condition_variable": variable,
                            "condition_value": str(level),
                            "mean_difference": float(np.mean(values)),
                            "median_difference": float(np.median(values)),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "n_observations": int(len(subset)),
                            "n_evaluation_groups": int(
                                subset["evaluation_group_id"].nunique()
                            ),
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                            "bootstrap_valid_replicates": valid,
                            "resampling_unit": "evaluation_group_id",
                        }
                    )
    return pd.DataFrame(records)


def _homologue_pair_details(position_summary: pd.DataFrame) -> pd.DataFrame:
    ndm = position_summary[
        position_summary["scheme_id"].astype(str).eq("ndm1_position_nested5")
    ].copy()
    vim = position_summary[
        position_summary["scheme_id"].astype(str).eq("vim2_2020_position_nested5")
    ].copy()
    ndm = ndm[
        ndm["reported_homologue_pair_position"].astype(str).str.lower().eq("true")
    ]
    ndm["partner_assay_position"] = pd.to_numeric(
        ndm["reported_homologue_partner_assay_position"], errors="coerce"
    )
    if ndm["partner_assay_position"].isna().any():
        raise BiologicalModelDifferenceError(
            "Reported NDM-1 homologue positions lack VIM-2 partner positions"
        )
    columns = ["assay_position", *[f"mean_{contrast}" for contrast in CONTRASTS]]
    vim = vim[columns].rename(
        columns={
            "assay_position": "vim_assay_position",
            **{f"mean_{contrast}": f"vim_{contrast}" for contrast in CONTRASTS},
        }
    )
    ndm_keep = ndm[
        [
            "assay_position",
            "partner_assay_position",
            *[f"mean_{contrast}" for contrast in CONTRASTS],
        ]
    ].rename(
        columns={
            "assay_position": "ndm_assay_position",
            "partner_assay_position": "vim_assay_position",
            **{f"mean_{contrast}": f"ndm_{contrast}" for contrast in CONTRASTS},
        }
    )
    ndm_keep["vim_assay_position"] = ndm_keep["vim_assay_position"].astype(int)
    paired = ndm_keep.merge(
        vim,
        on="vim_assay_position",
        how="inner",
        validate="one_to_one",
    )
    paired.insert(
        0,
        "homologue_pair_id",
        [
            f"NDM-{int(n)}__VIM-{int(v)}"
            for n, v in zip(
                paired["ndm_assay_position"], paired["vim_assay_position"], strict=True
            )
        ],
    )
    return paired


def _paired_position_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    comparison_id: str,
    contrast: str,
) -> tuple[float, float, float, float, float, float, int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    spearman = _safe_correlation(left, right, "spearman")
    pearson = _safe_correlation(left, right, "pearson")
    rng = np.random.default_rng(_seed("position-concordance", comparison_id, contrast))
    spearman_values = []
    pearson_values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(left), size=len(left))
        spearman_values.append(
            _safe_correlation(left[indices], right[indices], "spearman")
        )
        pearson_values.append(
            _safe_correlation(left[indices], right[indices], "pearson")
        )
    s_low, s_high, s_valid = _percentile_interval(spearman_values)
    p_low, p_high, p_valid = _percentile_interval(pearson_values)
    return spearman, s_low, s_high, pearson, p_low, p_high, min(s_valid, p_valid)


def _position_concordance(
    position_summary: pd.DataFrame,
    homologue_pairs: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    def add_comparison(
        comparison_id: str,
        left_scheme: str,
        right_scheme: str,
        left_position: str,
        right_position: str,
        joined: pd.DataFrame,
    ) -> None:
        for contrast in CONTRASTS:
            left_column = f"left_{contrast}"
            right_column = f"right_{contrast}"
            result = _paired_position_correlation(
                joined[left_column].to_numpy(dtype=float),
                joined[right_column].to_numpy(dtype=float),
                comparison_id=comparison_id,
                contrast=contrast,
            )
            spearman, s_low, s_high, pearson, p_low, p_high, valid = result
            records.append(
                {
                    "comparison_id": comparison_id,
                    "left_scheme_id": left_scheme,
                    "right_scheme_id": right_scheme,
                    "left_position_field": left_position,
                    "right_position_field": right_position,
                    "contrast": contrast,
                    "n_position_pairs": int(len(joined)),
                    "spearman": spearman,
                    "spearman_ci_lower": s_low,
                    "spearman_ci_upper": s_high,
                    "pearson": pearson,
                    "pearson_ci_lower": p_low,
                    "pearson_ci_upper": p_high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_valid_replicates": valid,
                    "resampling_unit": "paired_protein_positions",
                }
            )

    homologue_join = pd.DataFrame(
        {
            **{
                f"left_{contrast}": homologue_pairs[f"ndm_{contrast}"]
                for contrast in CONTRASTS
            },
            **{
                f"right_{contrast}": homologue_pairs[f"vim_{contrast}"]
                for contrast in CONTRASTS
            },
        }
    )
    add_comparison(
        "ndm1_vs_vim2_homologous_within_protein_positions",
        "ndm1_position_nested5",
        "vim2_2020_position_nested5",
        "ndm_assay_position",
        "vim_assay_position",
        homologue_join,
    )

    same_position_pairs = (
        (
            "ndm1_within_vs_transfer_target_same_positions",
            "ndm1_position_nested5",
            "vim2_to_ndm1_amp_source_cv5",
        ),
        (
            "vim2_within_vs_transfer_target_same_positions",
            "vim2_2020_position_nested5",
            "ndm1_to_vim2_amp_source_cv5",
        ),
        (
            "vim2_2020_vs_2022_same_positions",
            "vim2_2020_position_nested5",
            "vim2_2022_mutation_nested5",
        ),
    )
    for comparison_id, left_scheme, right_scheme in same_position_pairs:
        left = position_summary[
            position_summary["scheme_id"].astype(str).eq(left_scheme)
        ][["protein_id", "assay_position", *[f"mean_{c}" for c in CONTRASTS]]]
        right = position_summary[
            position_summary["scheme_id"].astype(str).eq(right_scheme)
        ][["protein_id", "assay_position", *[f"mean_{c}" for c in CONTRASTS]]]
        left = left.rename(columns={f"mean_{c}": f"left_{c}" for c in CONTRASTS})
        right = right.rename(columns={f"mean_{c}": f"right_{c}" for c in CONTRASTS})
        joined = left.merge(
            right,
            on=["protein_id", "assay_position"],
            how="inner",
            validate="one_to_one",
        )
        add_comparison(
            comparison_id,
            left_scheme,
            right_scheme,
            "assay_position",
            "assay_position",
            joined,
        )

    return pd.DataFrame(records)


def inspect_biological_model_difference_inputs(repo_root: Path) -> Mapping[str, object]:
    tables = _load_inputs(repo_root)
    _validate_residue_contract(tables["residue"])
    matched = _build_matched_predictions(tables)
    position_summary = _position_summary(matched, tables["residue"])
    counts = matched.groupby("scheme_id").size().astype(int).sort_index().to_dict()
    position_counts = (
        position_summary.groupby("scheme_id").size().astype(int).sort_index().to_dict()
    )
    return {
        "status": "PASS",
        "analysis_scope": "biological_model_difference_analysis",
        "analysis_role": "post_evaluation_biological_interpretation",
        "model_selection_performed": False,
        "final_labels_used_for_model_selection": False,
        "expected_matched_rows": EXPECTED_ROWS,
        "matched_rows": int(len(matched)),
        "scheme_row_counts": counts,
        "scheme_position_counts": position_counts,
        "model_ids": list(MODEL_COLUMNS),
        "contrasts": list(CONTRASTS),
        "primary_continuous_variables": list(PRIMARY_CONTINUOUS_VARIABLES),
        "sensitivity_continuous_variables": list(SENSITIVITY_CONTINUOUS_VARIABLES),
        "primary_category_variables": list(PRIMARY_CATEGORY_VARIABLES),
        "sensitivity_category_variables": list(SENSITIVITY_CATEGORY_VARIABLES),
        "structural_analysis_unit": "protein_position",
        "condition_analysis_resampling_unit": "evaluation_group_id",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "biological_interpretation_scope": "descriptive_association_only",
    }


def run_biological_model_difference_analysis(repo_root: Path) -> Mapping[str, object]:
    audit = inspect_biological_model_difference_inputs(repo_root)
    tables = _load_inputs(repo_root)
    matched = _build_matched_predictions(tables)
    metrics = _performance_metrics(matched)
    position_summary = _position_summary(matched, tables["residue"])
    associations = _structural_associations(position_summary)
    categories = _category_summaries(position_summary)
    conditions = _condition_summaries(matched)
    homologue_pairs = _homologue_pair_details(position_summary)
    concordance = _position_concordance(position_summary, homologue_pairs)

    output_dir = repo_root / "data/interim" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(matched, output_dir / "biological_model_difference_predictions.csv")
    _write_csv(metrics, output_dir / "biological_model_difference_metrics.csv")
    _write_csv(
        position_summary,
        output_dir / "biological_model_difference_position_summary.csv",
    )
    _write_csv(
        associations,
        output_dir / "biological_model_difference_structural_associations.csv",
    )
    _write_csv(
        categories,
        output_dir / "biological_model_difference_category_summaries.csv",
    )
    _write_csv(
        conditions,
        output_dir / "biological_model_difference_condition_summaries.csv",
    )
    _write_csv(
        homologue_pairs,
        output_dir / "biological_model_difference_homologue_pairs.csv",
    )
    _write_csv(
        concordance,
        output_dir / "biological_model_difference_position_concordance.csv",
    )

    summary = {
        **audit,
        "status": "PASS",
        "prediction_row_count": int(len(matched)),
        "metric_row_count": int(len(metrics)),
        "position_row_count": int(len(position_summary)),
        "structural_association_row_count": int(len(associations)),
        "category_summary_row_count": int(len(categories)),
        "condition_summary_row_count": int(len(conditions)),
        "homologue_pair_row_count": int(len(homologue_pairs)),
        "position_concordance_row_count": int(len(concordance)),
        "negative_contrast_interpretation": (
            "negative values indicate lower absolute error for the first named model"
        ),
        "multiple_testing_p_values_reported": False,
        "structural_associations_are_descriptive": True,
        "outputs": [
            "biological_model_difference_predictions.csv",
            "biological_model_difference_metrics.csv",
            "biological_model_difference_position_summary.csv",
            "biological_model_difference_structural_associations.csv",
            "biological_model_difference_category_summaries.csv",
            "biological_model_difference_condition_summaries.csv",
            "biological_model_difference_homologue_pairs.csv",
            "biological_model_difference_position_concordance.csv",
            "biological_model_difference_summary.json",
        ],
    }
    (output_dir / "biological_model_difference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
