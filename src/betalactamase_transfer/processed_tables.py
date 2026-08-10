# Prepare mutation-effect and residue-feature tables for analysis and modelling.

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


class ProcessedTableError(RuntimeError):
    """Raised when processed research tables cannot be prepared safely."""


EXPECTED_OBSERVATION_COUNT = 138_518
EXPECTED_MISSENSE_COUNT = 125_681
EXPECTED_CONTROL_COUNT = 12_837
EXPECTED_RESIDUE_COUNT = 537
EXPECTED_PAIR_GEOMETRY_COUNT = 54_768
EXPECTED_REPORTED_PAIR_COUNT = 132
EXPECTED_EC50_COUNT = 5_828

_RESIDUE_KEY = ["protein_id", "assay_position"]


OBSERVATION_ROLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "outcome": (
        "score_original",
        "score_analysis",
        "score_uncertainty",
    ),
    "mutation_input": (
        "wild_type",
        "mutant",
    ),
    "condition_input": (
        "antibiotic",
        "concentration",
        "temperature_c",
    ),
    "grouping": (
        "biological_observation_id",
        "underlying_experiment_id",
        "assay_position",
        "mutation_group_id",
        "position_group_id",
        "experiment_group_id",
    ),
    "evaluation_only": (
        "protein_id",
        "mutation",
        "assay_mutation",
        "mutation_class",
        "partition_rule",
        "structure_coverage_class",
        "structure_feature_status",
        "experimental_structure_available",
        "experimental_structure_matches_assay_residue",
        "alphafold_structure_available",
        "alphafold_structure_matches_assay_residue",
        "secondary_structure_available",
        "known_assay_reference_difference",
        "assay_specific_insertion",
        "reported_homologue_pair_position",
        "reported_homologue_partner_assay_position",
        "score_uncertainty_available",
        "sequence_task_eligible",
        "experimental_structure_task_eligible",
        "alphafold_structure_task_eligible",
    ),
    "provenance_quality": (
        "observation_id",
        "source_record_id",
        "study_id",
        "assay_id",
        "source_file",
        "source_sheet",
        "source_row",
        "source_score_column",
        "source_uncertainty_column",
        "publication_position",
        "source_position_label",
        "sequence_validation_status",
        "concentration_unit",
        "quality_status",
        "library_group",
        "identity_label",
        "score_direction",
        "wild_type_reference",
        "principal_outcome",
        "score_floor",
        "score_floor_interpretation",
        "imputation_policy",
        "host_system",
        "selection_duration_hours",
        "selection_temperature_c",
        "biological_replicates",
        "score_definition",
        "source_support",
    ),
}

DEFAULT_MODEL_INPUT_COLUMNS = {
    "wild_type",
    "mutant",
    "antibiotic",
    "concentration",
    "temperature_c",
}

OBSERVATION_COLUMN_NOTES = {
    "protein_id": (
        "Selects the protein representation and transfer direction; "
        "not a default direct feature."
    ),
    "assay_position": (
        "Used for grouping and representation lookup; not a default numeric feature."
    ),
    "score_original": (
        "Unmodified experimental outcome on the source-assay scale; retained for "
        "provenance and sensitivity analyses."
    ),
    "score_analysis": (
        "Publication-aligned DMS analysis outcome with the assay-defined lower "
        "fitness floor applied; primary regression target."
    ),
    "score_uncertainty": (
        "Outcome uncertainty for weighting or uncertainty analysis; "
        "not prediction input."
    ),
    "reported_homologue_pair_position": (
        "Membership in the Chen 2024 reported comparison set."
    ),
    "reported_homologue_partner_assay_position": (
        "Partner position in the Chen 2024 reported comparison set."
    ),
    "score_uncertainty_available": (
        "Outcome-availability indicator; must not become a predictive shortcut."
    ),
}


def _prepare_observation_column_roles(observations: pd.DataFrame) -> pd.DataFrame:
    """Classify every processed observation column before modelling begins."""
    observed = set(observations.columns)
    role_by_column = {
        column: role
        for role, columns in OBSERVATION_ROLE_COLUMNS.items()
        for column in columns
    }
    declared = set(role_by_column)
    missing = sorted(observed - declared)
    obsolete = sorted(declared - observed)
    if missing or obsolete:
        raise ProcessedTableError(
            "Observation column-role declaration is out of sync: "
            f"unclassified={missing}, absent_declared={obsolete}"
        )

    rows = []
    for column in observations.columns:
        role = role_by_column[column]
        rows.append(
            {
                "column_name": column,
                "role": role,
                "default_model_input_allowed": (column in DEFAULT_MODEL_INPUT_COLUMNS),
                "note": OBSERVATION_COLUMN_NOTES.get(
                    column,
                    f"Declared {role.replace('_', ' ')} field.",
                ),
            }
        )
    return pd.DataFrame(rows)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ProcessedTableError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ProcessedTableError(f"{label} is missing required columns: {missing}")


def _require_unique(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    if frame.duplicated(columns).any():
        raise ProcessedTableError(f"{label} contains duplicate keys: {columns}")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ProcessedTableError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    def convert(value: object) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0", "", "nan", "none"}:
            return False
        raise ProcessedTableError(f"Cannot interpret boolean value: {value}")

    return series.map(convert).astype(bool)


def _prepare_residue_features(
    evidence: pd.DataFrame,
    descriptors: pd.DataFrame,
    secondary: pd.DataFrame,
    homologue_pairs: pd.DataFrame,
) -> pd.DataFrame:
    _require_unique(evidence, _RESIDUE_KEY, "Residue evidence")
    _require_unique(descriptors, _RESIDUE_KEY, "Structural descriptors")
    _require_unique(secondary, _RESIDUE_KEY, "Secondary structure")

    evidence_keys = set(
        map(tuple, evidence[_RESIDUE_KEY].itertuples(index=False, name=None))
    )
    descriptor_keys = set(
        map(tuple, descriptors[_RESIDUE_KEY].itertuples(index=False, name=None))
    )
    secondary_keys = set(
        map(tuple, secondary[_RESIDUE_KEY].itertuples(index=False, name=None))
    )
    if evidence_keys != descriptor_keys or evidence_keys != secondary_keys:
        raise ProcessedTableError(
            "Residue evidence, structural descriptors, and secondary structure "
            "do not contain identical assay-position keys"
        )

    descriptor_columns = [
        "protein_id",
        "assay_position",
        "descriptor_status",
        "pdb_residue_name",
        "ca_x",
        "ca_y",
        "ca_z",
        "side_chain_centroid_x",
        "side_chain_centroid_y",
        "side_chain_centroid_z",
        "heavy_atom_count",
        "sasa_angstrom2",
        "relative_sasa_wilke",
        "ca_neighbour_count_8_angstrom",
        "ca_neighbour_count_10_angstrom",
        "ca_neighbour_count_12_angstrom",
        "heavy_atom_contact_count_4_5_angstrom",
        "ca_distance_to_catalytic_zinc_1_angstrom",
        "ca_distance_to_catalytic_zinc_2_angstrom",
        "minimum_atom_distance_to_catalytic_zinc_1_angstrom",
        "minimum_atom_distance_to_catalytic_zinc_2_angstrom",
        "ca_distance_to_catalytic_zinc_centroid_angstrom",
        "ca_distance_to_nearest_mcsa_reference_correspondence_angstrom",
        "minimum_atom_distance_to_nearest_mcsa_reference_correspondence_angstrom",
        "ca_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
        "minimum_atom_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
    ]
    secondary_columns = [
        "protein_id",
        "assay_position",
        "secondary_structure_status",
        "dssp_amino_acid",
        "dssp_coordinate_residue_identity",
        "dssp_assay_residue_identity",
        "dssp_class",
        "secondary_structure_3state",
    ]
    _require_columns(descriptors, descriptor_columns, "Structural descriptors")
    _require_columns(secondary, secondary_columns, "Secondary structure")

    frame = evidence.merge(
        descriptors[descriptor_columns],
        on=_RESIDUE_KEY,
        how="left",
        validate="one_to_one",
    ).merge(
        secondary[secondary_columns],
        on=_RESIDUE_KEY,
        how="left",
        validate="one_to_one",
    )

    _require_unique(
        homologue_pairs,
        ["vim_assay_position"],
        "Reported VIM-2 homologue positions",
    )
    _require_unique(
        homologue_pairs,
        ["ndm_assay_position"],
        "Reported NDM-1 homologue positions",
    )

    vim_map = {
        int(row.vim_assay_position): (
            int(row.ndm_assay_position),
            str(row.pair_source),
            str(row.pair_scope),
        )
        for row in homologue_pairs.itertuples(index=False)
    }
    ndm_map = {
        int(row.ndm_assay_position): (
            int(row.vim_assay_position),
            str(row.pair_source),
            str(row.pair_scope),
        )
        for row in homologue_pairs.itertuples(index=False)
    }

    partner_positions = []
    pair_sources = []
    pair_scopes = []
    pair_flags = []
    for row in frame.itertuples(index=False):
        position = int(row.assay_position)
        mapping = (
            vim_map.get(position)
            if row.protein_id == "VIM-2"
            else ndm_map.get(position)
        )
        if mapping is None:
            partner_positions.append(np.nan)
            pair_sources.append("")
            pair_scopes.append("")
            pair_flags.append(False)
        else:
            partner_positions.append(mapping[0])
            pair_sources.append(mapping[1])
            pair_scopes.append(mapping[2])
            pair_flags.append(True)

    frame["reported_homologue_pair_position"] = pair_flags
    frame["reported_homologue_partner_assay_position"] = partner_positions
    frame["reported_homologue_pair_source"] = pair_sources
    frame["reported_homologue_pair_scope"] = pair_scopes

    known_difference = _as_bool_series(frame["known_assay_reference_difference"])
    assay_insertion = _as_bool_series(frame["assay_specific_insertion"])
    experimental_available = (
        frame["descriptor_status"] == "experimental_coordinate_available"
    )
    alphafold_available = frame["alphafold_position"].notna()
    dssp_available = frame["secondary_structure_status"] == "dssp_assignment_available"

    frame["experimental_structure_available"] = experimental_available
    frame["experimental_structure_matches_assay_residue"] = (
        experimental_available & ~known_difference
    )
    frame["alphafold_structure_available"] = alphafold_available
    frame["alphafold_structure_matches_assay_residue"] = (
        alphafold_available & ~known_difference
    )
    frame["secondary_structure_available"] = dssp_available

    statuses = np.full(len(frame), "no_reference_coordinate", dtype=object)
    statuses[experimental_available.to_numpy() & ~known_difference.to_numpy()] = (
        "experimental_coordinate_matches_assay_residue"
    )
    statuses[experimental_available.to_numpy() & known_difference.to_numpy()] = (
        "experimental_coordinate_reference_residue_differs_from_assay"
    )
    statuses[
        ~experimental_available.to_numpy()
        & alphafold_available.to_numpy()
        & ~known_difference.to_numpy()
    ] = "alphafold_only_matches_assay_residue"
    statuses[assay_insertion.to_numpy()] = (
        "assay_specific_position_without_reference_coordinate"
    )
    frame["structure_feature_status"] = statuses

    if len(frame) != EXPECTED_RESIDUE_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_RESIDUE_COUNT} residue rows, found {len(frame)}"
        )
    if int(frame["reported_homologue_pair_position"].sum()) != 264:
        raise ProcessedTableError(
            "Expected 264 residue rows belonging to the 132 reported "
            "NDM-1/VIM-2 position pairs"
        )
    if int(known_difference.sum()) != 1:
        raise ProcessedTableError(
            "Expected exactly one documented assay/reference residue difference"
        )
    if int(assay_insertion.sum()) != 1:
        raise ProcessedTableError("Expected exactly one assay-specific insertion")

    return frame.sort_values(_RESIDUE_KEY).reset_index(drop=True)


def _prepare_observations(
    measurements: pd.DataFrame,
    groups: pd.DataFrame,
    assays: pd.DataFrame,
    residue_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(
        measurements,
        [
            "observation_id",
            "biological_observation_id",
            "underlying_experiment_id",
            "study_id",
            "assay_id",
            "protein_id",
            "wild_type",
            "assay_position",
            "mutant",
            "mutation",
            "assay_mutation",
            "mutation_class",
            "sequence_validation_status",
            "antibiotic",
            "concentration",
            "concentration_unit",
            "temperature_c",
            "score_original",
            "score_uncertainty",
            "quality_status",
            "library_group",
            "identity_label",
        ],
        "Mutation measurements",
    )
    _require_unique(measurements, ["observation_id"], "Mutation measurements")
    _require_unique(groups, ["observation_id"], "Measurement groups")
    _require_unique(assays, ["assay_id"], "Assay definitions")

    measurement_ids = set(measurements["observation_id"])
    group_ids = set(groups["observation_id"])
    if measurement_ids != group_ids:
        raise ProcessedTableError(
            "Measurement-group observation IDs do not exactly match "
            "mutation-measurement observation IDs"
        )

    group_columns = [
        "observation_id",
        "mutation_group_id",
        "position_group_id",
        "experiment_group_id",
        "partition_rule",
    ]
    assay_columns = [
        "assay_id",
        "score_direction",
        "wild_type_reference",
        "principal_outcome",
        "score_floor",
        "score_floor_interpretation",
        "imputation_policy",
        "host_system",
        "selection_duration_hours",
        "selection_temperature_c",
        "biological_replicates",
        "score_definition",
        "source_support",
    ]
    _require_columns(groups, group_columns, "Measurement groups")
    _require_columns(assays, assay_columns, "Assay definitions")

    residue_status_columns = [
        "protein_id",
        "assay_position",
        "structure_coverage_class",
        "structure_feature_status",
        "experimental_structure_available",
        "experimental_structure_matches_assay_residue",
        "alphafold_structure_available",
        "alphafold_structure_matches_assay_residue",
        "secondary_structure_available",
        "known_assay_reference_difference",
        "assay_specific_insertion",
        "reported_homologue_pair_position",
        "reported_homologue_partner_assay_position",
    ]

    frame = (
        measurements.merge(
            groups[group_columns],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            assays[assay_columns],
            on="assay_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            residue_features[residue_status_columns],
            on=_RESIDUE_KEY,
            how="left",
            validate="many_to_one",
        )
    )

    if len(frame) != EXPECTED_OBSERVATION_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_OBSERVATION_COUNT} observation rows, "
            f"found {len(frame)}"
        )
    if frame["structure_feature_status"].isna().any():
        missing = int(frame["structure_feature_status"].isna().sum())
        raise ProcessedTableError(
            f"{missing} observations failed to map to residue features"
        )
    if not (frame["quality_status"] == "valid").all():
        raise ProcessedTableError(
            "Processed observations contain non-valid quality rows"
        )
    if not (frame["sequence_validation_status"] == "valid").all():
        raise ProcessedTableError(
            "Processed observations contain non-valid sequence mappings"
        )

    score_floor = pd.to_numeric(frame["score_floor"], errors="raise")
    if score_floor.isna().any():
        raise ProcessedTableError("DMS observations are missing an assay score floor")
    frame["score_analysis"] = np.maximum(
        pd.to_numeric(frame["score_original"], errors="raise"),
        score_floor,
    )
    if (frame["score_analysis"] < score_floor).any():
        raise ProcessedTableError("Publication-aligned DMS score floor was not applied")
    if not frame.loc[frame["score_original"] >= score_floor, "score_analysis"].equals(
        frame.loc[frame["score_original"] >= score_floor, "score_original"]
    ):
        raise ProcessedTableError(
            "DMS score transformation changed values above the assay floor"
        )

    frame["score_uncertainty_available"] = frame["score_uncertainty"].notna()
    frame["sequence_task_eligible"] = frame["mutation_class"].eq("missense")
    frame["experimental_structure_task_eligible"] = frame["mutation_class"].eq(
        "missense"
    ) & _as_bool_series(frame["experimental_structure_matches_assay_residue"])
    frame["alphafold_structure_task_eligible"] = frame["mutation_class"].eq(
        "missense"
    ) & _as_bool_series(frame["alphafold_structure_matches_assay_residue"])

    missense = frame[frame["mutation_class"] == "missense"].copy()
    controls = frame[
        frame["mutation_class"].isin(["stop_control", "wild_type_control"])
    ].copy()

    if len(missense) != EXPECTED_MISSENSE_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_MISSENSE_COUNT} missense rows, found {len(missense)}"
        )
    if len(controls) != EXPECTED_CONTROL_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_CONTROL_COUNT} control rows, found {len(controls)}"
        )

    return (
        frame.sort_values("observation_id").reset_index(drop=True),
        missense.sort_values("observation_id").reset_index(drop=True),
        controls.sort_values("observation_id").reset_index(drop=True),
    )


def _prepare_cross_protein_comparisons(
    comparisons: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    if len(comparisons) != EXPECTED_REPORTED_PAIR_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_REPORTED_PAIR_COUNT} cross-protein comparisons, "
            f"found {len(comparisons)}"
        )
    if len(pairs) != EXPECTED_REPORTED_PAIR_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_REPORTED_PAIR_COUNT} reported position pairs, "
            f"found {len(pairs)}"
        )

    keys = ["vim_assay_position", "ndm_assay_position"]
    _require_unique(comparisons, keys, "Cross-protein comparisons")
    _require_unique(pairs, keys, "Reported homologue pairs")

    frame = comparisons.merge(
        pairs[
            keys
            + [
                "pair_source",
                "pair_scope",
                "vim_residue",
                "ndm_residue",
                "same_wild_type_residue",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if frame["pair_scope"].isna().any():
        raise ProcessedTableError(
            "Some cross-protein comparison rows are missing reported pair metadata"
        )
    return frame.sort_values(keys).reset_index(drop=True)


def _prepare_ec50(ec50: pd.DataFrame) -> pd.DataFrame:
    if len(ec50) != EXPECTED_EC50_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_EC50_COUNT} EC50 rows, found {len(ec50)}"
        )
    _require_unique(ec50, ["ec50_observation_id"], "EC50 observations")
    frame = ec50.copy()
    frame["analysis_eligible"] = (
        frame["quality_status"].eq("valid")
        & frame["sequence_validation_status"].eq("valid")
        & frame["mutation_class"].eq("missense")
    )
    return frame.sort_values("ec50_observation_id").reset_index(drop=True)


def prepare_processed_tables(repo_root: Path) -> Mapping[str, object]:
    """Prepare checked research tables without assigning train/test partitions."""
    repo_root = repo_root.resolve()
    output_dir = repo_root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements = _read_csv(
        repo_root / "data/interim/mutation_measurements/mutation_measurements.csv"
    )
    groups = _read_csv(
        repo_root / "data/interim/assay_definitions/measurement_groups.csv"
    )
    assays = _read_csv(repo_root / "data/interim/assay_definitions/assays.csv")
    evidence = _read_csv(
        repo_root / "data/interim/residue_evidence/residue_evidence.csv"
    )
    descriptors = _read_csv(
        repo_root
        / "data/interim/structural_descriptors/residue_structural_descriptors.csv"
    )
    secondary = _read_csv(
        repo_root / "data/interim/secondary_structure/secondary_structure.csv"
    )
    homologue_pairs = _read_csv(
        repo_root
        / "data/interim/residue_evidence/reported_homologue_position_pairs.csv"
    )
    comparisons = _read_csv(
        repo_root / "data/interim/mutation_measurements/cross_protein_comparisons.csv"
    )
    ec50 = _read_csv(
        repo_root / "data/interim/mutation_measurements/ec50_measurements.csv"
    )
    pair_geometry = _read_csv(
        repo_root / "data/interim/structural_descriptors/residue_pair_geometry.csv"
    )

    residue_features = _prepare_residue_features(
        evidence,
        descriptors,
        secondary,
        homologue_pairs,
    )
    observations, missense, controls = _prepare_observations(
        measurements,
        groups,
        assays,
        residue_features,
    )
    cross_protein = _prepare_cross_protein_comparisons(
        comparisons,
        homologue_pairs,
    )
    ec50_processed = _prepare_ec50(ec50)
    observation_column_roles = _prepare_observation_column_roles(observations)

    if len(pair_geometry) != EXPECTED_PAIR_GEOMETRY_COUNT:
        raise ProcessedTableError(
            f"Expected {EXPECTED_PAIR_GEOMETRY_COUNT} residue-pair rows, "
            f"found {len(pair_geometry)}"
        )
    _require_unique(
        pair_geometry,
        ["protein_id", "left_assay_position", "right_assay_position"],
        "Residue-pair geometry",
    )
    if not (
        pair_geometry["left_assay_position"] < pair_geometry["right_assay_position"]
    ).all():
        raise ProcessedTableError(
            "Residue-pair geometry contains reversed or self pairs"
        )

    _write_csv(output_dir / "observations.csv", observations)
    _write_csv(output_dir / "missense_observations.csv", missense)
    _write_csv(output_dir / "control_observations.csv", controls)
    _write_csv(output_dir / "residue_features.csv", residue_features)
    _write_csv(output_dir / "residue_pair_geometry.csv", pair_geometry)
    _write_csv(
        output_dir / "reported_cross_protein_comparisons.csv",
        cross_protein,
    )
    _write_csv(output_dir / "ec50_observations.csv", ec50_processed)
    _write_csv(output_dir / "assay_metadata.csv", assays)
    _write_csv(
        output_dir / "observation_column_roles.csv",
        observation_column_roles,
    )

    structure_status_counts = {
        str(key): int(value)
        for key, value in residue_features["structure_feature_status"]
        .value_counts()
        .sort_index()
        .items()
    }
    missense_by_assay = {
        str(key): int(value)
        for key, value in missense["assay_id"].value_counts().sort_index().items()
    }

    summary: Mapping[str, object] = {
        "status": "PASS",
        "observation_count": int(len(observations)),
        "missense_observation_count": int(len(missense)),
        "control_observation_count": int(len(controls)),
        "residue_feature_row_count": int(len(residue_features)),
        "residue_pair_geometry_row_count": int(len(pair_geometry)),
        "reported_cross_protein_comparison_count": int(len(cross_protein)),
        "ec50_observation_count": int(len(ec50_processed)),
        "assay_count": int(len(assays)),
        "observation_column_role_count": int(len(observation_column_roles)),
        "default_model_input_columns": observation_column_roles.loc[
            observation_column_roles["default_model_input_allowed"],
            "column_name",
        ].tolist(),
        "missense_observations_by_assay": missense_by_assay,
        "missense_sequence_task_eligible_count": int(
            missense["sequence_task_eligible"].sum()
        ),
        "missense_experimental_structure_task_eligible_count": int(
            missense["experimental_structure_task_eligible"].sum()
        ),
        "missense_alphafold_structure_task_eligible_count": int(
            missense["alphafold_structure_task_eligible"].sum()
        ),
        "residue_structure_feature_status_counts": structure_status_counts,
        "documented_assay_reference_difference_count": int(
            _as_bool_series(residue_features["known_assay_reference_difference"]).sum()
        ),
        "assay_specific_insertion_count": int(
            _as_bool_series(residue_features["assay_specific_insertion"]).sum()
        ),
        "score_policy": (
            "score_original remains unchanged on its source-assay scale; "
            "score_analysis applies only the publication-defined lower DMS fitness "
            "floor and no "
            "global pooling, rescaling, standardisation, or imputation"
        ),
        "partition_policy": (
            "group identifiers are retained, but train/validation/test assignments "
            "are created only by evaluation-partition preparation"
        ),
        "structure_policy": (
            "experimental and AlphaFold availability are recorded separately; "
            "coordinates whose reference residue differs from the assay residue "
            "are not marked eligible for the primary structure-dependent task"
        ),
        "homologue_pair_policy": (
            "the 132 Chen 2024 WT-reversion comparison rows remain a separate "
            "outcome-derived resource; the complete 217-position correspondence "
            "is prepared independently from publication position mapping"
        ),
        "outputs": [
            "observations.csv",
            "missense_observations.csv",
            "control_observations.csv",
            "residue_features.csv",
            "residue_pair_geometry.csv",
            "reported_cross_protein_comparisons.csv",
            "ec50_observations.csv",
            "assay_metadata.csv",
            "observation_column_roles.csv",
            "processed_summary.json",
        ],
    }
    (output_dir / "processed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
