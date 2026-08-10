# Prepare modelling-view contracts and observation membership.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


class ModellingViewError(RuntimeError):
    """Raised when modelling-view modelling views cannot be prepared safely."""


EXPECTED_OBSERVATION_COLUMN_COUNT = 62
EXPECTED_RESIDUE_FEATURE_COLUMN_COUNT = 74
EXPECTED_PAIR_FEATURE_COLUMN_COUNT = 13
EXPECTED_SCHEME_COUNT = 17
EXPECTED_MEMBERSHIP_ROW_COUNT = 496_748


OBSERVATION_CONTRACT_COLUMNS = [
    "column_name",
    "processed_table_role",
    "direct_predictor_allowed",
    "representation_lookup_allowed",
    "partition_or_selection_only",
    "target_allowed",
    "uncertainty_weighting_only",
    "modelling_contract_note",
]

RESIDUE_CONTRACT_COLUMNS = [
    "column_name",
    "feature_role",
    "primary_tabular_structural_input_allowed",
    "representation_construction_allowed",
    "biological_validation_reference",
    "quality_or_selection_only",
    "modelling_contract_note",
]

PAIR_CONTRACT_COLUMNS = [
    "column_name",
    "feature_role",
    "graph_construction_allowed",
    "direct_predictor_allowed",
    "modelling_contract_note",
]

REPRESENTATION_COLUMNS = [
    "representation_id",
    "status",
    "coordinate_source",
    "eligibility_column",
    "direct_observation_inputs",
    "lookup_keys",
    "permitted_residue_feature_roles",
    "scientific_use",
]

MEMBERSHIP_COLUMNS = [
    "scheme_id",
    "task_id",
    "scope_role",
    "observation_id",
    "protein_id",
    "assay_position",
    "mutation_group_id",
    "position_group_id",
    "group_type",
    "group_id",
    "sequence_task_eligible",
    "experimental_structure_task_eligible",
    "alphafold_structure_task_eligible",
]

REPRESENTATION_COUNT_COLUMNS = [
    "scheme_id",
    "task_id",
    "scope_role",
    "representation_id",
    "eligible_row_count",
    "total_scope_row_count",
    "coverage_fraction",
]

COMPARISON_POPULATION_COLUMNS = [
    "scheme_id",
    "task_id",
    "scope_role",
    "comparison_population",
    "required_eligibility",
    "row_count",
    "scientific_use",
]


DIRECT_OBSERVATION_INPUTS = {
    "wild_type",
    "mutant",
    "antibiotic",
    "concentration",
    "temperature_c",
}

OBSERVATION_LOOKUP_COLUMNS = {
    "protein_id",
    "assay_position",
}

OBSERVATION_TARGET_COLUMNS = {"score_analysis"}
OBSERVATION_UNCERTAINTY_COLUMNS = {"score_uncertainty"}


RESIDUE_LOOKUP_COLUMNS = {
    "protein_id",
    "assay_position",
    "assay_residue",
    "uniprot_position",
    "uniprot_residue",
}

RESIDUE_SEQUENCE_REFERENCE_COLUMNS = {
    "assay_uniprot_residue_identity",
    "assay_specific_insertion",
    "known_assay_reference_difference",
}

RESIDUE_EXPERIMENTAL_QUALITY_COLUMNS = {
    "selected_pdb_id",
    "selected_pdb_chain",
    "pdb_mapping_status",
    "pdb_author_position",
    "pdb_label_position",
    "pdb_modeled",
    "pdb_residue_identity_ok",
    "pdb_coordinate_present",
    "pdb_backbone_complete",
    "pdb_side_chain_complete",
    "pdb_alternate_location_present",
    "pdb_minimum_occupancy",
    "pdb_mean_b_factor",
    "descriptor_status",
    "pdb_residue_name",
}

RESIDUE_ALPHAFOLD_QUALITY_COLUMNS = {
    "alphafold_position",
    "alphafold_residue",
    "alphafold_mapping_status",
    "structure_coverage_class",
    "alphafold_plddt",
    "alphafold_confidence_category",
}

RESIDUE_GEOMETRY_CONSTRUCTION_COLUMNS = {
    "ca_x",
    "ca_y",
    "ca_z",
    "side_chain_centroid_x",
    "side_chain_centroid_y",
    "side_chain_centroid_z",
    "heavy_atom_count",
}

RESIDUE_GENERIC_STRUCTURAL_INPUT_COLUMNS = {
    "sasa_angstrom2",
    "relative_sasa_wilke",
    "ca_neighbour_count_8_angstrom",
    "ca_neighbour_count_10_angstrom",
    "ca_neighbour_count_12_angstrom",
    "heavy_atom_contact_count_4_5_angstrom",
    "secondary_structure_3state",
}

RESIDUE_MECHANISTIC_REFERENCE_COLUMNS = {
    "mcsa_reference_correspondence_count",
    "mcsa_reference_ids",
    "mcsa_reference_positions",
    "mcsa_correspondence_statuses",
    "mcsa_all_correspondences_identity_conserved",
    "mcsa_any_metal_ligand_role_correspondence",
    "mcsa_any_selected_catalytic_zinc_contact",
    "ca_distance_to_catalytic_zinc_1_angstrom",
    "ca_distance_to_catalytic_zinc_2_angstrom",
    "minimum_atom_distance_to_catalytic_zinc_1_angstrom",
    "minimum_atom_distance_to_catalytic_zinc_2_angstrom",
    "ca_distance_to_catalytic_zinc_centroid_angstrom",
    "ca_distance_to_nearest_mcsa_reference_correspondence_angstrom",
    "minimum_atom_distance_to_nearest_mcsa_reference_correspondence_angstrom",
    "ca_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
    "minimum_atom_distance_to_nearest_mcsa_metal_ligand_correspondence_angstrom",
}

RESIDUE_SECONDARY_STRUCTURE_QUALITY_COLUMNS = {
    "secondary_structure_status",
    "dssp_amino_acid",
    "dssp_coordinate_residue_identity",
    "dssp_assay_residue_identity",
    "dssp_class",
}

RESIDUE_REPORTED_PAIR_COLUMNS = {
    "reported_homologue_pair_position",
    "reported_homologue_partner_assay_position",
    "reported_homologue_pair_source",
    "reported_homologue_pair_scope",
}

RESIDUE_ELIGIBILITY_COLUMNS = {
    "experimental_structure_available",
    "experimental_structure_matches_assay_residue",
    "alphafold_structure_available",
    "alphafold_structure_matches_assay_residue",
    "secondary_structure_available",
    "structure_feature_status",
}


PAIR_LOOKUP_COLUMNS = {
    "protein_id",
    "pdb_id",
    "pdb_chain",
    "left_assay_position",
    "right_assay_position",
}

PAIR_GEOMETRY_COLUMNS = {
    "sequence_separation",
    "peptide_adjacent",
    "ca_distance_angstrom",
    "minimum_heavy_atom_distance_angstrom",
    "ca_within_8_angstrom",
    "ca_within_10_angstrom",
    "ca_within_12_angstrom",
    "heavy_atom_contact_4_5_angstrom",
}

REPRESENTATION_CONTRACTS = [
    {
        "representation_id": "tabular_mutation_condition",
        "status": "primary_baseline",
        "coordinate_source": "none",
        "eligibility_column": "sequence_task_eligible",
        "direct_observation_inputs": (
            "wild_type;mutant;antibiotic;concentration;temperature_c"
        ),
        "lookup_keys": "none",
        "permitted_residue_feature_roles": "none",
        "scientific_use": (
            "simple mutation-and-condition baseline without residue position or "
            "protein identity as direct numeric/categorical shortcuts"
        ),
    },
    {
        "representation_id": "sequence",
        "status": "primary",
        "coordinate_source": "none",
        "eligibility_column": "sequence_task_eligible",
        "direct_observation_inputs": (
            "wild_type;mutant;antibiotic;concentration;temperature_c"
        ),
        "lookup_keys": "protein_id;assay_position",
        "permitted_residue_feature_roles": "none",
        "scientific_use": (
            "sequence-based input built from the assay sequence and "
            "mutation identity; "
            "UniProt remains a mapping/reference source and lookup keys are not "
            "direct predictors"
        ),
    },
    {
        "representation_id": "experimental_structure",
        "status": "primary",
        "coordinate_source": "selected_experimental_pdb",
        "eligibility_column": "experimental_structure_task_eligible",
        "direct_observation_inputs": (
            "wild_type;mutant;antibiotic;concentration;temperature_c"
        ),
        "lookup_keys": "protein_id;assay_position",
        "permitted_residue_feature_roles": (
            "generic_structural_input;geometry_construction"
        ),
        "scientific_use": (
            "experimental-coordinate graph, structural-image, or controlled "
            "structure-based input; predicted mutant structures are "
            "not permitted"
        ),
    },
    {
        "representation_id": "alphafold_structure_supplementary",
        "status": "supplementary",
        "coordinate_source": "alphafold_reference_model",
        "eligibility_column": "alphafold_structure_task_eligible",
        "direct_observation_inputs": (
            "wild_type;mutant;antibiotic;concentration;temperature_c"
        ),
        "lookup_keys": "protein_id;assay_position",
        "permitted_residue_feature_roles": "geometry_construction",
        "scientific_use": (
            "supplementary reference-structure coverage/sensitivity analysis; "
            "AlphaFold does not replace the primary experimental coordinates"
        ),
    },
]


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _observation_contract_rows(roles: pd.DataFrame) -> list[dict[str, Any]]:
    if len(roles) != EXPECTED_OBSERVATION_COLUMN_COUNT:
        raise ModellingViewError("Unexpected processed-table observation-column count")
    if roles["column_name"].duplicated().any():
        raise ModellingViewError(
            "Duplicate processed-table observation column declaration"
        )
    if set(roles["column_name"]) != set(roles["column_name"].dropna()):
        raise ModellingViewError("Missing observation column name")
    if set(roles.loc[roles["default_model_input_allowed"], "column_name"]) != (
        DIRECT_OBSERVATION_INPUTS
    ):
        raise ModellingViewError(
            "processed-table default model inputs changed unexpectedly"
        )

    rows: list[dict[str, Any]] = []
    for record in roles.to_dict(orient="records"):
        column = str(record["column_name"])
        role = str(record["role"])
        direct_allowed = column in DIRECT_OBSERVATION_INPUTS
        lookup_allowed = column in OBSERVATION_LOOKUP_COLUMNS
        target_allowed = column in OBSERVATION_TARGET_COLUMNS
        uncertainty_only = column in OBSERVATION_UNCERTAINTY_COLUMNS
        selection_only = role in {"grouping", "evaluation_only"} and not lookup_allowed
        if direct_allowed:
            note = "direct mutation or assay-condition predictor permitted"
        elif lookup_allowed:
            note = (
                "protein/residue lookup and grouping key only; must not be passed as a "
                "default direct predictor"
            )
        elif target_allowed:
            note = "prediction target only"
        elif uncertainty_only:
            note = (
                "outcome uncertainty may be used only in a specified "
                "weighting or uncertainty analysis, never as a predictor"
            )
        elif role == "grouping":
            note = "partition/grouping metadata only"
        elif role == "evaluation_only":
            note = "evaluation or eligibility metadata only"
        else:
            note = "provenance/quality metadata; not a default predictor"
        rows.append(
            {
                "column_name": column,
                "processed_table_role": role,
                "direct_predictor_allowed": direct_allowed,
                "representation_lookup_allowed": lookup_allowed,
                "partition_or_selection_only": selection_only,
                "target_allowed": target_allowed,
                "uncertainty_weighting_only": uncertainty_only,
                "modelling_contract_note": note,
            }
        )
    return rows


def _residue_role(column: str) -> tuple[str, bool, bool, bool, bool, str]:
    if column in RESIDUE_LOOKUP_COLUMNS:
        return (
            "lookup_identifier",
            False,
            True,
            False,
            False,
            "residue/sequence lookup identity; not a default direct predictor",
        )
    if column in RESIDUE_SEQUENCE_REFERENCE_COLUMNS:
        return (
            "sequence_reference_quality",
            False,
            False,
            False,
            True,
            "mapping/reference status used for eligibility and scientific checks",
        )
    if column in RESIDUE_EXPERIMENTAL_QUALITY_COLUMNS:
        return (
            "experimental_structure_quality",
            False,
            False,
            False,
            True,
            "experimental-structure mapping or coordinate-quality metadata",
        )
    if column in RESIDUE_ALPHAFOLD_QUALITY_COLUMNS:
        return (
            "alphafold_quality",
            False,
            False,
            False,
            True,
            "AlphaFold mapping/confidence metadata for supplementary analysis",
        )
    if column in RESIDUE_GEOMETRY_CONSTRUCTION_COLUMNS:
        return (
            "geometry_construction",
            False,
            True,
            False,
            False,
            "coordinate/geometry field used to construct graph or vision inputs",
        )
    if column in RESIDUE_GENERIC_STRUCTURAL_INPUT_COLUMNS:
        return (
            "generic_structural_input",
            True,
            True,
            False,
            False,
            "generic experimental structural descriptor allowed for a controlled "
            "tabular structural baseline",
        )
    if column in RESIDUE_MECHANISTIC_REFERENCE_COLUMNS:
        return (
            "biological_validation_reference",
            False,
            False,
            True,
            False,
            "mechanistic/metal-reference information reserved from primary model "
            "inputs so biological validation remains independent of model inputs",
        )
    if column in RESIDUE_SECONDARY_STRUCTURE_QUALITY_COLUMNS:
        return (
            "secondary_structure_quality",
            False,
            False,
            False,
            True,
            "DSSP identity/status metadata; reduced 3-state assignment is the "
            "permitted structural input",
        )
    if column in RESIDUE_REPORTED_PAIR_COLUMNS:
        return (
            "reported_pair_evaluation",
            False,
            False,
            True,
            False,
            "Chen-2024 reported-pair information is evaluation/supporting evidence, "
            "not a default predictive input",
        )
    if column in RESIDUE_ELIGIBILITY_COLUMNS:
        return (
            "representation_eligibility",
            False,
            False,
            False,
            True,
            "model-input eligibility/status only",
        )
    raise ModellingViewError(f"Unclassified residue feature column: {column}")


def _residue_contract_rows(residues: pd.DataFrame) -> list[dict[str, Any]]:
    if len(residues.columns) != EXPECTED_RESIDUE_FEATURE_COLUMN_COUNT:
        raise ModellingViewError("Unexpected processed residue-feature column count")
    rows: list[dict[str, Any]] = []
    for column in residues.columns:
        role, direct, construct, biological, quality, note = _residue_role(column)
        rows.append(
            {
                "column_name": column,
                "feature_role": role,
                "primary_tabular_structural_input_allowed": direct,
                "representation_construction_allowed": construct,
                "biological_validation_reference": biological,
                "quality_or_selection_only": quality,
                "modelling_contract_note": note,
            }
        )
    return rows


def _pair_contract_rows(pairs: pd.DataFrame) -> list[dict[str, Any]]:
    if len(pairs.columns) != EXPECTED_PAIR_FEATURE_COLUMN_COUNT:
        raise ModellingViewError("Unexpected processed residue-pair column count")
    if set(pairs.columns) != PAIR_LOOKUP_COLUMNS | PAIR_GEOMETRY_COLUMNS:
        raise ModellingViewError("Unexpected processed residue-pair columns")
    rows: list[dict[str, Any]] = []
    for column in pairs.columns:
        if column in PAIR_LOOKUP_COLUMNS:
            role = "lookup_identifier"
            graph_allowed = True
            direct_allowed = False
            note = "pair/structure lookup key; not a direct predictive feature"
        else:
            role = "pair_geometry"
            graph_allowed = True
            direct_allowed = False
            note = (
                "experimental pair geometry permitted for graph construction; "
                "not passed as an observation-level shortcut"
            )
        rows.append(
            {
                "column_name": column,
                "feature_role": role,
                "graph_construction_allowed": graph_allowed,
                "direct_predictor_allowed": direct_allowed,
                "modelling_contract_note": note,
            }
        )
    return rows


def _scope_frames(missense: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    ndm = missense[missense["assay_id"].eq("ndm1_2024_dms")]
    vim20 = missense[missense["assay_id"].eq("vim2_2020_dms")]
    vim22 = missense[missense["assay_id"].eq("vim2_2022_amp_concentration_dms")]

    frames: dict[tuple[str, str], pd.DataFrame] = {
        ("ndm1_position_nested5", "analysis"): ndm,
        ("vim2_2020_position_nested5", "analysis"): vim20,
        ("vim2_2022_mutation_nested5", "analysis"): vim22,
    }

    ndm_amp = ndm[
        ndm["antibiotic"].eq("AMP")
        & ndm["concentration"].eq(256.0)
        & ndm["temperature_c"].eq(37.0)
    ]
    vim_amp = vim20[
        vim20["antibiotic"].eq("AMP")
        & vim20["concentration"].eq(128.0)
        & vim20["temperature_c"].eq(37.0)
    ]
    frames[("ndm1_to_vim2_amp_source_cv5", "source_model_selection")] = ndm_amp
    frames[("ndm1_to_vim2_amp_source_cv5", "final_test")] = vim_amp
    frames[("vim2_to_ndm1_amp_source_cv5", "source_model_selection")] = vim_amp
    frames[("vim2_to_ndm1_amp_source_cv5", "final_test")] = ndm_amp

    high_conditions = {"AMP": 128.0, "CTX": 4.0, "MEM": 0.031}
    high = vim20[vim20["temperature_c"].eq(37.0)].copy()
    high = high[
        high.apply(
            lambda row: high_conditions.get(row["antibiotic"]) == row["concentration"],
            axis=1,
        )
    ]
    mutation_sets = [
        set(high.loc[high["antibiotic"].eq(drug), "mutation_group_id"])
        for drug in ("AMP", "CTX", "MEM")
    ]
    common = set.intersection(*mutation_sets)
    high = high[high["mutation_group_id"].isin(common)]
    for drug in ("AMP", "CTX", "MEM"):
        scheme = f"vim2_2020_loo_antibiotic_{drug.lower()}"
        frames[(scheme, "final_test")] = high[high["antibiotic"].eq(drug)]
        frames[(scheme, "source_model_selection")] = high[high["antibiotic"].ne(drug)]

    temperatures = vim20[
        vim20["antibiotic"].eq("AMP")
        & vim20["concentration"].isin([2.0, 16.0, 128.0])
        & vim20["temperature_c"].isin([25.0, 37.0])
    ]
    groups25 = set(
        temperatures.loc[temperatures["temperature_c"].eq(25.0), "mutation_group_id"]
    )
    groups37 = set(
        temperatures.loc[temperatures["temperature_c"].eq(37.0), "mutation_group_id"]
    )
    shared = groups25 & groups37
    temperatures = temperatures[temperatures["mutation_group_id"].isin(shared)]
    for source_temp, target_temp in ((25.0, 37.0), (37.0, 25.0)):
        scheme = f"vim2_2020_temperature_{int(source_temp)}_to_{int(target_temp)}"
        frames[(scheme, "source_model_selection")] = temperatures[
            temperatures["temperature_c"].eq(source_temp)
        ]
        frames[(scheme, "final_test")] = temperatures[
            temperatures["temperature_c"].eq(target_temp)
        ]

    for concentration in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0):
        label = str(int(concentration))
        scheme = f"vim2_2022_loo_concentration_{label}"
        target = vim22[vim22["concentration"].eq(concentration)]
        target_groups = set(target["mutation_group_id"])
        source = vim22[
            vim22["concentration"].ne(concentration)
            & vim22["mutation_group_id"].isin(target_groups)
        ]
        frames[(scheme, "source_model_selection")] = source
        frames[(scheme, "final_test")] = target
    return frames


def _build_membership(
    missense: pd.DataFrame,
    schemes: pd.DataFrame,
    assignments: pd.DataFrame,
    scopes: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], pd.DataFrame]]:
    frames = _scope_frames(missense)
    expected_scope_keys = set(zip(scopes["scheme_id"], scopes["scope_role"]))
    if set(frames) != expected_scope_keys:
        missing = sorted(expected_scope_keys - set(frames))
        extra = sorted(set(frames) - expected_scope_keys)
        raise ModellingViewError(
            f"Modelling-view scope reconstruction mismatch: "
            f"missing={missing}, extra={extra}"
        )
    scheme_task = dict(zip(schemes["scheme_id"], schemes["task_id"]))
    scheme_group = dict(zip(schemes["scheme_id"], schemes["group_type"]))
    rows: list[dict[str, Any]] = []
    for (scheme_id, scope_role), frame in frames.items():
        expected_row = scopes[
            scopes["scheme_id"].eq(scheme_id) & scopes["scope_role"].eq(scope_role)
        ].iloc[0]
        if len(frame) != int(expected_row["row_count"]):
            raise ModellingViewError(
                f"Row-count mismatch for {scheme_id}/{scope_role}: "
                f"{len(frame)} != {int(expected_row['row_count'])}"
            )
        group_type = scheme_group[scheme_id]
        assignment_groups = set(
            assignments.loc[assignments["scheme_id"].eq(scheme_id), "group_id"]
        )
        if not set(frame[group_type]).issubset(assignment_groups):
            raise ModellingViewError(
                "Membership contains a group without an evaluation-partition "
                f"assignment: {scheme_id}"
            )
        for record in frame[
            [
                "observation_id",
                "protein_id",
                "assay_position",
                "mutation_group_id",
                "position_group_id",
                "sequence_task_eligible",
                "experimental_structure_task_eligible",
                "alphafold_structure_task_eligible",
            ]
        ].to_dict(orient="records"):
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": scheme_task[scheme_id],
                    "scope_role": scope_role,
                    **record,
                    "group_type": group_type,
                    "group_id": record[group_type],
                }
            )
    if len(rows) != EXPECTED_MEMBERSHIP_ROW_COUNT:
        raise ModellingViewError(
            f"Unexpected modelling-view membership row count: {len(rows)}"
        )
    return rows, frames


def _representation_count_rows(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    schemes: pd.DataFrame,
) -> list[dict[str, Any]]:
    task_map = dict(zip(schemes["scheme_id"], schemes["task_id"]))
    rows: list[dict[str, Any]] = []
    for (scheme_id, scope_role), frame in frames.items():
        for contract in REPRESENTATION_CONTRACTS:
            eligibility = str(contract["eligibility_column"])
            eligible = int(frame[eligibility].sum())
            total = len(frame)
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": task_map[scheme_id],
                    "scope_role": scope_role,
                    "representation_id": contract["representation_id"],
                    "eligible_row_count": eligible,
                    "total_scope_row_count": total,
                    "coverage_fraction": eligible / total,
                }
            )
    return rows


def _comparison_population_rows(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    schemes: pd.DataFrame,
) -> list[dict[str, Any]]:
    task_map = dict(zip(schemes["scheme_id"], schemes["task_id"]))
    definitions = [
        (
            "sequence_full",
            "sequence_task_eligible",
            "complete sequence-eligible population for sequence/basic results",
        ),
        (
            "experimental_structure_matched",
            "experimental_structure_task_eligible",
            "matched population for fair sequence-versus-experimental-structure "
            "comparison",
        ),
        (
            "alphafold_matched",
            "alphafold_structure_task_eligible",
            "matched population for supplementary sequence-versus-AlphaFold comparison",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for (scheme_id, scope_role), frame in frames.items():
        for population, eligibility, use in definitions:
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": task_map[scheme_id],
                    "scope_role": scope_role,
                    "comparison_population": population,
                    "required_eligibility": eligibility,
                    "row_count": int(frame[eligibility].sum()),
                    "scientific_use": use,
                }
            )
    return rows


def prepare_modelling_views(root: Path) -> dict[str, Any]:
    processed = root / "data/processed"
    partition_dir = root / "data/interim/evaluation_partitions"
    required = [
        processed / "missense_observations.csv",
        processed / "observations.csv",
        processed / "observation_column_roles.csv",
        processed / "residue_features.csv",
        processed / "residue_pair_geometry.csv",
        partition_dir / "partition_schemes.csv",
        partition_dir / "partition_scopes.csv",
        partition_dir / "group_assignments.csv",
        partition_dir / "evaluation_partition_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ModellingViewError(f"Missing modelling-view input files: {missing}")

    observations = pd.read_csv(processed / "observations.csv", low_memory=False)
    missense = pd.read_csv(processed / "missense_observations.csv", low_memory=False)
    roles = pd.read_csv(processed / "observation_column_roles.csv")
    residues = pd.read_csv(processed / "residue_features.csv", low_memory=False)
    pairs = pd.read_csv(processed / "residue_pair_geometry.csv", low_memory=False)
    schemes = pd.read_csv(partition_dir / "partition_schemes.csv")
    scopes = pd.read_csv(partition_dir / "partition_scopes.csv")
    assignments = pd.read_csv(partition_dir / "group_assignments.csv", low_memory=False)
    partition_summary = json.loads(
        (partition_dir / "evaluation_partition_summary.json").read_text(
            encoding="utf-8"
        )
    )

    if partition_summary.get("status") != "PASS":
        raise ModellingViewError(
            "Evaluation partitions are not in a verified PASS state"
        )
    if len(schemes) != EXPECTED_SCHEME_COUNT:
        raise ModellingViewError("Unexpected evaluation-partition scheme count")
    if set(roles["column_name"]) != set(observations.columns):
        raise ModellingViewError(
            "Observation-column roles no longer cover observations"
        )
    if not observations["observation_id"].is_unique:
        raise ModellingViewError("processed observation IDs are not unique")
    if not missense["observation_id"].is_unique:
        raise ModellingViewError("processed missense observation IDs are not unique")
    if missense["score_analysis"].isna().any():
        raise ModellingViewError(
            "modelling-view missense target contains missing values"
        )
    if set(missense["concentration_unit"].dropna().unique()) != {"ug/mL"}:
        raise ModellingViewError(
            "modelling-view task concentrations do not share one unit"
        )
    if not missense["mutation_class"].eq("missense").all():
        raise ModellingViewError(
            "modelling-view prediction membership must remain missense-only"
        )
    if not missense["sequence_task_eligible"].all():
        raise ModellingViewError("Unexpected sequence-ineligible missense observation")

    observation_contracts = _observation_contract_rows(roles)
    residue_contracts = _residue_contract_rows(residues)
    pair_contracts = _pair_contract_rows(pairs)
    membership, frames = _build_membership(missense, schemes, assignments, scopes)
    representation_counts = _representation_count_rows(frames, schemes)
    comparison_populations = _comparison_population_rows(frames, schemes)

    residue_contract_frame = pd.DataFrame(residue_contracts)
    biological_reference_columns = set(
        residue_contract_frame.loc[
            residue_contract_frame["biological_validation_reference"], "column_name"
        ]
    )
    if biological_reference_columns != (
        RESIDUE_MECHANISTIC_REFERENCE_COLUMNS | RESIDUE_REPORTED_PAIR_COLUMNS
    ):
        raise ModellingViewError(
            "Biological validation references changed unexpectedly"
        )
    if residue_contract_frame.loc[
        residue_contract_frame["biological_validation_reference"],
        "primary_tabular_structural_input_allowed",
    ].any():
        raise ModellingViewError(
            "Biological validation references must remain absent from primary inputs"
        )

    output_dir = root / "data/interim/modelling_views"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "observation_column_contracts.csv",
        observation_contracts,
        OBSERVATION_CONTRACT_COLUMNS,
    )
    _write_csv(
        output_dir / "residue_feature_contracts.csv",
        residue_contracts,
        RESIDUE_CONTRACT_COLUMNS,
    )
    _write_csv(
        output_dir / "residue_pair_feature_contracts.csv",
        pair_contracts,
        PAIR_CONTRACT_COLUMNS,
    )
    _write_csv(
        output_dir / "representation_contracts.csv",
        REPRESENTATION_CONTRACTS,
        REPRESENTATION_COLUMNS,
    )
    _write_csv(
        output_dir / "scheme_observation_membership.csv",
        membership,
        MEMBERSHIP_COLUMNS,
    )
    _write_csv(
        output_dir / "representation_scope_counts.csv",
        representation_counts,
        REPRESENTATION_COUNT_COLUMNS,
    )
    _write_csv(
        output_dir / "comparison_populations.csv",
        comparison_populations,
        COMPARISON_POPULATION_COLUMNS,
    )

    membership_frame = pd.DataFrame(membership)
    if membership_frame.duplicated(["scheme_id", "scope_role", "observation_id"]).any():
        raise ModellingViewError(
            "Duplicate observation within an evaluation-scheme scope"
        )
    if set(membership_frame["observation_id"]) - set(missense["observation_id"]):
        raise ModellingViewError(
            "Modelling-view membership contains a non-missense observation"
        )

    summary = {
        "status": "PASS",
        "evaluation_partition_scheme_count": len(schemes),
        "observation_column_contract_count": len(observation_contracts),
        "residue_feature_contract_count": len(residue_contracts),
        "residue_pair_feature_contract_count": len(pair_contracts),
        "representation_contract_count": len(REPRESENTATION_CONTRACTS),
        "scheme_observation_membership_row_count": len(membership_frame),
        "unique_membership_observation_count": membership_frame[
            "observation_id"
        ].nunique(),
        "representation_scope_count": len(representation_counts),
        "comparison_population_count": len(comparison_populations),
        "direct_observation_input_columns": sorted(DIRECT_OBSERVATION_INPUTS),
        "representation_lookup_columns": sorted(OBSERVATION_LOOKUP_COLUMNS),
        "target_column": "score_analysis",
        "target_in_membership_table": False,
        "score_uncertainty_predictor_policy": (
            "score_uncertainty is not a predictor; weighting or uncertainty "
            "analysis must be specified"
        ),
        "protein_position_policy": (
            "protein_id and assay_position are protein/residue lookup "
            "and grouping keys, "
            "not default direct predictors"
        ),
        "biological_validation_policy": (
            "M-CSA correspondence, mechanistic-reference distances, catalytic-zinc "
            "engineered distances, and Chen-2024 reported-pair fields are reserved "
            "from primary model inputs so biological validation is not "
            "circular"
        ),
        "matched_population_policy": (
            "model-input comparisons use the same eligibility-restricted rows; "
            "full sequence coverage may be reported separately"
        ),
        "experimental_structure_policy": (
            "experimental PDB remains primary; AlphaFold is supplementary and does "
            "not replace experimental coordinates"
        ),
        "partition_policy": (
            "Modelling-view preparation records observation membership but does "
            "not alter evaluation-partition group assignments"
        ),
        "feature_generation_status": (
            "no embeddings, graph tensors, structural images, or trained models are "
            "created during modelling-view preparation"
        ),
        "outputs": [
            "observation_column_contracts.csv",
            "residue_feature_contracts.csv",
            "residue_pair_feature_contracts.csv",
            "representation_contracts.csv",
            "scheme_observation_membership.csv",
            "representation_scope_counts.csv",
            "comparison_populations.csv",
            "modelling_view_summary.json",
        ],
    }
    (output_dir / "modelling_view_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
