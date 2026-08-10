# Define prediction tasks and evaluation rules before partition generation.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


class TaskDefinitionError(RuntimeError):
    """Raised when scientific task definitions cannot be prepared safely."""


EXPECTED_MISSENSE_COUNT = 125_681
EXPECTED_REPORTED_PAIR_COUNT = 132
EXPECTED_FULL_HOMOLOGUE_PAIR_COUNT = 217
EXPECTED_SHARED_WT_HOMOLOGUE_PAIR_COUNT = 81
EXPECTED_ALIGNED_SHARED_WT_MUTATION_PAIR_COUNT = 1_513
EXPECTED_VIM_CROSS_STUDY_SHARED_MUTATION_GROUP_COUNT = 5_005


TASK_DEFINITIONS: tuple[Mapping[str, Any], ...] = (
    {
        "task_id": "ndm1_within_position_regression",
        "task_family": "within_protein_regression",
        "priority": "primary",
        "scientific_question": (
            "How well can mutation effects be predicted at NDM-1 residue positions "
            "not used for model fitting when antibiotic and concentration remain "
            "specified?"
        ),
        "target_field": "score_analysis",
        "target_scale_policy": "retain_source_assay_scale_with_publication_lower_floor",
        "primary_metric": "spearman",
        "secondary_metrics": "pearson;mae;rmse",
        "representation_policy": "compare_modalities_on_matched_eligible_rows",
        "reported_pair_table_role": "not_used_as_training_labels",
        "full_homologue_alignment_required": False,
    },
    {
        "task_id": "vim2_2020_within_position_regression",
        "task_family": "within_protein_regression",
        "priority": "primary",
        "scientific_question": (
            "How well can mutation effects be predicted at VIM-2 residue positions "
            "not used for model fitting across the Chen 2020 antibiotic and "
            "temperature conditions?"
        ),
        "target_field": "score_analysis",
        "target_scale_policy": "retain_source_assay_scale_with_publication_lower_floor",
        "primary_metric": "spearman",
        "secondary_metrics": "pearson;mae;rmse",
        "representation_policy": "compare_modalities_on_matched_eligible_rows",
        "reported_pair_table_role": "not_used_as_training_labels",
        "full_homologue_alignment_required": False,
    },
    {
        "task_id": "ndm1_to_vim2_amp_transfer",
        "task_family": "cross_protein_transfer",
        "priority": "headline",
        "scientific_question": (
            "Can mutation effects learned from NDM-1 at the Chen 2024 matched AMP "
            "selection setting predict VIM-2 mutation effects at the corresponding "
            "study-defined AMP setting?"
        ),
        "target_field": "score_analysis",
        "target_scale_policy": (
            "retain_each_source_scale_with_publication_lower_floor_"
            "no_global_harmonisation"
        ),
        "primary_metric": "spearman",
        "secondary_metrics": "pearson;mae;rmse",
        "representation_policy": "compare_modalities_on_matched_eligible_rows",
        "reported_pair_table_role": "supporting_paired_analysis_only",
        "full_homologue_alignment_required": True,
    },
    {
        "task_id": "vim2_to_ndm1_amp_transfer",
        "task_family": "cross_protein_transfer",
        "priority": "headline",
        "scientific_question": (
            "Can mutation effects learned from VIM-2 at the Chen 2024 matched AMP "
            "selection setting predict NDM-1 mutation effects at the corresponding "
            "study-defined AMP setting?"
        ),
        "target_field": "score_analysis",
        "target_scale_policy": (
            "retain_each_source_scale_with_publication_lower_floor_"
            "no_global_harmonisation"
        ),
        "primary_metric": "spearman",
        "secondary_metrics": "pearson;mae;rmse",
        "representation_policy": "compare_modalities_on_matched_eligible_rows",
        "reported_pair_table_role": "supporting_paired_analysis_only",
        "full_homologue_alignment_required": True,
    },
    {
        "task_id": "vim2_2020_condition_transfer",
        "task_family": "condition_transfer",
        "priority": "secondary",
        "scientific_question": (
            "How well do VIM-2 mutation-effect relationships transfer across "
            "antibiotic or temperature conditions when the held-out condition is "
            "defined?"
        ),
        "target_field": "score_analysis",
        "target_scale_policy": "retain_source_assay_scale_with_publication_lower_floor",
        "primary_metric": "spearman",
        "secondary_metrics": "pearson;mae;rmse",
        "representation_policy": "compare_modalities_on_matched_eligible_rows",
        "reported_pair_table_role": "not_used_as_training_labels",
        "full_homologue_alignment_required": False,
    },
    {
        "task_id": "vim2_2022_concentration_response",
        "task_family": "concentration_response",
        "priority": "secondary",
        "scientific_question": (
            "How does VIM-2 mutation effect change with ampicillin concentration, "
            "and can the concentration-response relationship generalise to unseen "
            "mutations or held-out concentrations?"
        ),
        "target_field": "score_analysis",
        "target_scale_policy": "retain_source_assay_scale_with_publication_lower_floor",
        "primary_metric": "spearman",
        "secondary_metrics": "pearson;mae;rmse",
        "representation_policy": "compare_modalities_on_matched_eligible_rows",
        "reported_pair_table_role": "not_used_as_training_labels",
        "full_homologue_alignment_required": False,
    },
    {
        "task_id": "chen2024_wt_reversion_analysis",
        "task_family": "paired_biological_analysis",
        "priority": "supporting",
        "scientific_question": (
            "What do the 132 Chen 2024 reported WT-reversion comparisons show "
            "about background-dependent compatibility between NDM-1 and VIM-2?"
        ),
        "target_field": "not_applicable",
        "target_scale_policy": "use_reported_amp_values_without_reinterpretation",
        "primary_metric": "paired_effect_difference_and_reported_class",
        "secondary_metrics": "not_predefined",
        "representation_policy": "not_a_model_training_table",
        "reported_pair_table_role": "analysis_source_not_independent_labels",
        "full_homologue_alignment_required": False,
    },
)


TASK_SCOPES: tuple[Mapping[str, Any], ...] = (
    {
        "task_id": "ndm1_within_position_regression",
        "scope_role": "analysis",
        "data_table": "missense_observations.csv",
        "assay_id": "ndm1_2024_dms",
        "protein_id": "NDM-1",
        "antibiotic": "*",
        "concentration": "*",
        "temperature_c": "37",
    },
    {
        "task_id": "vim2_2020_within_position_regression",
        "scope_role": "analysis",
        "data_table": "missense_observations.csv",
        "assay_id": "vim2_2020_dms",
        "protein_id": "VIM-2",
        "antibiotic": "*",
        "concentration": "*",
        "temperature_c": "*",
    },
    {
        "task_id": "ndm1_to_vim2_amp_transfer",
        "scope_role": "source",
        "data_table": "missense_observations.csv",
        "assay_id": "ndm1_2024_dms",
        "protein_id": "NDM-1",
        "antibiotic": "AMP",
        "concentration": "256",
        "temperature_c": "37",
    },
    {
        "task_id": "ndm1_to_vim2_amp_transfer",
        "scope_role": "target",
        "data_table": "missense_observations.csv",
        "assay_id": "vim2_2020_dms",
        "protein_id": "VIM-2",
        "antibiotic": "AMP",
        "concentration": "128",
        "temperature_c": "37",
    },
    {
        "task_id": "vim2_to_ndm1_amp_transfer",
        "scope_role": "source",
        "data_table": "missense_observations.csv",
        "assay_id": "vim2_2020_dms",
        "protein_id": "VIM-2",
        "antibiotic": "AMP",
        "concentration": "128",
        "temperature_c": "37",
    },
    {
        "task_id": "vim2_to_ndm1_amp_transfer",
        "scope_role": "target",
        "data_table": "missense_observations.csv",
        "assay_id": "ndm1_2024_dms",
        "protein_id": "NDM-1",
        "antibiotic": "AMP",
        "concentration": "256",
        "temperature_c": "37",
    },
    {
        "task_id": "vim2_2020_condition_transfer",
        "scope_role": "analysis",
        "data_table": "missense_observations.csv",
        "assay_id": "vim2_2020_dms",
        "protein_id": "VIM-2",
        "antibiotic": "*",
        "concentration": "*",
        "temperature_c": "*",
    },
    {
        "task_id": "vim2_2022_concentration_response",
        "scope_role": "analysis",
        "data_table": "missense_observations.csv",
        "assay_id": "vim2_2022_amp_concentration_dms",
        "protein_id": "VIM-2",
        "antibiotic": "AMP",
        "concentration": "*",
        "temperature_c": "37",
    },
    {
        "task_id": "chen2024_wt_reversion_analysis",
        "scope_role": "analysis",
        "data_table": "reported_cross_protein_comparisons.csv",
        "assay_id": "not_applicable",
        "protein_id": "NDM-1;VIM-2",
        "antibiotic": "AMP",
        "concentration": "study_reported_matched_setting",
        "temperature_c": "37",
    },
)


EVALUATION_REGIMES: tuple[Mapping[str, Any], ...] = (
    {
        "regime_id": "position_grouped_within_protein",
        "applies_to": (
            "ndm1_within_position_regression;vim2_2020_within_position_regression"
        ),
        "status": "ready_for_partition_design",
        "partition_unit": "position_group_id",
        "held_out_dimension": "assay residue position",
        "same_mutation_may_cross_partitions": False,
        "same_position_may_cross_partitions": False,
        "same_condition_may_cross_partitions": True,
        "target_protein_visible_during_model_selection": True,
        "interpretation": (
            "generalisation to unseen residue positions within one protein"
        ),
        "required_rule": (
            "all mutations and all measured conditions at a residue position remain "
            "in one partition"
        ),
    },
    {
        "regime_id": "cross_protein_source_only_validation",
        "applies_to": "ndm1_to_vim2_amp_transfer;vim2_to_ndm1_amp_transfer",
        "status": "ready_for_partition_design",
        "partition_unit": (
            "source position_group_id for model selection; protein for final test"
        ),
        "held_out_dimension": "protein background",
        "same_mutation_may_cross_partitions": False,
        "same_position_may_cross_partitions": False,
        "same_condition_may_cross_partitions": False,
        "target_protein_visible_during_model_selection": False,
        "interpretation": (
            "transfer to a protein background absent from model selection"
        ),
        "required_rule": (
            "validation is drawn only from the source protein; the target protein is "
            "untouched until final evaluation"
        ),
    },
    {
        "regime_id": "leave_one_antibiotic_out_vim2_2020",
        "applies_to": "vim2_2020_condition_transfer",
        "status": "ready_for_partition_design",
        "partition_unit": "antibiotic",
        "held_out_dimension": "antibiotic identity",
        "same_mutation_may_cross_partitions": True,
        "same_position_may_cross_partitions": True,
        "same_condition_may_cross_partitions": False,
        "target_protein_visible_during_model_selection": True,
        "interpretation": (
            "37 C antibiotic transfer for mutations that may already be observed "
            "under other antibiotics; not unseen-mutation generalisation"
        ),
        "required_rule": (
            "restrict the antibiotic holdout to 37 C, where AMP, CTX, and MEM are "
            "all represented; all rows for the held-out antibiotic remain outside "
            "training"
        ),
    },
    {
        "regime_id": "directional_temperature_transfer_vim2_amp",
        "applies_to": "vim2_2020_condition_transfer",
        "status": "ready_for_partition_design",
        "partition_unit": "temperature_c within VIM-2 AMP rows",
        "held_out_dimension": "temperature",
        "same_mutation_may_cross_partitions": True,
        "same_position_may_cross_partitions": True,
        "same_condition_may_cross_partitions": False,
        "target_protein_visible_during_model_selection": True,
        "interpretation": (
            "temperature transfer for mutations measured at the other temperature; "
            "not unseen-mutation generalisation"
        ),
        "required_rule": (
            "compare 25 C to 37 C directionally using AMP concentrations represented "
            "at both temperatures"
        ),
    },
    {
        "regime_id": "mutation_grouped_vim2_2022_concentration",
        "applies_to": "vim2_2022_concentration_response",
        "status": "ready_for_partition_design",
        "partition_unit": "mutation_group_id",
        "held_out_dimension": "mutation identity",
        "same_mutation_may_cross_partitions": False,
        "same_position_may_cross_partitions": True,
        "same_condition_may_cross_partitions": True,
        "target_protein_visible_during_model_selection": True,
        "interpretation": "concentration-response prediction for unseen mutations",
        "required_rule": (
            "all concentrations for one mutation remain in the same partition"
        ),
    },
    {
        "regime_id": "leave_one_concentration_out_vim2_2022",
        "applies_to": "vim2_2022_concentration_response",
        "status": "ready_for_partition_design",
        "partition_unit": "concentration",
        "held_out_dimension": "ampicillin concentration",
        "same_mutation_may_cross_partitions": True,
        "same_position_may_cross_partitions": True,
        "same_condition_may_cross_partitions": False,
        "target_protein_visible_during_model_selection": True,
        "interpretation": (
            "concentration transfer for mutations observed at other concentrations; "
            "interior and edge concentrations must be reported separately"
        ),
        "required_rule": (
            "the held-out concentration remains outside training; do not describe "
            "this regime as unseen-mutation generalisation"
        ),
    },
    {
        "regime_id": "reported_wt_reversion_paired_analysis",
        "applies_to": "chen2024_wt_reversion_analysis",
        "status": "ready_for_analysis_design",
        "partition_unit": "not_applicable",
        "held_out_dimension": "not_applicable",
        "same_mutation_may_cross_partitions": False,
        "same_position_may_cross_partitions": False,
        "same_condition_may_cross_partitions": False,
        "target_protein_visible_during_model_selection": False,
        "interpretation": (
            "paired biological analysis of the reported WT-reversion comparison set; "
            "not an independent mutation-label dataset"
        ),
        "required_rule": (
            "retain reported classes and AMP values as reported comparison evidence"
        ),
    },
    {
        "regime_id": "aligned_shared_wild_type_transfer_reporting",
        "applies_to": "ndm1_to_vim2_amp_transfer;vim2_to_ndm1_amp_transfer",
        "status": "ready_for_reporting_population",
        "partition_unit": (
            "reuse cross-protein source-only model-selection partitions; final "
            "reporting pairs are keyed by homologue_position_id and mutant"
        ),
        "held_out_dimension": "protein background at directly comparable substitutions",
        "same_mutation_may_cross_partitions": True,
        "same_position_may_cross_partitions": True,
        "same_condition_may_cross_partitions": False,
        "target_protein_visible_during_model_selection": False,
        "interpretation": (
            "final held-out transfer performance on identical WT-to-mutant "
            "substitutions at aligned positions with the same WT residue"
        ),
        "required_rule": (
            "use the complete Chen 2024 aligned-position correspondence, restrict "
            "direct equivalent-mutation comparisons to the 81 shared-WT positions, "
            "pair observations without using score values, and never use target "
            "labels for model selection"
        ),
    },
)


EXPECTED_SCOPE_COUNTS: Mapping[tuple[str, str], int] = {
    ("ndm1_within_position_regression", "analysis"): 45_369,
    ("vim2_2020_within_position_regression", "analysis"): 45_222,
    ("ndm1_to_vim2_amp_transfer", "source"): 5_041,
    ("ndm1_to_vim2_amp_transfer", "target"): 5_023,
    ("vim2_to_ndm1_amp_transfer", "source"): 5_023,
    ("vim2_to_ndm1_amp_transfer", "target"): 5_041,
    ("vim2_2020_condition_transfer", "analysis"): 45_222,
    ("vim2_2022_concentration_response", "analysis"): 35_090,
    ("chen2024_wt_reversion_analysis", "analysis"): 132,
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise TaskDefinitionError(f"Missing required processed research table: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise TaskDefinitionError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _normalise_number_text(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def _filter_observations(frame: pd.DataFrame, scope: Mapping[str, Any]) -> pd.DataFrame:
    result = frame
    for column in ("assay_id", "protein_id", "antibiotic"):
        expected = str(scope[column])
        if expected != "*":
            result = result[result[column].astype(str).eq(expected)]

    for column in ("concentration", "temperature_c"):
        expected = str(scope[column])
        if expected != "*":
            expected_number = float(expected)
            numeric = pd.to_numeric(result[column], errors="coerce")
            result = result[numeric.eq(expected_number)]
    return result.copy()


def _scope_summary(
    scope: Mapping[str, Any],
    missense: pd.DataFrame,
    reported: pd.DataFrame,
) -> dict[str, Any]:
    row = dict(scope)
    if scope["data_table"] == "missense_observations.csv":
        selected = _filter_observations(missense, scope)
        row.update(
            {
                "row_count": len(selected),
                "mutation_group_count": selected["mutation_group_id"].nunique(),
                "position_group_count": selected["position_group_id"].nunique(),
                "experimental_structure_eligible_count": int(
                    selected["experimental_structure_task_eligible"].sum()
                ),
                "alphafold_structure_eligible_count": int(
                    selected["alphafold_structure_task_eligible"].sum()
                ),
                "score_min": float(selected["score_analysis"].min()),
                "score_median": float(selected["score_analysis"].median()),
                "score_max": float(selected["score_analysis"].max()),
            }
        )
        return row

    if scope["data_table"] == "reported_cross_protein_comparisons.csv":
        row.update(
            {
                "row_count": len(reported),
                "mutation_group_count": "not_applicable",
                "position_group_count": len(reported),
                "experimental_structure_eligible_count": "not_applicable",
                "alphafold_structure_eligible_count": "not_applicable",
                "score_min": "not_applicable",
                "score_median": "not_applicable",
                "score_max": "not_applicable",
            }
        )
        return row

    raise TaskDefinitionError(f"Unsupported task scope table: {scope['data_table']}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TaskDefinitionError(f"{label} is missing required columns: {missing}")


def _validate_definitions() -> None:
    task_ids = [str(task["task_id"]) for task in TASK_DEFINITIONS]
    if len(task_ids) != len(set(task_ids)):
        raise TaskDefinitionError("Task identifiers must be unique")

    scope_task_ids = {str(scope["task_id"]) for scope in TASK_SCOPES}
    if scope_task_ids != set(task_ids):
        raise TaskDefinitionError(
            "Task scopes do not cover exactly the declared task identifiers"
        )

    if any(
        "random" in str(regime["regime_id"]).lower() for regime in EVALUATION_REGIMES
    ):
        raise TaskDefinitionError(
            "Task definition does not define a random-row evaluation regime"
        )

    aligned_reporting = [
        regime
        for regime in EVALUATION_REGIMES
        if regime["regime_id"] == "aligned_shared_wild_type_transfer_reporting"
    ]
    if len(aligned_reporting) != 1 or aligned_reporting[0]["status"] != (
        "ready_for_reporting_population"
    ):
        raise TaskDefinitionError(
            "Aligned shared-WT transfer reporting must be defined"
        )


def prepare_task_definitions(repo_root: Path) -> dict[str, Any]:
    """Write task definitions without assigning partitions."""
    repo_root = repo_root.resolve()
    processed = repo_root / "data/processed"
    output = repo_root / "data/interim/task_definitions"

    missense = _read_csv(processed / "missense_observations.csv")
    reported = _read_csv(processed / "reported_cross_protein_comparisons.csv")
    homologue = _read_csv(
        repo_root
        / "data/interim/residue_evidence/homologue_position_correspondence.csv"
    )
    roles = _read_csv(processed / "observation_column_roles.csv")

    _require_columns(
        missense,
        [
            "observation_id",
            "assay_id",
            "protein_id",
            "mutation_group_id",
            "position_group_id",
            "antibiotic",
            "concentration",
            "temperature_c",
            "score_original",
            "score_analysis",
            "sequence_task_eligible",
            "experimental_structure_task_eligible",
            "alphafold_structure_task_eligible",
        ],
        "Missense observations",
    )
    _require_columns(
        reported,
        [
            "comparison_id",
            "reported_class",
            "independence_status",
            "pair_source",
            "pair_scope",
            "amp_vim",
            "amp_ndm",
        ],
        "Reported cross-protein comparisons",
    )
    _require_columns(
        roles,
        ["column_name", "role", "default_model_input_allowed"],
        "Observation-column roles",
    )

    if len(missense) != EXPECTED_MISSENSE_COUNT:
        raise TaskDefinitionError(
            f"Unexpected missense count: {len(missense)} != {EXPECTED_MISSENSE_COUNT}"
        )
    if len(reported) != EXPECTED_REPORTED_PAIR_COUNT:
        raise TaskDefinitionError(
            "Unexpected Chen 2024 reported-pair count: "
            f"{len(reported)} != {EXPECTED_REPORTED_PAIR_COUNT}"
        )
    if not missense["sequence_task_eligible"].astype(bool).all():
        raise TaskDefinitionError(
            "defined DMS tasks require sequence-valid missense observations"
        )
    if (
        not reported["pair_scope"]
        .eq("reported_comparison_positions_not_full_alignment")
        .all()
    ):
        raise TaskDefinitionError(
            "Chen 2024 comparison rows must not be relabelled as a full "
            "homologue alignment"
        )
    if not reported["independence_status"].eq("not_independent_comparison_table").all():
        raise TaskDefinitionError(
            "Chen 2024 comparison rows must remain marked as non-independent evidence"
        )

    _require_columns(
        homologue,
        [
            "homologue_position_id",
            "pair_source",
            "pair_scope",
            "mapping_uses_outcome_values",
            "ndm_assay_position",
            "ndm_residue",
            "vim_assay_position",
            "vim_residue",
            "same_wild_type_residue",
            "position_class",
        ],
        "Full homologue correspondence",
    )
    if len(homologue) != EXPECTED_FULL_HOMOLOGUE_PAIR_COUNT:
        raise TaskDefinitionError(
            f"Unexpected full homologue pair count: {len(homologue)}"
        )
    if homologue["mapping_uses_outcome_values"].astype(bool).any():
        raise TaskDefinitionError(
            "Homologue correspondence must not use mutation-effect outcomes"
        )
    shared_homologue = homologue[homologue["same_wild_type_residue"].astype(bool)]
    if len(shared_homologue) != EXPECTED_SHARED_WT_HOMOLOGUE_PAIR_COUNT:
        raise TaskDefinitionError(
            f"Unexpected shared-WT homologue count: {len(shared_homologue)}"
        )

    forbidden_default_roles = {
        "outcome",
        "grouping",
        "evaluation_only",
        "provenance_quality",
    }
    forbidden_enabled = roles[
        roles["role"].isin(forbidden_default_roles)
        & roles["default_model_input_allowed"].astype(bool)
    ]
    if not forbidden_enabled.empty:
        raise TaskDefinitionError(
            "Processed-table column roles permit unsafe default model inputs: "
            f"{forbidden_enabled['column_name'].tolist()}"
        )

    _validate_definitions()

    task_frame = pd.DataFrame(TASK_DEFINITIONS)
    scope_rows = [_scope_summary(scope, missense, reported) for scope in TASK_SCOPES]
    scope_frame = pd.DataFrame(scope_rows)

    for row in scope_frame.itertuples(index=False):
        key = (str(row.task_id), str(row.scope_role))
        expected = EXPECTED_SCOPE_COUNTS[key]
        if int(row.row_count) != expected:
            raise TaskDefinitionError(
                f"Unexpected scope count for {key}: {row.row_count} != {expected}"
            )

    evaluation_frame = pd.DataFrame(EVALUATION_REGIMES)

    vim_2020 = missense[missense["assay_id"].eq("vim2_2020_dms")]
    vim_2022 = missense[missense["assay_id"].eq("vim2_2022_amp_concentration_dms")]
    shared_mutation_groups = set(vim_2020["mutation_group_id"]).intersection(
        vim_2022["mutation_group_id"]
    )
    if (
        len(shared_mutation_groups)
        != EXPECTED_VIM_CROSS_STUDY_SHARED_MUTATION_GROUP_COUNT
    ):
        raise TaskDefinitionError(
            "Unexpected shared VIM-2 mutation-group count across Chen 2020/2022: "
            f"{len(shared_mutation_groups)} != "
            f"{EXPECTED_VIM_CROSS_STUDY_SHARED_MUTATION_GROUP_COUNT}"
        )

    ndm_headline = _filter_observations(
        missense,
        {
            "assay_id": "ndm1_2024_dms",
            "protein_id": "NDM-1",
            "antibiotic": "AMP",
            "concentration": "256",
            "temperature_c": "37",
        },
    )
    vim_headline = _filter_observations(
        missense,
        {
            "assay_id": "vim2_2020_dms",
            "protein_id": "VIM-2",
            "antibiotic": "AMP",
            "concentration": "128",
            "temperature_c": "37",
        },
    )
    pair_basis = shared_homologue[
        [
            "homologue_position_id",
            "ndm_assay_position",
            "vim_assay_position",
            "ndm_residue",
        ]
    ].rename(columns={"ndm_residue": "wild_type"})
    ndm_pairs = pair_basis.merge(
        ndm_headline[["observation_id", "assay_position", "wild_type", "mutant"]],
        left_on=["ndm_assay_position", "wild_type"],
        right_on=["assay_position", "wild_type"],
        how="inner",
        validate="one_to_many",
    ).rename(
        columns={
            "observation_id": "ndm_observation_id",
            "mutant": "mutant",
        }
    )
    aligned_pairs = ndm_pairs.merge(
        vim_headline[["observation_id", "assay_position", "wild_type", "mutant"]],
        left_on=["vim_assay_position", "wild_type", "mutant"],
        right_on=["assay_position", "wild_type", "mutant"],
        how="inner",
        validate="one_to_one",
        suffixes=("_ndm", "_vim"),
    ).rename(columns={"observation_id": "vim_observation_id"})
    aligned_pair_rows = aligned_pairs[
        [
            "homologue_position_id",
            "ndm_assay_position",
            "vim_assay_position",
            "wild_type",
            "mutant",
            "ndm_observation_id",
            "vim_observation_id",
        ]
    ].copy()
    aligned_pair_rows["pair_scope"] = "aligned_shared_wild_type_identical_substitution"
    aligned_pair_rows["pairing_uses_outcome_values"] = False
    if len(aligned_pair_rows) != EXPECTED_ALIGNED_SHARED_WT_MUTATION_PAIR_COUNT:
        raise TaskDefinitionError(
            "Unexpected aligned shared-WT mutation-pair count: "
            f"{len(aligned_pair_rows)} != "
            f"{EXPECTED_ALIGNED_SHARED_WT_MUTATION_PAIR_COUNT}"
        )
    if aligned_pair_rows["homologue_position_id"].nunique() != (
        EXPECTED_SHARED_WT_HOMOLOGUE_PAIR_COUNT
    ):
        raise TaskDefinitionError(
            "Aligned shared-WT mutation pairs do not cover all 81 positions"
        )

    condition_rows = (
        missense[
            [
                "assay_id",
                "protein_id",
                "antibiotic",
                "concentration",
                "temperature_c",
            ]
        ]
        .drop_duplicates()
        .sort_values(["assay_id", "antibiotic", "concentration", "temperature_c"])
    )

    scope_frame["concentration"] = scope_frame["concentration"].map(
        lambda value: (
            value
            if str(value) in {"*", "study_reported_matched_setting"}
            else _normalise_number_text(value)
        )
    )
    scope_frame["temperature_c"] = scope_frame["temperature_c"].map(
        lambda value: value if str(value) == "*" else _normalise_number_text(value)
    )

    _write_csv(output / "task_definitions.csv", task_frame)
    _write_csv(output / "task_scopes.csv", scope_frame)
    _write_csv(output / "evaluation_regimes.csv", evaluation_frame)
    _write_csv(output / "available_conditions.csv", condition_rows)
    _write_csv(
        output / "aligned_shared_wild_type_mutation_pairs.csv", aligned_pair_rows
    )

    summary = {
        "status": "PASS",
        "task_count": len(task_frame),
        "headline_cross_protein_task_count": int(
            (
                task_frame["task_family"].eq("cross_protein_transfer")
                & task_frame["priority"].eq("headline")
            ).sum()
        ),
        "evaluation_regime_count": len(evaluation_frame),
        "deferred_evaluation_regime_count": int(
            evaluation_frame["status"].astype(str).str.startswith("deferred_").sum()
        ),
        "available_condition_count": len(condition_rows),
        "vim_2020_2022_shared_mutation_group_count": len(shared_mutation_groups),
        "cross_study_vim_policy": (
            "Chen 2020 and Chen 2022 remain separate task sources; shared mutations "
            "must not be treated as independent evidence when an analysis "
            "combines them"
        ),
        "score_policy": (
            "Task definition preserves score_original unchanged while score_analysis "
            "applies the reported lower DMS fitness floor on each "
            "source-assay scale; "
            "defines no "
            "global harmonisation"
        ),
        "representation_comparison_policy": (
            "sequence models may use all task-eligible missense rows; "
            "experimental-structure models require "
            "experimental_structure_task_eligible; AlphaFold analyses require "
            "alphafold_structure_task_eligible; modality comparisons use the same "
            "eligible "
            "rows"
        ),
        "cross_protein_model_selection_policy": (
            "for headline transfer, target-protein observations remain absent from "
            "model "
            "selection; source-protein validation is defined before target evaluation"
        ),
        "homologue_correspondence_policy": (
            "the complete Chen 2024 aligned-position source provides 217 position "
            "pairs; direct equivalent-mutation reporting uses only the 81 shared-WT "
            "positions and pairs observations without consulting score values"
        ),
        "aligned_shared_wild_type_mutation_pair_count": int(len(aligned_pair_rows)),
        "reported_pair_policy": (
            "the 132 Chen 2024 WT-reversion rows remain outcome-derived supporting "
            "evidence and are never used to define model-selection populations"
        ),
        "random_row_split_policy": (
            "no random-row headline evaluation is defined in the task definitions"
        ),
        "partition_status": (
            "no train, validation, or test assignment is created here; "
            "evaluation-partition preparation generates and audits "
            "partitions from these task definitions"
        ),
        "outputs": [
            "task_definitions.csv",
            "task_scopes.csv",
            "evaluation_regimes.csv",
            "available_conditions.csv",
            "aligned_shared_wild_type_mutation_pairs.csv",
            "task_definition_summary.json",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "task_definition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
