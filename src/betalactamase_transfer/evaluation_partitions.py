# Define grouped evaluation partitions for mutation-effect analyses.

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


class EvaluationPartitionError(RuntimeError):
    """Raised when evaluation partitions cannot be prepared safely."""


OUTER_FOLD_COUNT = 5
SOURCE_CV_FOLD_COUNT = 5
INNER_VALIDATION_FOLD_COUNT = 5
PARTITION_METHOD = "sha256_sorted_round_robin"
PARTITION_SEED = "beta_lactamase_evaluation_partitions"
EXPECTED_MISSENSE_COUNT = 125_681
EXPECTED_TEMPERATURE_SHARED_MUTATION_GROUP_COUNT = 5_016
EXPECTED_VIM_2020_MUTATION_GROUP_COUNT_37C = 5_023
EXPECTED_VIM_2022_MUTATION_GROUP_COUNT = 5_015
EXPECTED_VIM_2022_COMPLETE_TRAJECTORY_MUTATION_GROUP_COUNT = 5_000


SCHEME_COLUMNS = [
    "scheme_id",
    "task_id",
    "regime_id",
    "priority",
    "status",
    "assignment_family",
    "group_type",
    "outer_fold_count",
    "source_cv_fold_count",
    "held_out_dimension",
    "held_out_value",
    "held_out_class",
    "held_out_labels_used_for_model_selection",
    "assignment_uses_outcome_values",
    "model_selection_grouping_key",
    "model_selection_rule",
    "final_evaluation_rule",
    "scientific_claim",
]

SCOPE_COLUMNS = [
    "scheme_id",
    "scope_role",
    "assay_id",
    "protein_id",
    "antibiotics",
    "concentrations",
    "temperatures_c",
    "mutation_population_policy",
    "row_count",
    "mutation_group_count",
    "position_group_count",
    "condition_count",
    "experimental_structure_eligible_count",
    "alphafold_structure_eligible_count",
]

ASSIGNMENT_COLUMNS = [
    "scheme_id",
    "task_id",
    "iteration",
    "group_type",
    "group_id",
    "split_role",
    "cv_fold",
]

AUDIT_COLUMNS = [
    "scheme_id",
    "iteration",
    "split_role",
    "cv_fold",
    "row_count",
    "group_count",
    "position_count",
    "condition_count",
    "experimental_structure_eligible_count",
    "alphafold_structure_eligible_count",
    "score_min",
    "score_median",
    "score_max",
]


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hash_key(seed: str, value: str) -> str:
    payload = f"{seed}|{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def _balanced_fold_map(
    group_ids: Iterable[str], *, fold_count: int, seed: str
) -> dict[str, int]:
    unique = sorted({str(group_id) for group_id in group_ids})
    if len(unique) < fold_count:
        raise EvaluationPartitionError(
            f"Cannot assign {len(unique)} groups across {fold_count} folds"
        )
    ordered = sorted(unique, key=lambda value: (_hash_key(seed, value), value))
    return {group_id: index % fold_count for index, group_id in enumerate(ordered)}


def _nested_assignments(
    *, scheme_id: str, task_id: str, group_type: str, group_ids: Iterable[str]
) -> list[dict[str, Any]]:
    groups = sorted({str(group_id) for group_id in group_ids})
    outer = _balanced_fold_map(
        groups,
        fold_count=OUTER_FOLD_COUNT,
        seed=f"{PARTITION_SEED}|{scheme_id}|outer",
    )
    rows: list[dict[str, Any]] = []
    for outer_fold in range(OUTER_FOLD_COUNT):
        test_groups = {
            group_id for group_id, fold in outer.items() if fold == outer_fold
        }
        remaining = [group_id for group_id in groups if group_id not in test_groups]
        inner = _balanced_fold_map(
            remaining,
            fold_count=INNER_VALIDATION_FOLD_COUNT,
            seed=f"{PARTITION_SEED}|{scheme_id}|outer={outer_fold}|inner",
        )
        validation_groups = {group_id for group_id, fold in inner.items() if fold == 0}
        for group_id in groups:
            if group_id in test_groups:
                split_role = "test"
            elif group_id in validation_groups:
                split_role = "validation"
            else:
                split_role = "train"
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "task_id": task_id,
                    "iteration": outer_fold,
                    "group_type": group_type,
                    "group_id": group_id,
                    "split_role": split_role,
                    "cv_fold": "",
                }
            )
    return rows


def _source_cv_assignments(
    *,
    scheme_id: str,
    task_id: str,
    group_type: str,
    source_group_ids: Iterable[str],
    target_group_ids: Iterable[str],
    fold_universe_group_ids: Iterable[str],
    grouping_key: str,
) -> list[dict[str, Any]]:
    source_groups = sorted({str(group_id) for group_id in source_group_ids})
    target_groups = sorted({str(group_id) for group_id in target_group_ids})
    fold_universe = sorted({str(group_id) for group_id in fold_universe_group_ids})
    if not set(source_groups).issubset(set(fold_universe)):
        raise EvaluationPartitionError(
            f"{scheme_id} source groups exceed the model-selection fold universe"
        )
    universe_folds = _balanced_fold_map(
        fold_universe,
        fold_count=SOURCE_CV_FOLD_COUNT,
        seed=f"{PARTITION_SEED}|{grouping_key}|source_cv",
    )
    folds = {group_id: universe_folds[group_id] for group_id in source_groups}
    rows: list[dict[str, Any]] = []
    for group_id in source_groups:
        rows.append(
            {
                "scheme_id": scheme_id,
                "task_id": task_id,
                "iteration": "",
                "group_type": group_type,
                "group_id": group_id,
                "split_role": "source_model_selection",
                "cv_fold": folds[group_id],
            }
        )
    for group_id in target_groups:
        rows.append(
            {
                "scheme_id": scheme_id,
                "task_id": task_id,
                "iteration": "",
                "group_type": group_type,
                "group_id": group_id,
                "split_role": "final_test",
                "cv_fold": "",
            }
        )
    return rows


def _condition_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(
        frame[["antibiotic", "concentration", "temperature_c"]]
        .drop_duplicates()
        .shape[0]
    )


def _values(values: pd.Series) -> str:
    items = sorted(values.dropna().unique().tolist())
    return ";".join(str(value) for value in items)


def _scope_row(
    *,
    scheme_id: str,
    scope_role: str,
    frame: pd.DataFrame,
    mutation_population_policy: str,
) -> dict[str, Any]:
    if frame.empty:
        raise EvaluationPartitionError(
            f"Empty partition scope: {scheme_id}/{scope_role}"
        )
    return {
        "scheme_id": scheme_id,
        "scope_role": scope_role,
        "assay_id": _values(frame["assay_id"]),
        "protein_id": _values(frame["protein_id"]),
        "antibiotics": _values(frame["antibiotic"]),
        "concentrations": _values(frame["concentration"]),
        "temperatures_c": _values(frame["temperature_c"]),
        "mutation_population_policy": mutation_population_policy,
        "row_count": len(frame),
        "mutation_group_count": frame["mutation_group_id"].nunique(),
        "position_group_count": frame["position_group_id"].nunique(),
        "condition_count": _condition_count(frame),
        "experimental_structure_eligible_count": int(
            frame["experimental_structure_task_eligible"].sum()
        ),
        "alphafold_structure_eligible_count": int(
            frame["alphafold_structure_task_eligible"].sum()
        ),
    }


def _score_summary(frame: pd.DataFrame) -> tuple[Any, Any, Any]:
    if frame.empty:
        return "", "", ""
    score = frame["score_original"]
    return float(score.min()), float(score.median()), float(score.max())


def _audit_row(
    *,
    scheme_id: str,
    iteration: Any,
    split_role: str,
    cv_fold: Any,
    frame: pd.DataFrame,
    group_type: str,
) -> dict[str, Any]:
    score_min, score_median, score_max = _score_summary(frame)
    return {
        "scheme_id": scheme_id,
        "iteration": iteration,
        "split_role": split_role,
        "cv_fold": cv_fold,
        "row_count": len(frame),
        "group_count": frame[group_type].nunique(),
        "position_count": frame["position_group_id"].nunique(),
        "condition_count": _condition_count(frame),
        "experimental_structure_eligible_count": int(
            frame["experimental_structure_task_eligible"].sum()
        ),
        "alphafold_structure_eligible_count": int(
            frame["alphafold_structure_task_eligible"].sum()
        ),
        "score_min": score_min,
        "score_median": score_median,
        "score_max": score_max,
    }


def _apply_nested_audit(
    *,
    scheme_id: str,
    frame: pd.DataFrame,
    group_type: str,
    assignments: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = assignments[assignments["scheme_id"].eq(scheme_id)]
    for iteration in range(OUTER_FOLD_COUNT):
        iteration_assignments = subset[subset["iteration"].eq(iteration)]
        for split_role in ("train", "validation", "test"):
            group_ids = set(
                iteration_assignments.loc[
                    iteration_assignments["split_role"].eq(split_role), "group_id"
                ]
            )
            part = frame[frame[group_type].isin(group_ids)]
            rows.append(
                _audit_row(
                    scheme_id=scheme_id,
                    iteration=iteration,
                    split_role=split_role,
                    cv_fold="",
                    frame=part,
                    group_type=group_type,
                )
            )
    return rows


def _apply_source_cv_audit(
    *,
    scheme_id: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    group_type: str,
    assignments: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = assignments[
        assignments["scheme_id"].eq(scheme_id)
        & assignments["split_role"].eq("source_model_selection")
    ]
    for cv_fold in range(SOURCE_CV_FOLD_COUNT):
        group_ids = set(subset.loc[subset["cv_fold"].eq(cv_fold), "group_id"])
        part = source[source[group_type].isin(group_ids)]
        rows.append(
            _audit_row(
                scheme_id=scheme_id,
                iteration="",
                split_role="source_cv_validation",
                cv_fold=cv_fold,
                frame=part,
                group_type=group_type,
            )
        )
    rows.append(
        _audit_row(
            scheme_id=scheme_id,
            iteration="",
            split_role="final_test",
            cv_fold="",
            frame=target,
            group_type=group_type,
        )
    )
    return rows


def _assert_group_partition_integrity(
    *,
    scheme_id: str,
    assignments: pd.DataFrame,
    expected_groups: set[str],
) -> None:
    subset = assignments[assignments["scheme_id"].eq(scheme_id)]
    for iteration in range(OUTER_FOLD_COUNT):
        current = subset[subset["iteration"].eq(iteration)]
        observed = set(current["group_id"])
        if observed != expected_groups:
            raise EvaluationPartitionError(
                f"{scheme_id} iteration {iteration} does not cover every group"
            )
        duplicate_count = current["group_id"].duplicated().sum()
        if duplicate_count:
            raise EvaluationPartitionError(
                f"{scheme_id} iteration {iteration} assigns a group more than once"
            )
        role_sets = {
            role: set(current.loc[current["split_role"].eq(role), "group_id"])
            for role in ("train", "validation", "test")
        }
        if role_sets["train"] & role_sets["validation"]:
            raise EvaluationPartitionError(f"{scheme_id} train/validation overlap")
        if role_sets["train"] & role_sets["test"]:
            raise EvaluationPartitionError(f"{scheme_id} train/test overlap")
        if role_sets["validation"] & role_sets["test"]:
            raise EvaluationPartitionError(f"{scheme_id} validation/test overlap")

    test_counts = (
        subset[subset["split_role"].eq("test")].groupby("group_id").size().to_dict()
    )
    if set(test_counts) != expected_groups or set(test_counts.values()) != {1}:
        raise EvaluationPartitionError(
            f"{scheme_id} must evaluate every group exactly once across outer folds"
        )


def _validate_task_definitions(root: Path) -> dict[str, Any]:
    task_dir = root / "data/interim/task_definitions"
    summary_path = task_dir / "task_definition_summary.json"
    regimes_path = task_dir / "evaluation_regimes.csv"
    if not summary_path.exists() or not regimes_path.exists():
        raise EvaluationPartitionError("task definitions are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise EvaluationPartitionError("task definitions are not in PASS state")
    regimes = pd.read_csv(regimes_path)
    required = {
        "position_grouped_within_protein",
        "cross_protein_source_only_validation",
        "leave_one_antibiotic_out_vim2_2020",
        "directional_temperature_transfer_vim2_amp",
        "mutation_grouped_vim2_2022_concentration",
        "leave_one_concentration_out_vim2_2022",
        "reported_wt_reversion_paired_analysis",
        "aligned_shared_wild_type_transfer_reporting",
    }
    if set(regimes["regime_id"]) != required:
        raise EvaluationPartitionError(
            "Unexpected task-definition evaluation-regime set"
        )
    aligned_reporting = regimes[
        regimes["regime_id"].eq("aligned_shared_wild_type_transfer_reporting")
    ]
    if len(aligned_reporting) != 1 or aligned_reporting.iloc[0]["status"] != (
        "ready_for_reporting_population"
    ):
        raise EvaluationPartitionError(
            "Aligned shared-WT transfer reporting is not ready"
        )
    return summary


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise EvaluationPartitionError(f"Missing required columns: {missing}")


def prepare_evaluation_partitions(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    task_definition_summary = _validate_task_definitions(root)
    missense_path = root / "data/processed/missense_observations.csv"
    if not missense_path.exists():
        raise EvaluationPartitionError("processed missense observations are required")
    missense = pd.read_csv(missense_path, low_memory=False)
    if len(missense) != EXPECTED_MISSENSE_COUNT:
        raise EvaluationPartitionError(
            f"Expected {EXPECTED_MISSENSE_COUNT} missense rows; found {len(missense)}"
        )
    _require_columns(
        missense,
        [
            "assay_id",
            "protein_id",
            "antibiotic",
            "concentration",
            "temperature_c",
            "score_original",
            "mutation_group_id",
            "position_group_id",
            "experimental_structure_task_eligible",
            "alphafold_structure_task_eligible",
        ],
    )

    schemes: list[dict[str, Any]] = []
    scopes: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame | None, str, str]] = {}

    def add_nested_scheme(
        *,
        scheme_id: str,
        task_id: str,
        regime_id: str,
        priority: str,
        frame: pd.DataFrame,
        group_type: str,
        scientific_claim: str,
    ) -> None:
        schemes.append(
            {
                "scheme_id": scheme_id,
                "task_id": task_id,
                "regime_id": regime_id,
                "priority": priority,
                "status": "ready",
                "assignment_family": "five_outer_folds_with_grouped_inner_validation",
                "group_type": group_type,
                "outer_fold_count": OUTER_FOLD_COUNT,
                "source_cv_fold_count": "",
                "held_out_dimension": group_type,
                "held_out_value": "rotating_outer_fold",
                "held_out_class": "unseen_group",
                "held_out_labels_used_for_model_selection": False,
                "assignment_uses_outcome_values": False,
                "model_selection_grouping_key": scheme_id,
                "model_selection_rule": (
                    "for each outer fold, validation groups are selected only from "
                    "the remaining groups; outer-test labels are not used"
                ),
                "final_evaluation_rule": (
                    "report performance across the five outer test folds; every "
                    "group is tested exactly once"
                ),
                "scientific_claim": scientific_claim,
            }
        )
        scopes.append(
            _scope_row(
                scheme_id=scheme_id,
                scope_role="analysis",
                frame=frame,
                mutation_population_policy="all_validated_task_rows",
            )
        )
        assignments.extend(
            _nested_assignments(
                scheme_id=scheme_id,
                task_id=task_id,
                group_type=group_type,
                group_ids=frame[group_type],
            )
        )
        frames[scheme_id] = (frame, None, group_type, "nested")

    def add_source_cv_scheme(
        *,
        scheme_id: str,
        task_id: str,
        regime_id: str,
        priority: str,
        source: pd.DataFrame,
        target: pd.DataFrame,
        group_type: str,
        held_out_dimension: str,
        held_out_value: str,
        held_out_class: str,
        scientific_claim: str,
        population_policy: str,
        model_selection_grouping_key: str,
        fold_universe_group_ids: Iterable[str],
    ) -> None:
        schemes.append(
            {
                "scheme_id": scheme_id,
                "task_id": task_id,
                "regime_id": regime_id,
                "priority": priority,
                "status": "ready",
                "assignment_family": "source_only_grouped_cv_then_final_test",
                "group_type": group_type,
                "outer_fold_count": "",
                "source_cv_fold_count": SOURCE_CV_FOLD_COUNT,
                "held_out_dimension": held_out_dimension,
                "held_out_value": held_out_value,
                "held_out_class": held_out_class,
                "held_out_labels_used_for_model_selection": False,
                "assignment_uses_outcome_values": False,
                "model_selection_grouping_key": model_selection_grouping_key,
                "model_selection_rule": (
                    "select model settings using five grouped folds from source "
                    "rows only; final-test labels are unavailable"
                ),
                "final_evaluation_rule": (
                    "after model settings are fixed, refit on all source rows and "
                    "evaluate the held-out target once"
                ),
                "scientific_claim": scientific_claim,
            }
        )
        scopes.extend(
            [
                _scope_row(
                    scheme_id=scheme_id,
                    scope_role="source_model_selection",
                    frame=source,
                    mutation_population_policy=population_policy,
                ),
                _scope_row(
                    scheme_id=scheme_id,
                    scope_role="final_test",
                    frame=target,
                    mutation_population_policy=population_policy,
                ),
            ]
        )
        assignments.extend(
            _source_cv_assignments(
                scheme_id=scheme_id,
                task_id=task_id,
                group_type=group_type,
                source_group_ids=source[group_type],
                target_group_ids=target[group_type],
                fold_universe_group_ids=fold_universe_group_ids,
                grouping_key=model_selection_grouping_key,
            )
        )
        frames[scheme_id] = (source, target, group_type, "source_cv")

    ndm = missense[missense["assay_id"].eq("ndm1_2024_dms")].copy()
    vim2020 = missense[missense["assay_id"].eq("vim2_2020_dms")].copy()
    vim2022 = missense[
        missense["assay_id"].eq("vim2_2022_amp_concentration_dms")
    ].copy()

    add_nested_scheme(
        scheme_id="ndm1_position_nested5",
        task_id="ndm1_within_position_regression",
        regime_id="position_grouped_within_protein",
        priority="primary",
        frame=ndm,
        group_type="position_group_id",
        scientific_claim="within-NDM-1 generalisation to residue positions not fitted",
    )
    add_nested_scheme(
        scheme_id="vim2_2020_position_nested5",
        task_id="vim2_2020_within_position_regression",
        regime_id="position_grouped_within_protein",
        priority="primary",
        frame=vim2020,
        group_type="position_group_id",
        scientific_claim="within-VIM-2 generalisation to residue positions not fitted",
    )

    ndm_amp = ndm[
        ndm["antibiotic"].eq("AMP")
        & ndm["concentration"].eq(256.0)
        & ndm["temperature_c"].eq(37.0)
    ].copy()
    vim_amp = vim2020[
        vim2020["antibiotic"].eq("AMP")
        & vim2020["concentration"].eq(128.0)
        & vim2020["temperature_c"].eq(37.0)
    ].copy()
    add_source_cv_scheme(
        scheme_id="ndm1_to_vim2_amp_source_cv5",
        task_id="ndm1_to_vim2_amp_transfer",
        regime_id="cross_protein_source_only_validation",
        priority="headline",
        source=ndm_amp,
        target=vim_amp,
        group_type="position_group_id",
        held_out_dimension="protein_background",
        held_out_value="VIM-2",
        held_out_class="cross_protein",
        scientific_claim="NDM-1 to VIM-2 transfer with target labels kept blind",
        population_policy="all_task_rows_per_protein_no_cross_protein_pairing_required",
        model_selection_grouping_key="ndm1_amp_position_cv5",
        fold_universe_group_ids=ndm_amp["position_group_id"],
    )
    add_source_cv_scheme(
        scheme_id="vim2_to_ndm1_amp_source_cv5",
        task_id="vim2_to_ndm1_amp_transfer",
        regime_id="cross_protein_source_only_validation",
        priority="headline",
        source=vim_amp,
        target=ndm_amp,
        group_type="position_group_id",
        held_out_dimension="protein_background",
        held_out_value="NDM-1",
        held_out_class="cross_protein",
        scientific_claim="VIM-2 to NDM-1 transfer with target labels kept blind",
        population_policy="all_task_rows_per_protein_no_cross_protein_pairing_required",
        model_selection_grouping_key="vim2_amp_position_cv5",
        fold_universe_group_ids=vim_amp["position_group_id"],
    )

    high_screening_conditions = {
        "AMP": 128.0,
        "CTX": 4.0,
        "MEM": 0.031,
    }
    vim37_high = vim2020[vim2020["temperature_c"].eq(37.0)].copy()
    vim37_high = vim37_high[
        vim37_high.apply(
            lambda row: (
                high_screening_conditions.get(row["antibiotic"]) == row["concentration"]
            ),
            axis=1,
        )
    ].copy()
    mutation_sets_37 = {
        antibiotic: set(group["mutation_group_id"])
        for antibiotic, group in vim37_high.groupby("antibiotic")
    }
    if set(mutation_sets_37) != {"AMP", "CTX", "MEM"}:
        raise EvaluationPartitionError("Unexpected VIM-2 37 C antibiotic set")
    common_37 = set.intersection(*mutation_sets_37.values())
    if len(common_37) != EXPECTED_VIM_2020_MUTATION_GROUP_COUNT_37C:
        raise EvaluationPartitionError("VIM-2 37 C antibiotic mutation sets differ")
    for antibiotic in ("AMP", "CTX", "MEM"):
        source = vim37_high[
            vim37_high["antibiotic"].ne(antibiotic)
            & vim37_high["mutation_group_id"].isin(common_37)
        ].copy()
        target = vim37_high[
            vim37_high["antibiotic"].eq(antibiotic)
            & vim37_high["mutation_group_id"].isin(common_37)
        ].copy()
        add_source_cv_scheme(
            scheme_id=f"vim2_2020_loo_antibiotic_{antibiotic.lower()}",
            task_id="vim2_2020_condition_transfer",
            regime_id="leave_one_antibiotic_out_vim2_2020",
            priority="secondary",
            source=source,
            target=target,
            group_type="mutation_group_id",
            held_out_dimension="antibiotic",
            held_out_value=antibiotic,
            held_out_class="antibiotic_transfer_37C",
            scientific_claim=(
                f"VIM-2 transfer to held-out {antibiotic} at the Chen-2020 "
                "high screening condition for mutations observed under the "
                "source antibiotics"
            ),
            population_policy=(
                "intersection_of_mutations_across_37C_high_screening_conditions"
            ),
            model_selection_grouping_key="vim2_2020_antibiotic_high_mutation_cv5",
            fold_universe_group_ids=common_37,
        )

    common_amp_concentrations = {2.0, 16.0, 128.0}
    amp25 = vim2020[
        vim2020["antibiotic"].eq("AMP")
        & vim2020["temperature_c"].eq(25.0)
        & vim2020["concentration"].isin(common_amp_concentrations)
    ].copy()
    amp37 = vim2020[
        vim2020["antibiotic"].eq("AMP")
        & vim2020["temperature_c"].eq(37.0)
        & vim2020["concentration"].isin(common_amp_concentrations)
    ].copy()
    amp25_mutations = set(amp25["mutation_group_id"])
    amp37_mutations = set(amp37["mutation_group_id"])
    common_temperature_mutations = amp25_mutations.intersection(amp37_mutations)
    temperature_25_only_count = len(amp25_mutations - amp37_mutations)
    temperature_37_only_count = len(amp37_mutations - amp25_mutations)
    if (
        len(common_temperature_mutations)
        != EXPECTED_TEMPERATURE_SHARED_MUTATION_GROUP_COUNT
    ):
        raise EvaluationPartitionError(
            "Unexpected VIM-2 mutation overlap between 25 C and 37 C AMP data"
        )
    amp25 = amp25[amp25["mutation_group_id"].isin(common_temperature_mutations)]
    amp37 = amp37[amp37["mutation_group_id"].isin(common_temperature_mutations)]
    for source_temp, target_temp, source, target in (
        (25, 37, amp25, amp37),
        (37, 25, amp37, amp25),
    ):
        add_source_cv_scheme(
            scheme_id=f"vim2_2020_temperature_{source_temp}_to_{target_temp}",
            task_id="vim2_2020_condition_transfer",
            regime_id="directional_temperature_transfer_vim2_amp",
            priority="secondary",
            source=source,
            target=target,
            group_type="mutation_group_id",
            held_out_dimension="temperature_c",
            held_out_value=str(target_temp),
            held_out_class="temperature_transfer_shared_mutations",
            scientific_claim=(
                f"VIM-2 AMP transfer from {source_temp} C to {target_temp} C "
                "using mutations measured at both temperatures"
            ),
            population_policy=(
                "intersection_of_mutations_at_25C_and_37C_for_AMP_2_16_128"
            ),
            model_selection_grouping_key="vim2_2020_temperature_shared_mutation_cv5",
            fold_universe_group_ids=common_temperature_mutations,
        )

    if vim2022["mutation_group_id"].nunique() != EXPECTED_VIM_2022_MUTATION_GROUP_COUNT:
        raise EvaluationPartitionError("Unexpected Chen 2022 mutation-group count")
    add_nested_scheme(
        scheme_id="vim2_2022_mutation_nested5",
        task_id="vim2_2022_concentration_response",
        regime_id="mutation_grouped_vim2_2022_concentration",
        priority="secondary",
        frame=vim2022,
        group_type="mutation_group_id",
        scientific_claim=(
            "VIM-2 concentration-response prediction for mutations whose entire "
            "concentration trajectories were absent from fitting"
        ),
    )

    concentration_values = sorted(vim2022["concentration"].unique().tolist())
    if concentration_values != [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]:
        raise EvaluationPartitionError("Unexpected Chen 2022 concentration set")
    complete_trajectory_count = (
        vim2022.groupby("mutation_group_id")["concentration"].nunique().eq(7).sum()
    )
    if (
        complete_trajectory_count
        != EXPECTED_VIM_2022_COMPLETE_TRAJECTORY_MUTATION_GROUP_COUNT
    ):
        raise EvaluationPartitionError(
            "Unexpected complete concentration-trajectory count"
        )
    for concentration in concentration_values:
        target = vim2022[vim2022["concentration"].eq(concentration)].copy()
        target_mutations = set(target["mutation_group_id"])
        source = vim2022[
            vim2022["concentration"].ne(concentration)
            & vim2022["mutation_group_id"].isin(target_mutations)
        ].copy()
        if not target_mutations.issubset(set(source["mutation_group_id"])):
            raise EvaluationPartitionError(
                "Held-out concentration "
                f"{concentration} contains source-unseen mutations"
            )
        holdout_class = (
            "edge_extrapolation" if concentration in {4.0, 256.0} else "interpolation"
        )
        value_label = str(int(concentration))
        add_source_cv_scheme(
            scheme_id=f"vim2_2022_loo_concentration_{value_label}",
            task_id="vim2_2022_concentration_response",
            regime_id="leave_one_concentration_out_vim2_2022",
            priority="secondary",
            source=source,
            target=target,
            group_type="mutation_group_id",
            held_out_dimension="ampicillin_concentration_ug_per_ml",
            held_out_value=value_label,
            held_out_class=holdout_class,
            scientific_claim=(
                f"VIM-2 transfer to held-out AMP concentration {value_label} ug/mL "
                "for mutations observed at other concentrations"
            ),
            population_policy="target_mutations_only_in_source_and_target",
            model_selection_grouping_key="vim2_2022_concentration_mutation_cv5",
            fold_universe_group_ids=set(vim2022["mutation_group_id"]),
        )

    scheme_frame = pd.DataFrame(schemes, columns=SCHEME_COLUMNS)
    assignment_frame = pd.DataFrame(assignments, columns=ASSIGNMENT_COLUMNS)
    if scheme_frame["scheme_id"].duplicated().any():
        raise EvaluationPartitionError("Duplicate partition scheme ID")

    for scheme_id, (frame, target, group_type, family) in frames.items():
        if family == "nested":
            _assert_group_partition_integrity(
                scheme_id=scheme_id,
                assignments=assignment_frame,
                expected_groups=set(frame[group_type]),
            )
        else:
            subset = assignment_frame[assignment_frame["scheme_id"].eq(scheme_id)]
            source_assignments = subset[
                subset["split_role"].eq("source_model_selection")
            ]
            target_assignments = subset[subset["split_role"].eq("final_test")]
            if source_assignments["group_id"].duplicated().any():
                raise EvaluationPartitionError(
                    f"{scheme_id} has duplicate source model-selection groups"
                )
            if target_assignments["group_id"].duplicated().any():
                raise EvaluationPartitionError(
                    f"{scheme_id} has duplicate target groups"
                )
            if set(source_assignments["cv_fold"].astype(int)) != set(
                range(SOURCE_CV_FOLD_COUNT)
            ):
                raise EvaluationPartitionError(f"{scheme_id} lacks a source CV fold")
            if target is None:
                raise EvaluationPartitionError(f"{scheme_id} lacks target data")

    audit_rows: list[dict[str, Any]] = []
    for scheme_id, (frame, target, group_type, family) in frames.items():
        if family == "nested":
            audit_rows.extend(
                _apply_nested_audit(
                    scheme_id=scheme_id,
                    frame=frame,
                    group_type=group_type,
                    assignments=assignment_frame,
                )
            )
        else:
            if target is None:
                raise EvaluationPartitionError(f"{scheme_id} lacks target data")
            audit_rows.extend(
                _apply_source_cv_audit(
                    scheme_id=scheme_id,
                    source=frame,
                    target=target,
                    group_type=group_type,
                    assignments=assignment_frame,
                )
            )
    audit_frame = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)

    if scheme_frame["assignment_uses_outcome_values"].any():
        raise EvaluationPartitionError(
            "Partition assignment must not use outcome values"
        )
    if scheme_frame["held_out_labels_used_for_model_selection"].any():
        raise EvaluationPartitionError("Held-out labels must remain unavailable")

    output_dir = root / "data/interim/evaluation_partitions"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "partition_schemes.csv", schemes, SCHEME_COLUMNS)
    _write_csv(output_dir / "partition_scopes.csv", scopes, SCOPE_COLUMNS)
    _write_csv(
        output_dir / "group_assignments.csv",
        assignments,
        ASSIGNMENT_COLUMNS,
    )
    _write_csv(output_dir / "partition_audit.csv", audit_rows, AUDIT_COLUMNS)

    summary = {
        "status": "PASS",
        "partition_method": PARTITION_METHOD,
        "partition_seed": PARTITION_SEED,
        "scheme_count": len(scheme_frame),
        "nested_group_scheme_count": int(
            scheme_frame["assignment_family"]
            .eq("five_outer_folds_with_grouped_inner_validation")
            .sum()
        ),
        "source_cv_final_test_scheme_count": int(
            scheme_frame["assignment_family"]
            .eq("source_only_grouped_cv_then_final_test")
            .sum()
        ),
        "group_assignment_row_count": len(assignment_frame),
        "audit_row_count": len(audit_frame),
        "temperature_shared_mutation_group_count": len(common_temperature_mutations),
        "temperature_25C_only_mutation_group_count": temperature_25_only_count,
        "temperature_37C_only_mutation_group_count": temperature_37_only_count,
        "vim_2020_37C_common_mutation_group_count": len(common_37),
        "vim_2022_complete_trajectory_mutation_group_count": int(
            complete_trajectory_count
        ),
        "vim_2022_incomplete_trajectory_mutation_group_count": int(
            EXPECTED_VIM_2022_MUTATION_GROUP_COUNT - complete_trajectory_count
        ),
        "target_label_policy": (
            "held-out and cross-protein target labels are not used for model "
            "selection; target sequence or structure may be supplied at final "
            "inference when required by the representation"
        ),
        "assignment_policy": (
            "group assignments use stable SHA-256 ordering and never use "
            "mutation-effect scores"
        ),
        "temperature_policy": (
            "directional 25 C/37 C VIM-2 AMP transfer is restricted to 2, 16, "
            "and 128 ug/mL and to mutation groups measured at both temperatures"
        ),
        "antibiotic_policy": (
            "VIM-2 leave-one-antibiotic-out transfer uses the Chen-2020 high "
            "screening conditions at 37 C (AMP 128, CTX 4, MEM 0.031 ug/mL) and "
            "the mutation population shared by all three antibiotics"
        ),
        "concentration_policy": (
            "unseen-mutation evaluation keeps complete available trajectories by "
            "mutation group; concentration holdouts use only mutations present in "
            "the held-out concentration and distinguish interpolation from edge "
            "extrapolation"
        ),
        "homologue_reporting_policy": (
            "the 217-position Chen 2024 correspondence is used only to define "
            "held-out reporting populations; the existing source-only cross-protein "
            "model-selection partitions are unchanged, and direct equivalent-mutation "
            "reporting is restricted to shared-WT aligned substitutions"
        ),
        "task_definition_count": task_definition_summary["task_count"],
        "outputs": [
            "partition_schemes.csv",
            "partition_scopes.csv",
            "group_assignments.csv",
            "partition_audit.csv",
            "evaluation_partition_summary.json",
        ],
    }
    (output_dir / "evaluation_partition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
