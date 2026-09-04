# Define assays, experimental conditions, and related measurements.

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Mapping

import pandas as pd


class AssayDefinitionError(RuntimeError):
    """Raised when assay-definition inputs are incomplete or internally inconsistent."""


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise AssayDefinitionError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise AssayDefinitionError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _score_registry() -> List[Mapping[str, object]]:
    common = {
        "score_direction": "higher_is_fitter_than_wild_type",
        "wild_type_reference": "wild-type-normalised enrichment ratio",
        "principal_outcome": "DMS fitness",
        "score_floor": -4.0,
        "score_floor_interpretation": "scores at or below -4 treated as nonsense-like",
        "imputation_policy": "do_not_impute_missing_scores",
    }
    return [
        {
            **common,
            "assay_id": "ndm1_2024_dms",
            "study_id": "chen_2024_natcomms",
            "protein_id": "NDM-1",
            "host_system": "Escherichia coli E. cloni 10G",
            "selection_duration_hours": 6,
            "selection_temperature_c": 37,
            "biological_replicates": 2,
            "score_definition": (
                "log2 selected/non-selected variant enrichment minus the "
                "corresponding wild-type enrichment"
            ),
            "source_support": "2024 article Methods, Eq. 1",
        },
        {
            **common,
            "assay_id": "vim2_2020_dms",
            "study_id": "chen_2020_elife",
            "protein_id": "VIM-2",
            "host_system": "Escherichia coli E. cloni 10G",
            "selection_duration_hours": 6,
            "selection_temperature_c": "condition-specific: 25 or 37",
            "biological_replicates": 2,
            "score_definition": (
                "log2 selected/non-selected variant enrichment normalised "
                "to wild-type; replicate fitness values averaged"
            ),
            "source_support": "2020 article Methods and Eq. 1",
        },
        {
            **common,
            "assay_id": "vim2_2022_amp_concentration_dms",
            "study_id": "chen_2022_natee",
            "protein_id": "VIM-2",
            "host_system": "Escherichia coli E. cloni 10G",
            "selection_duration_hours": 6,
            "selection_temperature_c": 37,
            "biological_replicates": 2,
            "score_definition": (
                "log2 selected/non-selected variant enrichment normalised "
                "to wild-type; replicate fitness values averaged"
            ),
            "source_support": (
                "Chen et al. 2022, Nature Ecology & Evolution, Methods: "
                "Calculation of fitness scores (Eq. 1)"
            ),
        },
    ]


def _condition_summary(observations: pd.DataFrame) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    keys = [
        "study_id",
        "assay_id",
        "protein_id",
        "antibiotic",
        "concentration",
        "concentration_unit",
        "temperature_c",
    ]
    for values, group in observations.groupby(keys, dropna=False):
        record = dict(zip(keys, values))
        missense = group[group["mutation_class"] == "missense"]
        wt = group[group["mutation_class"] == "wild_type_control"]
        stop = group[group["mutation_class"] == "stop_control"]
        record.update(
            {
                "observation_count": len(group),
                "missense_count": len(missense),
                "unique_missense_mutations": missense["assay_mutation"].nunique(),
                "score_min": group["score_original"].min(),
                "score_max": group["score_original"].max(),
                "score_mean": group["score_original"].mean(),
                "score_median": group["score_original"].median(),
                "score_sd": group["score_original"].std(),
                "wt_control_count": len(wt),
                "wt_control_mean": wt["score_original"].mean(),
                "wt_control_sd": wt["score_original"].std(),
                "stop_control_count": len(stop),
                "stop_control_mean": stop["score_original"].mean(),
                "stop_control_sd": stop["score_original"].std(),
                "reported_uncertainty_present": int(
                    group["score_uncertainty"].notna().sum()
                ),
            }
        )
        rows.append(record)
    return rows


def _group_assignments(observations: pd.DataFrame) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    for record in observations.to_dict("records"):
        mutation_group = _stable_id(
            "mutation_group", record["protein_id"], record["assay_mutation"]
        )
        position_group = _stable_id(
            "position_group", record["protein_id"], record["assay_position"]
        )
        experiment_group = _stable_id(
            "experiment_group",
            record["underlying_experiment_id"],
            record["protein_id"],
            record["antibiotic"],
            record["temperature_c"],
        )
        rows.append(
            {
                "observation_id": record["observation_id"],
                "biological_observation_id": record["biological_observation_id"],
                "underlying_experiment_id": record["underlying_experiment_id"],
                "mutation_group_id": mutation_group,
                "position_group_id": position_group,
                "experiment_group_id": experiment_group,
                "partition_rule": (
                    "keep identical mutation, residue position, and copied "
                    "experimental evidence within one partition"
                ),
            }
        )
    return rows


def _condition_candidates() -> List[Mapping[str, object]]:
    definitions = [
        {
            "antibiotic": "AMP",
            "ndm_concentration": 256.0,
            "vim_concentration": 128.0,
            "comparability_status": "study_defined_similar_selection_pressure",
            "direct_comparison_use": True,
            "basis": (
                "the 2024 study selected NDM-1 256 ug/mL and "
                "VIM-2 128 ug/mL at 37 C for the AMP50 comparison"
            ),
        },
        {
            "antibiotic": "CTX",
            "ndm_concentration": 16.0,
            "vim_concentration": 4.0,
            "comparability_status": "condition_specific_only",
            "direct_comparison_use": False,
            "basis": (
                "the source methods do not establish these concentrations as "
                "a matched-pressure NDM-1/VIM-2 comparison"
            ),
        },
        {
            "antibiotic": "MEM",
            "ndm_concentration": 0.25,
            "vim_concentration": 0.031,
            "comparability_status": "condition_specific_only",
            "direct_comparison_use": False,
            "basis": (
                "the source methods do not establish these concentrations as "
                "a matched-pressure NDM-1/VIM-2 comparison"
            ),
        },
    ]
    rows: List[Mapping[str, object]] = []
    for definition in definitions:
        rows.append(
            {
                "antibiotic": definition["antibiotic"],
                "ndm_assay_id": "ndm1_2024_dms",
                "ndm_concentration": definition["ndm_concentration"],
                "ndm_temperature_c": 37,
                "vim_assay_id": "vim2_2020_dms",
                "vim_concentration": definition["vim_concentration"],
                "vim_temperature_c": 37,
                "comparability_status": definition["comparability_status"],
                "basis": definition["basis"],
                "headline_use": definition["direct_comparison_use"],
                "direct_comparison_use": definition["direct_comparison_use"],
                "manual_confirmation_required": False,
            }
        )
    return rows


def _link_cross_homologue(
    observations: pd.DataFrame, comparisons: pd.DataFrame
) -> List[Mapping[str, object]]:
    ndm = observations[
        (observations["assay_id"] == "ndm1_2024_dms")
        & (observations["antibiotic"] == "AMP")
        & (observations["concentration"] == 256.0)
    ].set_index("assay_mutation")
    vim = observations[
        (observations["assay_id"] == "vim2_2020_dms")
        & (observations["antibiotic"] == "AMP")
        & (observations["concentration"] == 128.0)
        & (observations["temperature_c"] == 37.0)
    ].set_index("assay_mutation")
    rows: List[Mapping[str, object]] = []
    for pair in comparisons.to_dict("records"):
        vim_record = vim.loc[pair["vim_assay_mutation"]]
        ndm_record = ndm.loc[pair["ndm_assay_mutation"]]
        if isinstance(vim_record, pd.DataFrame) or isinstance(ndm_record, pd.DataFrame):
            raise AssayDefinitionError("Cross-homologue link is not one-to-one")
        vim_difference = float(pair["amp_vim"] - vim_record["score_original"])
        ndm_difference = float(pair["amp_ndm"] - ndm_record["score_original"])
        rows.append(
            {
                "comparison_id": pair["comparison_id"],
                "comparison_record_type": "cross_homologue_comparison",
                "vim_relationship": (
                    "exact_copy"
                    if abs(vim_difference) <= 1e-9
                    else "reported_value_differs_from_linked_observation"
                ),
                "ndm_relationship": (
                    "exact_copy"
                    if abs(ndm_difference) <= 1e-9
                    else "reported_value_differs_from_linked_observation"
                ),
                "independence_status": "not_independent_comparison_record",
                "vim_observation_id": vim_record["observation_id"],
                "ndm_observation_id": ndm_record["observation_id"],
                "vim_score_difference": vim_difference,
                "ndm_score_difference": ndm_difference,
                "vim_exact_within_1e-9": abs(vim_difference) <= 1e-9,
                "ndm_exact_within_1e-9": abs(ndm_difference) <= 1e-9,
                "vim_agrees_within_0_01": abs(vim_difference) <= 0.01,
                "ndm_agrees_within_0_01": abs(ndm_difference) <= 0.01,
                "reported_class": pair["reported_class"],
                "split_group_id": _stable_id(
                    "cross_pair_group",
                    vim_record["biological_observation_id"],
                    ndm_record["biological_observation_id"],
                ),
            }
        )
    return rows


def define_assays_and_conditions(repo_root: Path) -> Mapping[str, object]:
    root = repo_root.resolve()
    input_dir = root / "data/interim/mutation_measurements"
    output_dir = root / "data/interim/assay_definitions"
    observations = _read_csv(input_dir / "mutation_measurements.csv")
    comparisons = _read_csv(input_dir / "cross_protein_comparisons.csv")

    assays = _score_registry()
    condition_rows = _condition_summary(observations)
    assignments = _group_assignments(observations)
    candidates = _condition_candidates()
    links = _link_cross_homologue(observations, comparisons)

    _write_csv(output_dir / "assays.csv", assays)
    _write_csv(output_dir / "condition_score_statistics.csv", condition_rows)
    _write_csv(output_dir / "measurement_groups.csv", assignments)
    _write_csv(output_dir / "cross_protein_conditions.csv", candidates)
    _write_csv(output_dir / "comparison_value_relationships.csv", links)

    summary = {
        "status": "PASS",
        "validated_measurement_count": len(observations),
        "condition_count": len(condition_rows),
        "cross_protein_link_count": len(links),
        "cross_vim_exact_count": sum(row["vim_exact_within_1e-9"] for row in links),
        "cross_ndm_exact_count": sum(row["ndm_exact_within_1e-9"] for row in links),
        "headline_condition": (
            "NDM-1 AMP 256 ug/mL versus VIM-2 AMP 128 ug/mL at 37 C"
        ),
        "outputs": [
            "assays.csv",
            "condition_score_statistics.csv",
            "measurement_groups.csv",
            "cross_protein_conditions.csv",
            "comparison_value_relationships.csv",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "assay_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
