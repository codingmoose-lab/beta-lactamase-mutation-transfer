# Source-specific parsing of approved beta-lactamase experimental workbooks.
# ruff: noqa: E501, UP045

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from betalactamase_transfer.source_tables import SheetData, read_xlsx


class MutationMeasurementError(RuntimeError):
    """Raised when experimental data cannot be parsed without ambiguity."""


@dataclass(frozen=True)
class ConditionSpec:
    score_header: str
    uncertainty_header: str
    antibiotic: str
    concentration: float
    concentration_unit: str
    temperature_c: Optional[float]


PRIMARY_COLUMNS: Tuple[str, ...] = (
    "observation_id",
    "biological_observation_id",
    "underlying_experiment_id",
    "source_record_id",
    "study_id",
    "assay_id",
    "protein_id",
    "source_file",
    "source_sheet",
    "source_row",
    "source_score_column",
    "source_uncertainty_column",
    "wild_type",
    "publication_position",
    "source_position_label",
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
)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _sheet(path: Path, name: str) -> SheetData:
    matches = [sheet for sheet in read_xlsx(path) if sheet.name == name]
    if len(matches) != 1:
        raise MutationMeasurementError(
            f"Expected one worksheet named {name!r} in {path}, found {len(matches)}"
        )
    return matches[0]


def _table(
    sheet: SheetData, header_row: int
) -> Tuple[List[str], List[Mapping[str, object]]]:
    max_column = max((cell.column for cell in sheet.cells), default=0)
    max_row = max((cell.row for cell in sheet.cells), default=0)
    values = {(cell.row, cell.column): cell.value for cell in sheet.cells}
    headers = [
        str(values.get((header_row, col), "") or "").strip()
        for col in range(1, max_column + 1)
    ]
    nonempty_headers = [header for header in headers if header]
    if len(nonempty_headers) != len(set(nonempty_headers)):
        duplicates = sorted(
            {
                header
                for header in nonempty_headers
                if nonempty_headers.count(header) > 1
            }
        )
        raise MutationMeasurementError(
            f"Duplicate non-empty headers in {sheet.name}: {duplicates}"
        )
    rows: List[Mapping[str, object]] = []
    for row_number in range(header_row + 1, max_row + 1):
        record = {
            header: values.get((row_number, col))
            for col, header in enumerate(headers, start=1)
            if header
        }
        if any(value not in (None, "") for value in record.values()):
            rows.append({"__source_row__": row_number, **record})
    return headers, rows


def _require_headers(
    headers: Sequence[str], required: Sequence[str], context: str
) -> None:
    missing = [header for header in required if header not in headers]
    if missing:
        raise MutationMeasurementError(
            f"Missing required headers in {context}: {missing}"
        )


def _as_int(value: object, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise MutationMeasurementError(f"Boolean is invalid for {field} in {context}")
    if isinstance(value, str):
        rendered = value.strip().upper()
        match = re.fullmatch(r"[A-Z]?([1-9][0-9]*)", rendered)
        if match is not None:
            return int(match.group(1))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MutationMeasurementError(
            f"Invalid integer {field}={value!r} in {context}"
        ) from exc


def _map_publication_position(
    protein_id: str, value: object, field: str, context: str
) -> Tuple[str, int]:
    rendered = str(value).strip().upper()
    if protein_id == "VIM-2" and rendered == "G2":
        return "G2", 2
    publication_position = _as_int(value, field, context)
    if protein_id == "VIM-2" and publication_position >= 2:
        return str(publication_position), publication_position + 1
    return str(publication_position), publication_position


def _as_float_or_none(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MutationMeasurementError(
            f"Expected numeric value, found {value!r}"
        ) from exc


def _normalise_residue(value: object) -> str:
    return str(value or "").strip().upper()


def _mutation_validation(
    sequence: str,
    wild_type: str,
    position: int,
    mutant: str,
) -> Tuple[str, str]:
    if not wild_type or not mutant:
        return "missing_mutation_component", "unknown"
    if position < 1 or position > len(sequence):
        return "invalid_position", "unknown"
    expected = sequence[position - 1]
    if wild_type != expected:
        return "wild_type_mismatch", "unknown"
    if mutant == "*":
        return "valid", "stop_control"
    if mutant == wild_type:
        return "valid", "wild_type_control"
    if len(mutant) != 1 or mutant not in "ACDEFGHIKLMNPQRSTVWY":
        return "nonstandard_mutant", "unknown"
    return "valid", "missense"


def _load_assay_sequences(root: Path) -> Mapping[str, str]:
    path = root / "data/interim/sequence_mapping/sequence_sources.csv"
    if not path.is_file():
        raise MutationMeasurementError(f"Missing sequence-audit inventory: {path}")
    wanted = {
        "NDM1_2024_assay_construct": "NDM-1",
        "VIM2_2020_assay_construct": "VIM-2",
    }
    output: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            protein = wanted.get(row["record_id"])
            if protein:
                output[protein] = row["sequence"]
    if set(output) != {"NDM-1", "VIM-2"}:
        raise MutationMeasurementError(
            f"Could not load both fixed assay sequences; found {sorted(output)}"
        )
    return output


def _condition_specs_2024(headers: Sequence[str]) -> List[ConditionSpec]:
    # The workbook headers encode drug and concentration; Chen et al. (2024)
    # report a fixed 37 C selection temperature for these DMS conditions.
    pattern = re.compile(r"^(?P<concentration>[0-9.]+)_(?P<drug>AMP|CTX|MEM)_score$")
    output: List[ConditionSpec] = []
    for header in headers:
        match = pattern.fullmatch(header)
        if match:
            uncertainty = f"{header}_sd"
            if uncertainty not in headers:
                raise MutationMeasurementError(
                    f"Missing uncertainty column {uncertainty}"
                )
            output.append(
                ConditionSpec(
                    header,
                    uncertainty,
                    match.group("drug"),
                    float(match.group("concentration")),
                    "ug/mL",
                    37.0,
                )
            )
    return output


def _condition_specs_2020(headers: Sequence[str]) -> List[ConditionSpec]:
    pattern = re.compile(
        r"^(?P<concentration>[0-9.]+)ug/mL_(?P<drug>AMP|CTX|MEM)_(?P<temperature>[0-9]+)C$"
    )
    output: List[ConditionSpec] = []
    for header in headers:
        match = pattern.fullmatch(header)
        if match:
            uncertainty = f"{header}_SD"
            if uncertainty not in headers:
                raise MutationMeasurementError(
                    f"Missing uncertainty column {uncertainty}"
                )
            output.append(
                ConditionSpec(
                    header,
                    uncertainty,
                    match.group("drug"),
                    float(match.group("concentration")),
                    "ug/mL",
                    float(match.group("temperature")),
                )
            )
    return output


def _condition_specs_2022(headers: Sequence[str]) -> List[ConditionSpec]:
    # The workbook headers encode AMP concentration; Chen et al. (2022)
    # report a fixed 37 C selection temperature for these DMS conditions.
    pattern = re.compile(r"^(?P<concentration>[0-9.]+)ug/mL_fitness$")
    output: List[ConditionSpec] = []
    for header in headers:
        match = pattern.fullmatch(header)
        if match:
            uncertainty = header.replace("_fitness", "_sd")
            if uncertainty not in headers:
                raise MutationMeasurementError(
                    f"Missing uncertainty column {uncertainty}"
                )
            output.append(
                ConditionSpec(
                    header,
                    uncertainty,
                    "AMP",
                    float(match.group("concentration")),
                    "ug/mL",
                    37.0,
                )
            )
    return output


def _observation_record(
    *,
    study_id: str,
    assay_id: str,
    protein_id: str,
    source_file: str,
    source_sheet: str,
    source_row: int,
    condition: ConditionSpec,
    wild_type: str,
    publication_position: str,
    position: int,
    mutant: str,
    score: Optional[float],
    uncertainty: Optional[float],
    sequence: str,
    library_group: object,
    identity_label: object,
) -> Mapping[str, object]:
    validation, mutation_class = _mutation_validation(
        sequence, wild_type, position, mutant
    )
    mutation = f"{wild_type}{publication_position}{mutant}"
    assay_mutation = f"{wild_type}{position}{mutant}"
    biological_id = _stable_id(
        "bioobs",
        protein_id,
        assay_mutation,
        condition.antibiotic,
        condition.concentration,
        condition.concentration_unit,
        condition.temperature_c,
        assay_id,
    )
    source_record_id = _stable_id(
        "src", source_file, source_sheet, source_row, condition.score_header
    )
    quality_status = "missing_score" if score is None else validation
    return {
        "observation_id": _stable_id("obs", source_record_id, biological_id),
        "biological_observation_id": biological_id,
        "underlying_experiment_id": assay_id,
        "source_record_id": source_record_id,
        "study_id": study_id,
        "assay_id": assay_id,
        "protein_id": protein_id,
        "source_file": source_file,
        "source_sheet": source_sheet,
        "source_row": source_row,
        "source_score_column": condition.score_header,
        "source_uncertainty_column": condition.uncertainty_header,
        "wild_type": wild_type,
        "publication_position": publication_position,
        "source_position_label": publication_position,
        "assay_position": position,
        "mutant": mutant,
        "mutation": mutation,
        "assay_mutation": assay_mutation,
        "mutation_class": mutation_class,
        "sequence_validation_status": validation,
        "antibiotic": condition.antibiotic,
        "concentration": condition.concentration,
        "concentration_unit": condition.concentration_unit,
        "temperature_c": condition.temperature_c,
        "score_original": score,
        "score_uncertainty": uncertainty,
        "quality_status": quality_status,
        "library_group": library_group,
        "identity_label": identity_label,
    }


def _parse_primary_sheet(
    *,
    path: Path,
    sheet_name: str,
    header_row: int,
    study_id: str,
    assay_id: str,
    protein_id: str,
    sequence: str,
    wt_header: str,
    position_header: str,
    mutant_header: str,
    group_header: str,
    identity_header: str,
    condition_parser,
    relative_path: str,
) -> List[Mapping[str, object]]:
    sheet = _sheet(path, sheet_name)
    headers, rows = _table(sheet, header_row)
    _require_headers(
        headers,
        [wt_header, position_header, mutant_header, group_header, identity_header],
        f"{relative_path}:{sheet_name}",
    )
    conditions = condition_parser(headers)
    if not conditions:
        raise MutationMeasurementError(
            f"No condition columns recognised in {relative_path}:{sheet_name}"
        )
    output: List[Mapping[str, object]] = []
    for row in rows:
        source_row = int(row["__source_row__"])
        context = f"{relative_path}:{sheet_name}:row {source_row}"
        publication_position, position = _map_publication_position(
            protein_id, row.get(position_header), position_header, context
        )
        wild_type = _normalise_residue(row.get(wt_header))
        mutant = _normalise_residue(row.get(mutant_header))
        for condition in conditions:
            output.append(
                _observation_record(
                    study_id=study_id,
                    assay_id=assay_id,
                    protein_id=protein_id,
                    source_file=relative_path,
                    source_sheet=sheet_name,
                    source_row=source_row,
                    condition=condition,
                    wild_type=wild_type,
                    publication_position=publication_position,
                    position=position,
                    mutant=mutant,
                    score=_as_float_or_none(row.get(condition.score_header)),
                    uncertainty=_as_float_or_none(
                        row.get(condition.uncertainty_header)
                    ),
                    sequence=sequence,
                    library_group=row.get(group_header),
                    identity_label=row.get(identity_header),
                )
            )
    return output


def _parse_cross_homologue(
    root: Path, sequences: Mapping[str, str]
) -> List[Mapping[str, object]]:
    relative = (
        "data/raw/ndm_vim_2024/chen_2024_supplementary_data_4_ndm1_vim2_comparison.xlsx"
    )
    path = root / relative
    sheet = _sheet(path, "SD4_wt_reversion")
    headers, rows = _table(sheet, 1)
    required = [
        "vim_pos",
        "vim_wt",
        "vim_mut",
        "ndm_pos",
        "ndm_wt",
        "ndm_mut",
        "amp_vim",
        "amp_effect_vim",
        "amp_ndm",
        "amp_effect_ndm",
        "class",
    ]
    _require_headers(headers, required, f"{relative}:SD4_wt_reversion")
    output: List[Mapping[str, object]] = []
    for row in rows:
        source_row = int(row["__source_row__"])
        vim_publication_pos, vim_pos = _map_publication_position(
            "VIM-2", row["vim_pos"], "vim_pos", f"row {source_row}"
        )
        ndm_publication_pos, ndm_pos = _map_publication_position(
            "NDM-1", row["ndm_pos"], "ndm_pos", f"row {source_row}"
        )
        vim_wt = _normalise_residue(row["vim_wt"])
        vim_mut = _normalise_residue(row["vim_mut"])
        ndm_wt = _normalise_residue(row["ndm_wt"])
        ndm_mut = _normalise_residue(row["ndm_mut"])
        vim_status, _ = _mutation_validation(
            sequences["VIM-2"], vim_wt, vim_pos, vim_mut
        )
        ndm_status, _ = _mutation_validation(
            sequences["NDM-1"], ndm_wt, ndm_pos, ndm_mut
        )
        output.append(
            {
                "comparison_id": _stable_id(
                    "pair", vim_pos, vim_wt, vim_mut, ndm_pos, ndm_wt, ndm_mut
                ),
                "study_id": "chen_2024_natcomms",
                "source_file": relative,
                "source_sheet": "SD4_wt_reversion",
                "source_row": source_row,
                "vim_publication_position": vim_publication_pos,
                "vim_assay_position": vim_pos,
                "vim_wild_type": vim_wt,
                "vim_mutant": vim_mut,
                "vim_mutation": f"{vim_wt}{vim_publication_pos}{vim_mut}",
                "vim_assay_mutation": f"{vim_wt}{vim_pos}{vim_mut}",
                "vim_sequence_validation_status": vim_status,
                "ndm_publication_position": ndm_publication_pos,
                "ndm_assay_position": ndm_pos,
                "ndm_wild_type": ndm_wt,
                "ndm_mutant": ndm_mut,
                "ndm_mutation": f"{ndm_wt}{ndm_publication_pos}{ndm_mut}",
                "ndm_assay_mutation": f"{ndm_wt}{ndm_pos}{ndm_mut}",
                "ndm_sequence_validation_status": ndm_status,
                "amp_vim": _as_float_or_none(row["amp_vim"]),
                "amp_effect_vim": row["amp_effect_vim"],
                "amp_ndm": _as_float_or_none(row["amp_ndm"]),
                "amp_effect_ndm": row["amp_effect_ndm"],
                "reported_class": row["class"],
                "independence_status": "not_independent_comparison_table",
            }
        )
    return output


def _ec50_record(
    *,
    study_id: str,
    assay_id: str,
    protein_id: str,
    relative_path: str,
    sheet_name: str,
    source_row: int,
    publication_position: str,
    position: Optional[int],
    wild_type: str,
    mutant: str,
    sequence: str,
    antibiotic: str,
    measurement_type: str,
    ec50: Optional[float],
    ec50_scale: str,
    codon: object,
    replicate_row: object,
    replicate_column: object,
    uncertainty: Optional[float],
    observations: object,
) -> Mapping[str, object]:
    if position is None and mutant in {"WT", "VEC"}:
        validation = "not_applicable_control"
        mutation_class = (
            "global_wild_type_control" if mutant == "WT" else "vector_control"
        )
        mutation = mutant
    elif position is None:
        validation = "missing_mutation_component"
        mutation_class = "unknown"
        mutation = mutant or "UNKNOWN"
    else:
        validation, mutation_class = _mutation_validation(
            sequence, wild_type, position, mutant
        )
        mutation = f"{wild_type}{publication_position}{mutant}"
    source_record_id = _stable_id("ec50src", relative_path, sheet_name, source_row)
    return {
        "ec50_observation_id": _stable_id("ec50", source_record_id, mutation),
        "biological_mutation_id": _stable_id("mutation", protein_id, mutation),
        "source_record_id": source_record_id,
        "study_id": study_id,
        "assay_id": assay_id,
        "protein_id": protein_id,
        "source_file": relative_path,
        "source_sheet": sheet_name,
        "source_row": source_row,
        "wild_type": wild_type,
        "publication_position": publication_position,
        "assay_position": position,
        "mutant": mutant,
        "mutation": mutation,
        "mutation_class": mutation_class,
        "sequence_validation_status": validation,
        "antibiotic": antibiotic,
        "measurement_type": measurement_type,
        "ec50_value": ec50,
        "ec50_scale": ec50_scale,
        "concentration_unit": "ug/mL",
        "uncertainty": uncertainty,
        "observations": observations,
        "codon": codon,
        "replicate_row": replicate_row,
        "replicate_column": replicate_column,
        "quality_status": "missing_value" if ec50 is None else validation,
    }


def _parse_ec50(root: Path, sequences: Mapping[str, str]) -> List[Mapping[str, object]]:
    output: List[Mapping[str, object]] = []
    vim = sequences["VIM-2"]
    relative_2020 = "data/raw/vim2_2020/chen_2020_supplementary_file_2_vim2_dms_and_annotations.xlsx"
    sheet = _sheet(root / relative_2020, "SF2B Single Variant EC50")
    headers, rows = _table(sheet, 2)
    _require_headers(
        headers,
        ["row", "column", "antibiotic", "pos", "mut", "codon", "ec50 (ug/mL)"],
        "2020 SF2B",
    )
    for row in rows:
        source_row = int(row["__source_row__"])
        raw_position = row["pos"]
        raw_mutant = _normalise_residue(row["mut"])
        control_marker = str(raw_position).strip().lower()
        if control_marker in {"wt", "vec"}:
            publication_position = control_marker.upper()
            position = None
            wild_type = ""
            mutant = control_marker.upper()
        else:
            publication_position, position = _map_publication_position(
                "VIM-2", raw_position, "pos", f"2020 SF2B row {source_row}"
            )
            wild_type = vim[position - 1] if 1 <= position <= len(vim) else ""
            mutant = wild_type if raw_mutant == "WT" else raw_mutant
        output.append(
            _ec50_record(
                study_id="chen_2020_elife",
                assay_id="vim2_2020_individual_ec50",
                protein_id="VIM-2",
                relative_path=relative_2020,
                sheet_name=sheet.name,
                source_row=source_row,
                publication_position=publication_position,
                position=position,
                wild_type=wild_type,
                mutant=mutant,
                sequence=vim,
                antibiotic=str(row["antibiotic"] or "").upper(),
                measurement_type="individual_dose_response",
                ec50=_as_float_or_none(row["ec50 (ug/mL)"]),
                ec50_scale="linear",
                codon=row["codon"],
                replicate_row=row["row"],
                replicate_column=row["column"],
                uncertainty=None,
                observations=None,
            )
        )
    relative_2022 = "data/raw/vim2_2022/chen_2022_supplementary_data_vim2_concentration_landscape.xlsx"
    for sheet_name, header_row, measurement_type in (
        ("SD2 EC50", 6, "library_ec50"),
        ("SD3 Single variant EC50", 3, "individual_ec50"),
    ):
        sheet = _sheet(root / relative_2022, sheet_name)
        headers, rows = _table(sheet, header_row)
        position_header = (
            "variant position *" if sheet_name == "SD2 EC50" else "variant position"
        )
        mutant_header = (
            "variant residue **" if sheet_name == "SD2 EC50" else "variant residue"
        )
        _require_headers(
            headers,
            [position_header, mutant_header, "WT residue", "log2( EC50 in ug/mL )"],
            f"2022 {sheet_name}",
        )
        for row in rows:
            source_row = int(row["__source_row__"])
            raw_position = row[position_header]
            raw_wild_type = _normalise_residue(row["WT residue"])
            raw_mutant = _normalise_residue(row[mutant_header])
            if raw_position in (None, "") and raw_wild_type == "WT":
                publication_position = "WT"
                position = None
                wild_type = ""
                mutant = "WT"
            else:
                publication_position, position = _map_publication_position(
                    "VIM-2",
                    raw_position,
                    position_header,
                    f"2022 {sheet_name} row {source_row}",
                )
                wild_type = raw_wild_type
                mutant = raw_mutant
            output.append(
                _ec50_record(
                    study_id="chen_2022_natee",
                    assay_id=f"vim2_2022_{measurement_type}",
                    protein_id="VIM-2",
                    relative_path=relative_2022,
                    sheet_name=sheet_name,
                    source_row=source_row,
                    publication_position=publication_position,
                    position=position,
                    wild_type=wild_type,
                    mutant=mutant,
                    sequence=vim,
                    antibiotic="AMP",
                    measurement_type=measurement_type,
                    ec50=_as_float_or_none(row["log2( EC50 in ug/mL )"]),
                    ec50_scale="log2",
                    codon=row.get("variant codon"),
                    replicate_row=None,
                    replicate_column=None,
                    uncertainty=_as_float_or_none(row.get("sample SD ***")),
                    observations=row.get("observations"),
                )
            )
    return output


def _load_mapping(root: Path) -> Mapping[Tuple[str, int, str], Mapping[str, str]]:
    path = root / "data/interim/sequence_mapping/residue_number_mapping.csv"
    if not path.is_file():
        raise MutationMeasurementError(f"Missing residue mapping table: {path}")
    output: Dict[Tuple[str, int, str], Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["protein_id"], int(row["assay_position"]), row["pdb_chain"])
            if key in output:
                raise MutationMeasurementError(f"Duplicate residue mapping key: {key}")
            output[key] = row
    return output


def _coverage_rows(
    observations: Sequence[Mapping[str, object]],
    mapping: Mapping[Tuple[str, int, str], Mapping[str, str]],
) -> List[Mapping[str, object]]:
    unique = {}
    for row in observations:
        key = (row["protein_id"], int(row["assay_position"]), row["mutation"])
        unique[key] = row
    output: List[Mapping[str, object]] = []
    chains = {"NDM-1": ("A", "B", "C", "D", "E"), "VIM-2": ("A", "B")}
    for (protein, position, mutation), _row in sorted(unique.items()):
        for chain in chains[protein]:
            mapped = mapping.get((protein, position, chain))
            if mapped is None:
                output.append(
                    {
                        "protein_id": protein,
                        "mutation": mutation,
                        "assay_position": position,
                        "pdb_id": "3SPU" if protein == "NDM-1" else "5YD7",
                        "pdb_chain": chain,
                        "uniprot_position": "",
                        "pdb_entity_position": "",
                        "pdb_label_position": "",
                        "pdb_author_position": "",
                        "pdb_modeled": False,
                        "mapping_status": "not_in_assay_sequence",
                        "residue_identity_ok": False,
                        "provisional_preferred_chain": chain == "B",
                    }
                )
                continue
            output.append(
                {
                    "protein_id": protein,
                    "mutation": mutation,
                    "assay_position": position,
                    "pdb_id": mapped["pdb_id"],
                    "pdb_chain": chain,
                    "uniprot_position": mapped["uniprot_position"],
                    "pdb_entity_position": mapped["pdb_entity_position"],
                    "pdb_label_position": mapped["pdb_label_position"],
                    "pdb_author_position": mapped["pdb_author_position"],
                    "pdb_modeled": mapped["pdb_modeled"],
                    "mapping_status": mapped["mapping_status"],
                    "residue_identity_ok": mapped["residue_identity_ok"],
                    "provisional_preferred_chain": chain == "B",
                }
            )
    return output


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: List[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _count(rows: Sequence[Mapping[str, object]], field: str) -> Mapping[str, int]:
    output: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        output[key] = output.get(key, 0) + 1
    return dict(sorted(output.items()))


def extract_mutation_measurements(repo_root: Path) -> Mapping[str, object]:
    """Parse primary DMS, cross-homologue, and EC50 sources without merging scales."""
    root = repo_root.resolve()
    sequences = _load_assay_sequences(root)
    primary: List[Mapping[str, object]] = []
    specs = (
        {
            "relative": "data/raw/ndm_vim_2024/chen_2024_supplementary_data_3_ndm1_dms.xlsx",
            "sheet": "SD3_NDM_fitness_scores",
            "header": 1,
            "study": "chen_2024_natcomms",
            "assay": "ndm1_2024_dms",
            "protein": "NDM-1",
            "wt": "wt",
            "position": "pos",
            "mutant": "mut",
            "group": "group",
            "identity": "identity",
            "conditions": _condition_specs_2024,
        },
        {
            "relative": "data/raw/vim2_2020/chen_2020_supplementary_file_2_vim2_dms_and_annotations.xlsx",
            "sheet": "SF2A Fitness Scores",
            "header": 2,
            "study": "chen_2020_elife",
            "assay": "vim2_2020_dms",
            "protein": "VIM-2",
            "wt": "wt residue",
            "position": "position",
            "mutant": "variant residue",
            "group": "group",
            "identity": "identity",
            "conditions": _condition_specs_2020,
        },
        {
            "relative": "data/raw/vim2_2022/chen_2022_supplementary_data_vim2_concentration_landscape.xlsx",
            "sheet": "SD1 Fitness scores",
            "header": 6,
            "study": "chen_2022_natee",
            "assay": "vim2_2022_amp_concentration_dms",
            "protein": "VIM-2",
            "wt": "WT residue",
            "position": "variant position *",
            "mutant": "variant residue",
            "group": "library group",
            "identity": "identity",
            "conditions": _condition_specs_2022,
        },
    )
    for spec in specs:
        primary.extend(
            _parse_primary_sheet(
                path=root / spec["relative"],
                sheet_name=spec["sheet"],
                header_row=spec["header"],
                study_id=spec["study"],
                assay_id=spec["assay"],
                protein_id=spec["protein"],
                sequence=sequences[spec["protein"]],
                wt_header=spec["wt"],
                position_header=spec["position"],
                mutant_header=spec["mutant"],
                group_header=spec["group"],
                identity_header=spec["identity"],
                condition_parser=spec["conditions"],
                relative_path=spec["relative"],
            )
        )
    cross = _parse_cross_homologue(root, sequences)
    ec50 = _parse_ec50(root, sequences)
    mapping = _load_mapping(root)
    coverage = _coverage_rows(primary, mapping)
    validated = [row for row in primary if row["quality_status"] == "valid"]
    quarantine = [row for row in primary if row["quality_status"] != "valid"]
    output_dir = root / "data/interim/mutation_measurements"
    _write_csv(output_dir / "mutation_measurements_raw.csv", primary, PRIMARY_COLUMNS)
    _write_csv(output_dir / "mutation_measurements.csv", validated, PRIMARY_COLUMNS)
    _write_csv(output_dir / "excluded_measurements.csv", quarantine, PRIMARY_COLUMNS)
    _write_csv(output_dir / "cross_protein_comparisons.csv", cross)
    _write_csv(output_dir / "ec50_measurements.csv", ec50)
    _write_csv(output_dir / "mutation_structure_mapping.csv", coverage)
    parsing_summary = {
        "status": "PASS",
        "primary_observation_count": len(primary),
        "validated_observation_count": len(validated),
        "quarantine_observation_count": len(quarantine),
        "cross_homologue_comparison_count": len(cross),
        "ec50_observation_count": len(ec50),
        "unique_primary_mutations": len(
            {(row["protein_id"], row["mutation"]) for row in primary}
        ),
        "counts_by_assay": _count(primary, "assay_id"),
        "counts_by_antibiotic": _count(primary, "antibiotic"),
        "counts_by_mutation_class": _count(primary, "mutation_class"),
        "outputs": [
            "mutation_measurements_raw.csv",
            "mutation_measurements.csv",
            "excluded_measurements.csv",
            "cross_protein_comparisons.csv",
            "ec50_measurements.csv",
            "mutation_structure_mapping.csv",
        ],
    }
    validation_summary = {
        "sequence_validation_status": _count(primary, "sequence_validation_status"),
        "quality_status": _count(primary, "quality_status"),
        "ec50_sequence_validation_status": _count(ec50, "sequence_validation_status"),
        "cross_vim_sequence_validation_status": _count(
            cross, "vim_sequence_validation_status"
        ),
        "cross_ndm_sequence_validation_status": _count(
            cross, "ndm_sequence_validation_status"
        ),
        "structure_mapping_status": _count(coverage, "mapping_status"),
    }
    for name, payload in (
        ("measurement_summary.json", parsing_summary),
        ("validation_summary.json", validation_summary),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return parsing_summary
