# Read-only audit of published XLSX workbooks used by the study.
#
# The module intentionally uses the XLSX Open Packaging Convention directly so
# that auditing does not depend on Excel recalculation or mutate source files.
# It inventories workbook structure, headers, columns, formulas, mutation-like
# fields, and known source relationships. It does not harmonise biological
# observations or alter reported values.

from __future__ import annotations

# Python 3.9 compatibility requires Optional[...] annotations.
# ruff: noqa: E501, UP045
import csv
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"x": MAIN_NS, "r": REL_NS, "pr": PKG_REL_NS}
CELL_REF_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
POSITION_HEADER_RE = re.compile(r"(^|[^a-z])(pos|position|resi)([^a-z]|$)")
MUTATION_HEADER_TERMS = (
    "mut",
    "mutation",
    "variant residue",
    "aa_mut",
    "variant codon",
    "index",
)
WT_HEADER_TERMS = ("wt", "wild type", "starting residue", "vim residue")


class SourceTableError(RuntimeError):
    """Raised when a workbook cannot be audited safely."""


@dataclass(frozen=True)
class WorkbookSpec:
    workbook_id: str
    study_id: str
    relative_path: str
    primary_role: str


@dataclass(frozen=True)
class SheetRule:
    workbook_id: str
    sheet_name: str
    header_row: Optional[int]
    sheet_role: str
    include_in_primary_parsing: bool
    notes: str


@dataclass(frozen=True)
class CellValue:
    address: str
    row: int
    column: int
    value: object
    formula: Optional[str]


@dataclass(frozen=True)
class SheetData:
    name: str
    state: str
    dimension: str
    cells: Tuple[CellValue, ...]
    merged_ranges: Tuple[str, ...]


WORKBOOK_SPECS: Tuple[WorkbookSpec, ...] = (
    WorkbookSpec(
        "chen_2024_sd3",
        "chen_2024_natcomms",
        "data/raw/ndm_vim_2024/chen_2024_supplementary_data_3_ndm1_dms.xlsx",
        "primary_ndm1_dms",
    ),
    WorkbookSpec(
        "chen_2024_sd4",
        "chen_2024_natcomms",
        "data/raw/ndm_vim_2024/chen_2024_supplementary_data_4_ndm1_vim2_comparison.xlsx",
        "cross_homologue_wild_type_reversion_comparison",
    ),
    WorkbookSpec(
        "chen_2020_sf2",
        "chen_2020_elife",
        "data/raw/vim2_2020/chen_2020_supplementary_file_2_vim2_dms_and_annotations.xlsx",
        "primary_vim2_dms_and_annotations",
    ),
    WorkbookSpec(
        "chen_2020_sf3",
        "chen_2020_elife",
        "data/raw/vim2_2020/chen_2020_supplementary_file_3_ctx_mem_linear_models.xlsx",
        "published_linear_model_results",
    ),
    WorkbookSpec(
        "chen_2022_sd",
        "chen_2022_natee",
        "data/raw/vim2_2022/chen_2022_supplementary_data_vim2_concentration_landscape.xlsx",
        "vim2_ampicillin_concentration_landscape",
    ),
)


SHEET_RULES: Tuple[SheetRule, ...] = (
    SheetRule(
        "chen_2024_sd3",
        "SD3_NDM_fitness_scores",
        1,
        "primary_dms",
        True,
        "NDM-1 amino-acid-level DMS under AMP, CTX, and MEM concentrations.",
    ),
    SheetRule(
        "chen_2024_sd4",
        "SD4_wt_reversion",
        1,
        "cross_homologue_comparison",
        True,
        "Mapped VIM-2/NDM-1 wild-type reversion comparisons; not an independent DMS experiment.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "Legend",
        None,
        "documentation",
        False,
        "Workbook legend and sheet descriptions.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2A Fitness Scores",
        2,
        "primary_dms",
        True,
        "VIM-2 amino-acid fitness scores across antibiotics, concentrations, and temperatures.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2B Single Variant EC50",
        2,
        "individual_ec50",
        True,
        "Individually measured dose-response EC50 records; repeated wells/codons require grouping.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2C Linear Model Data",
        2,
        "derived_model_input",
        False,
        "Derived 37C fitness/biophysical table; overlaps SF2A and must not be counted as new observations.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2D Codon Fitness scores",
        2,
        "codon_level_dms",
        False,
        "Codon-level scores; retain separately from amino-acid-level primary labels.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2E signal peptide ddG",
        7,
        "signal_peptide_annotation",
        False,
        "Computed signal-peptide free-energy features.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2F Temperature Dependence",
        2,
        "position_annotation",
        False,
        "Position-level temperature-dependence classes derived from DMS.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2G Hydrogen bonds",
        2,
        "structure_annotation",
        False,
        "Hydrogen-bond list tied to 5YD7 residue numbering.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2H globally positive variants",
        2,
        "derived_variant_subset",
        False,
        "Selected positive variants derived from SF2A.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2I Specificity Variants",
        2,
        "derived_variant_subset",
        False,
        "Selected substrate-specific variants derived from SF2A.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2J Natural Variants",
        2,
        "natural_variant_summary",
        False,
        "External natural VIM entries; not DMS labels.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2K Natural Variant Mutations",
        1,
        "natural_variant_mutations",
        False,
        "Natural-variant substitutions relative to VIM-2.",
    ),
    SheetRule(
        "chen_2020_sf2",
        "SF2L Amplicon Sequencing Primer",
        4,
        "experimental_metadata",
        False,
        "Primer sequences; not modelling observations.",
    ),
    SheetRule(
        "chen_2020_sf3",
        "SF3A CTX linear model results",
        3,
        "published_model_result",
        False,
        "Published aggregate CTX regression output.",
    ),
    SheetRule(
        "chen_2020_sf3",
        "SF3B MEM linear model results",
        3,
        "published_model_result",
        False,
        "Published aggregate MEM regression output.",
    ),
    SheetRule(
        "chen_2022_sd",
        "SD1 Fitness scores",
        6,
        "primary_dms",
        True,
        "VIM-2 amino-acid fitness across seven AMP concentrations.",
    ),
    SheetRule(
        "chen_2022_sd",
        "SD2 EC50",
        6,
        "library_ec50",
        True,
        "Library-derived log2 EC50 table for VIM-2 variants.",
    ),
    SheetRule(
        "chen_2022_sd",
        "SD3 Single variant EC50",
        3,
        "individual_ec50",
        True,
        "Individually measured VIM-2 EC50 values.",
    ),
    SheetRule(
        "chen_2022_sd",
        "SD4 growth OD",
        4,
        "library_growth",
        False,
        "Library-level OD600 controls by concentration and replicate.",
    ),
)


SOURCE_RELATIONSHIPS: Tuple[Mapping[str, object], ...] = (
    {
        "source_id": "chen_2024_sd4:SD4_wt_reversion",
        "related_source_id": "chen_2020_sf2:SF2A Fitness Scores",
        "relationship": "reuses_or_compares_prior_vim2_measurements",
        "independence_status": "not_independent_until_row_level_reconciliation",
        "required_action": "Assign shared biological_observation_id where values derive from the same VIM-2 experiment.",
    },
    {
        "source_id": "chen_2020_sf2:SF2C Linear Model Data",
        "related_source_id": "chen_2020_sf2:SF2A Fitness Scores",
        "relationship": "derived_table",
        "independence_status": "not_independent",
        "required_action": "Use as annotation/model-reproduction input, not as additional mutation labels.",
    },
    {
        "source_id": "chen_2020_sf2:SF2H globally positive variants",
        "related_source_id": "chen_2020_sf2:SF2A Fitness Scores",
        "relationship": "selected_subset",
        "independence_status": "not_independent",
        "required_action": "Use as published classification/validation subset only.",
    },
    {
        "source_id": "chen_2020_sf2:SF2I Specificity Variants",
        "related_source_id": "chen_2020_sf2:SF2A Fitness Scores",
        "relationship": "selected_subset",
        "independence_status": "not_independent",
        "required_action": "Use as published specificity annotation, not duplicated labels.",
    },
    {
        "source_id": "chen_2022_sd:SD2 EC50",
        "related_source_id": "chen_2022_sd:SD1 Fitness scores",
        "relationship": "same_variant_library_different_outcome",
        "independence_status": "related_but_distinct_measurement",
        "required_action": "Keep EC50 and DMS fitness as separate outcomes while grouping the same mutation in splits.",
    },
    {
        "source_id": "chen_2022_sd:SD3 Single variant EC50",
        "related_source_id": "chen_2022_sd:SD2 EC50",
        "relationship": "individual_validation_subset",
        "independence_status": "partly_overlapping_variants",
        "required_action": "Retain measurement_type and do not average individual and library EC50 automatically.",
    },
)


def column_letters_to_number(letters: str) -> int:
    """Convert Excel column letters to a one-based integer."""
    result = 0
    for character in letters:
        if not "A" <= character <= "Z":
            raise SourceTableError(f"Invalid Excel column letters: {letters}")
        result = result * 26 + ord(character) - ord("A") + 1
    return result


def split_cell_reference(reference: str) -> Tuple[int, int]:
    match = CELL_REF_RE.fullmatch(reference)
    if match is None:
        raise SourceTableError(f"Invalid cell reference: {reference}")
    return int(match.group(2)), column_letters_to_number(match.group(1))


def _normalise_target(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return "xl/" + target.lstrip("/")


def _read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    filename = "xl/sharedStrings.xml"
    if filename not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(filename))
    values: List[str] = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        text = "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
        values.append(text)
    return values


def _cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        index = int(raw)
        if index >= len(shared_strings):
            raise SourceTableError("Shared-string index is outside the table")
        return shared_strings[index]
    if cell_type == "b":
        return raw == "1"
    if cell_type in {"str", "e"}:
        return raw
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer():
        return int(number)
    return number


def read_xlsx(path: Path) -> Tuple[SheetData, ...]:
    """Read workbook sheet cells without modifying or recalculating the file."""
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceTableError(f"Cannot open XLSX workbook: {path}") from exc
    with archive:
        shared_strings = _read_shared_strings(archive)
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        relations_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_targets = {
            item.attrib["Id"]: item.attrib["Target"] for item in relations_root
        }
        sheets_node = workbook_root.find(f"{{{MAIN_NS}}}sheets")
        if sheets_node is None:
            raise SourceTableError(f"Workbook has no sheets: {path}")
        output: List[SheetData] = []
        for sheet_node in sheets_node:
            name = sheet_node.attrib["name"]
            state = sheet_node.attrib.get("state", "visible")
            relationship_id = sheet_node.attrib[f"{{{REL_NS}}}id"]
            target = _normalise_target(relationship_targets[relationship_id])
            worksheet_root = ET.fromstring(archive.read(target))
            dimension_node = worksheet_root.find(f"{{{MAIN_NS}}}dimension")
            dimension = (
                dimension_node.attrib.get("ref", "")
                if dimension_node is not None
                else ""
            )
            cells: List[CellValue] = []
            sheet_data = worksheet_root.find(f"{{{MAIN_NS}}}sheetData")
            if sheet_data is not None:
                for row_node in sheet_data:
                    for cell_node in row_node.findall(f"{{{MAIN_NS}}}c"):
                        address = cell_node.attrib.get("r")
                        if not address:
                            continue
                        row_number, column_number = split_cell_reference(address)
                        formula_node = cell_node.find(f"{{{MAIN_NS}}}f")
                        formula = (
                            formula_node.text if formula_node is not None else None
                        )
                        cells.append(
                            CellValue(
                                address=address,
                                row=row_number,
                                column=column_number,
                                value=_cell_value(cell_node, shared_strings),
                                formula=formula,
                            )
                        )
            merged_ranges: List[str] = []
            merged_node = worksheet_root.find(f"{{{MAIN_NS}}}mergeCells")
            if merged_node is not None:
                merged_ranges = [item.attrib["ref"] for item in merged_node]
            output.append(
                SheetData(
                    name=name,
                    state=state,
                    dimension=dimension,
                    cells=tuple(cells),
                    merged_ranges=tuple(merged_ranges),
                )
            )
        return tuple(output)


def _sheet_rule(workbook_id: str, sheet_name: str) -> SheetRule:
    matches = [
        rule
        for rule in SHEET_RULES
        if rule.workbook_id == workbook_id and rule.sheet_name == sheet_name
    ]
    if len(matches) != 1:
        raise SourceTableError(
            f"Expected one sheet rule for {workbook_id}:{sheet_name}, found {len(matches)}"
        )
    return matches[0]


def _column_name(number: int) -> str:
    result = ""
    value = number
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _type_name(value: object) -> str:
    if value is None:
        return "empty"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    return "text"


def _header_for_column(sheet: SheetData, header_row: int, column: int) -> str:
    values = [
        cell.value
        for cell in sheet.cells
        if cell.row == header_row and cell.column == column and cell.value is not None
    ]
    if not values:
        return ""
    return str(values[0]).strip()


def _is_mutation_candidate(header: str) -> Tuple[bool, str]:
    normalised = re.sub(r"\s+", " ", header.lower()).strip()
    if not normalised:
        return False, ""
    if POSITION_HEADER_RE.search(normalised):
        return True, "position"
    if any(term in normalised for term in WT_HEADER_TERMS):
        return True, "wild_type"
    if any(term in normalised for term in MUTATION_HEADER_TERMS):
        return True, "mutation_or_variant"
    return False, ""


def _examples(values: Iterable[object], maximum: int = 10) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        if value is None or value == "":
            continue
        rendered = str(value)
        if rendered in seen:
            continue
        seen.add(rendered)
        output.append(rendered)
        if len(output) == maximum:
            break
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise SourceTableError(f"Refusing to write an empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def describe_source_tables(repo_root: Path) -> Mapping[str, object]:
    """Audit all five approved workbooks and write inventories to interim data."""
    root = repo_root.resolve()
    output_dir = root / "data/interim/source_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook_rows: List[Mapping[str, object]] = []
    sheet_rows: List[Mapping[str, object]] = []
    column_rows: List[Mapping[str, object]] = []
    mutation_candidate_rows: List[Mapping[str, object]] = []
    mutation_example_rows: List[Mapping[str, object]] = []
    encountered_rules = set()

    for spec in WORKBOOK_SPECS:
        path = root / spec.relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise SourceTableError(f"Missing or empty required workbook: {path}")
        sheets = read_xlsx(path)
        workbook_rows.append(
            {
                **asdict(spec),
                "file_size_bytes": path.stat().st_size,
                "sheet_count": len(sheets),
            }
        )
        for sheet in sheets:
            rule = _sheet_rule(spec.workbook_id, sheet.name)
            encountered_rules.add((spec.workbook_id, sheet.name))
            formula_count = sum(cell.formula is not None for cell in sheet.cells)
            nonempty_cells = [
                cell for cell in sheet.cells if cell.value not in (None, "")
            ]
            max_row = max((cell.row for cell in sheet.cells), default=0)
            max_column = max((cell.column for cell in sheet.cells), default=0)
            sheet_rows.append(
                {
                    "workbook_id": spec.workbook_id,
                    "study_id": spec.study_id,
                    "source_file": spec.relative_path,
                    "sheet_name": sheet.name,
                    "sheet_state": sheet.state,
                    "declared_dimension": sheet.dimension,
                    "observed_max_row": max_row,
                    "observed_max_column": max_column,
                    "nonempty_cell_count": len(nonempty_cells),
                    "formula_count": formula_count,
                    "merged_range_count": len(sheet.merged_ranges),
                    "header_row": rule.header_row,
                    "sheet_role": rule.sheet_role,
                    "include_in_primary_parsing": rule.include_in_primary_parsing,
                    "notes": rule.notes,
                }
            )
            if rule.header_row is None:
                continue
            for column in range(1, max_column + 1):
                header = _header_for_column(sheet, rule.header_row, column)
                data_cells = [
                    cell
                    for cell in sheet.cells
                    if cell.column == column and cell.row > rule.header_row
                ]
                values = [
                    cell.value for cell in data_cells if cell.value not in (None, "")
                ]
                types = sorted({_type_name(value) for value in values})
                unique_values = {str(value) for value in values}
                examples = _examples(values)
                formula_cells = sum(cell.formula is not None for cell in data_cells)
                column_letter = _column_name(column)
                column_record = {
                    "workbook_id": spec.workbook_id,
                    "study_id": spec.study_id,
                    "source_file": spec.relative_path,
                    "sheet_name": sheet.name,
                    "sheet_role": rule.sheet_role,
                    "header_row": rule.header_row,
                    "column_index": column,
                    "column_letter": column_letter,
                    "header": header,
                    "nonempty_count": len(values),
                    "missing_count_below_header": max(
                        0, max_row - rule.header_row - len(values)
                    ),
                    "unique_count": len(unique_values),
                    "value_types": "|".join(types),
                    "formula_count": formula_cells,
                    "example_values": " | ".join(examples),
                }
                column_rows.append(column_record)
                candidate, candidate_type = _is_mutation_candidate(header)
                if candidate:
                    candidate_record = {
                        **column_record,
                        "candidate_type": candidate_type,
                    }
                    mutation_candidate_rows.append(candidate_record)
                    for rank, example in enumerate(examples, start=1):
                        mutation_example_rows.append(
                            {
                                "workbook_id": spec.workbook_id,
                                "study_id": spec.study_id,
                                "sheet_name": sheet.name,
                                "column_letter": column_letter,
                                "header": header,
                                "candidate_type": candidate_type,
                                "example_rank": rank,
                                "example_value": example,
                            }
                        )

    expected_rules = {(rule.workbook_id, rule.sheet_name) for rule in SHEET_RULES}
    if encountered_rules != expected_rules:
        missing = sorted(expected_rules - encountered_rules)
        unexpected = sorted(encountered_rules - expected_rules)
        raise SourceTableError(
            f"Workbook sheet set differs from the audited specification. "
            f"Missing={missing}; unexpected={unexpected}"
        )

    _write_csv(output_dir / "workbook_metadata.csv", workbook_rows)
    _write_csv(output_dir / "worksheet_metadata.csv", sheet_rows)
    _write_csv(output_dir / "column_definitions.csv", column_rows)
    _write_csv(output_dir / "mutation_field_candidates.csv", mutation_candidate_rows)
    _write_csv(output_dir / "mutation_field_examples.csv", mutation_example_rows)
    _write_csv(
        output_dir / "source_table_relationships.csv", list(SOURCE_RELATIONSHIPS)
    )

    summary = {
        "audit_status": "PASS",
        "workbook_count": len(workbook_rows),
        "sheet_count": len(sheet_rows),
        "primary_or_measurement_sheet_count": sum(
            bool(row["include_in_primary_parsing"]) for row in sheet_rows
        ),
        "mutation_candidate_column_count": len(mutation_candidate_rows),
        "formula_count": sum(int(row["formula_count"]) for row in sheet_rows),
        "scientific_decisions": {
            "2024_sd3": "Primary NDM-1 amino-acid DMS label source.",
            "2024_sd4": "Cross-homologue comparison table; not independent labels.",
            "2020_sf2a": "Primary VIM-2 amino-acid DMS label source with condition-specific columns.",
            "2020_auxiliary_sheets": "Annotations or derived subsets; preserve but do not duplicate labels.",
            "2020_sf3": "Published aggregate regression outputs; not mutation-level training data.",
            "2022_sd1": "Primary concentration-conditioned VIM-2 fitness source.",
            "2022_sd2_sd3": "Distinct EC50 outcomes with overlap that requires mutation grouping.",
            "numbering": "VIM-2 sheets retain the in-house Gly2 convention.",
        },
        "outputs": [
            "workbook_metadata.csv",
            "worksheet_metadata.csv",
            "column_definitions.csv",
            "mutation_field_candidates.csv",
            "mutation_field_examples.csv",
            "source_table_relationships.csv",
        ],
    }
    with (output_dir / "source_table_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
