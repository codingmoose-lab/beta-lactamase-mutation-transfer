# Prepare AlphaFold reference models for residue-level scientific analysis.

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


class AlphaFoldPreparationError(RuntimeError):
    """Raised when AlphaFold preparation cannot be completed without ambiguity."""


PROTEINS: Mapping[str, Mapping[str, str]] = {
    "NDM-1": {
        "accession": "C7C422",
        "model_id": "AF-C7C422-F1",
        "pdb_id": "3SPU",
    },
    "VIM-2": {
        "accession": "Q9K2N0",
        "model_id": "AF-Q9K2N0-F1",
        "pdb_id": "5YD7",
    },
}
DEFAULT_RADII = (6.0, 8.0, 10.0, 12.0)
_CONFIDENCE_CODE_RANGES = {
    "D": (0.0, 50.0),
    "L": (50.0, 70.0),
    "M": (70.0, 90.0),
    "H": (90.0, 100.000001),
}
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AlphaFoldPreparationError(
            f"Invalid numeric value for {label}: {value}"
        ) from exc
    if not math.isfinite(number):
        raise AlphaFoldPreparationError(
            f"Non-finite numeric value for {label}: {value}"
        )
    return number


def _int(value: object, label: str) -> int:
    number = _float(value, label)
    if not number.is_integer():
        raise AlphaFoldPreparationError(f"Expected integer value for {label}: {value}")
    return int(number)


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise AlphaFoldPreparationError(f"Missing required JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AlphaFoldPreparationError(f"Invalid JSON file: {path}") from exc


def _read_fasta_sequence(path: Path) -> str:
    if not path.is_file():
        raise AlphaFoldPreparationError(f"Missing UniProt FASTA file: {path}")
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(">")
    ).upper()
    if not sequence:
        raise AlphaFoldPreparationError(f"UniProt FASTA contains no sequence: {path}")
    return sequence


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise AlphaFoldPreparationError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def parse_confidence_document(
    payload: object,
    *,
    expected_length: int,
) -> list[Mapping[str, object]]:
    """Validate an AlphaFold confidence document and return residue records."""
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise AlphaFoldPreparationError(
                "Confidence JSON list form must contain exactly one object"
            )
        document = payload[0]
    elif isinstance(payload, Mapping):
        document = payload
    else:
        raise AlphaFoldPreparationError(
            "Confidence JSON root must be an object or a one-object list"
        )

    required = ("residueNumber", "confidenceScore", "confidenceCategory")
    missing = [key for key in required if key not in document]
    if missing:
        raise AlphaFoldPreparationError(
            f"Confidence JSON is missing fields: {', '.join(missing)}"
        )

    numbers = list(document["residueNumber"])
    scores = list(document["confidenceScore"])
    categories = list(document["confidenceCategory"])
    lengths = {len(numbers), len(scores), len(categories)}
    if lengths != {expected_length}:
        raise AlphaFoldPreparationError(
            "Confidence arrays do not match expected sequence length "
            f"{expected_length}: residueNumber={len(numbers)}, "
            f"confidenceScore={len(scores)}, confidenceCategory={len(categories)}"
        )

    expected_numbers = list(range(1, expected_length + 1))
    parsed_numbers = [_int(value, "confidence residueNumber") for value in numbers]
    if parsed_numbers != expected_numbers:
        raise AlphaFoldPreparationError(
            "Confidence residueNumber values are not exactly 1..sequence_length"
        )

    rows: list[Mapping[str, object]] = []
    for position, score_value, category_value in zip(
        parsed_numbers, scores, categories
    ):
        score = _float(score_value, f"pLDDT at residue {position}")
        if not 0.0 <= score <= 100.0:
            raise AlphaFoldPreparationError(
                f"pLDDT outside [0, 100] at residue {position}: {score}"
            )
        category = str(category_value).strip()
        if category not in _CONFIDENCE_CODE_RANGES:
            raise AlphaFoldPreparationError(
                f"Unrecognized confidence category at residue {position}: {category}"
            )
        lower, upper = _CONFIDENCE_CODE_RANGES[category]
        if not lower <= score < upper:
            raise AlphaFoldPreparationError(
                "Confidence category is inconsistent with pLDDT at residue "
                f"{position}: category={category}, score={score}"
            )
        rows.append(
            {
                "alphafold_position": position,
                "plddt": score,
                "confidence_category": category,
                "confidence_category_score_consistent": True,
            }
        )
    return rows


def parse_pae_document(
    payload: object,
    *,
    expected_length: int,
) -> tuple[np.ndarray, float]:
    """Validate an AlphaFold PAE document and return its matrix and declared maximum."""
    if not isinstance(payload, list) or len(payload) != 1:
        raise AlphaFoldPreparationError(
            "PAE JSON root must be a one-object list for the acquired AlphaFold records"
        )
    document = payload[0]
    if not isinstance(document, Mapping):
        raise AlphaFoldPreparationError("PAE JSON list member is not an object")
    required = ("predicted_aligned_error", "max_predicted_aligned_error")
    missing = [key for key in required if key not in document]
    if missing:
        raise AlphaFoldPreparationError(
            f"PAE JSON is missing fields: {', '.join(missing)}"
        )

    try:
        matrix = np.asarray(document["predicted_aligned_error"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise AlphaFoldPreparationError("PAE matrix is not numeric") from exc
    if matrix.shape != (expected_length, expected_length):
        raise AlphaFoldPreparationError(
            f"PAE matrix shape {matrix.shape} does not match "
            f"({expected_length}, {expected_length})"
        )
    if not np.isfinite(matrix).all():
        raise AlphaFoldPreparationError("PAE matrix contains non-finite values")
    if (matrix < 0.0).any():
        raise AlphaFoldPreparationError("PAE matrix contains negative values")

    declared_maximum = _float(
        document["max_predicted_aligned_error"],
        "max_predicted_aligned_error",
    )
    if declared_maximum < 0.0:
        raise AlphaFoldPreparationError("Declared maximum PAE is negative")
    if float(matrix.max()) > declared_maximum + 1e-6:
        raise AlphaFoldPreparationError(
            "Observed PAE exceeds max_predicted_aligned_error"
        )
    return matrix, declared_maximum


def _mmcif_polymer_sequence(cif: Mapping[str, object]) -> str:
    values = cif.get("_entity_poly.pdbx_seq_one_letter_code_can")
    if values is None:
        raise AlphaFoldPreparationError(
            "AlphaFold mmCIF is missing _entity_poly.pdbx_seq_one_letter_code_can"
        )
    sequences = ["".join(value.split()).upper() for value in _as_list(values)]
    sequences = [sequence for sequence in sequences if sequence]
    if len(sequences) != 1:
        raise AlphaFoldPreparationError(
            f"Expected one AlphaFold polymer sequence, found {len(sequences)}"
        )
    return sequences[0]


def parse_alphafold_mmcif(path: Path) -> tuple[str, list[Mapping[str, object]]]:
    """Parse the AlphaFold monomer sequence and one C-alpha record per residue."""
    if not path.is_file():
        raise AlphaFoldPreparationError(f"Missing AlphaFold mmCIF: {path}")
    cif = MMCIF2Dict(str(path))
    sequence = _mmcif_polymer_sequence(cif)
    required = (
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    )
    missing = [key for key in required if key not in cif]
    if missing:
        raise AlphaFoldPreparationError(
            f"AlphaFold mmCIF is missing atom fields: {', '.join(missing)}"
        )
    columns = {key: _as_list(cif[key]) for key in required}
    count = len(columns["_atom_site.group_PDB"])
    if any(len(values) != count for values in columns.values()):
        raise AlphaFoldPreparationError("AlphaFold mmCIF atom columns differ in length")

    residue_rows: dict[int, Mapping[str, object]] = {}
    chain_ids: set[str] = set()
    for index in range(count):
        if columns["_atom_site.group_PDB"][index] != "ATOM":
            continue
        chain_ids.add(columns["_atom_site.label_asym_id"][index])
        if columns["_atom_site.label_atom_id"][index] != "CA":
            continue
        position = _int(columns["_atom_site.label_seq_id"][index], "label_seq_id")
        residue_name = columns["_atom_site.label_comp_id"][index].upper()
        residue = _THREE_TO_ONE.get(residue_name)
        if residue is None:
            raise AlphaFoldPreparationError(
                f"Unsupported AlphaFold residue name {residue_name} at {position}"
            )
        if position in residue_rows:
            raise AlphaFoldPreparationError(
                f"Multiple C-alpha coordinates for AlphaFold residue {position}"
            )
        residue_rows[position] = {
            "alphafold_position": position,
            "alphafold_residue": residue,
            "chain_id": columns["_atom_site.label_asym_id"][index],
            "x": _float(columns["_atom_site.Cartn_x"][index], "Cartn_x"),
            "y": _float(columns["_atom_site.Cartn_y"][index], "Cartn_y"),
            "z": _float(columns["_atom_site.Cartn_z"][index], "Cartn_z"),
            "occupancy": _float(columns["_atom_site.occupancy"][index], "occupancy"),
            "mmcif_plddt": _float(
                columns["_atom_site.B_iso_or_equiv"][index], "B_iso_or_equiv"
            ),
        }

    if chain_ids != {"A"}:
        raise AlphaFoldPreparationError(
            f"Expected AlphaFold monomer chain A, found {sorted(chain_ids)}"
        )
    expected_positions = set(range(1, len(sequence) + 1))
    if set(residue_rows) != expected_positions:
        missing_positions = sorted(expected_positions - set(residue_rows))
        extra_positions = sorted(set(residue_rows) - expected_positions)
        raise AlphaFoldPreparationError(
            "AlphaFold C-alpha coverage is incomplete or inconsistent: "
            f"missing={missing_positions[:10]}, extra={extra_positions[:10]}"
        )

    rows = [residue_rows[position] for position in range(1, len(sequence) + 1)]
    coordinate_sequence = "".join(str(row["alphafold_residue"]) for row in rows)
    if coordinate_sequence != sequence:
        raise AlphaFoldPreparationError(
            "AlphaFold coordinate residue sequence differs from the mmCIF polymer "
            "mmCIF sequence"
        )
    return sequence, rows


def _parse_selected_pdb_ca(
    cif_path: Path,
    *,
    chain_id: str,
) -> Mapping[tuple[str, str], Mapping[str, object]]:
    if not cif_path.is_file():
        raise AlphaFoldPreparationError(f"Missing experimental mmCIF: {cif_path}")
    cif = MMCIF2Dict(str(cif_path))
    required = (
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
    )
    missing = [key for key in required if key not in cif]
    if missing:
        raise AlphaFoldPreparationError(
            f"Experimental mmCIF is missing atom fields: {', '.join(missing)}"
        )
    columns = {key: _as_list(cif[key]) for key in required}
    count = len(columns["_atom_site.group_PDB"])
    altlocs = _as_list(cif.get("_atom_site.label_alt_id", ["."] * count))
    insertions = _as_list(cif.get("_atom_site.pdbx_PDB_ins_code", ["?"] * count))
    models = _as_list(cif.get("_atom_site.pdbx_PDB_model_num", ["1"] * count))

    candidates: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for index in range(count):
        if models[index] not in {"1", "1.0", ".", "?"}:
            continue
        if columns["_atom_site.group_PDB"][index] != "ATOM":
            continue
        if columns["_atom_site.auth_asym_id"][index] != chain_id:
            continue
        if columns["_atom_site.label_atom_id"][index] != "CA":
            continue
        insertion = insertions[index]
        if insertion in {".", ""}:
            insertion = "?"
        key = (columns["_atom_site.auth_seq_id"][index], insertion)
        residue_name = columns["_atom_site.label_comp_id"][index].upper()
        residue = _THREE_TO_ONE.get(residue_name)
        if residue is None:
            raise AlphaFoldPreparationError(
                f"Unsupported PDB residue name {residue_name} at {key}"
            )
        candidates.setdefault(key, []).append(
            {
                "pdb_residue": residue,
                "x": _float(columns["_atom_site.Cartn_x"][index], "Cartn_x"),
                "y": _float(columns["_atom_site.Cartn_y"][index], "Cartn_y"),
                "z": _float(columns["_atom_site.Cartn_z"][index], "Cartn_z"),
                "occupancy": _float(
                    columns["_atom_site.occupancy"][index], "occupancy"
                ),
                "altloc": altlocs[index],
            }
        )

    selected: dict[tuple[str, str], Mapping[str, object]] = {}
    for key, values in candidates.items():
        ordered = sorted(
            values,
            key=lambda row: (
                row["altloc"] not in {".", "?", ""},
                -float(row["occupancy"]),
                str(row["altloc"]),
            ),
        )
        selected[key] = ordered[0]
    if not selected:
        raise AlphaFoldPreparationError(
            f"No C-alpha coordinates found for experimental chain {chain_id}"
        )
    return selected


def kabsch_superpose(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rigidly superpose mobile coordinates onto target coordinates."""
    mobile_array = np.asarray(mobile, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if mobile_array.shape != target_array.shape:
        raise AlphaFoldPreparationError(
            f"Coordinate arrays have different shapes: {mobile_array.shape}, "
            f"{target_array.shape}"
        )
    if mobile_array.ndim != 2 or mobile_array.shape[1] != 3:
        raise AlphaFoldPreparationError("Coordinates must have shape (n, 3)")
    if mobile_array.shape[0] < 3:
        raise AlphaFoldPreparationError(
            "At least three residues are required to superpose"
        )
    if not np.isfinite(mobile_array).all() or not np.isfinite(target_array).all():
        raise AlphaFoldPreparationError(
            "Superposition coordinates contain non-finite values"
        )

    mobile_centroid = mobile_array.mean(axis=0)
    target_centroid = target_array.mean(axis=0)
    mobile_centered = mobile_array - mobile_centroid
    target_centered = target_array - target_centroid
    covariance = mobile_centered.T @ target_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = u_matrix @ vt_matrix
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1, :] *= -1
        rotation = u_matrix @ vt_matrix
    translation = target_centroid - mobile_centroid @ rotation
    transformed = mobile_array @ rotation + translation
    residuals = np.linalg.norm(transformed - target_array, axis=1)
    rmsd = float(np.sqrt(np.mean(np.square(residuals))))
    return transformed, rotation, translation, rmsd


def _local_confidence_rows(
    protein_id: str,
    assay_mapping: Sequence[Mapping[str, object]],
    af_rows: Sequence[Mapping[str, object]],
    pae: np.ndarray,
    radii: Sequence[float],
) -> list[Mapping[str, object]]:
    by_position = {int(row["alphafold_position"]): row for row in af_rows}
    coordinates = {
        position: np.array([row["x"], row["y"], row["z"]], dtype=float)
        for position, row in by_position.items()
    }
    rows: list[Mapping[str, object]] = []
    for mapping in assay_mapping:
        af_position = mapping["alphafold_position"]
        if af_position is None:
            continue
        center_position = int(af_position)
        center = coordinates[center_position]
        for radius in radii:
            neighbours = [
                position
                for position, coordinate in coordinates.items()
                if float(np.linalg.norm(coordinate - center)) <= radius + 1e-12
            ]
            plddt_values = [
                float(by_position[position]["plddt"]) for position in neighbours
            ]
            center_index = center_position - 1
            neighbour_indices = np.asarray([position - 1 for position in neighbours])
            rows.append(
                {
                    "protein_id": protein_id,
                    "assay_position": mapping["assay_position"],
                    "assay_residue": mapping["assay_residue"],
                    "uniprot_position": mapping["uniprot_position"],
                    "alphafold_position": center_position,
                    "radius_angstrom": radius,
                    "neighbour_residue_count": len(neighbours),
                    "mean_neighbour_plddt": statistics.fmean(plddt_values),
                    "median_neighbour_plddt": statistics.median(plddt_values),
                    "minimum_neighbour_plddt": min(plddt_values),
                    "maximum_neighbour_plddt": max(plddt_values),
                    "mean_pae_center_i_aligned_on_neighbour_j": float(
                        pae[center_index, neighbour_indices].mean()
                    ),
                    "mean_pae_neighbour_i_aligned_on_center_j": float(
                        pae[neighbour_indices, center_index].mean()
                    ),
                }
            )
    return rows


def _assay_to_alphafold_mapping(
    residue_mapping: pd.DataFrame,
    *,
    protein_id: str,
    primary_chain: str,
    af_sequence: str,
) -> list[Mapping[str, object]]:
    subset = residue_mapping[
        (residue_mapping["protein_id"] == protein_id)
        & (residue_mapping["pdb_chain"] == primary_chain)
    ].copy()
    subset = subset.sort_values("assay_position")
    if subset["assay_position"].duplicated().any():
        raise AlphaFoldPreparationError(
            f"Multiple mapping rows for {protein_id} primary chain {primary_chain}"
        )

    rows: list[Mapping[str, object]] = []
    for record in subset.to_dict("records"):
        uniprot_position_value = record["uniprot_position"]
        if pd.isna(uniprot_position_value):
            uniprot_position = None
            alphafold_position = None
            alphafold_residue = None
            af_identity_ok = False
        else:
            uniprot_position = int(uniprot_position_value)
            if not 1 <= uniprot_position <= len(af_sequence):
                raise AlphaFoldPreparationError(
                    f"UniProt position outside AlphaFold sequence for {protein_id}: "
                    f"{uniprot_position}"
                )
            alphafold_position = uniprot_position
            alphafold_residue = af_sequence[alphafold_position - 1]
            af_identity_ok = str(record["uniprot_residue"]) == alphafold_residue
            if not af_identity_ok:
                raise AlphaFoldPreparationError(
                    f"UniProt-to-AlphaFold residue mismatch for {protein_id} "
                    f"position {uniprot_position}: {record['uniprot_residue']} vs "
                    f"{alphafold_residue}"
                )

        assay_residue = str(record["assay_residue"])
        assay_uniprot_identity = (
            False
            if uniprot_position is None
            else assay_residue == str(record["uniprot_residue"])
        )
        assay_specific_insertion = record["mapping_status"] == "assay_insertion"
        experimental_pdb_modeled = bool(record["pdb_modeled"])
        if assay_specific_insertion:
            alphafold_mapping_status = "assay_specific_insertion"
            coverage_class = "assay_specific_no_reference_coordinate"
        elif alphafold_position is not None:
            alphafold_mapping_status = "mapped_via_uniprot"
            coverage_class = (
                "experimental_and_alphafold"
                if experimental_pdb_modeled
                else "alphafold_only"
            )
        else:
            alphafold_mapping_status = "unmapped"
            coverage_class = "unmapped"
        pdb_author_position = record["pdb_author_position"]
        rows.append(
            {
                "protein_id": protein_id,
                "assay_position": int(record["assay_position"]),
                "assay_residue": assay_residue,
                "uniprot_position": uniprot_position,
                "uniprot_residue": (
                    None
                    if pd.isna(record["uniprot_residue"])
                    else record["uniprot_residue"]
                ),
                "alphafold_position": alphafold_position,
                "alphafold_residue": alphafold_residue,
                "alphafold_mapping_status": alphafold_mapping_status,
                "source_residue_mapping_status": record["mapping_status"],
                "experimental_pdb_modeled": experimental_pdb_modeled,
                "experimental_pdb_author_position": (
                    None if pd.isna(pdb_author_position) else pdb_author_position
                ),
                "coverage_class": coverage_class,
                "assay_uniprot_residue_identity": assay_uniprot_identity,
                "uniprot_alphafold_residue_identity": af_identity_ok,
                "assay_specific_insertion": assay_specific_insertion,
                "known_assay_reference_difference": (
                    uniprot_position is not None and not assay_uniprot_identity
                ),
            }
        )
    return rows


def _pdb_alphafold_correspondence(
    residue_mapping: pd.DataFrame,
    *,
    protein_id: str,
    pdb_id: str,
    primary_chain: str,
    pdb_ca: Mapping[tuple[str, str], Mapping[str, object]],
    af_rows: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
    af_by_position = {int(row["alphafold_position"]): row for row in af_rows}
    subset = residue_mapping[
        (residue_mapping["protein_id"] == protein_id)
        & (residue_mapping["pdb_id"] == pdb_id)
        & (residue_mapping["pdb_chain"] == primary_chain)
        & residue_mapping["uniprot_position"].notna()
        & residue_mapping["pdb_modeled"].astype(bool)
    ].copy()
    subset = subset.sort_values("uniprot_position").drop_duplicates("uniprot_position")

    records: list[Mapping[str, object]] = []
    mobile: list[list[float]] = []
    target: list[list[float]] = []
    for record in subset.to_dict("records"):
        uniprot_position = int(record["uniprot_position"])
        af = af_by_position.get(uniprot_position)
        if af is None:
            raise AlphaFoldPreparationError(
                "AlphaFold residue missing at mapped UniProt position "
                f"{uniprot_position}"
            )
        author_position = str(int(float(record["pdb_author_position"])))
        insertion = record.get("pdb_insertion_code")
        insertion_text = (
            "?" if pd.isna(insertion) or insertion in {"", "."} else str(insertion)
        )
        pdb = pdb_ca.get((author_position, insertion_text))
        if pdb is None:
            raise AlphaFoldPreparationError(
                f"Mapped PDB C-alpha coordinate missing for {protein_id} "
                f"{pdb_id}-{primary_chain} author residue "
                f"{author_position}{insertion_text}"
            )
        if str(record["pdb_entity_residue"]) != str(af["alphafold_residue"]):
            raise AlphaFoldPreparationError(
                f"PDB entity and AlphaFold residues differ at {protein_id} "
                f"UniProt position {uniprot_position}"
            )
        records.append(
            {
                "protein_id": protein_id,
                "pdb_id": pdb_id,
                "pdb_chain": primary_chain,
                "uniprot_position": uniprot_position,
                "reference_residue": af["alphafold_residue"],
                "pdb_author_position": author_position,
                "pdb_insertion_code": insertion_text,
                "alphafold_position": uniprot_position,
                "alphafold_plddt": af["plddt"],
                "pdb_ca_altloc": pdb["altloc"],
                "pdb_ca_occupancy": pdb["occupancy"],
            }
        )
        mobile.append([float(af["x"]), float(af["y"]), float(af["z"])])
        target.append([float(pdb["x"]), float(pdb["y"]), float(pdb["z"])])

    if len(records) < 3:
        raise AlphaFoldPreparationError(
            f"Too few equivalent residues for {protein_id} PDB-AlphaFold superposition"
        )
    transformed, rotation, translation, rmsd = kabsch_superpose(
        np.asarray(mobile), np.asarray(target)
    )
    residuals = np.linalg.norm(transformed - np.asarray(target), axis=1)
    output_rows: list[Mapping[str, object]] = []
    for record, transformed_coordinate, displacement in zip(
        records, transformed, residuals
    ):
        output_rows.append(
            {
                **record,
                "superposed_alphafold_ca_x": float(transformed_coordinate[0]),
                "superposed_alphafold_ca_y": float(transformed_coordinate[1]),
                "superposed_alphafold_ca_z": float(transformed_coordinate[2]),
                "ca_displacement_angstrom": float(displacement),
            }
        )

    summary = {
        "protein_id": protein_id,
        "pdb_id": pdb_id,
        "pdb_chain": primary_chain,
        "equivalent_ca_count": len(records),
        "ca_rmsd_angstrom": rmsd,
        "mean_ca_displacement_angstrom": float(residuals.mean()),
        "median_ca_displacement_angstrom": float(np.median(residuals)),
        "p95_ca_displacement_angstrom": float(np.percentile(residuals, 95)),
        "maximum_ca_displacement_angstrom": float(residuals.max()),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
        "translation_z": float(translation[2]),
    }
    plddt_values = np.asarray([float(record["alphafold_plddt"]) for record in records])
    mobile_array = np.asarray(mobile)
    target_array = np.asarray(target)
    for label, threshold in (("confident", 70.0), ("very_high", 90.0)):
        mask = plddt_values >= threshold
        subset_count = int(mask.sum())
        summary[f"{label}_plddt_minimum"] = threshold
        summary[f"{label}_equivalent_ca_count"] = subset_count
        if subset_count >= 3:
            _, _, _, subset_rmsd = kabsch_superpose(
                mobile_array[mask], target_array[mask]
            )
            summary[f"{label}_ca_rmsd_angstrom"] = subset_rmsd
        else:
            summary[f"{label}_ca_rmsd_angstrom"] = None
    return output_rows, summary


def prepare_alphafold_models(
    repo_root: Path,
    *,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Mapping[str, object]:
    """Validate and prepare the two acquired AlphaFold reference models."""
    root = repo_root.resolve()
    output_dir = root / "data/interim/alphafold_preparation"
    mapping_path = root / "data/interim/sequence_mapping/residue_number_mapping.csv"
    selected_path = root / "data/interim/protein_structures/selected_structures.json"
    if not mapping_path.is_file():
        raise AlphaFoldPreparationError(
            f"Missing residue mapping table: {mapping_path}"
        )
    if not selected_path.is_file():
        raise AlphaFoldPreparationError(
            f"Missing selected-structure record: {selected_path}"
        )

    normalized_radii = tuple(float(radius) for radius in radii)
    if not normalized_radii or any(radius <= 0 for radius in normalized_radii):
        raise AlphaFoldPreparationError("Local confidence radii must be positive")
    if len(set(normalized_radii)) != len(normalized_radii):
        raise AlphaFoldPreparationError("Local confidence radii must be unique")

    residue_mapping = pd.read_csv(mapping_path, low_memory=False)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))

    model_rows: list[Mapping[str, object]] = []
    confidence_rows: list[Mapping[str, object]] = []
    pae_rows: list[Mapping[str, object]] = []
    assay_mapping_rows: list[Mapping[str, object]] = []
    local_rows: list[Mapping[str, object]] = []
    correspondence_rows: list[Mapping[str, object]] = []
    correspondence_summaries: list[Mapping[str, object]] = []

    for protein_id, definition in PROTEINS.items():
        accession = definition["accession"]
        expected_model_id = definition["model_id"]
        pdb_id = definition["pdb_id"]
        if protein_id not in selected:
            raise AlphaFoldPreparationError(
                f"Selected experimental structure is missing for {protein_id}"
            )
        primary_chain = str(selected[protein_id]["primary_chain"])
        if str(selected[protein_id]["pdb_id"]) != pdb_id:
            raise AlphaFoldPreparationError(
                "Selected PDB for "
                f"{protein_id} differs from recorded reference {pdb_id}"
            )

        raw_dir = root / "data/raw/alphafold" / accession
        metadata = _load_json(raw_dir / "metadata.json")
        if isinstance(metadata, list):
            if len(metadata) != 1 or not isinstance(metadata[0], Mapping):
                raise AlphaFoldPreparationError(
                    f"Unexpected AlphaFold metadata form for {accession}"
                )
            record = metadata[0]
        elif isinstance(metadata, Mapping):
            record = metadata
        else:
            raise AlphaFoldPreparationError(
                f"Unexpected AlphaFold metadata root for {accession}"
            )

        model_id = str(record.get("modelEntityId", "")).strip()
        metadata_accession = str(record.get("uniprotAccession", "")).strip().upper()
        api_sequence = "".join(str(record.get("sequence", "")).split()).upper()
        if model_id != expected_model_id:
            raise AlphaFoldPreparationError(
                f"Unexpected model identifier for {accession}: {model_id}"
            )
        if metadata_accession != accession:
            raise AlphaFoldPreparationError(
                f"Unexpected accession in metadata for {accession}: "
                f"{metadata_accession}"
            )
        if not api_sequence:
            raise AlphaFoldPreparationError(
                f"AlphaFold metadata has no sequence: {accession}"
            )
        sequence_start = _int(record.get("sequenceStart"), "sequenceStart")
        sequence_end = _int(record.get("sequenceEnd"), "sequenceEnd")
        if sequence_start != 1 or sequence_end != len(api_sequence):
            raise AlphaFoldPreparationError(
                f"AlphaFold metadata sequence range is inconsistent for {accession}"
            )

        uniprot_sequence = _read_fasta_sequence(
            root / "data/raw/uniprot" / accession / f"{accession}.fasta"
        )
        cif_sequence, af_coordinate_rows = parse_alphafold_mmcif(raw_dir / "model.cif")
        if api_sequence != uniprot_sequence or cif_sequence != uniprot_sequence:
            raise AlphaFoldPreparationError(
                "AlphaFold metadata/mmCIF sequence differs from UniProt for "
                f"{accession}"
            )
        sequence_length = len(uniprot_sequence)

        confidence = parse_confidence_document(
            _load_json(raw_dir / "confidence.json"),
            expected_length=sequence_length,
        )
        confidence_by_position = {
            int(row["alphafold_position"]): row for row in confidence
        }
        maximum_plddt_difference = 0.0
        enriched_af_rows: list[Mapping[str, object]] = []
        for coordinate in af_coordinate_rows:
            position = int(coordinate["alphafold_position"])
            confidence_record = confidence_by_position[position]
            difference = abs(
                float(coordinate["mmcif_plddt"]) - float(confidence_record["plddt"])
            )
            maximum_plddt_difference = max(maximum_plddt_difference, difference)
            enriched = {
                "protein_id": protein_id,
                "accession": accession,
                **coordinate,
                "plddt": confidence_record["plddt"],
                "confidence_category": confidence_record["confidence_category"],
                "confidence_category_score_consistent": confidence_record[
                    "confidence_category_score_consistent"
                ],
                "confidence_mmcif_absolute_difference": difference,
            }
            enriched_af_rows.append(enriched)
            confidence_rows.append(enriched)
        if maximum_plddt_difference > 1e-6:
            raise AlphaFoldPreparationError(
                f"Confidence JSON and mmCIF pLDDT differ for {accession}; "
                f"maximum absolute difference={maximum_plddt_difference}"
            )

        pae_matrix, declared_maximum_pae = parse_pae_document(
            _load_json(raw_dir / "pae.json"),
            expected_length=sequence_length,
        )
        for position in range(1, sequence_length + 1):
            row_values = pae_matrix[position - 1, :]
            column_values = pae_matrix[:, position - 1]
            pae_rows.append(
                {
                    "protein_id": protein_id,
                    "accession": accession,
                    "alphafold_position": position,
                    "mean_pae_residue_i_aligned_on_all_j": float(row_values.mean()),
                    "median_pae_residue_i_aligned_on_all_j": float(
                        np.median(row_values)
                    ),
                    "maximum_pae_residue_i_aligned_on_all_j": float(row_values.max()),
                    "mean_pae_all_i_aligned_on_residue_j": float(column_values.mean()),
                    "median_pae_all_i_aligned_on_residue_j": float(
                        np.median(column_values)
                    ),
                    "maximum_pae_all_i_aligned_on_residue_j": float(
                        column_values.max()
                    ),
                }
            )

        assay_mapping = _assay_to_alphafold_mapping(
            residue_mapping,
            protein_id=protein_id,
            primary_chain=primary_chain,
            af_sequence=uniprot_sequence,
        )
        assay_mapping_rows.extend(assay_mapping)
        local_rows.extend(
            _local_confidence_rows(
                protein_id,
                assay_mapping,
                enriched_af_rows,
                pae_matrix,
                normalized_radii,
            )
        )

        pdb_ca = _parse_selected_pdb_ca(
            root / "data/raw/pdb" / pdb_id / f"{pdb_id}.cif",
            chain_id=primary_chain,
        )
        correspondence, correspondence_summary = _pdb_alphafold_correspondence(
            residue_mapping,
            protein_id=protein_id,
            pdb_id=pdb_id,
            primary_chain=primary_chain,
            pdb_ca=pdb_ca,
            af_rows=enriched_af_rows,
        )
        correspondence_rows.extend(correspondence)
        correspondence_summaries.append(correspondence_summary)

        primary_mapping = residue_mapping[
            (residue_mapping["protein_id"] == protein_id)
            & (residue_mapping["pdb_chain"] == primary_chain)
        ]
        modeled_positions = int(
            primary_mapping.loc[
                primary_mapping["pdb_modeled"].astype(bool), "assay_position"
            ].nunique()
        )
        model_rows.append(
            {
                "protein_id": protein_id,
                "accession": accession,
                "model_id": model_id,
                "model_version": _int(record.get("latestVersion"), "latestVersion"),
                "sequence_length": sequence_length,
                "chain_id": "A",
                "coordinate_residue_count": len(enriched_af_rows),
                "coordinate_atom_count": sum(
                    1
                    for value in _as_list(
                        MMCIF2Dict(str(raw_dir / "model.cif"))["_atom_site.group_PDB"]
                    )
                    if value == "ATOM"
                ),
                "global_metric_value": _float(
                    record.get("globalMetricValue"), "globalMetricValue"
                ),
                "confidence_json_residue_count": len(confidence),
                "maximum_confidence_mmcif_absolute_difference": (
                    maximum_plddt_difference
                ),
                "pae_rows": pae_matrix.shape[0],
                "pae_columns": pae_matrix.shape[1],
                "declared_maximum_pae": declared_maximum_pae,
                "observed_maximum_pae": float(pae_matrix.max()),
                "assay_position_count": len(assay_mapping),
                "assay_positions_mapped_to_alphafold": sum(
                    row["alphafold_position"] is not None for row in assay_mapping
                ),
                "assay_specific_insertion_count": sum(
                    bool(row["assay_specific_insertion"]) for row in assay_mapping
                ),
                "known_assay_reference_difference_count": sum(
                    bool(row["known_assay_reference_difference"])
                    for row in assay_mapping
                ),
                "experimental_pdb_modeled_assay_position_count": modeled_positions,
                "alphafold_only_assay_position_count": sum(
                    row["coverage_class"] == "alphafold_only" for row in assay_mapping
                ),
                "alphafold_full_reference_coverage_fraction": len(enriched_af_rows)
                / sequence_length,
            }
        )

    _write_csv(output_dir / "model_metadata.csv", model_rows)
    _write_csv(output_dir / "residue_confidence.csv", confidence_rows)
    _write_csv(output_dir / "residue_pae_summary.csv", pae_rows)
    _write_csv(output_dir / "assay_alphafold_mapping.csv", assay_mapping_rows)
    _write_csv(output_dir / "local_confidence.csv", local_rows)
    _write_csv(output_dir / "pdb_alphafold_correspondence.csv", correspondence_rows)
    _write_csv(output_dir / "pdb_alphafold_summary.csv", correspondence_summaries)

    summary = {
        "status": "PASS",
        "protein_count": len(PROTEINS),
        "model_count": len(model_rows),
        "residue_confidence_row_count": len(confidence_rows),
        "residue_pae_summary_row_count": len(pae_rows),
        "assay_alphafold_mapping_row_count": len(assay_mapping_rows),
        "local_confidence_row_count": len(local_rows),
        "pdb_alphafold_correspondence_row_count": len(correspondence_rows),
        "local_radii_angstrom": list(normalized_radii),
        "models": model_rows,
        "pdb_alphafold": correspondence_summaries,
        "outputs": [
            "model_metadata.csv",
            "residue_confidence.csv",
            "residue_pae_summary.csv",
            "assay_alphafold_mapping.csv",
            "local_confidence.csv",
            "pdb_alphafold_correspondence.csv",
            "pdb_alphafold_summary.csv",
            "preparation_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "preparation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
