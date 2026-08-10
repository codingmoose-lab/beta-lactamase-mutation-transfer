# Prepare experimental protein structures for structural feature generation.

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
)

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


class ProteinStructureError(RuntimeError):
    """Raised when structure inputs are incomplete or inconsistent."""


_BACKBONE_ATOMS = {"N", "CA", "C", "O"}
_WATER_NAMES = {"HOH", "DOD", "WAT"}
_STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
_EXPECTED_SIDE_CHAIN_ATOMS = {
    "ALA": {"CB"},
    "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"CB", "CG", "OD1", "ND2"},
    "ASP": {"CB", "CG", "OD1", "OD2"},
    "CYS": {"CB", "SG"},
    "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"CB", "CG", "CD", "OE1", "OE2"},
    "GLY": set(),
    "HIS": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "LYS": {"CB", "CG", "CD", "CE", "NZ"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"},
    "SER": {"CB", "OG"},
    "THR": {"CB", "OG1", "CG2"},
    "TRP": {
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    },
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"CB", "CG1", "CG2"},
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ProteinStructureError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise ProteinStructureError(f"Refusing to write an empty table: {path}")
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


def _as_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _first(cif: Mapping[str, object], key: str, default: str = "") -> str:
    if key not in cif:
        return default
    values = _as_list(cif[key])
    return values[0] if values else default


def _float_or_none(value: object) -> float | None:
    text = str(value).strip()
    if text in {"", ".", "?", "None", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(value: object) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _normalize_position(value: object) -> str:
    number = _float_or_none(value)
    if number is not None and number.is_integer():
        return str(int(number))
    return str(value).strip()


def _distance(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    return math.sqrt(
        (float(left["x"]) - float(right["x"])) ** 2
        + (float(left["y"]) - float(right["y"])) ** 2
        + (float(left["z"]) - float(right["z"])) ** 2
    )


def _parse_atom_rows(cif_path: Path) -> List[Mapping[str, object]]:
    cif = MMCIF2Dict(str(cif_path))
    required = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    ]
    missing = [key for key in required if key not in cif]
    if missing:
        raise ProteinStructureError(
            f"Missing atom-site fields in {cif_path}: {', '.join(missing)}"
        )

    columns = {key: _as_list(cif[key]) for key in required}
    optional = {
        "_atom_site.label_alt_id": ".",
        "_atom_site.pdbx_PDB_ins_code": "?",
        "_atom_site.label_seq_id": "?",
        "_atom_site.pdbx_PDB_model_num": "1",
    }
    for key, default in optional.items():
        if key in cif:
            columns[key] = _as_list(cif[key])
        else:
            columns[key] = [default] * len(columns["_atom_site.id"])

    rows: List[Mapping[str, object]] = []
    count = len(columns["_atom_site.id"])
    for index in range(count):
        model = columns["_atom_site.pdbx_PDB_model_num"][index]
        if model not in {"1", "1.0", ".", "?"}:
            continue
        element = columns["_atom_site.type_symbol"][index].upper()
        if element in {"H", "D"}:
            continue
        rows.append(
            {
                "group": columns["_atom_site.group_PDB"][index],
                "atom_id": columns["_atom_site.id"][index],
                "element": element,
                "atom_name": columns["_atom_site.label_atom_id"][index],
                "residue_name": columns["_atom_site.label_comp_id"][index],
                "label_chain": columns["_atom_site.label_asym_id"][index],
                "chain_id": columns["_atom_site.auth_asym_id"][index],
                "label_seq_id": columns["_atom_site.label_seq_id"][index],
                "author_seq_id": columns["_atom_site.auth_seq_id"][index],
                "insertion_code": columns["_atom_site.pdbx_PDB_ins_code"][index],
                "altloc": columns["_atom_site.label_alt_id"][index],
                "x": float(columns["_atom_site.Cartn_x"][index]),
                "y": float(columns["_atom_site.Cartn_y"][index]),
                "z": float(columns["_atom_site.Cartn_z"][index]),
                "occupancy": float(columns["_atom_site.occupancy"][index]),
                "b_factor": float(columns["_atom_site.B_iso_or_equiv"][index]),
            }
        )
    if not rows:
        raise ProteinStructureError(f"No model-1 heavy atoms found in {cif_path}")
    return rows


def _structure_metadata(
    protein_id: str, pdb_id: str, cif_path: Path
) -> Mapping[str, object]:
    cif = MMCIF2Dict(str(cif_path))
    resolution = _float_or_none(
        _first(cif, "_refine.ls_d_res_high", _first(cif, "_reflns.d_resolution_high"))
    )
    return {
        "protein_id": protein_id,
        "pdb_id": pdb_id,
        "experimental_method": _first(cif, "_exptl.method"),
        "resolution_angstrom": resolution,
        "deposition_date": _first(
            cif, "_pdbx_database_status.recvd_initial_deposition_date"
        ),
        "release_date": _first(cif, "_pdbx_audit_revision_history.revision_date"),
        "polymer_entity_count": len(
            set(_as_list(cif.get("_entity_poly.entity_id", [])))
        ),
        "biological_assembly_count": len(
            set(_as_list(cif.get("_pdbx_struct_assembly.id", [])))
        ),
        "coordinate_file": str(cif_path),
    }


def _residue_key(atom: Mapping[str, object]) -> Tuple[str, str, str]:
    return (
        str(atom["chain_id"]),
        str(atom["author_seq_id"]),
        str(atom["insertion_code"]),
    )


def _group_residues(
    atoms: Sequence[Mapping[str, object]],
) -> Mapping[Tuple[str, str, str], List[Mapping[str, object]]]:
    grouped: MutableMapping[Tuple[str, str, str], List[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for atom in atoms:
        if atom["group"] == "ATOM":
            grouped[_residue_key(atom)].append(atom)
    return grouped


def _chain_metadata_rows(
    protein_id: str,
    pdb_id: str,
    atoms: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    protein_atoms = [atom for atom in atoms if atom["group"] == "ATOM"]
    chains = sorted({str(atom["chain_id"]) for atom in protein_atoms})
    rows: List[Mapping[str, object]] = []
    for chain_id in chains:
        chain_atoms = [atom for atom in protein_atoms if atom["chain_id"] == chain_id]
        residues = {_residue_key(atom) for atom in chain_atoms}
        hetero = [
            atom
            for atom in atoms
            if atom["group"] == "HETATM" and atom["chain_id"] == chain_id
        ]
        non_water = {
            (
                atom["residue_name"],
                atom["author_seq_id"],
                atom["insertion_code"],
            )
            for atom in hetero
            if atom["residue_name"] not in _WATER_NAMES and atom["element"] != "ZN"
        }
        zinc_count = sum(1 for atom in hetero if atom["element"] == "ZN")
        b_factors = [float(atom["b_factor"]) for atom in chain_atoms]
        rows.append(
            {
                "protein_id": protein_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "modeled_residue_count": len(residues),
                "coordinate_atom_count": len(chain_atoms),
                "alternate_location_atom_count": sum(
                    atom["altloc"] not in {".", "?", ""} for atom in chain_atoms
                ),
                "partial_occupancy_atom_count": sum(
                    float(atom["occupancy"]) < 0.999 for atom in chain_atoms
                ),
                "mean_b_factor": statistics.fmean(b_factors),
                "median_b_factor": statistics.median(b_factors),
                "zinc_atom_count": zinc_count,
                "non_water_ligand_count": len(non_water),
            }
        )
    return rows


def _structure_coverage(
    observations: pd.DataFrame, coverage: pd.DataFrame
) -> Tuple[List[Mapping[str, object]], Mapping[str, object]]:
    missense = observations[observations["mutation_class"] == "missense"]
    # ``coverage["mutation"]`` preserves the publication mutation string, while
    # ``assay_mutation`` can use a different residue number when the assay construct
    # contains an insertion.  VIM-2 has exactly this situation because its assay
    # construct contains the artificial Gly2 insertion.  Join on the publication
    # mutation plus the explicitly mapped assay position so that numbering systems
    # are never mixed.
    mutation_counts = (
        missense.groupby(["protein_id", "mutation", "assay_position"])
        .size()
        .rename("observation_count")
        .reset_index()
    )
    merged = coverage.merge(
        mutation_counts,
        on=["protein_id", "mutation", "assay_position"],
        how="inner",
    )
    rows: List[Mapping[str, object]] = []
    for (protein, pdb_id, chain), group in merged.groupby(
        ["protein_id", "pdb_id", "pdb_chain"]
    ):
        modeled = group[group["mapping_status"] == "mapped_modeled"]
        rows.append(
            {
                "protein_id": protein,
                "pdb_id": pdb_id,
                "pdb_chain": chain,
                "unique_missense_mutations": group["mutation"].nunique(),
                "modeled_unique_mutations": modeled["mutation"].nunique(),
                "modeled_unique_positions": modeled["assay_position"].nunique(),
                "observation_count": int(group["observation_count"].sum()),
                "modeled_observation_count": int(modeled["observation_count"].sum()),
                "modeled_observation_fraction": float(
                    modeled["observation_count"].sum()
                    / group["observation_count"].sum()
                ),
                "mapped_unmodeled_mutations": group.loc[
                    group["mapping_status"] == "mapped_unmodeled", "mutation"
                ].nunique(),
                "not_in_entity_mutations": group.loc[
                    group["mapping_status"] == "not_in_pdb_entity", "mutation"
                ].nunique(),
                "assay_insertion_mutations": group.loc[
                    group["mapping_status"] == "assay_insertion", "mutation"
                ].nunique(),
            }
        )
    decision = {
        "NDM-1": {
            "pdb_id": "3SPU",
            "primary_chain": "B",
            "decision_status": "selected_pending_coordinate_checks",
            "basis": (
                "highest modeled DMS-position coverage among chains A-E; "
                "coordinate-level checks are evaluated in this step"
            ),
        },
        "VIM-2": {
            "pdb_id": "5YD7",
            "primary_chain": "B",
            "secondary_chain": "A",
            "decision_status": "selected_with_tied_coverage_pending_coordinate_checks",
            "basis": (
                "chains A and B have equal DMS coverage; chain B is preferred "
                "because the validation report records no Ramachandran outliers, "
                "whereas chain A contains one at Ala178"
            ),
        },
    }
    return rows, decision


def _residue_coordinate_rows(
    mapping: pd.DataFrame,
    atom_lookup: Mapping[
        str, Mapping[Tuple[str, str, str], List[Mapping[str, object]]]
    ],
) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    for record in mapping.to_dict("records"):
        pdb_id = str(record["pdb_id"])
        chain_id = str(record["pdb_chain"])
        author_position = record["pdb_author_position"]
        insertion = record.get("pdb_insertion_code", "?")
        insertion_text = "?" if pd.isna(insertion) else str(insertion)
        key = (chain_id, _normalize_position(author_position), insertion_text)
        residue_atoms = atom_lookup.get(pdb_id, {}).get(key, [])
        atom_names = {str(atom["atom_name"]) for atom in residue_atoms}
        residue_name = str(residue_atoms[0]["residue_name"]) if residue_atoms else ""
        expected_side_chain = _EXPECTED_SIDE_CHAIN_ATOMS.get(residue_name, set())
        occupancies = [float(atom["occupancy"]) for atom in residue_atoms]
        b_factors = [float(atom["b_factor"]) for atom in residue_atoms]
        rows.append(
            {
                "protein_id": record["protein_id"],
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "assay_position": int(record["assay_position"]),
                "assay_residue": record["assay_residue"],
                "pdb_author_position": author_position,
                "pdb_insertion_code": insertion_text,
                "pdb_residue_name": residue_name,
                "mapping_status": record["mapping_status"],
                "coordinate_present": bool(residue_atoms),
                "backbone_complete": _BACKBONE_ATOMS.issubset(atom_names),
                "side_chain_complete": expected_side_chain.issubset(atom_names),
                "alternate_location_present": any(
                    atom["altloc"] not in {".", "?", ""} for atom in residue_atoms
                ),
                "minimum_occupancy": min(occupancies) if occupancies else None,
                "mean_b_factor": statistics.fmean(b_factors) if b_factors else None,
            }
        )
    return rows


def _metal_rows(
    protein_id: str,
    pdb_id: str,
    atoms: Sequence[Mapping[str, object]],
    cutoff: float = 3.0,
) -> Tuple[List[Mapping[str, object]], List[Mapping[str, object]]]:
    zinc_atoms = [atom for atom in atoms if atom["element"] == "ZN"]
    contacts: List[Mapping[str, object]] = []
    sites: List[Mapping[str, object]] = []
    for index, metal in enumerate(zinc_atoms, start=1):
        site_contacts: List[Mapping[str, object]] = []
        for atom in atoms:
            if atom is metal or atom["element"] in {"H", "D"}:
                continue
            distance = _distance(metal, atom)
            if distance > cutoff:
                continue
            if atom["group"] == "ATOM":
                contact_type = "protein"
            elif atom["residue_name"] in _WATER_NAMES:
                contact_type = "water"
            else:
                contact_type = "ligand_or_additive"
            contact = {
                "protein_id": protein_id,
                "pdb_id": pdb_id,
                "metal_site_id": f"{pdb_id}_{metal['chain_id']}_ZN_{index}",
                "metal_chain_id": metal["chain_id"],
                "metal_author_position": metal["author_seq_id"],
                "metal_atom_id": metal["atom_id"],
                "contact_chain_id": atom["chain_id"],
                "contact_residue_name": atom["residue_name"],
                "contact_author_position": atom["author_seq_id"],
                "contact_atom_name": atom["atom_name"],
                "distance_angstrom": distance,
                "contact_type": contact_type,
            }
            contacts.append(contact)
            site_contacts.append(contact)
        protein_residues = {
            (
                row["contact_chain_id"],
                row["contact_author_position"],
                row["contact_residue_name"],
            )
            for row in site_contacts
            if row["contact_type"] == "protein"
        }
        protein_atom_count = sum(
            row["contact_type"] == "protein" for row in site_contacts
        )
        if len(protein_residues) >= 2 and protein_atom_count >= 2:
            classification = "catalytic_site_candidate"
        else:
            classification = "non_catalytic_crystallographic_site"
        sites.append(
            {
                "protein_id": protein_id,
                "pdb_id": pdb_id,
                "metal_site_id": f"{pdb_id}_{metal['chain_id']}_ZN_{index}",
                "chain_id": metal["chain_id"],
                "author_position": metal["author_seq_id"],
                "occupancy": metal["occupancy"],
                "b_factor": metal["b_factor"],
                "protein_contact_residue_count": len(protein_residues),
                "protein_contact_atom_count": protein_atom_count,
                "water_contact_count": sum(
                    row["contact_type"] == "water" for row in site_contacts
                ),
                "ligand_or_additive_contact_count": sum(
                    row["contact_type"] == "ligand_or_additive" for row in site_contacts
                ),
                "site_classification": classification,
                "distance_cutoff_angstrom": cutoff,
            }
        )
    return sites, contacts


def _ligand_rows(
    protein_id: str,
    pdb_id: str,
    atoms: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    grouped: MutableMapping[Tuple[str, str, str, str], List[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for atom in atoms:
        if atom["group"] != "HETATM" or atom["residue_name"] in _WATER_NAMES:
            continue
        key = (
            str(atom["chain_id"]),
            str(atom["author_seq_id"]),
            str(atom["insertion_code"]),
            str(atom["residue_name"]),
        )
        grouped[key].append(atom)
    rows: List[Mapping[str, object]] = []
    for key, ligand_atoms in sorted(grouped.items()):
        chain_id, author_position, insertion, residue_name = key
        rows.append(
            {
                "protein_id": protein_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "author_position": author_position,
                "insertion_code": insertion,
                "ligand_id": residue_name,
                "atom_count": len(ligand_atoms),
                "element_set": ";".join(
                    sorted({str(atom["element"]) for atom in ligand_atoms})
                ),
                "minimum_occupancy": min(
                    float(atom["occupancy"]) for atom in ligand_atoms
                ),
                "mean_b_factor": statistics.fmean(
                    float(atom["b_factor"]) for atom in ligand_atoms
                ),
                "ligand_class": (
                    "metal" if residue_name == "ZN" else "ligand_or_additive"
                ),
            }
        )
    return rows


def _local_environment_rows(
    residue_rows: Sequence[Mapping[str, object]],
    atom_lookup: Mapping[
        str, Mapping[Tuple[str, str, str], List[Mapping[str, object]]]
    ],
    selected: Mapping[str, Mapping[str, object]],
    radii: Sequence[float] = (6.0, 8.0, 10.0, 12.0),
) -> List[Mapping[str, object]]:
    selected_pairs = {
        (protein, details["pdb_id"], details["primary_chain"])
        for protein, details in selected.items()
    }
    rows: List[Mapping[str, object]] = []
    for residue in residue_rows:
        pair = (residue["protein_id"], residue["pdb_id"], residue["chain_id"])
        if pair not in selected_pairs or not residue["coordinate_present"]:
            continue
        key = (
            str(residue["chain_id"]),
            _normalize_position(residue["pdb_author_position"]),
            str(residue["pdb_insertion_code"]),
        )
        center_atoms = atom_lookup[str(residue["pdb_id"])].get(key, [])
        center_ca = next(
            (atom for atom in center_atoms if atom["atom_name"] == "CA"),
            None,
        )
        if center_ca is None:
            continue
        all_residues = atom_lookup[str(residue["pdb_id"])]
        for radius in radii:
            neighbours: List[List[Mapping[str, object]]] = []
            for neighbour_atoms in all_residues.values():
                if (
                    not neighbour_atoms
                    or neighbour_atoms[0]["chain_id"] != residue["chain_id"]
                ):
                    continue
                neighbour_ca = next(
                    (atom for atom in neighbour_atoms if atom["atom_name"] == "CA"),
                    None,
                )
                if neighbour_ca is None or _distance(center_ca, neighbour_ca) > radius:
                    continue
                neighbours.append(neighbour_atoms)
            backbone_complete = 0
            side_chain_complete = 0
            altloc_count = 0
            partial_count = 0
            neighbour_b: List[float] = []
            for neighbour_atoms in neighbours:
                names = {str(atom["atom_name"]) for atom in neighbour_atoms}
                residue_name = str(neighbour_atoms[0]["residue_name"])
                expected = _EXPECTED_SIDE_CHAIN_ATOMS.get(residue_name, set())
                backbone_complete += _BACKBONE_ATOMS.issubset(names)
                side_chain_complete += expected.issubset(names)
                altloc_count += any(
                    atom["altloc"] not in {".", "?", ""} for atom in neighbour_atoms
                )
                partial_count += any(
                    float(atom["occupancy"]) < 0.999 for atom in neighbour_atoms
                )
                neighbour_b.extend(float(atom["b_factor"]) for atom in neighbour_atoms)
            total = len(neighbours)
            rows.append(
                {
                    "protein_id": residue["protein_id"],
                    "pdb_id": residue["pdb_id"],
                    "chain_id": residue["chain_id"],
                    "assay_position": residue["assay_position"],
                    "assay_residue": residue["assay_residue"],
                    "radius_angstrom": radius,
                    "neighbour_residue_count": total,
                    "complete_backbone_fraction": (
                        backbone_complete / total if total else None
                    ),
                    "complete_side_chain_fraction": (
                        side_chain_complete / total if total else None
                    ),
                    "alternate_location_neighbour_count": altloc_count,
                    "partial_occupancy_neighbour_count": partial_count,
                    "mean_neighbour_b_factor": (
                        statistics.fmean(neighbour_b) if neighbour_b else None
                    ),
                }
            )
    return rows


def _update_structure_decisions(
    decision: MutableMapping[str, MutableMapping[str, object]],
    chain_rows: Sequence[Mapping[str, object]],
    metal_rows: Sequence[Mapping[str, object]],
) -> None:
    for protein_id, details in decision.items():
        pdb_id = details["pdb_id"]
        chain_id = details["primary_chain"]
        chain = next(
            (
                row
                for row in chain_rows
                if row["protein_id"] == protein_id
                and row["pdb_id"] == pdb_id
                and row["chain_id"] == chain_id
            ),
            None,
        )
        if chain is None:
            raise ProteinStructureError(
                f"Selected chain not present in coordinate file: {pdb_id} {chain_id}"
            )
        catalytic_candidates = [
            row
            for row in metal_rows
            if row["protein_id"] == protein_id
            and row["pdb_id"] == pdb_id
            and row["chain_id"] == chain_id
            and row["site_classification"] == "catalytic_site_candidate"
        ]
        details["coordinate_checks"] = {
            "modeled_residue_count": chain["modeled_residue_count"],
            "alternate_location_atom_count": chain["alternate_location_atom_count"],
            "partial_occupancy_atom_count": chain["partial_occupancy_atom_count"],
            "zinc_atom_count": chain["zinc_atom_count"],
            "catalytic_zinc_candidate_count": len(catalytic_candidates),
        }
        if len(catalytic_candidates) >= 2:
            details["decision_status"] = "selected_after_coordinate_checks"
        else:
            details["decision_status"] = "manual_review_required"


def prepare_protein_structures(repo_root: Path) -> Mapping[str, object]:
    root = repo_root.resolve()
    input_dir = root / "data/interim/mutation_measurements"
    mapping_dir = root / "data/interim/sequence_mapping"
    output_dir = root / "data/interim/protein_structures"
    observations = _read_csv(input_dir / "mutation_measurements.csv")
    coverage = _read_csv(input_dir / "mutation_structure_mapping.csv")
    residue_mapping = _read_csv(mapping_dir / "residue_number_mapping.csv")

    structures = {
        "3SPU": {
            "protein_id": "NDM-1",
            "path": root / "data/raw/pdb/3SPU/3SPU.cif",
        },
        "5YD7": {
            "protein_id": "VIM-2",
            "path": root / "data/raw/pdb/5YD7/5YD7.cif",
        },
    }
    for pdb_id, record in structures.items():
        if not record["path"].is_file():
            raise ProteinStructureError(
                f"Missing coordinate file for {pdb_id}: {record['path']}"
            )

    chain_coverage, structure_decision = _structure_coverage(observations, coverage)
    structure_rows: List[Mapping[str, object]] = []
    chain_rows: List[Mapping[str, object]] = []
    metal_sites: List[Mapping[str, object]] = []
    metal_contacts: List[Mapping[str, object]] = []
    ligand_rows: List[Mapping[str, object]] = []
    atom_lookup: Dict[
        str, Mapping[Tuple[str, str, str], List[Mapping[str, object]]]
    ] = {}

    for pdb_id, record in structures.items():
        protein_id = str(record["protein_id"])
        cif_path = Path(record["path"])
        atoms = _parse_atom_rows(cif_path)
        atom_lookup[pdb_id] = _group_residues(atoms)
        metadata = dict(_structure_metadata(protein_id, pdb_id, cif_path))
        metadata["coordinate_file"] = cif_path.relative_to(root).as_posix()
        structure_rows.append(metadata)
        chain_rows.extend(_chain_metadata_rows(protein_id, pdb_id, atoms))
        sites, contacts = _metal_rows(protein_id, pdb_id, atoms)
        metal_sites.extend(sites)
        metal_contacts.extend(contacts)
        ligand_rows.extend(_ligand_rows(protein_id, pdb_id, atoms))

    coordinate_rows = _residue_coordinate_rows(residue_mapping, atom_lookup)
    local_rows = _local_environment_rows(
        coordinate_rows,
        atom_lookup,
        structure_decision,
    )
    _update_structure_decisions(structure_decision, chain_rows, metal_sites)

    _write_csv(output_dir / "chain_coverage.csv", chain_coverage)
    _write_csv(output_dir / "measured_position_coverage.csv", chain_coverage)
    _write_csv(output_dir / "structure_metadata.csv", structure_rows)
    _write_csv(output_dir / "chain_metadata.csv", chain_rows)
    _write_csv(output_dir / "residue_coordinate_status.csv", coordinate_rows)
    _write_csv(output_dir / "metal_sites.csv", metal_sites)
    _write_csv(output_dir / "metal_contacts.csv", metal_contacts)
    _write_csv(output_dir / "ligand_sites.csv", ligand_rows)
    _write_csv(output_dir / "local_environment_completeness.csv", local_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_structures.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(structure_decision, handle, indent=2, sort_keys=True)
        handle.write("\n")

    selected_chain_checks = {
        protein_id: details["coordinate_checks"]
        for protein_id, details in structure_decision.items()
    }
    summary = {
        "status": "PASS",
        "selected_structures": structure_decision,
        "selected_chain_checks": selected_chain_checks,
        "structure_count": len(structure_rows),
        "chain_count": len(chain_rows),
        "metal_site_count": len(metal_sites),
        "metal_contact_count": len(metal_contacts),
        "residue_coordinate_row_count": len(coordinate_rows),
        "local_environment_row_count": len(local_rows),
        "outputs": [
            "chain_coverage.csv",
            "measured_position_coverage.csv",
            "structure_metadata.csv",
            "chain_metadata.csv",
            "residue_coordinate_status.csv",
            "metal_sites.csv",
            "metal_contacts.csv",
            "ligand_sites.csv",
            "local_environment_completeness.csv",
            "selected_structures.json",
        ],
    }
    with (output_dir / "structure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
