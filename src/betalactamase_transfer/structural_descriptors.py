# Generate selected-structure residue descriptors and pairwise geometry.

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.DSSP import residue_max_acc
from Bio.PDB.SASA import ShrakeRupley


class StructuralDescriptorError(RuntimeError):
    """Raised when selected-structure descriptors cannot be generated safely."""


SELECTED_STRUCTURES: Mapping[str, Tuple[str, str]] = {
    "NDM-1": ("3SPU", "B"),
    "VIM-2": ("5YD7", "B"),
}

_STANDARD_AA3 = set(residue_max_acc["Wilke"])


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise StructuralDescriptorError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise StructuralDescriptorError(f"Refusing to write an empty table: {path}")
    fields: List[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _optional_int(value: object) -> int | None:
    if pd.isna(value) or str(value).strip() in {"", "?"}:
        return None
    number = float(value)
    if not number.is_integer():
        raise StructuralDescriptorError(f"Expected integer position, found {value}")
    return int(number)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "nan", "none"}:
        return False
    raise StructuralDescriptorError(f"Cannot interpret boolean value: {value}")


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _minimum_atom_distance(
    left_atoms: Sequence[np.ndarray],
    right_atoms: Sequence[np.ndarray],
) -> float:
    left = np.asarray(left_atoms, dtype=float)
    right = np.asarray(right_atoms, dtype=float)
    differences = left[:, None, :] - right[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return float(distances.min())


def _protein_chain_copy(chain: object) -> object:
    copied = deepcopy(chain)
    for residue in list(copied):
        if residue.id[0] != " " or residue.resname not in _STANDARD_AA3:
            copied.detach_child(residue.id)
    return copied


def _selected_residue_records(
    repo_root: Path,
    protein_id: str,
    pdb_id: str,
    chain_id: str,
    evidence: pd.DataFrame,
    catalytic_zinc_positions: Sequence[int],
) -> Tuple[Dict[int, Mapping[str, object]], List[np.ndarray]]:
    cif_path = repo_root / f"data/raw/pdb/{pdb_id}/{pdb_id}.cif"
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(cif_path))
    model = structure[0]
    if chain_id not in model:
        raise StructuralDescriptorError(f"Missing selected chain {pdb_id}-{chain_id}")
    chain = model[chain_id]

    protein_chain = _protein_chain_copy(chain)
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(protein_chain, level="R")
    sasa_by_author_position = {
        int(residue.id[1]): float(residue.sasa) for residue in protein_chain
    }

    zinc_coordinates: List[np.ndarray] = []
    for author_position in catalytic_zinc_positions:
        matches = [
            residue
            for residue in chain
            if residue.resname == "ZN" and int(residue.id[1]) == author_position
        ]
        if len(matches) != 1:
            raise StructuralDescriptorError(
                f"Expected one selected catalytic zinc at {pdb_id}-{chain_id} "
                f"position {author_position}, found {len(matches)}"
            )
        atom = next(matches[0].get_atoms())
        zinc_coordinates.append(np.asarray(atom.coord, dtype=float))

    records: Dict[int, Mapping[str, object]] = {}
    protein_evidence = evidence[evidence["protein_id"] == protein_id]
    for row in protein_evidence.itertuples(index=False):
        assay_position = int(row.assay_position)
        if not _as_bool(row.pdb_modeled):
            continue
        author_position = _optional_int(row.pdb_author_position)
        if author_position is None:
            raise StructuralDescriptorError(
                f"Modeled residue lacks author position: {protein_id} {assay_position}"
            )
        residue_id = (" ", author_position, " ")
        if residue_id not in chain:
            raise StructuralDescriptorError(
                f"Selected residue not found in coordinates: {pdb_id}-{chain_id} "
                f"author position {author_position}"
            )
        residue = chain[residue_id]
        if residue.resname not in _STANDARD_AA3:
            raise StructuralDescriptorError(
                f"Unexpected residue {residue.resname} at {pdb_id}-{chain_id} "
                f"author position {author_position}"
            )
        if "CA" not in residue:
            raise StructuralDescriptorError(
                f"Missing C-alpha atom at {pdb_id}-{chain_id} {author_position}"
            )

        heavy_atoms = [
            atom
            for atom in residue.get_atoms()
            if str(atom.element).upper() not in {"H", "D"}
        ]
        atom_coordinates = [np.asarray(atom.coord, dtype=float) for atom in heavy_atoms]
        ca_coordinate = np.asarray(residue["CA"].coord, dtype=float)
        side_chain_atoms = [
            atom
            for atom in heavy_atoms
            if atom.get_name().strip().upper() not in {"N", "CA", "C", "O", "OXT"}
        ]
        if side_chain_atoms:
            side_chain_centroid = np.mean(
                [np.asarray(atom.coord, dtype=float) for atom in side_chain_atoms],
                axis=0,
            )
        else:
            side_chain_centroid = None

        sasa = sasa_by_author_position.get(author_position)
        if sasa is None:
            raise StructuralDescriptorError(
                "Missing SASA for "
                f"{pdb_id}-{chain_id} author position {author_position}"
            )
        max_sasa = float(residue_max_acc["Wilke"][residue.resname])
        relative_sasa = sasa / max_sasa

        records[assay_position] = {
            "assay_position": assay_position,
            "pdb_author_position": author_position,
            "pdb_residue_name": residue.resname,
            "ca_coordinate": ca_coordinate,
            "atom_coordinates": atom_coordinates,
            "heavy_atom_count": len(atom_coordinates),
            "side_chain_centroid": side_chain_centroid,
            "sasa_angstrom2": sasa,
            "relative_sasa_wilke": relative_sasa,
        }

    return records, zinc_coordinates


def _reference_position_sets(
    evidence: pd.DataFrame, protein_id: str
) -> Tuple[set, set]:
    protein = evidence[evidence["protein_id"] == protein_id]
    all_reference = set(
        protein.loc[
            protein["mcsa_reference_correspondence_count"] > 0, "assay_position"
        ].astype(int)
    )
    metal_ligands = set(
        protein.loc[
            protein["mcsa_any_metal_ligand_role_correspondence"].map(_as_bool),
            "assay_position",
        ].astype(int)
    )
    if len(all_reference) != 8:
        raise StructuralDescriptorError(
            f"Expected 8 M-CSA-reference-corresponding residues for {protein_id}, "
            f"found {len(all_reference)}"
        )
    if len(metal_ligands) != 6:
        raise StructuralDescriptorError(
            f"Expected 6 metal-ligand-corresponding residues for {protein_id}, "
            f"found {len(metal_ligands)}"
        )
    return all_reference, metal_ligands


def _nearest_reference_distances(
    assay_position: int,
    records: Mapping[int, Mapping[str, object]],
    reference_positions: set,
) -> Tuple[float, float]:
    current = records[assay_position]
    modeled_reference_positions = [
        position for position in sorted(reference_positions) if position in records
    ]
    if not modeled_reference_positions:
        raise StructuralDescriptorError("No modeled reference positions available")
    ca_distances = [
        _distance(
            np.asarray(current["ca_coordinate"]),
            np.asarray(records[position]["ca_coordinate"]),
        )
        for position in modeled_reference_positions
    ]
    atom_distances = [
        _minimum_atom_distance(
            current["atom_coordinates"], records[position]["atom_coordinates"]
        )
        for position in modeled_reference_positions
    ]
    return min(ca_distances), min(atom_distances)


def prepare_structural_descriptors(repo_root: Path) -> Mapping[str, object]:
    """Generate selected-PDB residue descriptors and all modeled residue pairs."""
    repo_root = repo_root.resolve()
    output_dir = repo_root / "data/interim/structural_descriptors"
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = _read_csv(
        repo_root / "data/interim/residue_evidence/residue_evidence.csv"
    )
    metal_sites = _read_csv(
        repo_root / "data/interim/protein_structures/metal_sites.csv"
    )
    if len(evidence) != 537:
        raise StructuralDescriptorError(
            f"Expected 537 residue-evidence rows, found {len(evidence)}"
        )
    if evidence.duplicated(["protein_id", "assay_position"]).any():
        raise StructuralDescriptorError("Residue evidence contains duplicate positions")

    descriptor_rows: List[Mapping[str, object]] = []
    pair_rows: List[Mapping[str, object]] = []
    modeled_counts: Dict[str, int] = {}
    pair_counts: Dict[str, int] = {}

    for protein_id, (pdb_id, chain_id) in SELECTED_STRUCTURES.items():
        selected_zinc = metal_sites[
            (metal_sites["protein_id"] == protein_id)
            & (metal_sites["pdb_id"] == pdb_id)
            & (metal_sites["chain_id"] == chain_id)
            & (metal_sites["site_classification"] == "catalytic_site_candidate")
        ].sort_values("author_position")
        if len(selected_zinc) != 2:
            raise StructuralDescriptorError(
                f"Expected two selected catalytic zinc sites for {protein_id}, "
                f"found {len(selected_zinc)}"
            )
        catalytic_zinc_positions = [
            int(value) for value in selected_zinc["author_position"]
        ]
        records, zinc_coordinates = _selected_residue_records(
            repo_root,
            protein_id,
            pdb_id,
            chain_id,
            evidence,
            catalytic_zinc_positions,
        )
        modeled_counts[protein_id] = len(records)
        all_reference, metal_ligands = _reference_position_sets(evidence, protein_id)

        positions = sorted(records)
        pair_count = 0
        neighbour_counts = {
            position: {
                "ca_within_8": 0,
                "ca_within_10": 0,
                "ca_within_12": 0,
                "heavy_atom_contact_4_5": 0,
            }
            for position in positions
        }

        for left_index, left_position in enumerate(positions):
            left = records[left_position]
            for right_position in positions[left_index + 1 :]:
                right = records[right_position]
                ca_distance = _distance(
                    np.asarray(left["ca_coordinate"]),
                    np.asarray(right["ca_coordinate"]),
                )
                heavy_distance = _minimum_atom_distance(
                    left["atom_coordinates"], right["atom_coordinates"]
                )
                ca_8 = ca_distance <= 8.0
                ca_10 = ca_distance <= 10.0
                ca_12 = ca_distance <= 12.0
                heavy_contact = heavy_distance <= 4.5
                for position in (left_position, right_position):
                    neighbour_counts[position]["ca_within_8"] += int(ca_8)
                    neighbour_counts[position]["ca_within_10"] += int(ca_10)
                    neighbour_counts[position]["ca_within_12"] += int(ca_12)
                    neighbour_counts[position]["heavy_atom_contact_4_5"] += int(
                        heavy_contact
                    )
                pair_rows.append(
                    {
                        "protein_id": protein_id,
                        "pdb_id": pdb_id,
                        "pdb_chain": chain_id,
                        "left_assay_position": left_position,
                        "right_assay_position": right_position,
                        "sequence_separation": abs(right_position - left_position),
                        "peptide_adjacent": abs(right_position - left_position) == 1,
                        "ca_distance_angstrom": ca_distance,
                        "minimum_heavy_atom_distance_angstrom": heavy_distance,
                        "ca_within_8_angstrom": ca_8,
                        "ca_within_10_angstrom": ca_10,
                        "ca_within_12_angstrom": ca_12,
                        "heavy_atom_contact_4_5_angstrom": heavy_contact,
                    }
                )
                pair_count += 1
        pair_counts[protein_id] = pair_count

        for row in (
            evidence[evidence["protein_id"] == protein_id]
            .sort_values("assay_position")
            .itertuples(index=False)
        ):
            assay_position = int(row.assay_position)
            base = {
                "protein_id": protein_id,
                "assay_position": assay_position,
                "assay_residue": str(row.assay_residue),
                "pdb_id": pdb_id,
                "pdb_chain": chain_id,
                "descriptor_status": (
                    "experimental_coordinate_available"
                    if assay_position in records
                    else "no_experimental_coordinate"
                ),
            }
            if assay_position not in records:
                descriptor_rows.append(base)
                continue

            record = records[assay_position]
            ca = np.asarray(record["ca_coordinate"])
            side_chain = record["side_chain_centroid"]
            zinc_ca_distances = [_distance(ca, zinc) for zinc in zinc_coordinates]
            zinc_atom_distances = [
                min(
                    _distance(np.asarray(atom), zinc)
                    for atom in record["atom_coordinates"]
                )
                for zinc in zinc_coordinates
            ]
            zinc_centroid = np.mean(zinc_coordinates, axis=0)
            reference_ca, reference_atom = _nearest_reference_distances(
                assay_position, records, all_reference
            )
            metal_ca, metal_atom = _nearest_reference_distances(
                assay_position, records, metal_ligands
            )
            counts = neighbour_counts[assay_position]

            descriptor_rows.append(
                {
                    **base,
                    "pdb_author_position": record["pdb_author_position"],
                    "pdb_residue_name": record["pdb_residue_name"],
                    "ca_x": float(ca[0]),
                    "ca_y": float(ca[1]),
                    "ca_z": float(ca[2]),
                    "side_chain_centroid_x": (
                        None if side_chain is None else float(side_chain[0])
                    ),
                    "side_chain_centroid_y": (
                        None if side_chain is None else float(side_chain[1])
                    ),
                    "side_chain_centroid_z": (
                        None if side_chain is None else float(side_chain[2])
                    ),
                    "heavy_atom_count": record["heavy_atom_count"],
                    "sasa_angstrom2": record["sasa_angstrom2"],
                    "relative_sasa_wilke": record["relative_sasa_wilke"],
                    "ca_neighbour_count_8_angstrom": counts["ca_within_8"],
                    "ca_neighbour_count_10_angstrom": counts["ca_within_10"],
                    "ca_neighbour_count_12_angstrom": counts["ca_within_12"],
                    "heavy_atom_contact_count_4_5_angstrom": counts[
                        "heavy_atom_contact_4_5"
                    ],
                    "ca_distance_to_catalytic_zinc_1_angstrom": zinc_ca_distances[0],
                    "ca_distance_to_catalytic_zinc_2_angstrom": zinc_ca_distances[1],
                    "minimum_atom_distance_to_catalytic_zinc_1_angstrom": (
                        zinc_atom_distances[0]
                    ),
                    "minimum_atom_distance_to_catalytic_zinc_2_angstrom": (
                        zinc_atom_distances[1]
                    ),
                    "ca_distance_to_catalytic_zinc_centroid_angstrom": _distance(
                        ca, zinc_centroid
                    ),
                    "ca_distance_to_nearest_mcsa_reference_correspondence_angstrom": (
                        reference_ca
                    ),
                    (
                        "minimum_atom_distance_to_nearest_mcsa_"
                        "reference_correspondence_angstrom"
                    ): reference_atom,
                    (
                        "ca_distance_to_nearest_mcsa_metal_ligand_"
                        "correspondence_angstrom"
                    ): metal_ca,
                    (
                        "minimum_atom_distance_to_nearest_mcsa_metal_"
                        "ligand_correspondence_angstrom"
                    ): metal_atom,
                }
            )

    expected_modeled = {"NDM-1": 238, "VIM-2": 231}
    if modeled_counts != expected_modeled:
        raise StructuralDescriptorError(
            f"Unexpected modeled-residue counts: {modeled_counts}"
        )
    expected_pairs = {
        protein_id: count * (count - 1) // 2
        for protein_id, count in expected_modeled.items()
    }
    if pair_counts != expected_pairs:
        raise StructuralDescriptorError(f"Unexpected pair counts: {pair_counts}")
    if len(descriptor_rows) != 537:
        raise StructuralDescriptorError(
            f"Expected 537 descriptor rows, found {len(descriptor_rows)}"
        )

    _write_csv(output_dir / "residue_structural_descriptors.csv", descriptor_rows)
    _write_csv(output_dir / "residue_pair_geometry.csv", pair_rows)

    descriptor_frame = pd.DataFrame(descriptor_rows)
    pair_frame = pd.DataFrame(pair_rows)
    summary: Mapping[str, object] = {
        "status": "PASS",
        "protein_count": 2,
        "residue_descriptor_row_count": int(len(descriptor_frame)),
        "modeled_residue_counts": modeled_counts,
        "residue_pair_counts": pair_counts,
        "residue_pair_row_count": int(len(pair_frame)),
        "sasa_method": "Shrake-Rupley protein-chain-only",
        "sasa_probe_radius_angstrom": 1.4,
        "sasa_sphere_points": 100,
        "relative_sasa_reference": "Wilke/Tien maximum residue accessibility",
        "pair_distance_scope": "all modeled residue pairs within each selected chain",
        "alternate_location_policy": (
            "Biopython selected atom coordinates; alternate-location status remains "
            "available in residue_evidence.csv for later sensitivity analysis"
        ),
        "secondary_structure_status": "not_computed_in_this_stage",
        "residue_depth_status": "not_computed_in_this_stage",
        "nonmetal_ligand_feature_status": (
            "not_used_from_selected references; 5YD7 nonmetal hetero residues are "
            "formate additives rather than an antibiotic/substrate feature"
        ),
        "outputs": [
            "residue_structural_descriptors.csv",
            "residue_pair_geometry.csv",
            "structural_descriptor_summary.json",
        ],
    }
    (output_dir / "structural_descriptor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
