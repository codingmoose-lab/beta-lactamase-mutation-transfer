# Prepare residue-level evidence for the selected NDM-1 and VIM-2 references.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


class ResidueEvidenceError(RuntimeError):
    """Raised when residue-level evidence cannot be joined without ambiguity."""


SELECTED_STRUCTURES: Mapping[str, Tuple[str, str]] = {
    "NDM-1": ("3SPU", "B"),
    "VIM-2": ("5YD7", "B"),
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ResidueEvidenceError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise ResidueEvidenceError(f"Refusing to write an empty table: {path}")
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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "nan", "none"}:
        return False
    raise ResidueEvidenceError(f"Cannot interpret boolean value: {value}")


def _optional_int(value: object) -> int | None:
    if pd.isna(value) or str(value).strip() in {"", "?"}:
        return None
    number = float(value)
    if not number.is_integer():
        raise ResidueEvidenceError(f"Expected an integer position, found: {value}")
    return int(number)


def _optional_float(value: object) -> float | None:
    if pd.isna(value) or str(value).strip() in {"", "?"}:
        return None
    return float(value)


def _optional_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"nan", "None"} else text


def _require_unique_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    duplicated = frame.duplicated(list(columns), keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, list(columns)].head(10).to_dict("records")
        raise ResidueEvidenceError(
            f"{label} contains duplicate key rows for {list(columns)}: {sample}"
        )


def _selected_pdb_mapping(repo_root: Path) -> pd.DataFrame:
    mapping = _read_csv(
        repo_root / "data/interim/sequence_mapping/residue_number_mapping.csv"
    )
    selected_parts: List[pd.DataFrame] = []
    for protein_id, (pdb_id, chain_id) in SELECTED_STRUCTURES.items():
        part = mapping[
            (mapping["protein_id"] == protein_id)
            & (mapping["pdb_id"] == pdb_id)
            & (mapping["pdb_chain"] == chain_id)
        ].copy()
        if part.empty:
            raise ResidueEvidenceError(
                f"No selected PDB mapping rows for {protein_id} {pdb_id}-{chain_id}"
            )
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)
    _require_unique_keys(selected, ["protein_id", "assay_position"], "PDB mapping")
    return selected


def _selected_coordinate_status(repo_root: Path) -> pd.DataFrame:
    status = _read_csv(
        repo_root / "data/interim/protein_structures/residue_coordinate_status.csv"
    )
    selected_parts: List[pd.DataFrame] = []
    for protein_id, (pdb_id, chain_id) in SELECTED_STRUCTURES.items():
        part = status[
            (status["protein_id"] == protein_id)
            & (status["pdb_id"] == pdb_id)
            & (status["chain_id"] == chain_id)
        ].copy()
        if part.empty:
            raise ResidueEvidenceError(
                "No selected coordinate-status rows for "
                f"{protein_id} {pdb_id}-{chain_id}"
            )
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)
    _require_unique_keys(
        selected, ["protein_id", "assay_position"], "coordinate status"
    )
    return selected


def _mcsa_by_target_position(
    repo_root: Path,
) -> Dict[Tuple[str, int], Mapping[str, object]]:
    path = (
        repo_root
        / "data/interim/mcsa_correspondence/catalytic_residue_correspondence.csv"
    )
    frame = _read_csv(path)
    grouped: Dict[Tuple[str, int], Mapping[str, object]] = {}
    for (protein_id, position), group in frame.groupby(
        ["target_protein", "target_assay_position"], sort=True
    ):
        position_int = int(position)
        mcsa_ids = sorted({int(value) for value in group["mcsa_id"]})
        reference_positions = sorted(
            {
                f"{int(row.mcsa_id)}:{int(row.reference_position)}"
                f"{row.reference_residue}"
                for row in group.itertuples(index=False)
            }
        )
        statuses = sorted(
            {
                _optional_text(value)
                for value in group["mcsa_role_correspondence_status"]
            }
        )
        grouped[(str(protein_id), position_int)] = {
            "mcsa_reference_correspondence_count": int(len(group)),
            "mcsa_reference_ids": ";".join(str(value) for value in mcsa_ids),
            "mcsa_reference_positions": ";".join(reference_positions),
            "mcsa_correspondence_statuses": ";".join(statuses),
            "mcsa_all_correspondences_identity_conserved": bool(
                group["residue_identity_conserved"].map(_as_bool).all()
            ),
            "mcsa_any_metal_ligand_role_correspondence": bool(
                group["has_mcsa_metal_ligand_role"].map(_as_bool).any()
            ),
            "mcsa_any_selected_catalytic_zinc_contact": bool(
                group["target_contacts_selected_catalytic_zinc"].map(_as_bool).any()
            ),
        }
    return grouped


def _reported_homologue_pairs(repo_root: Path) -> List[Mapping[str, object]]:
    path = (
        repo_root / "data/interim/mutation_measurements/cross_protein_comparisons.csv"
    )
    frame = _read_csv(path)
    required = [
        "vim_assay_position",
        "vim_wild_type",
        "ndm_assay_position",
        "ndm_wild_type",
    ]
    unique = frame[required].drop_duplicates().copy()
    if len(unique) != len(frame):
        raise ResidueEvidenceError(
            "Chen 2024 comparison rows do not define one unique position pair per row"
        )
    if unique["vim_assay_position"].duplicated().any():
        raise ResidueEvidenceError(
            "One VIM-2 assay position maps to multiple reported NDM-1 "
            "comparison positions"
        )
    if unique["ndm_assay_position"].duplicated().any():
        raise ResidueEvidenceError(
            "One NDM-1 assay position maps to multiple reported VIM-2 "
            "comparison positions"
        )

    rows: List[Mapping[str, object]] = []
    for row in unique.sort_values(
        ["vim_assay_position", "ndm_assay_position"]
    ).itertuples(index=False):
        rows.append(
            {
                "pair_source": "chen_2024_reported_comparison",
                "pair_scope": "reported_comparison_positions_not_full_alignment",
                "vim_assay_position": int(row.vim_assay_position),
                "vim_residue": str(row.vim_wild_type),
                "ndm_assay_position": int(row.ndm_assay_position),
                "ndm_residue": str(row.ndm_wild_type),
                "same_wild_type_residue": str(row.vim_wild_type)
                == str(row.ndm_wild_type),
            }
        )
    return rows


def _full_homologue_correspondence(repo_root: Path) -> List[Mapping[str, object]]:
    source_path = repo_root / "data/raw/ndm_vim_2024/chen_2024_source_data.xlsx"
    if not source_path.is_file():
        raise ResidueEvidenceError(f"Missing Chen 2024 source data: {source_path}")

    source = pd.read_excel(source_path, sheet_name="fig2ce")
    required = ["pos_ndm", "pos_vim"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ResidueEvidenceError(
            f"Chen 2024 fig2ce is missing aligned-position columns: {missing}"
        )
    positions = source[required].dropna().copy()
    if len(positions) != 217:
        raise ResidueEvidenceError(
            "Expected 217 aligned NDM-1/VIM-2 positions from Chen 2024 fig2ce, "
            f"found {len(positions)}"
        )
    if (
        positions["pos_ndm"].duplicated().any()
        or positions["pos_vim"].duplicated().any()
    ):
        raise ResidueEvidenceError(
            "Chen 2024 aligned-position source is not one-to-one"
        )

    mapping = _read_csv(
        repo_root / "data/interim/sequence_mapping/residue_number_mapping.csv"
    )
    mapping = mapping[
        [
            "protein_id",
            "assay_position",
            "assay_residue",
            "uniprot_position",
            "uniprot_residue",
        ]
    ].drop_duplicates()
    mapping = mapping[mapping["uniprot_position"].notna()].copy()
    _require_unique_keys(
        mapping, ["protein_id", "uniprot_position"], "assay/UniProt mapping"
    )
    lookup = {
        (str(row.protein_id), int(row.uniprot_position)): row
        for row in mapping.itertuples(index=False)
    }

    rows: List[Mapping[str, object]] = []
    for alignment_index, row in enumerate(positions.itertuples(index=False), start=1):
        ndm_publication_position = int(row.pos_ndm)
        vim_publication_position = int(row.pos_vim)
        ndm_key = ("NDM-1", ndm_publication_position)
        vim_key = ("VIM-2", vim_publication_position)
        if ndm_key not in lookup or vim_key not in lookup:
            raise ResidueEvidenceError(
                "Chen 2024 aligned position cannot be mapped to an assay construct: "
                f"NDM={ndm_publication_position}, VIM={vim_publication_position}"
            )
        ndm = lookup[ndm_key]
        vim = lookup[vim_key]
        same_wild_type = str(ndm.assay_residue) == str(vim.assay_residue)
        rows.append(
            {
                "homologue_position_id": f"homologue_position_{alignment_index:03d}",
                "pair_source": "chen_2024_source_data_fig2ce",
                "pair_scope": "complete_publication_aligned_dms_positions",
                "mapping_uses_outcome_values": False,
                "ndm_publication_position": ndm_publication_position,
                "ndm_assay_position": int(ndm.assay_position),
                "ndm_residue": str(ndm.assay_residue),
                "vim_publication_position": vim_publication_position,
                "vim_assay_position": int(vim.assay_position),
                "vim_residue": str(vim.assay_residue),
                "same_wild_type_residue": same_wild_type,
                "position_class": "shared_wild_type"
                if same_wild_type
                else "differing_wild_type",
            }
        )

    frame = pd.DataFrame(rows)
    if int(frame["same_wild_type_residue"].sum()) != 81:
        raise ResidueEvidenceError(
            "Chen 2024 aligned-position mapping does not reproduce "
            "81 shared-WT positions"
        )
    if int((~frame["same_wild_type_residue"]).sum()) != 136:
        raise ResidueEvidenceError(
            "Chen 2024 aligned-position mapping does not reproduce "
            "136 differing-WT positions"
        )
    return rows


def prepare_residue_evidence(repo_root: Path) -> Mapping[str, object]:
    """Prepare selected-reference residue rows and reported position pairs."""
    repo_root = repo_root.resolve()
    output_dir = repo_root / "data/interim/residue_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_mapping = _selected_pdb_mapping(repo_root)
    coordinate_status = _selected_coordinate_status(repo_root)
    alphafold = _read_csv(
        repo_root / "data/interim/alphafold_preparation/assay_alphafold_mapping.csv"
    )
    confidence = _read_csv(
        repo_root / "data/interim/alphafold_preparation/residue_confidence.csv"
    )

    _require_unique_keys(
        alphafold, ["protein_id", "assay_position"], "AlphaFold mapping"
    )
    _require_unique_keys(
        confidence, ["protein_id", "alphafold_position"], "AlphaFold confidence"
    )

    confidence_lookup = {
        (str(row.protein_id), int(row.alphafold_position)): row
        for row in confidence.itertuples(index=False)
    }
    coordinate_lookup = {
        (str(row.protein_id), int(row.assay_position)): row
        for row in coordinate_status.itertuples(index=False)
    }
    pdb_lookup = {
        (str(row.protein_id), int(row.assay_position)): row
        for row in pdb_mapping.itertuples(index=False)
    }
    mcsa_lookup = _mcsa_by_target_position(repo_root)

    residue_rows: List[Mapping[str, object]] = []
    for af_row in alphafold.sort_values(["protein_id", "assay_position"]).itertuples(
        index=False
    ):
        protein_id = str(af_row.protein_id)
        assay_position = int(af_row.assay_position)
        key = (protein_id, assay_position)
        if key not in pdb_lookup or key not in coordinate_lookup:
            raise ResidueEvidenceError(
                "Missing selected experimental-structure row for "
                f"{protein_id} {assay_position}"
            )
        pdb_row = pdb_lookup[key]
        coord_row = coordinate_lookup[key]

        if str(af_row.assay_residue) != str(pdb_row.assay_residue):
            raise ResidueEvidenceError(
                f"Assay residue disagreement for {protein_id} {assay_position}: "
                f"AlphaFold mapping={af_row.assay_residue}, "
                f"PDB mapping={pdb_row.assay_residue}"
            )
        if _as_bool(af_row.experimental_pdb_modeled) != _as_bool(pdb_row.pdb_modeled):
            raise ResidueEvidenceError(
                "Experimental PDB modeled status disagreement for "
                f"{protein_id} {assay_position}"
            )

        alphafold_position = _optional_int(af_row.alphafold_position)
        confidence_row = (
            confidence_lookup.get((protein_id, alphafold_position))
            if alphafold_position is not None
            else None
        )
        if alphafold_position is not None and confidence_row is None:
            raise ResidueEvidenceError(
                "Missing AlphaFold confidence for "
                f"{protein_id} position {alphafold_position}"
            )

        mcsa = mcsa_lookup.get(
            key,
            {
                "mcsa_reference_correspondence_count": 0,
                "mcsa_reference_ids": "",
                "mcsa_reference_positions": "",
                "mcsa_correspondence_statuses": "",
                "mcsa_all_correspondences_identity_conserved": False,
                "mcsa_any_metal_ligand_role_correspondence": False,
                "mcsa_any_selected_catalytic_zinc_contact": False,
            },
        )

        residue_rows.append(
            {
                "protein_id": protein_id,
                "assay_position": assay_position,
                "assay_residue": str(af_row.assay_residue),
                "uniprot_position": _optional_int(af_row.uniprot_position),
                "uniprot_residue": _optional_text(af_row.uniprot_residue),
                "assay_uniprot_residue_identity": _as_bool(
                    af_row.assay_uniprot_residue_identity
                ),
                "assay_specific_insertion": _as_bool(af_row.assay_specific_insertion),
                "known_assay_reference_difference": _as_bool(
                    af_row.known_assay_reference_difference
                ),
                "selected_pdb_id": str(pdb_row.pdb_id),
                "selected_pdb_chain": str(pdb_row.pdb_chain),
                "pdb_mapping_status": str(pdb_row.mapping_status),
                "pdb_author_position": _optional_text(pdb_row.pdb_author_position),
                "pdb_label_position": _optional_int(pdb_row.pdb_label_position),
                "pdb_modeled": _as_bool(pdb_row.pdb_modeled),
                "pdb_residue_identity_ok": _as_bool(pdb_row.residue_identity_ok),
                "pdb_coordinate_present": _as_bool(coord_row.coordinate_present),
                "pdb_backbone_complete": _as_bool(coord_row.backbone_complete),
                "pdb_side_chain_complete": _as_bool(coord_row.side_chain_complete),
                "pdb_alternate_location_present": _as_bool(
                    coord_row.alternate_location_present
                ),
                "pdb_minimum_occupancy": _optional_float(coord_row.minimum_occupancy),
                "pdb_mean_b_factor": _optional_float(coord_row.mean_b_factor),
                "alphafold_position": alphafold_position,
                "alphafold_residue": _optional_text(af_row.alphafold_residue),
                "alphafold_mapping_status": str(af_row.alphafold_mapping_status),
                "structure_coverage_class": str(af_row.coverage_class),
                "alphafold_plddt": (
                    _optional_float(confidence_row.plddt)
                    if confidence_row is not None
                    else None
                ),
                "alphafold_confidence_category": (
                    _optional_text(confidence_row.confidence_category)
                    if confidence_row is not None
                    else ""
                ),
                **mcsa,
            }
        )

    if len(residue_rows) != 537:
        raise ResidueEvidenceError(
            "Expected 537 assay-residue rows (270 NDM-1 + 267 VIM-2), "
            f"found {len(residue_rows)}"
        )

    full_homologue_rows = _full_homologue_correspondence(repo_root)
    full_homologue_frame = pd.DataFrame(full_homologue_rows)

    homologue_rows = _reported_homologue_pairs(repo_root)
    if len(homologue_rows) != 132:
        raise ResidueEvidenceError(
            "Expected 132 reported Chen 2024 position pairs, "
            f"found {len(homologue_rows)}"
        )

    reported_frame = pd.DataFrame(homologue_rows)
    merged_reported = reported_frame.merge(
        full_homologue_frame,
        on=["vim_assay_position", "ndm_assay_position"],
        how="left",
        validate="one_to_one",
        suffixes=("_reported", "_full"),
    )
    if merged_reported["homologue_position_id"].isna().any():
        raise ResidueEvidenceError(
            "A Chen 2024 WT-reversion comparison position is absent from the full "
            "publication-aligned correspondence"
        )
    if merged_reported["same_wild_type_residue_full"].any():
        raise ResidueEvidenceError(
            "WT-reversion comparison positions must belong to the differing-WT class"
        )
    residue_by_key = {
        (str(row["protein_id"]), int(row["assay_position"])): str(row["assay_residue"])
        for row in residue_rows
    }
    for row in full_homologue_rows:
        ndm_key = ("NDM-1", int(row["ndm_assay_position"]))
        vim_key = ("VIM-2", int(row["vim_assay_position"]))
        if residue_by_key.get(ndm_key) != str(row["ndm_residue"]):
            raise ResidueEvidenceError(f"NDM homologue residue mismatch at {ndm_key}")
        if residue_by_key.get(vim_key) != str(row["vim_residue"]):
            raise ResidueEvidenceError(f"VIM homologue residue mismatch at {vim_key}")

    _write_csv(output_dir / "residue_evidence.csv", residue_rows)
    _write_csv(
        output_dir / "homologue_position_correspondence.csv", full_homologue_rows
    )
    _write_csv(output_dir / "reported_homologue_position_pairs.csv", homologue_rows)

    frame = pd.DataFrame(residue_rows)
    summary = {
        "status": "PASS",
        "protein_count": int(frame["protein_id"].nunique()),
        "residue_evidence_row_count": int(len(frame)),
        "residue_rows_by_protein": {
            protein_id: int(count)
            for protein_id, count in frame.groupby("protein_id").size().items()
        },
        "selected_pdb_modeled_rows": int(frame["pdb_modeled"].sum()),
        "alphafold_mapped_rows": int(frame["alphafold_position"].notna().sum()),
        "assay_specific_insertion_rows": int(frame["assay_specific_insertion"].sum()),
        "known_assay_reference_difference_rows": int(
            frame["known_assay_reference_difference"].sum()
        ),
        "mcsa_reference_correspondence_residue_count": int(
            (frame["mcsa_reference_correspondence_count"] > 0).sum()
        ),
        "mcsa_metal_ligand_correspondence_residue_count": int(
            frame["mcsa_any_metal_ligand_role_correspondence"].sum()
        ),
        "full_homologue_position_pair_count": int(len(full_homologue_rows)),
        "full_homologue_shared_wild_type_pair_count": int(
            full_homologue_frame["same_wild_type_residue"].sum()
        ),
        "full_homologue_differing_wild_type_pair_count": int(
            (~full_homologue_frame["same_wild_type_residue"]).sum()
        ),
        "full_homologue_pair_scope": "complete_publication_aligned_dms_positions",
        "reported_homologue_position_pair_count": int(len(homologue_rows)),
        "reported_homologue_pair_scope": (
            "reported_comparison_positions_not_full_alignment"
        ),
        "outputs": [
            "residue_evidence.csv",
            "homologue_position_correspondence.csv",
            "reported_homologue_position_pairs.csv",
            "residue_evidence_summary.json",
        ],
    }
    (output_dir / "residue_evidence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
