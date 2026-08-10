# Map curated M-CSA catalytic residues onto the study proteins by sequence.

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices


class MCSACorrespondenceError(RuntimeError):
    """Raised when catalytic-residue correspondence cannot be established safely."""


_TARGETS = {
    "NDM-1": "C7C422",
    "VIM-2": "Q9K2N0",
}
_MAX_OPTIMAL_ALIGNMENTS = 100
_ALIGNMENT_METHOD = "global_mature_chain_blosum62_affine_gap"


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise MCSACorrespondenceError(f"Missing required input: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MCSACorrespondenceError(f"Invalid JSON input: {path}") from exc


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise MCSACorrespondenceError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise MCSACorrespondenceError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sequence_and_chain(path: Path, accession: str) -> tuple[str, int, int]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise MCSACorrespondenceError(f"UniProt JSON is not an object: {path}")
    observed = str(payload.get("primaryAccession") or "").strip().upper()
    if observed != accession.upper():
        raise MCSACorrespondenceError(
            f"UniProt accession mismatch for {path}: expected {accession}, "
            f"observed {observed}"
        )
    sequence_record = payload.get("sequence")
    if not isinstance(sequence_record, Mapping):
        raise MCSACorrespondenceError(f"UniProt JSON has no sequence object: {path}")
    sequence = "".join(str(sequence_record.get("value") or "").split()).upper()
    if not sequence:
        raise MCSACorrespondenceError(f"UniProt sequence is empty: {path}")

    chains: list[tuple[int, int]] = []
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise MCSACorrespondenceError(f"UniProt features are not a list: {path}")
    for feature in features:
        if not isinstance(feature, Mapping) or feature.get("type") != "Chain":
            continue
        location = feature.get("location")
        if not isinstance(location, Mapping):
            continue
        start_record = location.get("start")
        end_record = location.get("end")
        if not isinstance(start_record, Mapping) or not isinstance(end_record, Mapping):
            continue
        try:
            start = int(start_record.get("value"))
            end = int(end_record.get("value"))
        except (TypeError, ValueError):
            continue
        if 1 <= start <= end <= len(sequence):
            chains.append((start, end))

    if len(chains) != 1:
        raise MCSACorrespondenceError(
            f"Expected exactly one valid UniProt Chain feature for {accession}, "
            f"found {chains}"
        )
    start, end = chains[0]
    return sequence, start, end


def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner


def _alignment_position_map(
    alignment: Any,
    reference_start: int,
    target_start: int,
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    reference_blocks = alignment.aligned[0]
    target_blocks = alignment.aligned[1]
    if len(reference_blocks) != len(target_blocks):
        raise MCSACorrespondenceError(
            "Alignment contains unequal numbers of reference and target blocks"
        )
    for ref_block, target_block in zip(reference_blocks, target_blocks):
        ref_begin, ref_end = map(int, ref_block)
        target_begin, target_end = map(int, target_block)
        if ref_end - ref_begin != target_end - target_begin:
            raise MCSACorrespondenceError("Aligned residue blocks have unequal lengths")
        for offset in range(ref_end - ref_begin):
            mapping[reference_start + ref_begin + offset] = (
                target_start + target_begin + offset
            )
    return mapping


def _alignment_statistics(
    alignment: Any,
    reference_sequence: str,
    reference_start: int,
    target_sequence: str,
    target_start: int,
) -> tuple[int, int, float]:
    mapping = _alignment_position_map(alignment, reference_start, target_start)
    matches = sum(
        reference_sequence[ref_pos - 1] == target_sequence[target_pos - 1]
        for ref_pos, target_pos in mapping.items()
    )
    aligned_pairs = len(mapping)
    identity = matches / aligned_pairs if aligned_pairs else 0.0
    return aligned_pairs, matches, identity


def _optimal_alignment_consensus(
    reference_sequence: str,
    reference_chain: tuple[int, int],
    target_sequence: str,
    target_chain: tuple[int, int],
    reference_positions: Sequence[int],
) -> tuple[dict[int, tuple[str, int | None]], dict[str, object]]:
    ref_start, ref_end = reference_chain
    target_start, target_end = target_chain
    ref_mature = reference_sequence[ref_start - 1 : ref_end]
    target_mature = target_sequence[target_start - 1 : target_end]

    alignments = _make_aligner().align(ref_mature, target_mature)
    optimal_count = len(alignments)
    if optimal_count < 1:
        raise MCSACorrespondenceError("Pairwise alignment returned no alignments")
    if optimal_count > _MAX_OPTIMAL_ALIGNMENTS:
        raise MCSACorrespondenceError(
            f"Pairwise alignment has {optimal_count} equally optimal solutions; "
            "refusing to infer catalytic correspondence from an incompletely "
            "enumerated alignment set"
        )

    maps: list[dict[int, int]] = []
    statistics: list[tuple[int, int, float]] = []
    score: float | None = None
    for alignment in alignments:
        maps.append(_alignment_position_map(alignment, ref_start, target_start))
        statistics.append(
            _alignment_statistics(
                alignment,
                reference_sequence,
                ref_start,
                target_sequence,
                target_start,
            )
        )
        if score is None:
            score = float(alignment.score)

    consensus: dict[int, tuple[str, int | None]] = {}
    for position in reference_positions:
        observed = {mapping.get(position) for mapping in maps}
        if len(observed) == 1:
            target_position = next(iter(observed))
            status = (
                "consensus_mapped" if target_position is not None else "consensus_gap"
            )
            consensus[position] = (status, target_position)
        else:
            consensus[position] = ("alignment_ambiguous", None)

    aligned_pair_counts = [value[0] for value in statistics]
    match_counts = [value[1] for value in statistics]
    identities = [value[2] for value in statistics]
    summary = {
        "alignment_method": _ALIGNMENT_METHOD,
        "substitution_matrix": "BLOSUM62",
        "gap_open_score": -10.0,
        "gap_extend_score": -0.5,
        "alignment_score": score,
        "optimal_alignment_count": optimal_count,
        "reference_chain_start": ref_start,
        "reference_chain_end": ref_end,
        "reference_chain_length": len(ref_mature),
        "target_chain_start": target_start,
        "target_chain_end": target_end,
        "target_chain_length": len(target_mature),
        "aligned_pair_count_min": min(aligned_pair_counts),
        "aligned_pair_count_max": max(aligned_pair_counts),
        "identity_min": min(identities),
        "identity_max": max(identities),
        "match_count_min": min(match_counts),
        "match_count_max": max(match_counts),
    }
    return consensus, summary


def _mcsa_residue_records(path: Path) -> list[Mapping[str, Any]]:
    payload = _read_json(path)
    records = payload.get("results") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list):
        raise MCSACorrespondenceError("M-CSA residue source is not a list")
    normalized: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise MCSACorrespondenceError("M-CSA residue record is not an object")
        normalized.append(record)
    if not normalized:
        raise MCSACorrespondenceError("M-CSA residue source is empty")
    return normalized


def _reference_residue(record: Mapping[str, Any]) -> tuple[str, int, str]:
    sequences = record.get("residue_sequences")
    if not isinstance(sequences, list):
        raise MCSACorrespondenceError("M-CSA residue has no residue_sequences list")
    references = [
        item
        for item in sequences
        if isinstance(item, Mapping) and item.get("is_reference") is True
    ]
    if len(references) != 1:
        raise MCSACorrespondenceError(
            "M-CSA residue must have exactly one reference-sequence annotation"
        )
    item = references[0]
    accession = str(item.get("uniprot_id") or "").strip().upper()
    residue = str(item.get("code") or "").strip()
    try:
        position = int(item.get("resid"))
    except (TypeError, ValueError) as exc:
        raise MCSACorrespondenceError(
            "Invalid M-CSA reference residue position"
        ) from exc
    if not accession or not residue:
        raise MCSACorrespondenceError("Incomplete M-CSA reference residue annotation")
    return accession, position, residue


def _unique_roles(record: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    roles = record.get("roles", [])
    if not isinstance(roles, list):
        raise MCSACorrespondenceError("M-CSA residue roles are not a list")
    unique: set[tuple[str, str, str, str]] = set()
    for role in roles:
        if not isinstance(role, Mapping):
            continue
        unique.add(
            (
                str(role.get("function_type") or "").strip(),
                str(role.get("group_function") or "").strip(),
                str(role.get("function") or "").strip(),
                str(role.get("emo") or "").strip(),
            )
        )
    return sorted(unique)


def _one_letter(code: str) -> str:
    mapping = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V",
    }
    if code not in mapping:
        raise MCSACorrespondenceError(f"Unsupported M-CSA residue code: {code}")
    return mapping[code]


def _target_assay_map(
    mapping: pd.DataFrame, protein_id: str, chain: str
) -> dict[int, dict[str, object]]:
    subset = mapping[
        (mapping["protein_id"] == protein_id) & (mapping["pdb_chain"] == chain)
    ].copy()
    subset = subset[subset["uniprot_position"].notna()]
    by_position: dict[int, dict[str, object]] = {}
    for _, row in subset.iterrows():
        position = int(row["uniprot_position"])
        record = {
            "assay_position": int(row["assay_position"]),
            "assay_residue": str(row["assay_residue"]),
            "pdb_author_position": (
                ""
                if pd.isna(row["pdb_author_position"])
                else str(row["pdb_author_position"])
            ),
            "pdb_modeled": bool(row["pdb_modeled"]),
        }
        previous = by_position.get(position)
        if previous is not None and previous != record:
            raise MCSACorrespondenceError(
                f"Non-unique assay/PDB mapping for {protein_id} "
                f"UniProt position {position}"
            )
        by_position[position] = record
    return by_position


def _selected_catalytic_contacts(
    contacts: pd.DataFrame,
    sites: pd.DataFrame,
    protein_id: str,
    chain: str,
) -> set[str]:
    selected_sites = sites[
        (sites["protein_id"] == protein_id)
        & (sites["chain_id"] == chain)
        & (sites["site_classification"] == "catalytic_site_candidate")
    ]
    site_ids = set(selected_sites["metal_site_id"].astype(str))
    subset = contacts[
        (contacts["protein_id"] == protein_id)
        & (contacts["contact_chain_id"] == chain)
        & (contacts["contact_type"] == "protein")
        & (contacts["metal_site_id"].astype(str).isin(site_ids))
    ]
    return set(subset["contact_author_position"].astype(str))


def prepare_mcsa_correspondence(repo_root: Path) -> Mapping[str, object]:
    """Map direct M-CSA curation to NDM-1/VIM-2 without numbering assumptions."""
    mcsa_records = _mcsa_residue_records(
        repo_root / "data/raw/mcsa/residues_15_16.json"
    )
    selected = _read_json(
        repo_root / "data/interim/protein_structures/selected_structures.json"
    )
    if not isinstance(selected, Mapping):
        raise MCSACorrespondenceError("Selected-structure record is not an object")

    residue_mapping = _read_csv(
        repo_root / "data/interim/sequence_mapping/residue_number_mapping.csv"
    )
    metal_contacts = _read_csv(
        repo_root / "data/interim/protein_structures/metal_contacts.csv"
    )
    metal_sites = _read_csv(
        repo_root / "data/interim/protein_structures/metal_sites.csv"
    )

    reference_accessions: dict[int, str] = {}
    positions_by_mcsa: dict[int, list[int]] = defaultdict(list)
    record_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in mcsa_records:
        try:
            mcsa_id = int(record.get("mcsa_id"))
        except (TypeError, ValueError) as exc:
            raise MCSACorrespondenceError("Invalid M-CSA residue mcsa_id") from exc
        accession, position, code = _reference_residue(record)
        reference_accessions.setdefault(mcsa_id, accession)
        if reference_accessions[mcsa_id] != accession:
            raise MCSACorrespondenceError(
                f"M-CSA {mcsa_id} has inconsistent reference accessions"
            )
        key = (mcsa_id, position)
        if key in record_by_key:
            raise MCSACorrespondenceError(
                f"Duplicate M-CSA catalytic residue {mcsa_id}:{position}"
            )
        record_by_key[key] = record
        positions_by_mcsa[mcsa_id].append(position)
        reference_sequence, _, _ = _sequence_and_chain(
            repo_root / f"data/raw/uniprot/{accession}/{accession}.json",
            accession,
        )
        if position > len(reference_sequence) or reference_sequence[
            position - 1
        ] != _one_letter(code):
            raise MCSACorrespondenceError(
                f"M-CSA {mcsa_id} residue {code}{position} does not match "
                f"UniProt {accession}"
            )

    if set(reference_accessions) != {15, 16}:
        raise MCSACorrespondenceError(
            f"Expected M-CSA entries 15 and 16, observed {sorted(reference_accessions)}"
        )

    alignment_rows: list[dict[str, object]] = []
    correspondence_rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []

    target_cache: dict[str, tuple[str, int, int]] = {}
    for protein_id, accession in _TARGETS.items():
        target_cache[protein_id] = _sequence_and_chain(
            repo_root / f"data/raw/uniprot/{accession}/{accession}.json", accession
        )

    for mcsa_id in sorted(reference_accessions):
        reference_accession = reference_accessions[mcsa_id]
        reference_sequence, ref_start, ref_end = _sequence_and_chain(
            repo_root
            / f"data/raw/uniprot/{reference_accession}/{reference_accession}.json",
            reference_accession,
        )
        positions = sorted(positions_by_mcsa[mcsa_id])
        roles_by_position: dict[int, list[tuple[str, str, str, str]]] = {}
        for reference_position in positions:
            record = record_by_key[(mcsa_id, reference_position)]
            _, _, reference_code = _reference_residue(record)
            reference_residue = _one_letter(reference_code)
            roles = _unique_roles(record)
            roles_by_position[reference_position] = roles
            for function_type, group_function, function, emo in roles:
                role_rows.append(
                    {
                        "mcsa_id": mcsa_id,
                        "reference_accession": reference_accession,
                        "reference_position": reference_position,
                        "reference_residue": reference_residue,
                        "function_type": function_type,
                        "group_function": group_function,
                        "function": function,
                        "emo_id": emo,
                    }
                )

        for protein_id, target_accession in _TARGETS.items():
            target_sequence, target_start, target_end = target_cache[protein_id]
            consensus, alignment_summary = _optimal_alignment_consensus(
                reference_sequence,
                (ref_start, ref_end),
                target_sequence,
                (target_start, target_end),
                positions,
            )
            selected_record = selected.get(protein_id)
            if not isinstance(selected_record, Mapping):
                raise MCSACorrespondenceError(
                    f"No selected structure for target protein {protein_id}"
                )
            pdb_id = str(selected_record.get("pdb_id") or "")
            chain = str(selected_record.get("primary_chain") or "")
            if not pdb_id or not chain:
                raise MCSACorrespondenceError(
                    f"Incomplete selected structure for target protein {protein_id}"
                )
            assay_map = _target_assay_map(residue_mapping, protein_id, chain)
            catalytic_contacts = _selected_catalytic_contacts(
                metal_contacts, metal_sites, protein_id, chain
            )

            alignment_rows.append(
                {
                    "mcsa_id": mcsa_id,
                    "reference_accession": reference_accession,
                    "target_protein": protein_id,
                    "target_accession": target_accession,
                    **alignment_summary,
                }
            )

            for reference_position in positions:
                record = record_by_key[(mcsa_id, reference_position)]
                _, _, reference_code = _reference_residue(record)
                reference_residue = _one_letter(reference_code)
                status, target_position = consensus[reference_position]
                target_residue = ""
                identity = False
                assay_position: object = ""
                assay_residue = ""
                pdb_author_position = ""
                pdb_modeled = False
                contacts_catalytic_zinc = False
                if target_position is not None:
                    target_residue = target_sequence[target_position - 1]
                    identity = target_residue == reference_residue
                    mapped = assay_map.get(target_position)
                    if mapped is not None:
                        assay_position = mapped["assay_position"]
                        assay_residue = str(mapped["assay_residue"])
                        pdb_author_position = str(mapped["pdb_author_position"])
                        pdb_modeled = bool(mapped["pdb_modeled"])
                        contacts_catalytic_zinc = (
                            pdb_author_position in catalytic_contacts
                        )

                roles = roles_by_position[reference_position]
                functions = sorted({role[2] for role in roles if role[2]})
                emos = sorted({role[3] for role in roles if role[3]})
                has_metal_ligand_role = "metal ligand" in functions
                if status != "consensus_mapped":
                    role_correspondence_status = "reference_role_not_mapped"
                elif identity:
                    role_correspondence_status = (
                        "reference_role_correspondence_identity_conserved"
                    )
                else:
                    role_correspondence_status = (
                        "reference_role_correspondence_nonidentical_residue"
                    )
                correspondence_rows.append(
                    {
                        "mcsa_id": mcsa_id,
                        "reference_accession": reference_accession,
                        "reference_position": reference_position,
                        "reference_residue": reference_residue,
                        "mcsa_reference_roles_summary_source": str(
                            record.get("roles_summary") or ""
                        ),
                        "mcsa_reference_unique_functions": ";".join(functions),
                        "mcsa_reference_unique_emo_ids": ";".join(emos),
                        "has_mcsa_metal_ligand_role": has_metal_ligand_role,
                        "annotation_relation": "homology_mapped_from_mcsa_reference",
                        "mcsa_role_correspondence_status": role_correspondence_status,
                        "target_protein": protein_id,
                        "target_accession": target_accession,
                        "mapping_status": status,
                        "target_uniprot_position": target_position or "",
                        "target_residue": target_residue,
                        "residue_identity_conserved": identity,
                        "target_assay_position": assay_position,
                        "target_assay_residue": assay_residue,
                        "selected_pdb_id": pdb_id,
                        "selected_pdb_chain": chain,
                        "selected_pdb_author_position": pdb_author_position,
                        "selected_pdb_modeled": pdb_modeled,
                        "target_contacts_selected_catalytic_zinc": (
                            contacts_catalytic_zinc
                        ),
                    }
                )

    role_keys = [
        (
            row["mcsa_id"],
            row["reference_accession"],
            row["reference_position"],
            row["reference_residue"],
            row["function_type"],
            row["group_function"],
            row["function"],
            row["emo_id"],
        )
        for row in role_rows
    ]
    if len(role_keys) != len(set(role_keys)):
        raise MCSACorrespondenceError(
            "Normalized M-CSA reference-role table contains duplicate annotations"
        )

    ambiguous = sum(
        row["mapping_status"] == "alignment_ambiguous" for row in correspondence_rows
    )
    gaps = sum(row["mapping_status"] == "consensus_gap" for row in correspondence_rows)
    mapped = [
        row
        for row in correspondence_rows
        if row["mapping_status"] == "consensus_mapped"
    ]
    metal_ligand_rows = [row for row in mapped if row["has_mcsa_metal_ligand_role"]]
    metal_ligand_identity_failures = sum(
        not bool(row["residue_identity_conserved"]) for row in metal_ligand_rows
    )
    metal_ligand_contact_failures = sum(
        not bool(row["target_contacts_selected_catalytic_zinc"])
        for row in metal_ligand_rows
    )
    dimetallic_rows = [row for row in metal_ligand_rows if int(row["mcsa_id"]) == 15]
    dimetallic_identity_failures = sum(
        not bool(row["residue_identity_conserved"]) for row in dimetallic_rows
    )
    dimetallic_contact_failures = sum(
        not bool(row["target_contacts_selected_catalytic_zinc"])
        for row in dimetallic_rows
    )
    dimetallic_reference_positions = {
        int(row["reference_position"]) for row in dimetallic_rows
    }
    dimetallic_expected_target_rows = len(dimetallic_reference_positions) * len(
        _TARGETS
    )
    biological_crosscheck = (
        "PASS"
        if ambiguous == 0
        and gaps == 0
        and metal_ligand_identity_failures == 0
        and metal_ligand_contact_failures == 0
        and len(dimetallic_reference_positions) == 6
        and dimetallic_expected_target_rows == len(dimetallic_rows)
        and dimetallic_identity_failures == 0
        and dimetallic_contact_failures == 0
        else "REVIEW_REQUIRED"
    )

    output_dir = repo_root / "data/interim/mcsa_correspondence"
    _write_csv(output_dir / "alignment_summary.csv", alignment_rows)
    _write_csv(output_dir / "catalytic_residue_correspondence.csv", correspondence_rows)
    _write_csv(output_dir / "catalytic_role_annotations.csv", role_rows)

    summary: Mapping[str, object] = {
        "status": "PASS",
        "biological_crosscheck_status": biological_crosscheck,
        "alignment_method": _ALIGNMENT_METHOD,
        "mcsa_entry_count": len(reference_accessions),
        "target_protein_count": len(_TARGETS),
        "alignment_pair_count": len(alignment_rows),
        "curated_reference_residue_count": len(record_by_key),
        "correspondence_row_count": len(correspondence_rows),
        "unique_role_annotation_count": len(role_rows),
        "alignment_ambiguous_row_count": ambiguous,
        "consensus_gap_row_count": gaps,
        "metal_ligand_correspondence_row_count": len(metal_ligand_rows),
        "metal_ligand_identity_failure_count": metal_ligand_identity_failures,
        "metal_ligand_selected_pdb_contact_failure_count": (
            metal_ligand_contact_failures
        ),
        "dimetallic_reference_mcsa_id": 15,
        "dimetallic_reference_unique_metal_ligand_count": len(
            dimetallic_reference_positions
        ),
        "dimetallic_reference_target_metal_ligand_row_count": len(dimetallic_rows),
        "dimetallic_reference_identity_failure_count": dimetallic_identity_failures,
        "dimetallic_reference_selected_pdb_contact_failure_count": (
            dimetallic_contact_failures
        ),
        "output_directory": output_dir.relative_to(repo_root).as_posix(),
        "outputs": [
            "alignment_summary.csv",
            "catalytic_residue_correspondence.csv",
            "catalytic_role_annotations.csv",
            "correspondence_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "correspondence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
