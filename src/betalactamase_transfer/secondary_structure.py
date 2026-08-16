# Assign DSSP secondary structure to selected experimental residues.

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Mapping

import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.DSSP import make_dssp_dict


class SecondaryStructureError(RuntimeError):
    """Raised when DSSP assignments cannot be prepared safely."""


SELECTED_STRUCTURES: Mapping[str, tuple[str, str]] = {
    "NDM-1": ("3SPU", "B"),
    "VIM-2": ("5YD7", "B"),
}

EXPECTED_MODELED_COUNTS: Mapping[str, int] = {
    "NDM-1": 238,
    "VIM-2": 231,
}

HELIX_CODES = {"H", "G", "I"}
STRAND_CODES = {"E", "B"}
COIL_CODES = {"T", "S", "P", "-"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SecondaryStructureError(f"Missing required input: {path}")
    return pd.read_csv(path, low_memory=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise SecondaryStructureError(f"Refusing to write an empty table: {path}")
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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "nan", "none"}:
        return False
    raise SecondaryStructureError(f"Cannot interpret boolean value: {value}")


def _optional_int(value: object) -> int | None:
    if pd.isna(value) or str(value).strip() in {"", "?"}:
        return None
    number = float(value)
    if not number.is_integer():
        raise SecondaryStructureError(f"Expected integer position, found {value}")
    return int(number)


def _normalize_insertion_code(value: object) -> str:
    text = str(value).strip()
    return text if text not in {"", "?", "nan"} else ""


def _reduce_dssp_class(code: str) -> str:
    if code in HELIX_CODES:
        return "helix"
    if code in STRAND_CODES:
        return "strand"
    if code in COIL_CODES:
        return "coil"
    raise SecondaryStructureError(f"Unexpected DSSP class: {code!r}")


def _resolve_mkdssp(supplied_path: Path | None) -> Path:
    if supplied_path is not None:
        candidate = supplied_path.expanduser().resolve()
        if not candidate.is_file():
            raise SecondaryStructureError(f"mkdssp not found: {candidate}")
        return candidate

    from_path = shutil.which("mkdssp")
    if from_path:
        return Path(from_path).resolve()

    local = Path.home() / ".local/bin/mkdssp"
    if local.is_file():
        return local.resolve()

    raise SecondaryStructureError(
        "mkdssp was not found in PATH or ~/.local/bin; supply --mkdssp"
    )


def _validate_mkdssp_version(version_text: str) -> None:
    """Require the DSSP 4.x command-line interface used here."""
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", version_text)
    if match is None:
        raise SecondaryStructureError(
            f"Could not parse mkdssp version from: {version_text!r}"
        )
    major = int(match.group(1))
    if major != 4:
        raise SecondaryStructureError(
            f"Secondary-structure preparation requires mkdssp 4.x; "
            f"found version text: {version_text!r}"
        )


def _mkdssp_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SecondaryStructureError(
            f"mkdssp --version failed with exit code {result.returncode}"
        )
    text = (result.stdout or result.stderr).strip()
    if not text:
        raise SecondaryStructureError("mkdssp --version returned no version text")
    return text.splitlines()[0]


def _run_dssp(executable: Path, cif_path: Path, output_path: Path) -> str:
    result = subprocess.run(
        [
            str(executable),
            "--output-format",
            "dssp",
            str(cif_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stderr = result.stderr.strip()
    if result.returncode != 0:
        raise SecondaryStructureError(
            f"mkdssp failed for {cif_path.name} with exit code "
            f"{result.returncode}: {stderr}"
        )
    if "Unknown compound:" in stderr:
        raise SecondaryStructureError(
            f"mkdssp reported unknown compounds for {cif_path.name}; "
            "verify the libcifpp Chemical Component Dictionary"
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise SecondaryStructureError(
            f"mkdssp did not create a non-empty output for {cif_path.name}"
        )
    return stderr


def prepare_secondary_structure(
    repo_root: Path,
    mkdssp_executable: Path | None = None,
) -> Mapping[str, object]:
    """Assign DSSP classes to selected PDB residues and assay positions."""
    repo_root = repo_root.resolve()
    output_dir = repo_root / "data/interim/secondary_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = _read_csv(
        repo_root / "data/interim/residue_evidence/residue_evidence.csv"
    )
    descriptors = _read_csv(
        repo_root
        / "data/interim/structural_descriptors/residue_structural_descriptors.csv"
    )
    if len(evidence) != 537 or len(descriptors) != 537:
        raise SecondaryStructureError(
            "Expected 537 rows in both residue evidence and structural descriptors"
        )
    keys = ["protein_id", "assay_position"]
    if evidence.duplicated(keys).any() or descriptors.duplicated(keys).any():
        raise SecondaryStructureError(
            "Residue inputs contain duplicate assay positions"
        )

    descriptor_lookup = descriptors.set_index(keys)
    executable = _resolve_mkdssp(mkdssp_executable)
    version = _mkdssp_version(executable)
    _validate_mkdssp_version(version)

    rows: List[Mapping[str, object]] = []
    modeled_counts: dict[str, int] = {}
    warning_line_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    assay_identity_mismatch_count = 0
    coordinate_identity_mismatch_count = 0

    with tempfile.TemporaryDirectory(prefix="beta-lactamase-dssp-") as temp_dir:
        temp_root = Path(temp_dir)
        for protein_id, (pdb_id, chain_id) in SELECTED_STRUCTURES.items():
            cif_path = repo_root / f"data/raw/pdb/{pdb_id}/{pdb_id}.cif"
            if not cif_path.is_file():
                raise SecondaryStructureError(f"Missing coordinate file: {cif_path}")
            dssp_path = temp_root / f"{pdb_id}.dssp"
            stderr = _run_dssp(executable, cif_path, dssp_path)
            warning_line_counts[protein_id] = len(
                [line for line in stderr.splitlines() if line.strip()]
            )
            dssp, dssp_keys = make_dssp_dict(str(dssp_path))
            chain_keys = [key for key in dssp_keys if key[0] == chain_id]
            if len(chain_keys) != EXPECTED_MODELED_COUNTS[protein_id]:
                raise SecondaryStructureError(
                    f"Unexpected DSSP residue count for {pdb_id}-{chain_id}: "
                    f"{len(chain_keys)}"
                )

            dssp_by_author_key = {}
            for _, residue_id in chain_keys:
                author_position = int(residue_id[1])
                insertion_code = _normalize_insertion_code(residue_id[2])
                author_key = (author_position, insertion_code)
                if author_key in dssp_by_author_key:
                    raise SecondaryStructureError(
                        f"Duplicate DSSP author key for {pdb_id}-{chain_id}: "
                        f"{author_key}"
                    )
                dssp_by_author_key[author_key] = dssp[(chain_id, residue_id)]

            protein_rows = evidence[evidence["protein_id"] == protein_id].sort_values(
                "assay_position"
            )
            modeled_count = 0
            protein_class_counts = {"helix": 0, "strand": 0, "coil": 0}
            for row in protein_rows.itertuples(index=False):
                assay_position = int(row.assay_position)
                base = {
                    "protein_id": protein_id,
                    "assay_position": assay_position,
                    "assay_residue": str(row.assay_residue),
                    "pdb_id": pdb_id,
                    "pdb_chain": chain_id,
                }
                if not _as_bool(row.pdb_modeled):
                    rows.append(
                        {
                            **base,
                            "secondary_structure_status": "no_experimental_coordinate",
                        }
                    )
                    continue

                author_position = _optional_int(row.pdb_author_position)
                if author_position is None:
                    raise SecondaryStructureError(
                        f"Modeled residue lacks PDB author position: "
                        f"{protein_id} {assay_position}"
                    )
                author_key = (author_position, "")
                if author_key not in dssp_by_author_key:
                    raise SecondaryStructureError(
                        f"No DSSP assignment for {pdb_id}-{chain_id} author "
                        f"position {author_position}"
                    )

                dssp_record = dssp_by_author_key[author_key]
                dssp_aa = str(dssp_record[0]).upper()
                dssp_class = str(dssp_record[1])
                reduced_class = _reduce_dssp_class(dssp_class)

                descriptor = descriptor_lookup.loc[(protein_id, assay_position)]
                pdb_residue_name = str(descriptor["pdb_residue_name"]).upper()
                coordinate_aa = protein_letters_3to1.get(pdb_residue_name)
                if coordinate_aa is None:
                    raise SecondaryStructureError(
                        f"Unsupported PDB residue name {pdb_residue_name} for "
                        f"{protein_id} {assay_position}"
                    )
                coordinate_identity = dssp_aa == coordinate_aa
                assay_identity = dssp_aa == str(row.assay_residue).upper()
                coordinate_identity_mismatch_count += int(not coordinate_identity)
                assay_identity_mismatch_count += int(not assay_identity)
                if not coordinate_identity:
                    raise SecondaryStructureError(
                        f"DSSP/PDB residue mismatch for {protein_id} "
                        f"{assay_position}: DSSP={dssp_aa}, PDB={coordinate_aa}"
                    )

                rows.append(
                    {
                        **base,
                        "secondary_structure_status": "dssp_assignment_available",
                        "pdb_author_position": author_position,
                        "pdb_insertion_code": "",
                        "pdb_residue_name": pdb_residue_name,
                        "dssp_amino_acid": dssp_aa,
                        "dssp_coordinate_residue_identity": coordinate_identity,
                        "dssp_assay_residue_identity": assay_identity,
                        "known_assay_reference_difference": _as_bool(
                            row.known_assay_reference_difference
                        ),
                        "dssp_class": dssp_class,
                        "secondary_structure_3state": reduced_class,
                    }
                )
                modeled_count += 1
                protein_class_counts[reduced_class] += 1

            modeled_counts[protein_id] = modeled_count
            class_counts[protein_id] = protein_class_counts

    if modeled_counts != dict(EXPECTED_MODELED_COUNTS):
        raise SecondaryStructureError(
            f"Unexpected mapped DSSP counts: {modeled_counts}"
        )
    if len(rows) != 537:
        raise SecondaryStructureError(
            f"Expected 537 secondary-structure rows, found {len(rows)}"
        )
    if coordinate_identity_mismatch_count != 0:
        raise SecondaryStructureError("Unexpected DSSP/PDB identity mismatches")

    frame = pd.DataFrame(rows)
    no_coordinate_count = int(
        (frame["secondary_structure_status"] == "no_experimental_coordinate").sum()
    )
    if no_coordinate_count != 68:
        raise SecondaryStructureError(
            f"Expected 68 residues without experimental coordinates, found "
            f"{no_coordinate_count}"
        )

    assignment_available = (
        frame["secondary_structure_status"] == "dssp_assignment_available"
    )
    assay_identity = frame["dssp_assay_residue_identity"].eq(True)
    known_difference = frame["known_assay_reference_difference"].eq(True)

    known_difference_mismatches = frame[
        assignment_available & (~assay_identity) & known_difference
    ]
    unexpected_assay_mismatches = frame[
        assignment_available & (~assay_identity) & (~known_difference)
    ]
    if not unexpected_assay_mismatches.empty:
        raise SecondaryStructureError(
            "DSSP/assay residue mismatches exist outside documented "
            "assay-reference differences"
        )

    _write_csv(output_dir / "secondary_structure.csv", rows)
    summary: Mapping[str, object] = {
        "status": "PASS",
        "protein_count": 2,
        "residue_row_count": int(len(frame)),
        "modeled_dssp_counts": modeled_counts,
        "no_experimental_coordinate_count": no_coordinate_count,
        "coordinate_identity_mismatch_count": coordinate_identity_mismatch_count,
        "assay_identity_mismatch_count": assay_identity_mismatch_count,
        "documented_assay_reference_mismatch_count": int(
            len(known_difference_mismatches)
        ),
        "secondary_structure_3state_counts": class_counts,
        "mkdssp_version": version,
        "dssp_output_format": "classic dssp",
        "dssp_role": "secondary_structure_assignment_only",
        "dssp_accessibility_used_as_feature": False,
        "mkdssp_warning_line_counts": warning_line_counts,
        "selected_structures": {
            protein_id: {"pdb_id": pdb_id, "chain": chain_id}
            for protein_id, (pdb_id, chain_id) in SELECTED_STRUCTURES.items()
        },
        "outputs": ["secondary_structure.csv", "secondary_structure_summary.json"],
    }
    (output_dir / "secondary_structure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
