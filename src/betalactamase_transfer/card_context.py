# Acquire and prepare targeted CARD Antibiotic Resistance Ontology context.

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tarfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


class CARDContextError(RuntimeError):
    """Raised when CARD ontology acquisition or annotation is not reproducible."""


CARD_ONTOLOGY_VERSION = "4.0.1"
CARD_ONTOLOGY_RELEASE_DATE = "2025-05-29"
CARD_ONTOLOGY_OBO_DATE = "29:05:2025 10:16"
CARD_ONTOLOGY_URL = "https://card.mcmaster.ca/latest/ontology"
CARD_DOWNLOAD_PAGE = "https://card.mcmaster.ca/download"
CARD_ONTOLOGY_LICENSE = "CC BY 4.0"
CARD_ONTOLOGY_ARCHIVE_SHA256 = (
    "55c8a9efb255f3804735cbf4b13006fc67326dd80da3ec1d59c5f8dcefe8502d"
)
CARD_PRIMARY_CITATION = (
    "Alcock et al. 2023. Nucleic Acids Research 51:D690-D699. PMID:36263822"
)

TARGETS: Mapping[str, Mapping[str, str]] = {
    "NDM-1": {
        "aro_accession": "ARO:3000589",
        "card_name": "NDM-1",
        "family_aro": "ARO:3000057",
        "family_name": "NDM beta-lactamase",
    },
    "VIM-2": {
        "aro_accession": "ARO:3002272",
        "card_name": "VIM-2",
        "family_aro": "ARO:3000021",
        "family_name": "VIM beta-lactamase",
    },
}

EXPECTED_CLASSIFICATION_CHAIN: Tuple[Tuple[str, str], ...] = (
    ("ARO:3000568", "subclass B1 (metallo-) beta-lactamase"),
    ("ARO:3000004", "class B (metallo-) beta-lactamase"),
    ("ARO:3000001", "beta-lactamase"),
    ("ARO:3000557", "antibiotic inactivation enzyme"),
    ("ARO:3000000", "determinant of antibiotic resistance"),
)

EXPECTED_DRUG_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("ARO:0000020", "carbapenem"),
    ("ARO:0000032", "cephalosporin"),
    ("ARO:3000008", "penicillin beta-lactam"),
)

ASSAY_ANTIBIOTIC_NAMES: Mapping[str, str] = {
    "AMP": "ampicillin",
    "CTX": "cefotaxime",
    "MEM": "meropenem",
}

_REQUIRED_ARCHIVE_MEMBERS = {
    "aro.tsv",
    "ro.json",
    "aro.owl",
    "ro.obo",
    "mo.obo",
    "ncbi_taxonomy.obo",
    "ro.tsv",
    "viro.owl",
    "Ontology-Download-README.txt",
    "mo.json",
    "mo.tsv",
    "aro.obo",
    "viro.obo",
    "ncbi_taxonomy.tsv",
    "aro.json",
    "ncbi_taxonomy.json",
}


@dataclass(frozen=True)
class OBOTerm:
    """Minimal ARO term representation needed by this study."""

    term_id: str
    name: str
    definition: str
    parents: Tuple[str, ...]
    relationships: Tuple[Tuple[str, str], ...]
    synonyms: Tuple[str, ...]
    xrefs: Tuple[str, ...]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalise_member_name(name: str) -> str:
    value = name
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def _validate_archive_members(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        name = _normalise_member_name(member.name)
        if not name or name == ".":
            if member.isdir():
                continue
            raise CARDContextError(
                f"Unexpected unnamed member in CARD ontology archive: {member.name}"
            )
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise CARDContextError(
                f"Unsafe path in CARD ontology archive: {member.name}"
            )
        if member.issym() or member.islnk():
            raise CARDContextError(
                f"Links are not allowed in CARD ontology archive: {member.name}"
            )
        if not member.isfile():
            raise CARDContextError(
                f"Unexpected non-file member in CARD ontology archive: {member.name}"
            )
        if len(candidate.parts) != 1:
            raise CARDContextError(
                f"Unexpected nested member in CARD ontology archive: {member.name}"
            )
        if name in members:
            raise CARDContextError(f"Duplicate CARD ontology archive member: {name}")
        members[name] = member

    missing = sorted(_REQUIRED_ARCHIVE_MEMBERS - set(members))
    if missing:
        raise CARDContextError(
            "CARD ontology archive is missing required member(s): " + ", ".join(missing)
        )
    return members


def _write_bytes_without_silent_replacement(
    destination: Path,
    data: bytes,
    *,
    overwrite: bool,
) -> str:
    incoming_hash = hashlib.sha256(data).hexdigest()
    existed = destination.exists()
    if existed:
        existing_hash = _sha256_path(destination)
        if existing_hash == incoming_hash:
            return "already_present_identical"
        if not overwrite:
            raise CARDContextError(
                f"Refusing to replace non-identical existing file: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return "overwritten" if existed else "written"


def _copy_without_silent_replacement(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> str:
    source_hash = _sha256_path(source)
    if destination.exists():
        destination_hash = _sha256_path(destination)
        if destination_hash == source_hash:
            return "already_present_identical"
        if not overwrite:
            raise CARDContextError(
                f"Refusing to replace non-identical existing file: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return "copied" if not overwrite else "copied_or_overwritten"


def _read_obo_header(path: Path) -> Dict[str, str]:
    header: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                break
            if ": " in line:
                key, value = line.split(": ", 1)
                header[key] = value
    return header


def _unquote_definition(value: str) -> str:
    match = re.match(r'^"(.*)"\s+\[.*\]$', value)
    if match:
        return match.group(1).replace(r"\"", '"')
    return value


def _synonym_text(value: str) -> str:
    match = re.match(r'^"(.*?)"\s+', value)
    return match.group(1) if match else value


def parse_aro_obo(path: Path) -> Dict[str, OBOTerm]:
    """Parse the subset of OBO fields required for targeted CARD context."""

    terms: Dict[str, OBOTerm] = {}
    current: Dict[str, object] = {}

    def flush() -> None:
        if not current:
            return
        term_id = str(current.get("id", ""))
        if not term_id:
            current.clear()
            return
        if term_id in terms:
            raise CARDContextError(f"Duplicate ARO term in OBO: {term_id}")
        terms[term_id] = OBOTerm(
            term_id=term_id,
            name=str(current.get("name", "")),
            definition=str(current.get("definition", "")),
            parents=tuple(current.get("parents", ())),
            relationships=tuple(current.get("relationships", ())),
            synonyms=tuple(current.get("synonyms", ())),
            xrefs=tuple(current.get("xrefs", ())),
        )
        current.clear()

    with path.open("r", encoding="utf-8") as handle:
        in_term = False
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                flush()
                in_term = True
                continue
            if line.startswith("[") and line.endswith("]"):
                flush()
                in_term = False
                continue
            if not in_term:
                continue
            if not line:
                flush()
                in_term = False
                continue
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if key == "id":
                current["id"] = value
            elif key == "name":
                current["name"] = value
            elif key == "def":
                current["definition"] = _unquote_definition(value)
            elif key == "is_a":
                parent = value.split(" ! ", 1)[0].strip()
                current.setdefault("parents", []).append(parent)
            elif key == "relationship":
                parts = value.split()
                if len(parts) >= 2:
                    current.setdefault("relationships", []).append((parts[0], parts[1]))
            elif key == "synonym":
                current.setdefault("synonyms", []).append(_synonym_text(value))
            elif key == "xref":
                current.setdefault("xrefs", []).append(value)
        flush()

    if not terms:
        raise CARDContextError(f"No ARO terms parsed from {path}")
    return terms


def _validate_target_terms(terms: Mapping[str, OBOTerm]) -> None:
    for protein_id, expected in TARGETS.items():
        target_id = expected["aro_accession"]
        target = terms.get(target_id)
        if target is None:
            raise CARDContextError(
                f"Missing target ARO term for {protein_id}: {target_id}"
            )
        if target.name != expected["card_name"]:
            raise CARDContextError(
                f"Unexpected CARD name for {target_id}: {target.name!r}"
            )
        family_id = expected["family_aro"]
        family = terms.get(family_id)
        if family is None or family.name != expected["family_name"]:
            raise CARDContextError(
                f"Missing or mismatched CARD family term for {protein_id}: {family_id}"
            )
        if family_id not in target.parents:
            raise CARDContextError(
                f"{target_id} is not directly assigned to expected family {family_id}"
            )


def _ancestor_depths(terms: Mapping[str, OBOTerm], start_id: str) -> Dict[str, int]:
    depths: Dict[str, int] = {}
    queue = deque([(start_id, 0)])
    while queue:
        term_id, depth = queue.popleft()
        term = terms.get(term_id)
        if term is None:
            raise CARDContextError(f"ARO parent term is missing: {term_id}")
        for parent_id in term.parents:
            next_depth = depth + 1
            if parent_id not in depths or next_depth < depths[parent_id]:
                depths[parent_id] = next_depth
                queue.append((parent_id, next_depth))
    return depths


def _validate_expected_classification(terms: Mapping[str, OBOTerm]) -> None:
    for protein_id, target in TARGETS.items():
        depths = _ancestor_depths(terms, target["aro_accession"])
        for term_id, expected_name in EXPECTED_CLASSIFICATION_CHAIN:
            term = terms.get(term_id)
            if term is None or term.name != expected_name:
                raise CARDContextError(
                    f"Expected classification term missing or renamed: {term_id}"
                )
            if term_id not in depths:
                raise CARDContextError(
                    f"{protein_id} does not descend from expected term {term_id}"
                )


def _validate_expected_family_drug_classes(terms: Mapping[str, OBOTerm]) -> None:
    expected = {term_id for term_id, _ in EXPECTED_DRUG_CLASSES}
    for protein_id, target in TARGETS.items():
        family = terms[target["family_aro"]]
        observed = {
            object_id
            for relation, object_id in family.relationships
            if relation == "confers_resistance_to_drug_class"
        }
        if observed != expected:
            raise CARDContextError(
                f"Unexpected CARD family drug classes for {protein_id}: "
                f"{sorted(observed)}"
            )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def acquire_card_ontology(
    repo_root: Path,
    source_archive: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Preserve and validate the frozen official CARD ontology 4.0.1 archive."""

    repo_root = repo_root.resolve()
    source_archive = source_archive.expanduser().resolve()
    if not source_archive.is_file():
        raise CARDContextError(f"CARD ontology archive not found: {source_archive}")

    source_hash = _sha256_path(source_archive)
    if source_hash != CARD_ONTOLOGY_ARCHIVE_SHA256:
        raise CARDContextError(
            "CARD ontology archive SHA-256 does not match the frozen 4.0.1 source: "
            f"{source_hash}"
        )

    raw_dir = repo_root / "data/raw/card/ontology" / CARD_ONTOLOGY_VERSION
    output_dir = repo_root / "data/interim/card_acquisition"
    archive_destination = raw_dir / "card-ontology.tar.bz2"

    archive_status = _copy_without_silent_replacement(
        source_archive,
        archive_destination,
        overwrite=overwrite,
    )

    extracted_records: List[Dict[str, object]] = []
    with tarfile.open(source_archive, mode="r:bz2") as archive:
        members = _validate_archive_members(archive)
        for name in sorted(members):
            file_handle = archive.extractfile(members[name])
            if file_handle is None:
                raise CARDContextError(f"Could not read archive member: {name}")
            data = file_handle.read()
            destination = raw_dir / name
            status = _write_bytes_without_silent_replacement(
                destination,
                data,
                overwrite=overwrite,
            )
            extracted_records.append(
                {
                    "relative_path": destination.relative_to(repo_root).as_posix(),
                    "source_member": name,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "status": status,
                }
            )

    readme = (raw_dir / "Ontology-Download-README.txt").read_text(encoding="utf-8")
    if "Creative Commons CC-BY license version 4.0" not in readme:
        raise CARDContextError(
            "CARD ontology README does not state expected CC BY 4.0 license"
        )
    if "Alcock et al. 2023" not in readme:
        raise CARDContextError(
            "CARD ontology README does not contain expected primary citation"
        )

    aro_obo = raw_dir / "aro.obo"
    header = _read_obo_header(aro_obo)
    if header.get("date") != CARD_ONTOLOGY_OBO_DATE:
        raise CARDContextError(
            f"Unexpected ARO OBO date: {header.get('date')!r}; "
            f"expected {CARD_ONTOLOGY_OBO_DATE!r}"
        )
    if header.get("saved-by") != "CARD":
        raise CARDContextError("ARO OBO is not marked as saved by CARD")

    terms = parse_aro_obo(aro_obo)
    _validate_target_terms(terms)
    _validate_expected_classification(terms)
    _validate_expected_family_drug_classes(terms)

    manifest_lines = [
        f"{_sha256_path(archive_destination)}  card-ontology.tar.bz2",
    ]
    for record in extracted_records:
        manifest_lines.append(f"{record['sha256']}  {record['source_member']}")
    (raw_dir / "SHA256SUMS.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_inventory = [
        {
            "source_name": "CARD Antibiotic Resistance Ontology",
            "source_kind": "official_ontology_archive",
            "source_url": CARD_ONTOLOGY_URL,
            "download_page": CARD_DOWNLOAD_PAGE,
            "version": CARD_ONTOLOGY_VERSION,
            "release_date": CARD_ONTOLOGY_RELEASE_DATE,
            "obo_date": CARD_ONTOLOGY_OBO_DATE,
            "license": CARD_ONTOLOGY_LICENSE,
            "primary_citation": CARD_PRIMARY_CITATION,
            "acquisition_mode": "manual_official_download_then_verified_import",
            "source_archive_name": source_archive.name,
            "preserved_archive_path": archive_destination.relative_to(
                repo_root
            ).as_posix(),
            "archive_sha256": source_hash,
            "archive_status": archive_status,
        }
    ]
    _write_csv(
        output_dir / "source_inventory.csv",
        list(source_inventory[0]),
        source_inventory,
    )
    _write_csv(
        output_dir / "extracted_files.csv",
        ["relative_path", "source_member", "size_bytes", "sha256", "status"],
        extracted_records,
    )

    target_rows = []
    for protein_id, target_spec in TARGETS.items():
        target = terms[target_spec["aro_accession"]]
        family = terms[target_spec["family_aro"]]
        target_rows.append(
            {
                "protein_id": protein_id,
                "aro_accession": target.term_id,
                "card_name": target.name,
                "family_aro": family.term_id,
                "family_name": family.name,
                "status": "verified",
            }
        )
    _write_csv(
        output_dir / "target_presence.csv",
        [
            "protein_id",
            "aro_accession",
            "card_name",
            "family_aro",
            "family_name",
            "status",
        ],
        target_rows,
    )

    summary: Dict[str, object] = {
        "status": "PASS",
        "source": "CARD Antibiotic Resistance Ontology",
        "source_url": CARD_ONTOLOGY_URL,
        "version": CARD_ONTOLOGY_VERSION,
        "release_date": CARD_ONTOLOGY_RELEASE_DATE,
        "archive_sha256": source_hash,
        "archive_member_count": len(extracted_records),
        "aro_term_count": len(terms),
        "target_count": len(TARGETS),
        "target_accessions": sorted(
            target["aro_accession"] for target in TARGETS.values()
        ),
        "license": CARD_ONTOLOGY_LICENSE,
        "output_directory": "data/interim/card_acquisition",
        "raw_directory": f"data/raw/card/ontology/{CARD_ONTOLOGY_VERSION}",
        "outputs": [
            "source_inventory.csv",
            "extracted_files.csv",
            "target_presence.csv",
            "acquisition_summary.json",
        ],
    }
    (output_dir / "acquisition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _classification_rows(
    protein_id: str,
    target_id: str,
    terms: Mapping[str, OBOTerm],
) -> List[Dict[str, object]]:
    depths = _ancestor_depths(terms, target_id)
    rows = []
    for ancestor_id, depth in sorted(
        depths.items(), key=lambda item: (item[1], item[0])
    ):
        ancestor = terms[ancestor_id]
        rows.append(
            {
                "protein_id": protein_id,
                "target_aro": target_id,
                "ancestor_aro": ancestor_id,
                "ancestor_name": ancestor.name,
                "minimum_is_a_depth": depth,
                "evidence_type": "aro_is_a_ancestry",
            }
        )
    return rows


def _context_relationship_rows(
    protein_id: str,
    target_id: str,
    terms: Mapping[str, OBOTerm],
    relation_names: Sequence[str],
) -> List[Dict[str, object]]:
    depths = {target_id: 0, **_ancestor_depths(terms, target_id)}
    rows = []
    for source_id, depth in sorted(depths.items(), key=lambda item: (item[1], item[0])):
        source_term = terms[source_id]
        for relation, object_id in source_term.relationships:
            if relation not in relation_names:
                continue
            object_term = terms.get(object_id)
            if object_term is None:
                raise CARDContextError(
                    "ARO relationship target is missing: "
                    f"{source_id} {relation} {object_id}"
                )
            rows.append(
                {
                    "protein_id": protein_id,
                    "target_aro": target_id,
                    "source_term_aro": source_id,
                    "source_term_name": source_term.name,
                    "source_depth": depth,
                    "assertion_scope": "direct" if depth == 0 else "inherited_context",
                    "relationship": relation,
                    "object_aro": object_id,
                    "object_name": object_term.name,
                }
            )
    return rows


def _term_by_exact_name(terms: Mapping[str, OBOTerm], name: str) -> OBOTerm:
    matches = [term for term in terms.values() if term.name == name]
    if len(matches) != 1:
        raise CARDContextError(
            f"Expected exactly one ARO term named {name!r}; found {len(matches)}"
        )
    return matches[0]


def _assay_crosswalk_rows(
    assay_conditions_path: Path,
    terms: Mapping[str, OBOTerm],
) -> List[Dict[str, object]]:
    if not assay_conditions_path.is_file():
        raise CARDContextError(
            f"Assay condition table not found: {assay_conditions_path}"
        )
    with assay_conditions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed_codes = sorted(
        {row["antibiotic"] for row in rows if row.get("antibiotic")}
    )
    unsupported = sorted(set(observed_codes) - set(ASSAY_ANTIBIOTIC_NAMES))
    if unsupported:
        raise CARDContextError(
            "Assay table contains antibiotic code(s) without frozen CARD crosswalk: "
            + ", ".join(unsupported)
        )

    drug_class_ids = {term_id for term_id, _ in EXPECTED_DRUG_CLASSES}
    output_rows = []
    for code in observed_codes:
        antibiotic_name = ASSAY_ANTIBIOTIC_NAMES[code]
        antibiotic_term = _term_by_exact_name(terms, antibiotic_name)
        ancestry = _ancestor_depths(terms, antibiotic_term.term_id)
        matching_classes = [
            (term_id, depth)
            for term_id, depth in ancestry.items()
            if term_id in drug_class_ids
        ]
        if len(matching_classes) != 1:
            raise CARDContextError(
                f"Expected one CARD study drug class for {antibiotic_name}; "
                f"found {matching_classes}"
            )
        class_id, class_depth = matching_classes[0]
        class_term = terms[class_id]
        output_rows.append(
            {
                "assay_antibiotic_code": code,
                "assay_antibiotic_name": antibiotic_name,
                "antibiotic_aro": antibiotic_term.term_id,
                "card_drug_class_aro": class_id,
                "card_drug_class_name": class_term.name,
                "minimum_is_a_depth": class_depth,
                "mapping_basis": "assay_antibiotic_identity_plus_aro_is_a_ancestry",
                "mapping_status": "verified",
            }
        )
    return output_rows


def prepare_card_context(repo_root: Path) -> Dict[str, object]:
    """Prepare targeted NDM-1/VIM-2 CARD ontology context without model labels."""

    repo_root = repo_root.resolve()
    raw_dir = repo_root / "data/raw/card/ontology" / CARD_ONTOLOGY_VERSION
    aro_obo = raw_dir / "aro.obo"
    if not aro_obo.is_file():
        raise CARDContextError(
            "CARD ontology raw source is missing; run 11_acquire_card_ontology.py first"
        )

    if _sha256_path(raw_dir / "card-ontology.tar.bz2") != CARD_ONTOLOGY_ARCHIVE_SHA256:
        raise CARDContextError(
            "Preserved CARD ontology archive hash no longer matches source"
        )

    terms = parse_aro_obo(aro_obo)
    _validate_target_terms(terms)
    _validate_expected_classification(terms)
    _validate_expected_family_drug_classes(terms)

    output_dir = repo_root / "data/interim/card_context"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows: List[Dict[str, object]] = []
    ancestry_rows: List[Dict[str, object]] = []
    relationship_rows: List[Dict[str, object]] = []

    relation_names = (
        "confers_resistance_to_drug_class",
        "confers_resistance_to_antibiotic",
        "participates_in",
    )

    for protein_id, target_spec in TARGETS.items():
        target = terms[target_spec["aro_accession"]]
        family = terms[target_spec["family_aro"]]
        target_rows.append(
            {
                "protein_id": protein_id,
                "aro_accession": target.term_id,
                "card_name": target.name,
                "definition": target.definition,
                "family_aro": family.term_id,
                "family_name": family.name,
                "family_assignment_evidence": "direct_aro_is_a",
                "source_version": CARD_ONTOLOGY_VERSION,
            }
        )
        ancestry_rows.extend(_classification_rows(protein_id, target.term_id, terms))
        relationship_rows.extend(
            _context_relationship_rows(
                protein_id,
                target.term_id,
                terms,
                relation_names,
            )
        )

    drug_class_rows = [
        row
        for row in relationship_rows
        if row["relationship"] == "confers_resistance_to_drug_class"
    ]
    direct_antibiotic_rows = [
        row
        for row in relationship_rows
        if row["relationship"] == "confers_resistance_to_antibiotic"
    ]
    mechanism_rows = [
        row for row in relationship_rows if row["relationship"] == "participates_in"
    ]

    expected_drug_classes = {term_id for term_id, _ in EXPECTED_DRUG_CLASSES}
    for protein_id in TARGETS:
        observed = {
            row["object_aro"]
            for row in drug_class_rows
            if row["protein_id"] == protein_id
        }
        if observed != expected_drug_classes:
            raise CARDContextError(
                "Unexpected inherited drug-class set for "
                f"{protein_id}: {sorted(observed)}"
            )

    crosswalk_rows = _assay_crosswalk_rows(
        repo_root / "data/interim/assay_definitions/condition_score_statistics.csv",
        terms,
    )

    _write_csv(
        output_dir / "target_terms.csv",
        [
            "protein_id",
            "aro_accession",
            "card_name",
            "definition",
            "family_aro",
            "family_name",
            "family_assignment_evidence",
            "source_version",
        ],
        target_rows,
    )
    _write_csv(
        output_dir / "ontology_ancestry.csv",
        [
            "protein_id",
            "target_aro",
            "ancestor_aro",
            "ancestor_name",
            "minimum_is_a_depth",
            "evidence_type",
        ],
        ancestry_rows,
    )
    relationship_fields = [
        "protein_id",
        "target_aro",
        "source_term_aro",
        "source_term_name",
        "source_depth",
        "assertion_scope",
        "relationship",
        "object_aro",
        "object_name",
    ]
    _write_csv(
        output_dir / "drug_class_annotations.csv",
        relationship_fields,
        drug_class_rows,
    )
    _write_csv(
        output_dir / "direct_antibiotic_annotations.csv",
        relationship_fields,
        direct_antibiotic_rows,
    )
    _write_csv(
        output_dir / "mechanism_annotations.csv",
        relationship_fields,
        mechanism_rows,
    )
    _write_csv(
        output_dir / "assay_drug_class_crosswalk.csv",
        [
            "assay_antibiotic_code",
            "assay_antibiotic_name",
            "antibiotic_aro",
            "card_drug_class_aro",
            "card_drug_class_name",
            "minimum_is_a_depth",
            "mapping_basis",
            "mapping_status",
        ],
        crosswalk_rows,
    )

    ndm_direct_antibiotics = {
        row["object_name"]
        for row in direct_antibiotic_rows
        if row["protein_id"] == "NDM-1" and row["assertion_scope"] == "direct"
    }
    vim_direct_antibiotics = {
        row["object_name"]
        for row in direct_antibiotic_rows
        if row["protein_id"] == "VIM-2" and row["assertion_scope"] == "direct"
    }

    summary: Dict[str, object] = {
        "status": "PASS",
        "source": "CARD Antibiotic Resistance Ontology",
        "source_version": CARD_ONTOLOGY_VERSION,
        "source_release_date": CARD_ONTOLOGY_RELEASE_DATE,
        "target_count": len(target_rows),
        "target_accessions": sorted(row["aro_accession"] for row in target_rows),
        "classification_required_term_count": len(EXPECTED_CLASSIFICATION_CHAIN),
        "drug_class_annotation_row_count": len(drug_class_rows),
        "shared_expected_drug_class_count_per_target": len(EXPECTED_DRUG_CLASSES),
        "direct_antibiotic_annotation_row_count": len(direct_antibiotic_rows),
        "ndm1_direct_antibiotics": sorted(ndm_direct_antibiotics),
        "vim2_direct_antibiotics": sorted(vim_direct_antibiotics),
        "mechanism_annotation_row_count": len(mechanism_rows),
        "assay_drug_class_crosswalk_row_count": len(crosswalk_rows),
        "assay_drug_class_crosswalk_status": "PASS",
        "direct_vs_inherited_context_preserved": True,
        "card_sequence_used_as_numbering_authority": False,
        "card_fields_used_as_mutation_effect_labels": False,
        "output_directory": "data/interim/card_context",
        "outputs": [
            "target_terms.csv",
            "ontology_ancestry.csv",
            "drug_class_annotations.csv",
            "direct_antibiotic_annotations.csv",
            "mechanism_annotations.csv",
            "assay_drug_class_crosswalk.csv",
            "context_summary.json",
        ],
    }
    (output_dir / "context_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
