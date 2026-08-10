# Acquire targeted M-CSA catalytic and mechanistic records.

from __future__ import annotations

import csv
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MCSAAnnotationError(RuntimeError):
    """Raised when targeted M-CSA records cannot be acquired safely."""


MCSA_API_ROOT = "https://www.ebi.ac.uk/thornton-srv/m-csa/api"
DEFAULT_MCSA_IDS = (15, 16)
ENTRY_ROLES: Mapping[int, str] = {
    15: "class_b1_dimetallic_reference",
    16: "class_b1_monometallic_reference",
}
_ALLOWED_HOST = "www.ebi.ac.uk"

FetchBytes = Callable[[str, int], tuple[bytes, str]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_fetch_bytes(url: str, timeout: int) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "beta-lactamase-mutation-transfer/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise MCSAAnnotationError(f"Unexpected HTTP status {status} for {url}")
            return response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        raise MCSAAnnotationError(f"HTTP {exc.code} while retrieving {url}") from exc
    except urllib.error.URLError as exc:
        raise MCSAAnnotationError(f"Could not retrieve {url}: {exc.reason}") from exc


def _validate_api_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise MCSAAnnotationError(f"M-CSA API URL is not HTTPS: {url}")
    if (parsed.hostname or "").lower() != _ALLOWED_HOST:
        raise MCSAAnnotationError(
            f"Unexpected host in M-CSA API URL: {parsed.hostname}"
        )
    if not parsed.path.startswith("/thornton-srv/m-csa/api/"):
        raise MCSAAnnotationError(f"Unexpected M-CSA API path: {parsed.path}")


def _target_url(view: str, mcsa_ids: Sequence[int]) -> str:
    if view not in {"entries", "residues"}:
        raise MCSAAnnotationError(f"Unsupported M-CSA API view: {view}")
    id_text = ",".join(str(value) for value in mcsa_ids)
    url = f"{MCSA_API_ROOT}/{view}/?format=json&entries.mcsa_ids={id_text}"
    _validate_api_url(url)
    return url


def _validate_json(data: bytes, label: str, url: str) -> object:
    if not data:
        raise MCSAAnnotationError(f"Empty {label} response from {url}")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MCSAAnnotationError(f"Invalid {label} JSON from {url}") from exc


def _normalize_api_page(payload: object, label: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        records: Sequence[object] = payload
    elif isinstance(payload, Mapping):
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise MCSAAnnotationError(
                f"{label} response has no list-valued results field"
            )
        if payload.get("next") not in {None, ""}:
            raise MCSAAnnotationError(
                f"{label} response is paginated; targeted acquisition "
                "would be incomplete"
            )
        declared_count = payload.get("count")
        if declared_count is not None:
            try:
                count = int(declared_count)
            except (TypeError, ValueError) as exc:
                raise MCSAAnnotationError(
                    f"{label} response has a non-integer count"
                ) from exc
            if count != len(raw_results):
                raise MCSAAnnotationError(
                    f"{label} response count {count} does not match "
                    f"returned records {len(raw_results)}"
                )
        records = raw_results
    else:
        raise MCSAAnnotationError(
            f"{label} response is neither a JSON object nor a list"
        )

    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MCSAAnnotationError(f"{label} record {index} is not a JSON object")
        normalized.append(record)
    if not normalized:
        raise MCSAAnnotationError(f"{label} response contains no records")
    return normalized


def _entry_id(record: Mapping[str, Any]) -> int:
    value = record.get("mcsa_id")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MCSAAnnotationError(
            f"M-CSA entry has invalid mcsa_id: {value!r}"
        ) from exc


def _validate_entry_selection(
    records: Sequence[Mapping[str, Any]], expected_ids: Sequence[int]
) -> list[Mapping[str, Any]]:
    expected = set(expected_ids)
    by_id: dict[int, Mapping[str, Any]] = {}
    for record in records:
        mcsa_id = _entry_id(record)
        if mcsa_id in by_id:
            raise MCSAAnnotationError(f"Duplicate M-CSA entry {mcsa_id}")
        by_id[mcsa_id] = record

    observed = set(by_id)
    if observed != expected:
        raise MCSAAnnotationError(
            f"Targeted M-CSA entry mismatch: expected {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )

    selected: list[Mapping[str, Any]] = []
    for mcsa_id in expected_ids:
        record = by_id[mcsa_id]
        enzyme_name = str(record.get("enzyme_name") or "").strip()
        if (
            "beta-lactamase" not in enzyme_name.lower()
            or "class b1" not in enzyme_name.lower()
        ):
            raise MCSAAnnotationError(
                f"M-CSA entry {mcsa_id} is not labelled beta-lactamase (Class B1): "
                f"{enzyme_name!r}"
            )
        ecs = record.get("all_ecs")
        if not isinstance(ecs, list) or "3.5.2.6" not in {str(value) for value in ecs}:
            raise MCSAAnnotationError(
                f"M-CSA entry {mcsa_id} does not report EC 3.5.2.6"
            )
        reference_uniprot = str(record.get("reference_uniprot_id") or "").strip()
        if not reference_uniprot:
            raise MCSAAnnotationError(
                f"M-CSA entry {mcsa_id} has no reference UniProt identifier"
            )
        residues = record.get("residues")
        if not isinstance(residues, list) or not residues:
            raise MCSAAnnotationError(
                f"M-CSA entry {mcsa_id} contains no curated catalytic residues"
            )
        selected.append(record)
    return selected


def _reference_residue_signature(
    record: Mapping[str, Any],
) -> tuple[int, str, str, int, str]:
    mcsa_id = _entry_id(record)
    sequences = record.get("residue_sequences")
    if not isinstance(sequences, list):
        raise MCSAAnnotationError(
            f"M-CSA residue record for entry {mcsa_id} has no residue_sequences list"
        )

    reference_sequences = [
        sequence
        for sequence in sequences
        if isinstance(sequence, Mapping) and bool(sequence.get("is_reference"))
    ]
    if len(reference_sequences) != 1:
        raise MCSAAnnotationError(
            f"M-CSA residue record for entry {mcsa_id} has "
            f"{len(reference_sequences)} reference sequence residues; "
            "expected exactly 1"
        )

    sequence = reference_sequences[0]
    uniprot_id = str(sequence.get("uniprot_id") or "").strip()
    code = str(sequence.get("code") or "").strip()
    resid_value = sequence.get("resid")
    try:
        resid = int(resid_value)
    except (TypeError, ValueError) as exc:
        raise MCSAAnnotationError(
            f"M-CSA residue record for entry {mcsa_id} has invalid reference residue "
            f"number: {resid_value!r}"
        ) from exc

    if not uniprot_id or not code:
        raise MCSAAnnotationError(
            f"M-CSA residue record for entry {mcsa_id} has incomplete reference "
            "sequence annotation"
        )

    return mcsa_id, uniprot_id, code, resid, str(record.get("roles_summary") or "")


def _validate_residue_selection(
    residue_records: Sequence[Mapping[str, Any]],
    selected_entries: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[int],
) -> Mapping[int, int]:
    expected = set(expected_ids)
    residue_counts = {mcsa_id: 0 for mcsa_id in expected_ids}
    api_signatures: set[tuple[int, str, str, int, str]] = set()

    for record in residue_records:
        mcsa_id = _entry_id(record)
        if mcsa_id not in expected:
            raise MCSAAnnotationError(
                f"Targeted M-CSA residue response contains unexpected entry {mcsa_id}"
            )
        signature = _reference_residue_signature(record)
        if signature in api_signatures:
            raise MCSAAnnotationError(
                f"Duplicate curated M-CSA residue record in API response: {signature}"
            )
        api_signatures.add(signature)
        residue_counts[mcsa_id] += 1

    missing_ids = [mcsa_id for mcsa_id, count in residue_counts.items() if count == 0]
    if missing_ids:
        raise MCSAAnnotationError(
            "Targeted M-CSA residue response contains no records for entries "
            f"{missing_ids}"
        )

    embedded_signatures: set[tuple[int, str, str, int, str]] = set()
    embedded_counts: dict[int, int] = {}
    for entry in selected_entries:
        mcsa_id = _entry_id(entry)
        residues = entry.get("residues")
        if not isinstance(residues, list):
            raise MCSAAnnotationError(
                f"M-CSA entry {mcsa_id} has no embedded curated residue list"
            )
        embedded_counts[mcsa_id] = len(residues)
        for residue in residues:
            if not isinstance(residue, Mapping):
                raise MCSAAnnotationError(
                    f"M-CSA entry {mcsa_id} contains a non-object residue record"
                )
            signature = _reference_residue_signature(residue)
            if signature in embedded_signatures:
                raise MCSAAnnotationError(
                    "Duplicate curated M-CSA residue embedded in entry response: "
                    f"{signature}"
                )
            embedded_signatures.add(signature)

    if residue_counts != embedded_counts:
        raise MCSAAnnotationError(
            "M-CSA curated residue counts disagree between entry and residue API "
            f"views: entry={embedded_counts}, residue_api={residue_counts}"
        )
    if api_signatures != embedded_signatures:
        missing_from_api = sorted(embedded_signatures - api_signatures)
        extra_in_api = sorted(api_signatures - embedded_signatures)
        raise MCSAAnnotationError(
            "M-CSA curated residue identities disagree between entry and residue API "
            f"views; missing_from_residue_api={missing_from_api}, "
            f"extra_in_residue_api={extra_in_api}"
        )

    return residue_counts


def _mechanism_summary(record: Mapping[str, Any]) -> tuple[int, int, str]:
    reaction = record.get("reaction")
    if not isinstance(reaction, Mapping):
        return 0, 0, ""
    mechanisms = reaction.get("mechanisms")
    if not isinstance(mechanisms, list):
        return 0, 0, str(reaction.get("ec") or "")
    detailed = sum(
        1
        for mechanism in mechanisms
        if isinstance(mechanism, Mapping) and bool(mechanism.get("is_detailed"))
    )
    return len(mechanisms), detailed, str(reaction.get("ec") or "")


def _reference_pdbs(record: Mapping[str, Any]) -> list[str]:
    residues = record.get("residues")
    if not isinstance(residues, list):
        return []
    pdb_ids: set[str] = set()
    for residue in residues:
        if not isinstance(residue, Mapping):
            continue
        chains = residue.get("residue_chains")
        if not isinstance(chains, list):
            continue
        for chain in chains:
            if not isinstance(chain, Mapping) or not bool(chain.get("is_reference")):
                continue
            pdb_id = str(chain.get("pdb_id") or "").strip().lower()
            if pdb_id:
                pdb_ids.add(pdb_id)
    return sorted(pdb_ids)


def _entry_summary_row(record: Mapping[str, Any]) -> Mapping[str, object]:
    mcsa_id = _entry_id(record)
    mechanism_count, detailed_count, reaction_ec = _mechanism_summary(record)
    residues = record.get("residues")
    residue_count = len(residues) if isinstance(residues, list) else 0
    all_ecs = record.get("all_ecs")
    ecs = [str(value) for value in all_ecs] if isinstance(all_ecs, list) else []
    return {
        "mcsa_id": mcsa_id,
        "requested_role": ENTRY_ROLES.get(mcsa_id, "targeted_reference"),
        "enzyme_name": str(record.get("enzyme_name") or ""),
        "reference_uniprot_id": str(record.get("reference_uniprot_id") or ""),
        "reference_pdb_ids": ";".join(_reference_pdbs(record)),
        "ec_numbers": ";".join(ecs),
        "reaction_ec": reaction_ec,
        "curated_catalytic_residue_count": residue_count,
        "mechanism_count": mechanism_count,
        "detailed_mechanism_count": detailed_count,
        "entry_url": str(record.get("url") or ""),
    }


def _write_bytes_safely(path: Path, data: bytes, overwrite: bool) -> str:
    digest = _sha256(data)
    existed = path.exists()
    if existed:
        if _sha256(path.read_bytes()) == digest:
            return "already_present"
        if not overwrite:
            raise MCSAAnnotationError(
                "Refusing to replace a different existing file "
                f"without --overwrite: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "replaced" if existed else "downloaded"


def _write_text_safely(path: Path, text: str, overwrite: bool) -> str:
    return _write_bytes_safely(path, text.encode("utf-8"), overwrite)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise MCSAAnnotationError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _checksum_text(files: Sequence[Path]) -> str:
    lines = [f"{_sha256(path.read_bytes())}  {path.name}" for path in sorted(files)]
    return "\n".join(lines) + "\n"


def acquire_mcsa_annotations(
    repo_root: Path,
    mcsa_ids: Sequence[int] = DEFAULT_MCSA_IDS,
    *,
    timeout: int = 60,
    overwrite: bool = False,
    fetch_bytes: FetchBytes = _default_fetch_bytes,
    retrieved_at: str | None = None,
) -> Mapping[str, object]:
    """Acquire the targeted manually curated M-CSA B1 records."""
    if timeout <= 0:
        raise MCSAAnnotationError("Timeout must be a positive integer")

    normalized_ids = tuple(dict.fromkeys(int(value) for value in mcsa_ids))
    if not normalized_ids or any(value <= 0 for value in normalized_ids):
        raise MCSAAnnotationError("At least one positive M-CSA identifier is required")

    timestamp = retrieved_at or _utc_now()
    entries_url = _target_url("entries", normalized_ids)
    residues_url = _target_url("residues", normalized_ids)

    entries_bytes, entries_content_type = fetch_bytes(entries_url, timeout)
    entries_payload = _validate_json(entries_bytes, "M-CSA entries", entries_url)
    entries_records = _normalize_api_page(entries_payload, "M-CSA entries")
    selected_entries = _validate_entry_selection(entries_records, normalized_ids)

    residues_bytes, residues_content_type = fetch_bytes(residues_url, timeout)
    residues_payload = _validate_json(residues_bytes, "M-CSA residues", residues_url)
    residue_records = _normalize_api_page(residues_payload, "M-CSA residues")
    residue_counts = _validate_residue_selection(
        residue_records, selected_entries, normalized_ids
    )

    raw_dir = repo_root / "data/raw/mcsa"
    output_dir = repo_root / "data/interim/mcsa_acquisition"
    id_text = "_".join(str(value) for value in normalized_ids)
    entries_path = raw_dir / f"entries_{id_text}.json"
    residues_path = raw_dir / f"residues_{id_text}.json"

    entries_status = _write_bytes_safely(entries_path, entries_bytes, overwrite)
    residues_status = _write_bytes_safely(residues_path, residues_bytes, overwrite)

    checksum_path = raw_dir / "SHA256SUMS.txt"
    checksum_status = _write_text_safely(
        checksum_path,
        _checksum_text([entries_path, residues_path]),
        overwrite,
    )

    downloaded_rows = [
        {
            "file_role": "targeted_entries",
            "local_path": entries_path.relative_to(repo_root).as_posix(),
            "source_url": entries_url,
            "content_type": entries_content_type,
            "sha256": _sha256(entries_path.read_bytes()),
            "size_bytes": entries_path.stat().st_size,
            "retrieved_at_utc": timestamp,
            "status": entries_status,
            "licence": "CC-BY-4.0",
        },
        {
            "file_role": "targeted_curated_residues",
            "local_path": residues_path.relative_to(repo_root).as_posix(),
            "source_url": residues_url,
            "content_type": residues_content_type,
            "sha256": _sha256(residues_path.read_bytes()),
            "size_bytes": residues_path.stat().st_size,
            "retrieved_at_utc": timestamp,
            "status": residues_status,
            "licence": "CC-BY-4.0",
        },
        {
            "file_role": "checksums",
            "local_path": checksum_path.relative_to(repo_root).as_posix(),
            "source_url": "generated_locally",
            "content_type": "text/plain",
            "sha256": _sha256(checksum_path.read_bytes()),
            "size_bytes": checksum_path.stat().st_size,
            "retrieved_at_utc": timestamp,
            "status": checksum_status,
            "licence": "CC-BY-4.0",
        },
    ]
    entry_rows = [_entry_summary_row(record) for record in selected_entries]

    _write_csv(output_dir / "downloaded_files.csv", downloaded_rows)
    _write_csv(output_dir / "entries.csv", entry_rows)

    summary: Mapping[str, object] = {
        "status": "PASS",
        "mcsa_ids": list(normalized_ids),
        "entry_count": len(selected_entries),
        "curated_residue_api_record_count": len(residue_records),
        "curated_residue_counts_by_mcsa_id": {
            str(mcsa_id): residue_counts[mcsa_id] for mcsa_id in normalized_ids
        },
        "entry_residue_api_crosscheck": "PASS",
        "raw_file_record_count": len(downloaded_rows),
        "homologue_bulk_file_acquired": False,
        "output_directory": output_dir.relative_to(repo_root).as_posix(),
        "outputs": [
            "downloaded_files.csv",
            "entries.csv",
            "acquisition_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acquisition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
