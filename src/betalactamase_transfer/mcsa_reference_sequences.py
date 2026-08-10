# Acquire UniProt reference sequences used by the selected M-CSA entries.

from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MCSAReferenceSequenceError(RuntimeError):
    """Raised when an M-CSA reference UniProt record cannot be acquired safely."""


UNIPROT_REST_ROOT = "https://rest.uniprot.org/uniprotkb"
_ALLOWED_HOST = "rest.uniprot.org"
_SUPPORTED_FORMATS = ("fasta", "json", "txt")
_VALID_SEQUENCE = re.compile(r"^[A-Z]+$")

FetchResponse = Callable[[str, int], tuple[bytes, str, Mapping[str, str]]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _default_fetch_response(
    url: str, timeout: int
) -> tuple[bytes, str, Mapping[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "*/*",
            "User-Agent": "beta-lactamase-mutation-transfer/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise MCSAReferenceSequenceError(
                    f"Unexpected HTTP status {status} for {url}"
                )
            headers = {str(key): str(value) for key, value in response.headers.items()}
            return response.read(), response.headers.get_content_type(), headers
    except urllib.error.HTTPError as exc:
        raise MCSAReferenceSequenceError(
            f"HTTP {exc.code} while retrieving {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise MCSAReferenceSequenceError(
            f"Could not retrieve {url}: {exc.reason}"
        ) from exc


def _validate_uniprot_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise MCSAReferenceSequenceError(f"UniProt URL is not HTTPS: {url}")
    if (parsed.hostname or "").lower() != _ALLOWED_HOST:
        raise MCSAReferenceSequenceError(
            f"Unexpected host in UniProt URL: {parsed.hostname}"
        )
    if not parsed.path.startswith("/uniprotkb/"):
        raise MCSAReferenceSequenceError(f"Unexpected UniProt path: {parsed.path}")


def _uniprot_url(accession: str, fmt: str) -> str:
    if fmt not in _SUPPORTED_FORMATS:
        raise MCSAReferenceSequenceError(f"Unsupported UniProt format: {fmt}")
    normalized = accession.strip().upper()
    if not normalized:
        raise MCSAReferenceSequenceError("Empty UniProt accession")
    url = f"{UNIPROT_REST_ROOT}/{normalized}.{fmt}"
    _validate_uniprot_url(url)
    return url


def _load_mcsa_reference_accessions(mcsa_path: Path) -> dict[int, str]:
    if not mcsa_path.is_file():
        raise MCSAReferenceSequenceError(
            f"M-CSA entry source file does not exist: {mcsa_path}"
        )
    try:
        payload = json.loads(mcsa_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MCSAReferenceSequenceError(
            f"Invalid M-CSA entry JSON: {mcsa_path}"
        ) from exc

    records = payload.get("results") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list):
        raise MCSAReferenceSequenceError(
            "M-CSA entry source has no list-valued records"
        )

    references: dict[int, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise MCSAReferenceSequenceError("M-CSA entry record is not an object")
        try:
            mcsa_id = int(record.get("mcsa_id"))
        except (TypeError, ValueError) as exc:
            raise MCSAReferenceSequenceError(
                f"Invalid M-CSA entry identifier: {record.get('mcsa_id')!r}"
            ) from exc
        accession = str(record.get("reference_uniprot_id") or "").strip().upper()
        if not accession:
            raise MCSAReferenceSequenceError(
                f"M-CSA entry {mcsa_id} has no reference UniProt accession"
            )
        if mcsa_id in references:
            raise MCSAReferenceSequenceError(f"Duplicate M-CSA entry {mcsa_id}")
        references[mcsa_id] = accession

    if not references:
        raise MCSAReferenceSequenceError("M-CSA entry source contains no references")
    return references


def _validate_plain_sequence(sequence: str, label: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise MCSAReferenceSequenceError(f"{label} contains an empty sequence")
    if not _VALID_SEQUENCE.fullmatch(normalized):
        raise MCSAReferenceSequenceError(
            f"{label} contains non-alphabetic protein-sequence characters"
        )
    return normalized


def _parse_fasta(data: bytes, accession: str) -> tuple[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MCSAReferenceSequenceError(
            f"UniProt FASTA for {accession} is not UTF-8"
        ) from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise MCSAReferenceSequenceError(
            f"UniProt FASTA for {accession} has an invalid header"
        )
    header = lines[0][1:]
    tokens = header.split("|")
    header_accessions = {token.strip().upper() for token in tokens[:3]}
    if accession.upper() not in header_accessions:
        raise MCSAReferenceSequenceError(
            f"UniProt FASTA header does not identify {accession}: {header!r}"
        )
    sequence = _validate_plain_sequence("".join(lines[1:]), f"FASTA {accession}")
    return sequence, header


def _parse_json(data: bytes, accession: str) -> dict[str, Any]:
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MCSAReferenceSequenceError(
            f"UniProt JSON for {accession} is invalid"
        ) from exc
    if not isinstance(record, Mapping):
        raise MCSAReferenceSequenceError(
            f"UniProt JSON for {accession} is not an object"
        )
    primary = str(record.get("primaryAccession") or "").strip().upper()
    if primary != accession.upper():
        raise MCSAReferenceSequenceError(
            f"UniProt JSON accession mismatch: expected {accession}, observed {primary}"
        )
    sequence_record = record.get("sequence")
    if not isinstance(sequence_record, Mapping):
        raise MCSAReferenceSequenceError(
            f"UniProt JSON for {accession} has no sequence object"
        )
    sequence = _validate_plain_sequence(
        str(sequence_record.get("value") or ""), f"JSON {accession}"
    )
    declared_length = sequence_record.get("length")
    if declared_length is not None:
        try:
            length = int(declared_length)
        except (TypeError, ValueError) as exc:
            raise MCSAReferenceSequenceError(
                f"UniProt JSON sequence length for {accession} is invalid"
            ) from exc
        if length != len(sequence):
            raise MCSAReferenceSequenceError(
                f"UniProt JSON sequence length mismatch for {accession}: "
                f"declared {length}, observed {len(sequence)}"
            )
    organism_record = record.get("organism")
    organism = ""
    if isinstance(organism_record, Mapping):
        organism = str(organism_record.get("scientificName") or "").strip()
    audit = record.get("entryAudit")
    audit_record = audit if isinstance(audit, Mapping) else {}
    return {
        "sequence": sequence,
        "entry_type": str(record.get("entryType") or "").strip(),
        "uniprotkb_id": str(record.get("uniProtkbId") or "").strip(),
        "organism": organism,
        "sequence_version": str(audit_record.get("sequenceVersion") or "").strip(),
        "last_sequence_update_date": str(
            audit_record.get("lastSequenceUpdateDate") or ""
        ).strip(),
    }


def _parse_flat_file(data: bytes, accession: str) -> tuple[str, list[str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MCSAReferenceSequenceError(
            f"UniProt flat file for {accession} is not UTF-8"
        ) from exc

    accessions: list[str] = []
    sequence_lines: list[str] = []
    in_sequence = False
    terminated = False
    for line in text.splitlines():
        if line.startswith("AC   "):
            accessions.extend(
                value.strip().upper() for value in line[5:].split(";") if value.strip()
            )
        elif line.startswith("SQ   "):
            in_sequence = True
        elif in_sequence and line.startswith("//"):
            terminated = True
            break
        elif in_sequence:
            sequence_lines.append(
                "".join(character for character in line if character.isalpha())
            )

    if not accessions or accessions[0] != accession.upper():
        raise MCSAReferenceSequenceError(
            "UniProt flat-file primary accession mismatch for "
            f"{accession}: {accessions}"
        )
    if not terminated:
        raise MCSAReferenceSequenceError(
            f"UniProt flat file for {accession} has no terminal // record"
        )
    sequence = _validate_plain_sequence(
        "".join(sequence_lines), f"flat file {accession}"
    )
    return sequence, accessions


def _release_metadata(headers: Mapping[str, str]) -> tuple[str, str]:
    normalized = _normalize_headers(headers)
    return (
        normalized.get("x-uniprot-release", "").strip(),
        normalized.get("x-uniprot-release-date", "").strip(),
    )


def _write_bytes_safely(path: Path, data: bytes) -> str:
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise MCSAReferenceSequenceError(
                f"Refusing to replace different UniProt source file: {path}"
            )
        return "existing_verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "downloaded"


def _checksum_text(repo_root: Path, uniprot_root: Path) -> str:
    paths = sorted(
        path
        for path in uniprot_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [
        f"{_sha256(path.read_bytes())}  {path.relative_to(repo_root).as_posix()}"
        for path in paths
    ]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise MCSAReferenceSequenceError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def acquire_mcsa_reference_sequences(
    repo_root: Path,
    *,
    timeout: int = 60,
    fetch_response: FetchResponse = _default_fetch_response,
    retrieved_at: str | None = None,
) -> Mapping[str, object]:
    """Acquire and cross-check exact UniProt records referenced by M-CSA."""
    if timeout <= 0:
        raise MCSAReferenceSequenceError("Timeout must be a positive integer")

    mcsa_path = repo_root / "data/raw/mcsa/entries_15_16.json"
    references = _load_mcsa_reference_accessions(mcsa_path)
    expected_ids = {15, 16}
    if set(references) != expected_ids:
        raise MCSAReferenceSequenceError(
            f"Expected frozen M-CSA entries {sorted(expected_ids)}, "
            f"observed {sorted(references)}"
        )

    timestamp = retrieved_at or _utc_now()
    uniprot_root = repo_root / "data/raw/uniprot"
    output_dir = repo_root / "data/interim/mcsa_reference_sequences"

    fetched: dict[str, dict[str, tuple[bytes, str, Mapping[str, str], str]]] = {}
    reference_rows: list[dict[str, object]] = []

    for mcsa_id in sorted(references):
        accession = references[mcsa_id]
        format_records: dict[str, tuple[bytes, str, Mapping[str, str], str]] = {}
        for fmt in _SUPPORTED_FORMATS:
            url = _uniprot_url(accession, fmt)
            data, content_type, headers = fetch_response(url, timeout)
            if not data:
                raise MCSAReferenceSequenceError(
                    f"Empty UniProt {fmt} response for {accession}"
                )
            head = data[:4096].decode("utf-8", errors="replace").lower()
            if "<html" in head or "<!doctype html" in head:
                raise MCSAReferenceSequenceError(
                    f"HTML returned instead of UniProt {fmt} data for {accession}"
                )
            format_records[fmt] = (data, content_type, headers, url)

        fasta_sequence, _ = _parse_fasta(format_records["fasta"][0], accession)
        json_record = _parse_json(format_records["json"][0], accession)
        flat_sequence, _ = _parse_flat_file(format_records["txt"][0], accession)
        json_sequence = str(json_record["sequence"])

        if not (fasta_sequence == json_sequence == flat_sequence):
            raise MCSAReferenceSequenceError(
                f"UniProt sequence representations disagree for {accession}"
            )

        release_pairs = {
            _release_metadata(format_records[fmt][2]) for fmt in _SUPPORTED_FORMATS
        }
        nonempty_release_pairs = {pair for pair in release_pairs if pair != ("", "")}
        if len(nonempty_release_pairs) > 1:
            raise MCSAReferenceSequenceError(
                f"UniProt release headers disagree across formats for {accession}: "
                f"{sorted(nonempty_release_pairs)}"
            )
        release, release_date = next(iter(nonempty_release_pairs), ("", ""))

        fetched[accession] = format_records
        reference_rows.append(
            {
                "mcsa_id": mcsa_id,
                "accession": accession,
                "uniprotkb_id": json_record["uniprotkb_id"],
                "entry_type": json_record["entry_type"],
                "organism": json_record["organism"],
                "sequence_length": len(json_sequence),
                "sequence_sha256": _sha256(json_sequence.encode("ascii")),
                "sequence_version": json_record["sequence_version"],
                "last_sequence_update_date": json_record["last_sequence_update_date"],
                "uniprot_release": release,
                "uniprot_release_date": release_date,
                "sequence_representation_crosscheck": "PASS",
            }
        )

    release_pairs = {
        (str(row["uniprot_release"]), str(row["uniprot_release_date"]))
        for row in reference_rows
        if row["uniprot_release"] or row["uniprot_release_date"]
    }
    if len(release_pairs) > 1:
        raise MCSAReferenceSequenceError(
            "UniProt release headers disagree between M-CSA reference accessions: "
            f"{sorted(release_pairs)}"
        )

    downloaded_rows: list[dict[str, object]] = []
    for row in reference_rows:
        accession = str(row["accession"])
        accession_dir = uniprot_root / accession
        for fmt in _SUPPORTED_FORMATS:
            data, content_type, headers, url = fetched[accession][fmt]
            path = accession_dir / f"{accession}.{fmt}"
            status = _write_bytes_safely(path, data)
            release, release_date = _release_metadata(headers)
            downloaded_rows.append(
                {
                    "mcsa_id": row["mcsa_id"],
                    "accession": accession,
                    "format": fmt,
                    "local_path": path.relative_to(repo_root).as_posix(),
                    "source_url": url,
                    "content_type": content_type,
                    "sha256": _sha256(path.read_bytes()),
                    "size_bytes": path.stat().st_size,
                    "uniprot_release": release,
                    "uniprot_release_date": release_date,
                    "retrieved_at_utc": timestamp,
                    "status": status,
                }
            )

    checksum_path = uniprot_root / "SHA256SUMS.txt"
    checksum_path.write_text(
        _checksum_text(repo_root, uniprot_root),
        encoding="utf-8",
    )

    _write_csv(output_dir / "reference_sequences.csv", reference_rows)
    _write_csv(output_dir / "downloaded_files.csv", downloaded_rows)

    summary: Mapping[str, object] = {
        "status": "PASS",
        "mcsa_reference_accessions": {
            str(mcsa_id): references[mcsa_id] for mcsa_id in sorted(references)
        },
        "accession_count": len(reference_rows),
        "downloaded_source_file_count": len(downloaded_rows),
        "sequence_representation_crosscheck": "PASS",
        "uniprot_release_consistency": "PASS",
        "output_directory": output_dir.relative_to(repo_root).as_posix(),
        "outputs": [
            "reference_sequences.csv",
            "downloaded_files.csv",
            "acquisition_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acquisition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
