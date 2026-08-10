# Acquire exact AlphaFold DB models for the study reference accessions.

from __future__ import annotations

import csv
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AlphaFoldAcquisitionError(RuntimeError):
    """Raised when AlphaFold metadata or model files cannot be acquired safely."""


API_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
DEFAULT_ACCESSIONS = ("C7C422", "Q9K2N0")
_ALLOWED_DOWNLOAD_HOST_SUFFIXES = ("alphafold.ebi.ac.uk", "ebi.ac.uk")

FetchBytes = Callable[[str, int], tuple[bytes, str]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_fetch_bytes(url: str, timeout: int) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json,chemical/x-mmcif,text/plain,*/*",
            "User-Agent": "beta-lactamase-mutation-transfer/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise AlphaFoldAcquisitionError(
                    f"Unexpected HTTP status {status} for {url}"
                )
            content_type = response.headers.get_content_type()
            return response.read(), content_type
    except urllib.error.HTTPError as exc:
        raise AlphaFoldAcquisitionError(
            f"HTTP {exc.code} while retrieving {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise AlphaFoldAcquisitionError(
            f"Could not retrieve {url}: {exc.reason}"
        ) from exc


def _normalize_records(payload: object) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        records: Sequence[object] = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise AlphaFoldAcquisitionError(
            "AlphaFold prediction API returned neither an object nor a list"
        )

    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise AlphaFoldAcquisitionError(
                f"AlphaFold prediction record {index} is not a JSON object"
            )
        normalized.append(record)
    if not normalized:
        raise AlphaFoldAcquisitionError("AlphaFold prediction API returned no records")
    return normalized


def _first_value(record: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return None


def _record_accession(record: Mapping[str, Any]) -> str:
    value = _first_value(
        record,
        (
            "uniprotAccession",
            "uniprot_accession",
            "accession",
        ),
    )
    return str(value or "").strip().upper()


def _model_identifier(record: Mapping[str, Any]) -> str:
    value = _first_value(
        record,
        (
            "modelEntityId",
            "entryId",
            "model_id",
        ),
    )
    return str(value or "").strip()


def _record_sequence(record: Mapping[str, Any]) -> str:
    value = _first_value(record, ("sequence", "uniprotSequence"))
    return "".join(str(value or "").split()).upper()


def _record_version(record: Mapping[str, Any]) -> str:
    value = _first_value(record, ("latestVersion", "version", "modelVersion"))
    return str(value or "").strip()


def _record_url(record: Mapping[str, Any], names: Sequence[str]) -> str | None:
    value = _first_value(record, names)
    if value is None:
        return None
    url = str(value).strip()
    return url or None


def _record_rank(
    record: Mapping[str, Any], accession: str
) -> tuple[int, int, int, int]:
    model_id = _model_identifier(record).upper()
    sequence = _record_sequence(record)
    reviewed = bool(_first_value(record, ("isUniProtReviewed", "isReviewed")) or False)
    exact_model = model_id == f"AF-{accession}-F1"
    f1_model = model_id.startswith(f"AF-{accession}-F1")
    return (
        int(exact_model),
        int(f1_model),
        int(reviewed),
        len(sequence),
    )


def select_exact_prediction(
    records: Sequence[Mapping[str, Any]], accession: str
) -> Mapping[str, Any]:
    # Select the F1 single-chain prediction for one exact UniProt accession.
    normalized_accession = accession.strip().upper()
    exact = [
        record
        for record in records
        if _record_accession(record) == normalized_accession
    ]
    if not exact:
        returned = sorted({_record_accession(record) for record in records})
        raise AlphaFoldAcquisitionError(
            "No exact AlphaFold record for "
            f"{normalized_accession}; returned: {returned}"
        )

    ranked = sorted(
        exact,
        key=lambda record: _record_rank(record, normalized_accession),
        reverse=True,
    )
    selected = ranked[0]
    model_id = _model_identifier(selected)
    sequence = _record_sequence(selected)
    if not model_id:
        raise AlphaFoldAcquisitionError(
            "Selected AlphaFold record for "
            f"{normalized_accession} has no model identifier"
        )
    if not sequence:
        raise AlphaFoldAcquisitionError(
            f"Selected AlphaFold record for {normalized_accession} has no sequence"
        )
    return selected


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise AlphaFoldAcquisitionError(f"AlphaFold download URL is not HTTPS: {url}")
    hostname = (parsed.hostname or "").lower()
    if not any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in _ALLOWED_DOWNLOAD_HOST_SUFFIXES
    ):
        raise AlphaFoldAcquisitionError(
            f"Unexpected host in AlphaFold download URL: {hostname}"
        )


def _validate_mmcif(data: bytes, content_type: str, url: str) -> None:
    if not data:
        raise AlphaFoldAcquisitionError(f"Empty AlphaFold mmCIF response from {url}")
    head = data[:4096].decode("utf-8", errors="replace").lower()
    if "<html" in head or "<!doctype html" in head:
        raise AlphaFoldAcquisitionError(
            f"HTML was returned instead of mmCIF from {url}"
        )
    if "data_" not in head and "_atom_site." not in head:
        raise AlphaFoldAcquisitionError(
            f"Downloaded file does not look like PDBx/mmCIF: {url} ({content_type})"
        )


def _validate_json(data: bytes, label: str, url: str) -> object:
    if not data:
        raise AlphaFoldAcquisitionError(f"Empty {label} response from {url}")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AlphaFoldAcquisitionError(f"Invalid {label} JSON from {url}") from exc


def _write_bytes_safely(path: Path, data: bytes, overwrite: bool) -> str:
    digest = _sha256(data)
    existed = path.exists()
    if existed:
        existing = path.read_bytes()
        if _sha256(existing) == digest:
            return "already_present"
        if not overwrite:
            raise AlphaFoldAcquisitionError(
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
        raise AlphaFoldAcquisitionError(f"Refusing to write an empty table: {path}")
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


def acquire_alphafold_models(
    repo_root: Path,
    accessions: Sequence[str] = DEFAULT_ACCESSIONS,
    *,
    timeout: int = 60,
    overwrite: bool = False,
    fetch_bytes: FetchBytes = _default_fetch_bytes,
    retrieved_at: str | None = None,
) -> Mapping[str, object]:
    """Acquire exact AlphaFold DB model files and record their provenance."""
    if timeout <= 0:
        raise AlphaFoldAcquisitionError("Timeout must be a positive integer")
    normalized_accessions = tuple(
        dict.fromkeys(accession.strip().upper() for accession in accessions)
    )
    if not normalized_accessions or any(
        not accession for accession in normalized_accessions
    ):
        raise AlphaFoldAcquisitionError("At least one non-empty accession is required")

    timestamp = retrieved_at or _utc_now()
    raw_root = repo_root / "data/raw/alphafold"
    output_dir = repo_root / "data/interim/alphafold_acquisition"
    acquired_rows: list[Mapping[str, object]] = []
    model_rows: list[Mapping[str, object]] = []

    for accession in normalized_accessions:
        api_url = API_URL_TEMPLATE.format(accession=accession)
        metadata_bytes, metadata_content_type = fetch_bytes(api_url, timeout)
        metadata_payload = _validate_json(
            metadata_bytes, "prediction metadata", api_url
        )
        records = _normalize_records(metadata_payload)
        selected = select_exact_prediction(records, accession)

        model_id = _model_identifier(selected)
        sequence = _record_sequence(selected)
        model_version = _record_version(selected)
        cif_url = _record_url(selected, ("modelCifUrl", "cifUrl", "model_cif_url"))
        pae_url = _record_url(selected, ("paeDocUrl", "pae_url", "paeUrl"))
        plddt_url = _record_url(selected, ("plddtDocUrl", "plddt_url", "confidenceUrl"))
        if cif_url is None:
            raise AlphaFoldAcquisitionError(
                f"Selected AlphaFold record for {accession} has no mmCIF URL"
            )
        _validate_download_url(cif_url)
        if pae_url is not None:
            _validate_download_url(pae_url)
        if plddt_url is not None:
            _validate_download_url(plddt_url)

        accession_dir = raw_root / accession
        selected_metadata = json.dumps(selected, indent=2, sort_keys=True) + "\n"
        metadata_path = accession_dir / "metadata.json"
        metadata_status = _write_text_safely(
            metadata_path, selected_metadata, overwrite=overwrite
        )

        cif_bytes, cif_content_type = fetch_bytes(cif_url, timeout)
        _validate_mmcif(cif_bytes, cif_content_type, cif_url)
        cif_path = accession_dir / "model.cif"
        cif_status = _write_bytes_safely(cif_path, cif_bytes, overwrite=overwrite)

        local_files = [metadata_path, cif_path]
        file_records = [
            (
                "metadata",
                metadata_path,
                api_url,
                metadata_content_type,
                metadata_status,
            ),
            ("model_cif", cif_path, cif_url, cif_content_type, cif_status),
        ]

        plddt_available = False
        if plddt_url is not None:
            plddt_bytes, plddt_content_type = fetch_bytes(plddt_url, timeout)
            _validate_json(plddt_bytes, "pLDDT confidence", plddt_url)
            plddt_path = accession_dir / "confidence.json"
            plddt_status = _write_bytes_safely(
                plddt_path, plddt_bytes, overwrite=overwrite
            )
            local_files.append(plddt_path)
            file_records.append(
                (
                    "plddt_confidence",
                    plddt_path,
                    plddt_url,
                    plddt_content_type,
                    plddt_status,
                )
            )
            plddt_available = True

        pae_available = False
        if pae_url is not None:
            pae_bytes, pae_content_type = fetch_bytes(pae_url, timeout)
            _validate_json(pae_bytes, "PAE", pae_url)
            pae_path = accession_dir / "pae.json"
            pae_status = _write_bytes_safely(pae_path, pae_bytes, overwrite=overwrite)
            local_files.append(pae_path)
            file_records.append(
                ("pae", pae_path, pae_url, pae_content_type, pae_status)
            )
            pae_available = True

        checksum_path = accession_dir / "SHA256SUMS.txt"
        checksum_status = _write_text_safely(
            checksum_path,
            _checksum_text(local_files),
            overwrite=overwrite,
        )
        file_records.append(
            (
                "checksums",
                checksum_path,
                "generated_locally",
                "text/plain",
                checksum_status,
            )
        )

        for file_role, path, source_url, content_type, status in file_records:
            acquired_rows.append(
                {
                    "accession": accession,
                    "model_id": model_id,
                    "model_version": model_version,
                    "file_role": file_role,
                    "local_path": path.relative_to(repo_root).as_posix(),
                    "source_url": source_url,
                    "content_type": content_type,
                    "sha256": _sha256(path.read_bytes()),
                    "size_bytes": path.stat().st_size,
                    "retrieved_at_utc": timestamp,
                    "status": status,
                    "licence": "CC-BY-4.0",
                }
            )

        model_rows.append(
            {
                "accession": accession,
                "model_id": model_id,
                "model_version": model_version,
                "sequence_length": len(sequence),
                "sequence_sha256": _sha256(sequence.encode("ascii")),
                "plddt_available": plddt_available,
                "pae_available": pae_available,
                "api_url": api_url,
                "model_cif_url": cif_url,
                "plddt_url": plddt_url or "",
                "pae_url": pae_url or "",
                "retrieved_at_utc": timestamp,
                "licence": "CC-BY-4.0",
            }
        )

    _write_csv(output_dir / "downloaded_files.csv", acquired_rows)
    _write_csv(output_dir / "models.csv", model_rows)
    summary: Mapping[str, object] = {
        "status": "PASS",
        "accessions": list(normalized_accessions),
        "model_count": len(model_rows),
        "downloaded_file_record_count": len(acquired_rows),
        "output_directory": output_dir.relative_to(repo_root).as_posix(),
        "outputs": [
            "downloaded_files.csv",
            "models.csv",
            "acquisition_summary.json",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acquisition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
