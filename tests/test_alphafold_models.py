import json
from pathlib import Path

import pytest

from betalactamase_transfer.alphafold_models import (
    AlphaFoldAcquisitionError,
    acquire_alphafold_models,
    select_exact_prediction,
)


def _record(accession: str, model_id: str, *, pae: bool = True) -> dict[str, object]:
    record: dict[str, object] = {
        "modelEntityId": model_id,
        "uniprotAccession": accession,
        "sequence": "ACDEFG",
        "latestVersion": 6,
        "modelCifUrl": f"https://alphafold.ebi.ac.uk/files/{model_id}.cif",
        "isUniProtReviewed": True,
    }
    record["plddtDocUrl"] = (
        f"https://alphafold.ebi.ac.uk/files/{model_id}-confidence.json"
    )
    if pae:
        record["paeDocUrl"] = f"https://alphafold.ebi.ac.uk/files/{model_id}-pae.json"
    return record


def test_select_exact_prediction_prefers_f1_model() -> None:
    records = [
        _record("C7C422", "AF-C7C422-F2"),
        _record("C7C422", "AF-C7C422-F1"),
        _record("Q9K2N0", "AF-Q9K2N0-F1"),
    ]
    selected = select_exact_prediction(records, "C7C422")
    assert selected["modelEntityId"] == "AF-C7C422-F1"


def test_select_exact_prediction_rejects_nonmatching_accession() -> None:
    with pytest.raises(AlphaFoldAcquisitionError, match="No exact AlphaFold record"):
        select_exact_prediction([_record("Q9K2N0", "AF-Q9K2N0-F1")], "C7C422")


def test_acquisition_writes_only_validated_files(tmp_path: Path) -> None:
    accession = "C7C422"
    record = _record(accession, "AF-C7C422-F1")
    responses = {
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}": (
            json.dumps([record]).encode(),
            "application/json",
        ),
        record["modelCifUrl"]: (
            b"data_AF-C7C422-F1\n_atom_site.id 1\n",
            "chemical/x-mmcif",
        ),
        record["plddtDocUrl"]: (
            json.dumps(
                [
                    {
                        "residueNumber": [1],
                        "confidenceScore": [95.0],
                        "confidenceCategory": ["Very high"],
                    }
                ]
            ).encode(),
            "application/json",
        ),
        record["paeDocUrl"]: (
            json.dumps(
                [
                    {
                        "predicted_aligned_error": [[0.0]],
                        "max_predicted_aligned_error": 31.75,
                    }
                ]
            ).encode(),
            "application/json",
        ),
    }

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        assert timeout == 30
        return responses[url]

    summary = acquire_alphafold_models(
        tmp_path,
        [accession],
        timeout=30,
        fetch_bytes=fake_fetch,
        retrieved_at="2026-08-06T00:00:00+00:00",
    )

    assert summary["model_count"] == 1
    raw_dir = tmp_path / "data/raw/alphafold/C7C422"
    assert (raw_dir / "metadata.json").is_file()
    assert (raw_dir / "model.cif").is_file()
    assert (raw_dir / "confidence.json").is_file()
    assert (raw_dir / "pae.json").is_file()
    assert (raw_dir / "SHA256SUMS.txt").is_file()
    assert (
        tmp_path / "data/interim/alphafold_acquisition/downloaded_files.csv"
    ).is_file()


def test_acquisition_refuses_conflicting_existing_model(tmp_path: Path) -> None:
    accession = "C7C422"
    record = _record(accession, "AF-C7C422-F1", pae=False)
    model_path = tmp_path / "data/raw/alphafold/C7C422/model.cif"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("data_old\n_atom_site.id 1\n", encoding="utf-8")

    responses = {
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}": (
            json.dumps([record]).encode(),
            "application/json",
        ),
        record["modelCifUrl"]: (
            b"data_new\n_atom_site.id 1\n",
            "chemical/x-mmcif",
        ),
    }

    def fake_fetch(url: str, timeout: int) -> tuple[bytes, str]:
        return responses[url]

    with pytest.raises(AlphaFoldAcquisitionError, match="Refusing to replace"):
        acquire_alphafold_models(
            tmp_path,
            [accession],
            fetch_bytes=fake_fetch,
            retrieved_at="2026-08-06T00:00:00+00:00",
        )
