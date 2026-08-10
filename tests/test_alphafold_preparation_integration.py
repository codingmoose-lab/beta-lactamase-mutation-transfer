import json
from pathlib import Path

import numpy as np
import pandas as pd

from betalactamase_transfer.alphafold_preparation import prepare_alphafold_models

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_alphafold_models_outputs() -> None:
    observed_summary = prepare_alphafold_models(REPO_ROOT)
    assert observed_summary["status"] == "PASS"

    output = REPO_ROOT / "data/interim/alphafold_preparation"
    summary = json.loads(
        (output / "preparation_summary.json").read_text(encoding="utf-8")
    )
    assert summary == observed_summary
    assert summary["status"] == "PASS"
    assert summary["model_count"] == 2
    assert summary["local_radii_angstrom"] == [6.0, 8.0, 10.0, 12.0]

    models = pd.read_csv(output / "model_metadata.csv")
    by_accession = models.set_index("accession")
    assert int(by_accession.loc["C7C422", "sequence_length"]) == 270
    assert int(by_accession.loc["Q9K2N0", "sequence_length"]) == 266
    assert (
        by_accession["maximum_confidence_mmcif_absolute_difference"].astype(float)
        <= 1e-6
    ).all()
    assert (by_accession["alphafold_full_reference_coverage_fraction"] == 1.0).all()

    mapping = pd.read_csv(output / "assay_alphafold_mapping.csv")
    vim_gly2 = mapping[
        (mapping["protein_id"] == "VIM-2") & (mapping["assay_position"] == 2)
    ].iloc[0]
    assert bool(vim_gly2["assay_specific_insertion"])
    assert pd.isna(vim_gly2["alphafold_position"])

    vim_difference = mapping[
        (mapping["protein_id"] == "VIM-2") & (mapping["assay_position"] == 186)
    ].iloc[0]
    assert int(vim_difference["uniprot_position"]) == 185
    assert vim_difference["assay_residue"] == "I"
    assert vim_difference["alphafold_residue"] == "V"
    assert bool(vim_difference["known_assay_reference_difference"])

    correspondence = pd.read_csv(output / "pdb_alphafold_summary.csv")
    assert set(correspondence["protein_id"]) == {"NDM-1", "VIM-2"}
    assert (correspondence["equivalent_ca_count"] >= 3).all()
    assert np.isfinite(correspondence["ca_rmsd_angstrom"].to_numpy()).all()
    counts = correspondence.set_index("protein_id")["equivalent_ca_count"]
    assert int(counts["NDM-1"]) == 238
    assert int(counts["VIM-2"]) == 231
