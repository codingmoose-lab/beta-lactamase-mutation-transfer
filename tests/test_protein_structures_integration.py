import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_protein_structures() -> None:
    output = REPO_ROOT / "data/interim/protein_structures"
    summary = json.loads(
        (output / "structure_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    expected = {
        "chain_coverage.csv",
        "measured_position_coverage.csv",
        "structure_metadata.csv",
        "chain_metadata.csv",
        "residue_coordinate_status.csv",
        "metal_sites.csv",
        "metal_contacts.csv",
        "ligand_sites.csv",
        "local_environment_completeness.csv",
        "selected_structures.json",
    }
    assert expected.issubset(set(summary["outputs"]))
    for filename in expected:
        assert (output / filename).is_file()

    selected = summary["selected_structures"]
    assert selected["NDM-1"]["primary_chain"] == "B"
    assert selected["VIM-2"]["primary_chain"] == "B"

    chains = pd.read_csv(output / "chain_metadata.csv")
    assert {"3SPU", "5YD7"}.issubset(set(chains["pdb_id"]))
    coverage = pd.read_csv(output / "chain_coverage.csv")
    vim_b = coverage[
        (coverage["protein_id"] == "VIM-2")
        & (coverage["pdb_id"] == "5YD7")
        & (coverage["pdb_chain"] == "B")
    ].iloc[0]
    assert int(vim_b["unique_missense_mutations"]) == 5045
    assert int(vim_b["modeled_unique_mutations"]) == 4364
    assert int(vim_b["modeled_unique_positions"]) == 231
    assert int(vim_b["observation_count"]) == 80312
    assert int(vim_b["modeled_observation_count"]) == 69464
    assert abs(float(vim_b["modeled_observation_fraction"]) - 0.864926785536408) < 1e-12

    structures = pd.read_csv(output / "structure_metadata.csv")
    assert (
        structures["coordinate_file"]
        .map(lambda value: not Path(value).is_absolute())
        .all()
    )

    metals = pd.read_csv(output / "metal_sites.csv")
    assert (metals["site_classification"] == "catalytic_site_candidate").any()
