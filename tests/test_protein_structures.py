import pandas as pd

from betalactamase_transfer.protein_structures import (
    _distance,
    _structure_coverage,
)


def test_structure_module_is_importable() -> None:
    assert callable(_structure_coverage)


def test_coordinate_distance() -> None:
    left = {"x": 0.0, "y": 0.0, "z": 0.0}
    right = {"x": 3.0, "y": 4.0, "z": 0.0}
    assert _distance(left, right) == 5.0


def test_structure_coverage_keeps_publication_and_assay_numbering_distinct() -> None:
    observations = pd.DataFrame(
        [
            {
                "protein_id": "VIM-2",
                "mutation": "L96W",
                "assay_mutation": "L97W",
                "assay_position": 97,
                "mutation_class": "missense",
            },
            {
                "protein_id": "VIM-2",
                "mutation": "L96W",
                "assay_mutation": "L97W",
                "assay_position": 97,
                "mutation_class": "missense",
            },
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "protein_id": "VIM-2",
                "mutation": "L96W",
                "assay_position": 97,
                "pdb_id": "5YD7",
                "pdb_chain": "B",
                "mapping_status": "mapped_modeled",
            }
        ]
    )

    rows, _ = _structure_coverage(observations, coverage)

    assert len(rows) == 1
    assert rows[0]["unique_missense_mutations"] == 1
    assert rows[0]["modeled_unique_mutations"] == 1
    assert rows[0]["modeled_unique_positions"] == 1
    assert rows[0]["observation_count"] == 2
    assert rows[0]["modeled_observation_count"] == 2
    assert rows[0]["modeled_observation_fraction"] == 1.0
