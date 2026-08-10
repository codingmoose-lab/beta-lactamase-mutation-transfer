import numpy as np
import pytest

from betalactamase_transfer.alphafold_preparation import (
    AlphaFoldPreparationError,
    kabsch_superpose,
    parse_confidence_document,
    parse_pae_document,
)


def test_parse_confidence_document_accepts_current_object_schema() -> None:
    rows = parse_confidence_document(
        {
            "residueNumber": [1, 2, 3],
            "confidenceScore": [98.1, 83.4, 61.2],
            "confidenceCategory": ["H", "M", "L"],
        },
        expected_length=3,
    )
    assert [row["alphafold_position"] for row in rows] == [1, 2, 3]
    assert [row["plddt"] for row in rows] == [98.1, 83.4, 61.2]


def test_parse_confidence_document_rejects_noncontiguous_numbering() -> None:
    with pytest.raises(AlphaFoldPreparationError, match="exactly 1..sequence_length"):
        parse_confidence_document(
            {
                "residueNumber": [1, 3],
                "confidenceScore": [90.0, 90.0],
                "confidenceCategory": ["Very high", "Very high"],
            },
            expected_length=2,
        )


def test_parse_confidence_document_rejects_category_score_disagreement() -> None:
    with pytest.raises(AlphaFoldPreparationError, match="inconsistent with pLDDT"):
        parse_confidence_document(
            {
                "residueNumber": [1],
                "confidenceScore": [95.0],
                "confidenceCategory": ["L"],
            },
            expected_length=1,
        )


def test_parse_pae_document_validates_matrix_dimensions() -> None:
    matrix, declared_maximum = parse_pae_document(
        [
            {
                "predicted_aligned_error": [[0.0, 2.0], [3.0, 0.0]],
                "max_predicted_aligned_error": 31.75,
            }
        ],
        expected_length=2,
    )
    assert matrix.shape == (2, 2)
    assert declared_maximum == 31.75


def test_parse_pae_document_rejects_wrong_matrix_dimensions() -> None:
    with pytest.raises(AlphaFoldPreparationError, match="does not match"):
        parse_pae_document(
            [
                {
                    "predicted_aligned_error": [[0.0, 2.0]],
                    "max_predicted_aligned_error": 31.75,
                }
            ],
            expected_length=2,
        )


def test_kabsch_superpose_recovers_rigid_transform() -> None:
    mobile = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = mobile @ rotation + np.array([4.0, -2.0, 7.0])
    transformed, fitted_rotation, _, rmsd = kabsch_superpose(mobile, target)
    assert np.allclose(transformed, target, atol=1e-10)
    assert np.linalg.det(fitted_rotation) == pytest.approx(1.0)
    assert rmsd == pytest.approx(0.0, abs=1e-10)
