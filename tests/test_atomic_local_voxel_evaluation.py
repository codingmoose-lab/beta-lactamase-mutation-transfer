from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from betalactamase_transfer.atomic_local_voxel_evaluation import (
    VOXEL_CHANNELS,
    VOXEL_DIMENSION,
    AtomicLocalVoxelCNN,
    AtomicLocalVoxelEvaluationError,
    NonOverlappingMaxPool3d,
    _average_repeat_predictions,
    _deposit_trilinear,
    _prepare_checkpoint_directory,
    _residue_local_frame,
    _voxelise_residue_environment,
)


def _atoms() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "element": "N",
                "atom_name": "N",
                "author_position": "10",
                "insertion_code": "",
                "x": -1.2,
                "y": 0.4,
                "z": 0.0,
            },
            {
                "element": "C",
                "atom_name": "CA",
                "author_position": "10",
                "insertion_code": "",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "element": "C",
                "atom_name": "C",
                "author_position": "10",
                "insertion_code": "",
                "x": 1.4,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "element": "O",
                "atom_name": "O",
                "author_position": "10",
                "insertion_code": "",
                "x": 2.0,
                "y": 0.8,
                "z": 0.0,
            },
            {
                "element": "C",
                "atom_name": "CB",
                "author_position": "10",
                "insertion_code": "",
                "x": -0.2,
                "y": -1.1,
                "z": 0.9,
            },
            {
                "element": "S",
                "atom_name": "SG",
                "author_position": "11",
                "insertion_code": "",
                "x": 3.0,
                "y": 2.0,
                "z": 1.0,
            },
        ]
    )


def test_local_frame_is_right_handed_and_orthonormal() -> None:
    atoms = _atoms()
    residue = atoms.loc[atoms["author_position"] == "10"]
    origin, rotation = _residue_local_frame(residue)
    assert np.allclose(origin, np.zeros(3))
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) > 0.999


def test_voxelisation_is_rigid_motion_invariant() -> None:
    atoms = _atoms()
    first = _voxelise_residue_environment(atoms, "10", "")
    theta = 0.73
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = atoms.copy()
    xyz = moved[["x", "y", "z"]].to_numpy(dtype=float)
    xyz = xyz @ rotation.T + np.array([4.0, -2.0, 7.0])
    moved[["x", "y", "z"]] = xyz
    second = _voxelise_residue_environment(moved, "10", "")
    assert np.allclose(first, second, atol=1e-5)


def test_voxel_channels_are_atom_level_and_centre_specific() -> None:
    grid = _voxelise_residue_environment(_atoms(), "10", "")
    assert grid.shape == (
        len(VOXEL_CHANNELS),
        VOXEL_DIMENSION,
        VOXEL_DIMENSION,
        VOXEL_DIMENSION,
    )
    assert grid[0].sum() > 0.0
    assert grid[1].sum() > 0.0
    assert grid[2].sum() > 0.0
    assert grid[3].sum() > 0.0
    assert grid[5].sum() > 0.0


def test_trilinear_deposition_conserves_density_inside_grid() -> None:
    grid = np.zeros(
        (len(VOXEL_CHANNELS), VOXEL_DIMENSION, VOXEL_DIMENSION, VOXEL_DIMENSION),
        dtype=np.float32,
    )
    _deposit_trilinear(grid, 0, np.array([0.1, -0.2, 0.3]))
    assert np.isclose(grid[0].sum(), 1.0, atol=1e-6)


def test_cnn_forward_shape() -> None:
    model = AtomicLocalVoxelCNN(condition_dim=3)
    bank = torch.zeros(
        (4, len(VOXEL_CHANNELS), VOXEL_DIMENSION, VOXEL_DIMENSION, VOXEL_DIMENSION)
    )
    voxel_index = torch.tensor([0, 2], dtype=torch.long)
    substitution = torch.zeros((2, 380))
    condition = torch.zeros((2, 3))
    output = model(bank, voxel_index, substitution, condition)
    assert output.shape == (2,)


def test_non_overlapping_max_pool_matches_native_max_pool() -> None:
    generator = torch.Generator().manual_seed(43091)
    values = torch.randn((2, 4, 20, 20, 20), generator=generator)

    native_values = values.clone().requires_grad_(True)
    custom_values = values.clone().requires_grad_(True)

    native = torch.nn.functional.max_pool3d(
        native_values,
        kernel_size=2,
        stride=2,
    )
    custom = NonOverlappingMaxPool3d()(custom_values)

    assert torch.equal(native, custom)

    native.square().sum().backward()
    custom.square().sum().backward()

    assert torch.equal(native_values.grad, custom_values.grad)


def test_non_overlapping_max_pool_matches_native_max_pool_for_ties() -> None:
    cases = (
        torch.zeros((1, 1, 2, 2, 2)),
        torch.ones((1, 1, 2, 2, 2)),
        torch.tensor([[[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [9.0, 9.0]]]]]),
    )

    for values in cases:
        native_values = values.clone().requires_grad_(True)
        custom_values = values.clone().requires_grad_(True)

        native = torch.nn.functional.max_pool3d(
            native_values,
            kernel_size=2,
            stride=2,
        )
        custom = NonOverlappingMaxPool3d()(custom_values)

        assert torch.equal(native, custom)

        native.sum().backward()
        custom.sum().backward()

        assert torch.equal(native_values.grad, custom_values.grad)


def test_non_overlapping_max_pool_cuda_backward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.manual_seed(43092)
        torch.cuda.manual_seed_all(43092)

        values = torch.randn(
            (2, 4, 20, 20, 20),
            device="cuda",
            requires_grad=True,
        )
        output = NonOverlappingMaxPool3d().cuda()(values)
        loss = output.square().mean()
        loss.backward()
        torch.cuda.synchronize()

        assert output.shape == (2, 4, 10, 10, 10)
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
    finally:
        torch.use_deterministic_algorithms(
            original_enabled,
            warn_only=original_warn_only,
        )


def test_average_repeat_predictions_normalises_evaluation_iteration() -> None:
    base = {
        "scheme_id": "ndm1_position_nested5",
        "evaluation_iteration": 0,
        "evaluation_role": "outer_test",
        "observation_id": "obs_test",
        "task_id": "ndm1_within_position_regression",
        "evaluation_group_id": "position_10",
        "protein_id": "NDM-1",
        "assay_position": 10,
        "wild_type": "A",
        "mutant": "V",
        "antibiotic": "AMP",
        "concentration": 256.0,
        "temperature_c": 37.0,
        "y_true": 0.5,
        "model_id": "atomic_local_voxel_cnn",
    }

    frame = pd.DataFrame(
        [
            {
                **base,
                "training_repeat": 0,
                "y_pred": 0.25,
            },
            {
                **base,
                "training_repeat": 1,
                "y_pred": 0.75,
            },
        ]
    )

    averaged = _average_repeat_predictions(frame)

    assert len(averaged) == 1
    assert averaged.loc[0, "evaluation_iteration"] == "0"
    assert averaged.loc[0, "y_pred"] == pytest.approx(0.5)


def test_postprocess_only_accepts_only_source_hash_difference(tmp_path) -> None:
    output_dir = tmp_path / "atomic"

    original = {
        "model_id": "atomic_local_voxel_cnn",
        "batch_size": 16,
        "source_sha256": "training-source",
    }

    checkpoint_dir = _prepare_checkpoint_directory(
        output_dir,
        original,
    )

    source_only_change = {
        **original,
        "source_sha256": "postprocessing-source",
    }

    reused = _prepare_checkpoint_directory(
        output_dir,
        source_only_change,
        postprocess_only=True,
    )

    assert reused == checkpoint_dir

    incompatible = {
        **source_only_change,
        "batch_size": 32,
    }

    with pytest.raises(AtomicLocalVoxelEvaluationError):
        _prepare_checkpoint_directory(
            output_dir,
            incompatible,
            postprocess_only=True,
        )
