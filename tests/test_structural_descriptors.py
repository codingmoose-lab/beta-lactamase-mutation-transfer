from __future__ import annotations

import numpy as np

from betalactamase_transfer.structural_descriptors import (
    _distance,
    _minimum_atom_distance,
)


def test_distance_is_euclidean() -> None:
    left = np.asarray([0.0, 0.0, 0.0])
    right = np.asarray([3.0, 4.0, 0.0])
    assert _distance(left, right) == 5.0


def test_minimum_atom_distance_uses_closest_pair() -> None:
    left = [np.asarray([0.0, 0.0, 0.0]), np.asarray([10.0, 0.0, 0.0])]
    right = [np.asarray([2.0, 0.0, 0.0]), np.asarray([20.0, 0.0, 0.0])]
    assert _minimum_atom_distance(left, right) == 2.0
