from __future__ import annotations

import pytest

from betalactamase_transfer.secondary_structure import (
    SecondaryStructureError,
    _normalize_insertion_code,
    _reduce_dssp_class,
    _validate_mkdssp_version,
)


@pytest.mark.parametrize("code", ["H", "G", "I"])
def test_reduce_dssp_class_helix(code: str) -> None:
    assert _reduce_dssp_class(code) == "helix"


@pytest.mark.parametrize("code", ["E", "B"])
def test_reduce_dssp_class_strand(code: str) -> None:
    assert _reduce_dssp_class(code) == "strand"


@pytest.mark.parametrize("code", ["T", "S", "P", "-"])
def test_reduce_dssp_class_coil(code: str) -> None:
    assert _reduce_dssp_class(code) == "coil"


def test_reduce_dssp_class_rejects_unknown_code() -> None:
    with pytest.raises(SecondaryStructureError):
        _reduce_dssp_class("X")


def test_normalize_insertion_code_handles_blank_forms() -> None:
    assert _normalize_insertion_code(" ") == ""
    assert _normalize_insertion_code("") == ""
    assert _normalize_insertion_code("A") == "A"


def test_validate_mkdssp_version_accepts_supported_major_version() -> None:
    _validate_mkdssp_version("mkdssp version 4.6.1")


def test_validate_mkdssp_version_rejects_other_major_version() -> None:
    with pytest.raises(SecondaryStructureError, match="requires mkdssp 4.x"):
        _validate_mkdssp_version("mkdssp version 5.0.0")


def test_validate_mkdssp_version_rejects_unparseable_text() -> None:
    with pytest.raises(SecondaryStructureError, match="Could not parse"):
        _validate_mkdssp_version("mkdssp unknown")
