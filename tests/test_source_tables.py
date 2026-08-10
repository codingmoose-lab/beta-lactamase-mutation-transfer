import pytest

from betalactamase_transfer.source_tables import (
    SourceTableError,
    column_letters_to_number,
    split_cell_reference,
)


def test_excel_column_conversion() -> None:
    assert column_letters_to_number("A") == 1
    assert column_letters_to_number("Z") == 26
    assert column_letters_to_number("AA") == 27
    assert column_letters_to_number("XFD") == 16384


def test_cell_reference_split() -> None:
    assert split_cell_reference("A1") == (1, 1)
    assert split_cell_reference("W5555") == (5555, 23)


def test_invalid_cell_reference_is_rejected() -> None:
    with pytest.raises(SourceTableError):
        split_cell_reference("A0")


def test_invalid_column_letters_are_rejected() -> None:
    with pytest.raises(SourceTableError):
        column_letters_to_number("A1")
