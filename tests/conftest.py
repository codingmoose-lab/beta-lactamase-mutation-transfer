from pathlib import Path

import pytest

from betalactamase_transfer.assay_definitions import define_assays_and_conditions
from betalactamase_transfer.mutation_measurements import extract_mutation_measurements
from betalactamase_transfer.protein_structures import prepare_protein_structures
from betalactamase_transfer.sequence_mapping import build_sequence_mapping
from betalactamase_transfer.source_tables import describe_source_tables

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def prepared_data() -> None:
    """Generate intermediate tables once for integration tests."""
    build_sequence_mapping(REPO_ROOT)
    describe_source_tables(REPO_ROOT)
    extract_mutation_measurements(REPO_ROOT)
    define_assays_and_conditions(REPO_ROOT)
    prepare_protein_structures(REPO_ROOT)
