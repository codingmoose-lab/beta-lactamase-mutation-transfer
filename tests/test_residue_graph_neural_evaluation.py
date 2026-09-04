from __future__ import annotations

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module(
    "betalactamase_transfer.residue_graph_neural_evaluation"
)
AMINO_ACIDS = module.AMINO_ACIDS
CENTRE_MODEL = module.CENTRE_MODEL
GRAPH_MODEL = module.GRAPH_MODEL
SUBSTITUTIONS = module.SUBSTITUTIONS
CentreResidueNeuralRegressor = module.CentreResidueNeuralRegressor
ConditionTransform = module.ConditionTransform
ResidueGraphNeuralRegressor = module.ResidueGraphNeuralRegressor
_fit_continuous_transform = module._fit_continuous_transform


def test_predefined_residue_and_substitution_dimensions() -> None:
    assert len(AMINO_ACIDS) == 20
    assert len(SUBSTITUTIONS) == 380
    assert len(set(SUBSTITUTIONS)) == 380


def test_continuous_transform_handles_constant_column() -> None:
    values = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    transform = _fit_continuous_transform(values)
    transformed = transform.apply(values)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[:, 0], 0.0)


def test_condition_transform_none_has_zero_columns() -> None:
    transform = ConditionTransform("none")
    assert transform.output_dimension == 0


def test_both_neural_models_return_one_value_per_observation() -> None:
    node_count = 6
    node_dim = 30
    edge_dim = 5
    condition_dim = 3
    node_features = torch.randn(node_count, node_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    edge_features = torch.randn(edge_index.shape[1], edge_dim)
    target_node_index = torch.tensor([0, 2, 5], dtype=torch.long)
    substitution = torch.zeros(3, 380)
    substitution[:, 0] = 1.0
    condition = torch.randn(3, condition_dim)

    models = {
        CENTRE_MODEL: CentreResidueNeuralRegressor(node_dim, condition_dim),
        GRAPH_MODEL: ResidueGraphNeuralRegressor(node_dim, edge_dim, condition_dim),
    }
    for model in models.values():
        result = model(
            node_features,
            edge_index,
            edge_features,
            target_node_index,
            substitution,
            condition,
        )
        assert result.shape == (3,)
        assert torch.isfinite(result).all()


def test_deterministic_execution_configuration() -> None:
    module._configure_deterministic_execution()

    assert torch.are_deterministic_algorithms_enabled()
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
