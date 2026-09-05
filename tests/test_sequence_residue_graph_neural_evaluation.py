from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

module = importlib.import_module(
    "betalactamase_transfer.sequence_residue_graph_neural_evaluation"
)


def test_embedding_transform_replaces_zero_sd() -> None:
    values = np.ones((3, module.ESMC_INPUT_DIMENSION), dtype=np.float32)
    mean, sd = module._fit_embedding_transform(values)
    assert mean.shape == (module.ESMC_INPUT_DIMENSION,)
    assert sd.shape == (module.ESMC_INPUT_DIMENSION,)
    assert np.all(sd == 1.0)


def test_attach_embedding_rows_requires_complete_mapping() -> None:
    frame = pd.DataFrame(
        {
            "protein_id": ["NDM-1"],
            "assay_position": [10],
            "wild_type": ["A"],
            "mutant": ["V"],
            "observation_id": ["obs-1"],
        }
    )
    index = pd.DataFrame(
        {
            "protein_id": ["NDM-1"],
            "assay_position": [11],
            "wild_type": ["A"],
            "mutant": ["V"],
            "embedding_row": [0],
        }
    )
    with pytest.raises(module.SequenceResidueGraphNeuralEvaluationError):
        module._attach_embedding_rows(frame, index)


def test_metric_difference_uses_positive_as_model_a_better() -> None:
    a = {"spearman": 0.7, "pearson": 0.6, "mae": 1.0, "rmse": 1.2}
    b = {"spearman": 0.6, "pearson": 0.5, "mae": 1.1, "rmse": 1.3}
    assert module._metric_difference("spearman", a, b) == pytest.approx(0.1)
    assert module._metric_difference("pearson", a, b) == pytest.approx(0.1)
    assert module._metric_difference("mae", a, b) == pytest.approx(0.1)
    assert module._metric_difference("rmse", a, b) == pytest.approx(0.1)


def test_model_uses_sequence_projection_and_gine() -> None:
    model = module.SequenceResidueGraphRegressor(30, 5, 2)
    assert model.sequence_projection[0].in_features == module.ESMC_INPUT_DIMENSION
    assert model.sequence_projection[0].out_features == module.ESMC_PROJECTION_WIDTH
    assert len(model.convolutions) == module.graph_stage.GRAPH_LAYER_COUNT


def test_validate_prediction_population_rejects_context_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fusion = pd.DataFrame(
        {
            "scheme_id": ["scheme-a"],
            "evaluation_iteration": ["0"],
            "evaluation_role": ["outer_test"],
            "observation_id": ["obs-1"],
            "y_true": [1.0],
        }
    )
    graph_reference = pd.DataFrame(
        {
            "scheme_id": ["scheme-a", "scheme-a"],
            "evaluation_iteration": ["1", "1"],
            "evaluation_role": ["outer_test", "outer_test"],
            "observation_id": ["obs-1", "obs-1"],
            "y_true": [1.0, 1.0],
            "model_id": [module.PRIMARY_COMPARATOR, module.SECONDARY_REFERENCE],
        }
    )

    monkeypatch.setattr(module, "EXPECTED_PREDICTION_COUNT", 1)
    with pytest.raises(module.SequenceResidueGraphNeuralEvaluationError):
        module._validate_prediction_population(fusion, graph_reference)


def test_fitting_rows_for_context_rejects_duplicate_observations() -> None:
    train = pd.DataFrame({"observation_id": ["obs-1"]})
    validation = pd.DataFrame({"observation_id": ["obs-1"]})
    context = {"train": train, "validation": validation}

    with pytest.raises(module.SequenceResidueGraphNeuralEvaluationError):
        module._fitting_rows_for_context(context)


def test_fitting_rows_for_context_preserves_nested_population() -> None:
    train = pd.DataFrame({"observation_id": ["obs-1", "obs-2"]})
    validation = pd.DataFrame({"observation_id": ["obs-3"]})
    context = {"train": train, "validation": validation}

    fitting = module._fitting_rows_for_context(context)

    assert fitting["observation_id"].tolist() == ["obs-1", "obs-2", "obs-3"]
