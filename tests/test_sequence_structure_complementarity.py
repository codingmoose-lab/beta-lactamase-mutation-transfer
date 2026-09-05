from __future__ import annotations

import numpy as np
import pandas as pd

from betalactamase_transfer import sequence_structure_complementarity as module


def test_select_weight_uses_validation_mae() -> None:
    rows = pd.DataFrame(
        {
            "y_true": [0.0, 0.0, 1.0],
            "sequence_prediction": [0.0, 0.0, 0.0],
            "graph_prediction": [0.0, 0.0, 2.0],
        }
    )
    table, weight = module._select_weight(rows, "example")
    assert weight == 0.5
    assert table.loc[table["selected"], "graph_weight"].tolist() == [0.5]


def test_select_weight_exact_tie_prefers_graph_reference() -> None:
    rows = pd.DataFrame(
        {
            "y_true": [1.0, 1.0],
            "sequence_prediction": [1.0, 1.0],
            "graph_prediction": [1.0, 1.0],
        }
    )
    _table, weight = module._select_weight(rows, "example")
    assert weight == 1.0


def test_metric_difference_convention_is_positive_for_better_model_a() -> None:
    truth = np.array([0.0, 1.0, 2.0])
    a = module._metric_values(truth, truth)
    b = module._metric_values(truth, np.array([0.0, 0.2, 0.1]))
    assert module.graph_stage._metric_difference("mae", a, b) > 0
    assert module.graph_stage._metric_difference("rmse", a, b) > 0


def test_validation_frame_uses_scheme_grouping_column() -> None:
    rows = pd.DataFrame(
        {
            "observation_id": ["obs-1", "obs-2"],
            "position_group_id": ["pos-1", "pos-2"],
            module.graph_stage.TARGET_COLUMN: [0.5, -0.5],
        }
    )
    setting = pd.Series({"hidden_state_index": 36, "alpha": 10000.0})
    frame = module._validation_frame(
        "ndm1_position_nested5",
        "0",
        "inner_validation",
        rows,
        "position_group_id",
        np.array([0.4, -0.4]),
        np.array([0.6, -0.6]),
        setting,
        20,
    )
    assert frame["evaluation_group_id"].tolist() == ["pos-1", "pos-2"]


def test_validation_frame_rejects_missing_scheme_grouping_column() -> None:
    rows = pd.DataFrame(
        {
            "observation_id": ["obs-1"],
            module.graph_stage.TARGET_COLUMN: [0.5],
        }
    )
    setting = pd.Series({"hidden_state_index": 36, "alpha": 10000.0})
    try:
        module._validation_frame(
            "ndm1_position_nested5",
            "0",
            "inner_validation",
            rows,
            "position_group_id",
            np.array([0.4]),
            np.array([0.6]),
            setting,
            20,
        )
    except module.SequenceStructureComplementarityError as exc:
        assert "position_group_id" in str(exc)
    else:
        raise AssertionError("Missing scheme grouping column was accepted")
