from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import betalactamase_transfer.sequence_plm_evaluation as plm_eval
from betalactamase_transfer.sequence_plm_evaluation import (
    EMBEDDING_HIDDEN_STATE_INDICES,
    ESMC_RIDGE_ALPHAS,
    SequencePLMEvaluationError,
    _hidden_state_array_index,
    _metric_rows,
    _select_model_setting,
    _site_delta,
    _validate_inputs,
)


def _embedding_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "protein_id": ["NDM-1", "VIM-2"],
            "assay_mutation": ["M1A", "M1A"],
            "assay_position": [1, 1],
            "wild_type": ["M", "M"],
            "mutant": ["A", "A"],
            "mutant_sequence_sha256": ["a", "b"],
        }
    )


def test_validate_inputs_requires_contiguous_embedding_rows() -> None:
    index = _embedding_index()
    index["embedding_row"] = [1, 0]

    observations = pd.DataFrame({"score_analysis": [0.0]})
    membership = pd.DataFrame({"observation_id": ["obs_1"]})

    with pytest.raises(SequencePLMEvaluationError, match="contiguous range"):
        _validate_inputs(
            observations,
            membership,
            index,
            np.empty((0,)),
            np.empty((0,)),
        )


def test_validate_inputs_rejects_unexpected_embedding_protein() -> None:
    index = _embedding_index()
    index.loc[1, "protein_id"] = "OTHER"

    observations = pd.DataFrame({"score_analysis": [0.0]})
    membership = pd.DataFrame({"observation_id": ["obs_1"]})

    with pytest.raises(SequencePLMEvaluationError, match="Unexpected embedding-index"):
        _validate_inputs(
            observations,
            membership,
            index,
            np.empty((0,)),
            np.empty((0,)),
        )


def test_hidden_state_array_index_matches_extraction_order() -> None:
    assert EMBEDDING_HIDDEN_STATE_INDICES == (9, 18, 27, 36)
    assert [
        _hidden_state_array_index(value) for value in EMBEDDING_HIDDEN_STATE_INDICES
    ] == [
        0,
        1,
        2,
        3,
    ]

    with pytest.raises(SequencePLMEvaluationError, match="Unsupported ESMC"):
        _hidden_state_array_index(17)


def test_esmc_regularisation_grid_extends_beyond_previous_boundary() -> None:
    assert ESMC_RIDGE_ALPHAS[-4:] == (1000.0, 10000.0, 100000.0, 1000000.0)
    assert len(ESMC_RIDGE_ALPHAS) == len(set(ESMC_RIDGE_ALPHAS))
    assert all(alpha > 0 for alpha in ESMC_RIDGE_ALPHAS)


def test_site_delta_rejects_unexpected_protein() -> None:
    frame = pd.DataFrame(
        {
            "embedding_row": [0],
            "protein_id": ["OTHER"],
            "assay_position": [1],
        }
    )
    mutant = np.zeros((1, 4, 1152), dtype=np.float16)
    wild_type = np.zeros((2, 4, 270, 1152), dtype=np.float16)

    with pytest.raises(SequencePLMEvaluationError, match="Unexpected protein"):
        _site_delta(
            frame,
            mutant,
            wild_type,
            hidden_state_index=36,
        )


def test_site_delta_uses_requested_hidden_state_and_vim2_residue_identity() -> None:
    frame = pd.DataFrame(
        {
            "embedding_row": [0],
            "protein_id": ["VIM-2"],
            "assay_position": [267],
        }
    )
    mutant = np.zeros((1, 4, 1152), dtype=np.float16)
    wild_type = np.zeros((2, 4, 270, 1152), dtype=np.float16)

    mutant[0, 0, :] = 7.0
    wild_type[1, 0, 266, :] = 2.0
    mutant[0, 3, :] = 3.0
    wild_type[1, 3, 266, :] = 1.0

    layer_9 = _site_delta(
        frame,
        mutant,
        wild_type,
        hidden_state_index=9,
    )
    layer_36 = _site_delta(
        frame,
        mutant,
        wild_type,
        hidden_state_index=36,
    )

    assert layer_9.shape == (1, 1152)
    assert layer_36.shape == (1, 1152)
    assert np.all(layer_9 == 5.0)
    assert np.all(layer_36 == 2.0)


def test_model_selection_chooses_hidden_state_alpha_pair_from_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame({"score_analysis": [0.0, 0.0]})
    validation = pd.DataFrame({"score_analysis": [0.0, 0.0]})

    def fake_fit_transform(
        train_frame: pd.DataFrame,
        mutant_embeddings: np.ndarray,
        wild_type_embeddings: np.ndarray,
        *,
        condition_mode: str,
        hidden_state_index: int,
    ) -> object:
        del train_frame, mutant_embeddings, wild_type_embeddings, condition_mode
        return type("Transform", (), {"hidden_state_index": hidden_state_index})()

    def fake_ridge_statistics(
        train_frame: pd.DataFrame,
        transform: object,
        mutant_embeddings: np.ndarray,
        wild_type_embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del train_frame, mutant_embeddings, wild_type_embeddings
        hidden_state_index = int(transform.hidden_state_index)
        return np.asarray([hidden_state_index], dtype=float), np.asarray([0.0])

    def fake_ridge_coefficients(
        gram: np.ndarray,
        target_product: np.ndarray,
        *,
        alpha: float,
    ) -> np.ndarray:
        del target_product
        return np.asarray([gram[0], alpha], dtype=float)

    def fake_predict(
        frame: pd.DataFrame,
        transform: object,
        coefficients: np.ndarray,
        mutant_embeddings: np.ndarray,
        wild_type_embeddings: np.ndarray,
    ) -> np.ndarray:
        del transform, mutant_embeddings, wild_type_embeddings
        hidden_state_index = int(coefficients[0])
        alpha = float(coefficients[1])
        error = abs(hidden_state_index - 27) + abs(np.log10(alpha) - 2.0)
        return np.full(len(frame), error, dtype=float)

    monkeypatch.setattr(plm_eval, "_fit_transform", fake_fit_transform)
    monkeypatch.setattr(plm_eval, "_ridge_statistics", fake_ridge_statistics)
    monkeypatch.setattr(plm_eval, "_ridge_coefficients", fake_ridge_coefficients)
    monkeypatch.setattr(plm_eval, "_predict", fake_predict)

    hidden_state_index, alpha, validation_mae, detail = _select_model_setting(
        [(train, validation)],
        np.empty((0,)),
        np.empty((0,)),
        condition_mode="none",
    )

    assert hidden_state_index == 27
    assert alpha == 100.0
    assert validation_mae == 0.0
    assert len(detail) == len(EMBEDDING_HIDDEN_STATE_INDICES) * len(ESMC_RIDGE_ALPHAS)


def test_metric_rows_marks_pooled_mixed_hidden_state_selection() -> None:
    predictions = pd.DataFrame(
        {
            "scheme_id": ["scheme", "scheme", "scheme", "scheme"],
            "model_id": ["model", "model", "model", "model"],
            "hidden_state_index": [9, 9, 36, 36],
            "evaluation_iteration": [0, 0, 1, 1],
            "evaluation_group_id": ["a", "b", "c", "d"],
            "y_true": [0.0, 1.0, 0.0, 1.0],
            "y_pred": [0.1, 0.9, 0.2, 0.8],
        }
    )

    metrics = _metric_rows(predictions)

    by_iteration = metrics.set_index("evaluation_iteration")
    assert by_iteration.loc[0, "hidden_state_index"] == 9
    assert by_iteration.loc[1, "hidden_state_index"] == 36
    assert by_iteration.loc["pooled", "hidden_state_index"] == "mixed"
