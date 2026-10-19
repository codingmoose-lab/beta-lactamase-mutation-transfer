"""Checks source-condition selection and finite ridge fitting values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.source_full_position_baselines import (
    SourcePositionEvaluationError,
    prepare_direction,
    ridge_fit,
)

AA = "CDEFGHIKLMNPQRSTVWY"


def example_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    paired, observations = [], []
    for protein, prefix, concentration, assay in (
        ("NDM-1", "n", 256.0, "N_assay"),
        ("VIM-2", "v", 128.0, "V_assay"),
    ):
        for position in range(81):
            for mut in AA[:18] + (AA[18] if position < 55 else ""):
                observations.append(
                    {
                        "observation_id": f"{prefix}_{position}_{mut}",
                        "assay_id": assay,
                        "protein_id": protein,
                        "assay_position": position + (300 if prefix == "v" else 10),
                        "wild_type": "A",
                        "mutant": mut,
                        "score_analysis": -1.0 + 0.01 * position,
                        "antibiotic": "AMP",
                        "concentration": concentration,
                        "temperature_c": 37,
                    }
                )
        # Both additional measurements have the same assay ID but differ
        # in antibiotic or concentration, so neither may enter fitting.
        for suffix, drug, conc in [
            ("other_drug", "MEM", 0.25),
            ("other_amp", "AMP", 64.0),
        ]:
            observations.append(
                {
                    "observation_id": f"{prefix}_{suffix}",
                    "assay_id": assay,
                    "protein_id": protein,
                    "assay_position": 900,
                    "wild_type": "A",
                    "mutant": "C",
                    "score_analysis": -3.0,
                    "antibiotic": drug,
                    "concentration": conc,
                    "temperature_c": 37,
                }
            )
    for position in range(81):
        for mut in AA[:18] + (AA[18] if position < 55 else ""):
            for direction, src, target in (
                ("NDM-1_to_VIM-2", "n", "v"),
                ("VIM-2_to_NDM-1", "v", "n"),
            ):
                paired.append(
                    {
                        "direction": direction,
                        "source_observation_id": f"{src}_{position}_{mut}",
                        "target_observation_id": f"{target}_{position}_{mut}",
                        "homologue_position_id": f"pos_{position}",
                        "wild_type": "A",
                        "mutant": mut,
                        "source_effect": -1.0 + 0.01 * position,
                        "target_effect": -1.0 + 0.01 * position,
                        "minimum_score_class": "both_above_minimum",
                    }
                )
    return pd.DataFrame(paired), pd.DataFrame(observations)


@pytest.mark.parametrize(
    "direction",
    [
        "NDM-1_to_VIM-2",
        "VIM-2_to_NDM-1",
    ],
)
def test_source_condition_excludes_other_conditions(direction: str) -> None:
    paired, measurements = example_data()
    _, selected, _ = prepare_direction(paired, measurements, direction)
    expected = 256 if direction.startswith("NDM-1") else 128
    assert len(selected) == 1513
    assert selected.concentration.eq(expected).all()
    assert selected.antibiotic.eq("AMP").all()
    assert selected.temperature_c.eq(37).all()
    assert not selected.observation_id.str.contains("other_").any()


def test_mixed_paired_conditions_are_rejected() -> None:
    paired, measurements = example_data()
    measurements.loc[measurements.observation_id.eq("n_0_C"), "concentration"] = 64.0
    with pytest.raises(SourcePositionEvaluationError, match="multiple concentration"):
        prepare_direction(paired, measurements, "NDM-1_to_VIM-2")


def test_missing_paired_condition_is_rejected() -> None:
    paired, measurements = example_data()
    measurements.loc[measurements.observation_id.eq("v_0_C"), "temperature_c"] = np.nan
    with pytest.raises(SourcePositionEvaluationError, match="missing experimental"):
        prepare_direction(paired, measurements, "VIM-2_to_NDM-1")


def test_ridge_fits_finite_with_no_normal_equation_warnings() -> None:
    training = pd.DataFrame(
        [
            {
                "assay_position": p,
                "wild_type": "A",
                "mutant": a,
                "score_analysis": -0.5 + p / 20 + i / 30,
            }
            for p in range(10)
            for i, a in enumerate(AA)
        ]
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        coefficients = ridge_fit(training, 1.0)
    assert len(coefficients) == 41
    assert np.isfinite(coefficients).all()
