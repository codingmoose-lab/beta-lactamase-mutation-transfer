"""Check source-position fitting and paired ESM-C comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalactamase_transfer.source_position_withheld_esmc import (
    PositionWithheldESMCError,
    add_embedding_rows,
    choose_alpha,
    fit_esmc,
    summarize_positions,
)


def test_source_cv_penalty_selection_and_prediction_features() -> None:
    rng = np.random.default_rng(77)
    positions = np.repeat(np.arange(18), 3)
    x = rng.normal(size=(54, 6))
    y = 1.5 * x[:, 0] - 0.3 * x[:, 2]
    alpha, records = choose_alpha(x, y, positions, (1.0, 10.0))
    assert alpha in (1.0, 10.0)
    assert len(records) == 2
    test = x[:3].copy()
    original = fit_esmc(x, y, test, alpha)
    changed = fit_esmc(x, y, test + 100, alpha)
    assert not np.allclose(original, changed)


def test_missing_mutation_and_duplicates_rejected() -> None:
    obs = pd.DataFrame(
        [{"protein_id": "NDM-1", "assay_position": 10, "wild_type": "A", "mutant": "C"}]
    )
    index = obs.assign(embedding_row=0)
    assert int(add_embedding_rows(obs, index).embedding_row.iloc[0]) == 0
    with pytest.raises(PositionWithheldESMCError):
        add_embedding_rows(obs, index.iloc[:0])
    with pytest.raises(PositionWithheldESMCError):
        add_embedding_rows(obs, pd.concat([index, index], ignore_index=True))


def test_position_bootstrap_and_directional_mae() -> None:
    records = []
    for direction in ["NDM-1_to_VIM-2", "VIM-2_to_NDM-1"]:
        for position in range(4):
            for j in range(3):
                records.append(
                    dict(
                        direction=direction,
                        homologue_position_id=str(position),
                        minimum_score_class="both_above_minimum",
                        target_effect=float(j),
                        model_prediction=float(j) + 0.1,
                        reference_prediction=float(j) + 0.4,
                    )
                )
    result = summarize_positions(pd.DataFrame(records), 150)
    assert len(result) == 4
    assert (result.mae_improvement_model > 0).all()
    assert np.allclose(result.mae_improvement_model, 0.3)


def test_full_source_only_direction_execution(tmp_path, monkeypatch) -> None:
    """Exercise both directions with target labels absent from fitted features."""
    import json

    from betalactamase_transfer import source_position_withheld_esmc as module

    roots = tmp_path / "data/interim/baseline_evaluation/homologue_analysis"
    roots.mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    directory = tmp_path / "data/interim/sequence_plm_embeddings"
    directory.mkdir(parents=True)
    mutants = list("CDEFGHIKLMNPQRSTVWY")
    dms_rows, paired, indices = [], [], []
    emb = []
    for protein, drugconc in [("NDM-1", 256.0), ("VIM-2", 128.0)]:
        for pos in range(1, 82):
            for aa in mutants:
                if (pos == 81 and mutants.index(aa) >= 1) or (
                    pos == 80 and mutants.index(aa) >= 11
                ):
                    continue
                oid = f"{protein}_{pos}_{aa}"
                val = (
                    -2
                    + pos / 100
                    + mutants.index(aa) / 20
                    + (protein == "VIM-2") * 0.25
                )
                dms_rows.append(
                    dict(
                        observation_id=oid,
                        assay_id=protein,
                        protein_id=protein,
                        assay_position=pos,
                        wild_type="A",
                        mutant=aa,
                        score_analysis=val,
                        antibiotic="AMP",
                        concentration=drugconc,
                        temperature_c=37,
                    )
                )
                indices.append(
                    dict(
                        protein_id=protein,
                        assay_position=pos,
                        wild_type="A",
                        mutant=aa,
                        embedding_row=len(indices),
                        assay_mutation=f"A{pos}{aa}",
                    )
                )
                base = float(mutants.index(aa)) / 10 + pos / 300
                emb.append(np.full((4, 8), base + float(protein == "VIM-2") * 0.25))
    dms = pd.DataFrame(dms_rows)
    for pos in range(1, 82):
        subset = mutants[:1] if pos == 81 else mutants[:11] if pos == 80 else mutants
        for aa in subset:
            ndm = dms.loc[dms.observation_id.eq(f"NDM-1_{pos}_{aa}")].iloc[0]
            vim = dms.loc[dms.observation_id.eq(f"VIM-2_{pos}_{aa}")].iloc[0]
            for src, tgt in [(ndm, vim), (vim, ndm)]:
                paired.append(
                    dict(
                        direction=(
                            "NDM-1_to_VIM-2"
                            if src.protein_id == "NDM-1"
                            else "VIM-2_to_NDM-1"
                        ),
                        source_observation_id=src.observation_id,
                        target_observation_id=tgt.observation_id,
                        source_effect=src.score_analysis,
                        target_effect=tgt.score_analysis,
                        wild_type="A",
                        mutant=aa,
                        minimum_score_class="both_above_minimum",
                        homologue_position_id=f"h{pos}",
                    )
                )
    assert len(paired) == 3026
    dms.to_csv(tmp_path / "data/processed/missense_observations.csv", index=False)
    pd.DataFrame(paired).to_csv(
        roots / "matched_homologue_predictions.csv", index=False
    )
    pd.DataFrame(indices).to_csv(
        directory / "mutation_embedding_index.csv", index=False
    )
    np.save(directory / "mutant_site_embeddings.npy", np.asarray(emb, dtype=np.float16))
    np.save(
        directory / "wild_type_residue_embeddings.npy",
        np.zeros((2, 4, 270, 8), dtype=np.float16),
    )
    for name in ("embedding_summary.json", "extraction_summary.json"):
        (directory / name).write_text(json.dumps({"synthetic": True}))
    monkeypatch.setattr(module, "_validate_embedding_provenance", lambda *args: None)
    result = module.run(
        tmp_path, tmp_path.parent / "source_position_esmc_test_output", 100
    )
    assert result["prediction_count"] == 3026
    comparisons = (
        tmp_path.parent
        / "source_position_esmc_test_output"
        / "source_position_withheld_esmc_comparisons.csv"
    )
    assert len(pd.read_csv(comparisons)) == 4
    reference_cv = (
        tmp_path.parent
        / "source_position_esmc_test_output"
        / "source_position_withheld_esmc_reference_source_cv.csv"
    )
    assert len(pd.read_csv(reference_cv)) == 40


def test_float16_feature_statistics_use_float64() -> None:
    import warnings

    x = np.asarray(
        [[-10000.0, 2000.0], [10000.0, -2000.0]] * 12,
        dtype=np.float16,
    )
    y = np.asarray([-2.0, 2.0] * 12)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predictions = fit_esmc(x, y, x, alpha=1.0)
    assert np.isfinite(predictions).all()
    assert predictions[1] > predictions[0]
    assert np.max(np.abs(predictions - y)) < 0.2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_esmc_rejects_nonfinite_features_and_labels(bad: float) -> None:
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)
    bad_source = x.copy()
    bad_source[0, 0] = bad
    with pytest.raises(PositionWithheldESMCError, match="fitting inputs"):
        fit_esmc(bad_source, y, x, 1.0)
    bad_labels = y.copy()
    bad_labels[0] = bad
    with pytest.raises(PositionWithheldESMCError, match="fitting inputs"):
        fit_esmc(x, bad_labels, x, 1.0)


def test_esmc_rejects_nonfinite_solver_coefficients(monkeypatch) -> None:
    from betalactamase_transfer import source_position_withheld_esmc as module

    monkeypatch.setattr(
        module,
        "lsqr",
        lambda *args, **kwargs: (np.full(args[0].shape[1], np.nan), 1),
    )
    with pytest.raises(PositionWithheldESMCError, match="coefficients or predictions"):
        fit_esmc(np.eye(5), np.arange(5.0), np.eye(5), 1.0)


def test_esmc_cv_does_not_select_invalid_candidate(monkeypatch) -> None:
    from betalactamase_transfer import source_position_withheld_esmc as module

    monkeypatch.setattr(
        module,
        "fit_esmc",
        lambda train, labels, test, alpha: np.full(len(test), np.nan),
    )
    with pytest.raises(PositionWithheldESMCError, match="source CV"):
        choose_alpha(np.ones((54, 3)), np.ones(54), np.repeat(np.arange(18), 3))
