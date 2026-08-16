from __future__ import annotations

from pathlib import Path

from betalactamase_transfer.experimental_structure_evaluation import (
    CORE_SCHEME_IDS,
    EXPERIMENTAL_STRUCTURE_MODEL,
    MUTATION_CONDITION_MODEL,
    prepare_experimental_structure_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_experimental_structure_evaluation_if_available() -> None:
    required = [
        REPO_ROOT / "data/processed/missense_observations.csv",
        REPO_ROOT / "data/processed/residue_features.csv",
        REPO_ROOT / "data/interim/modelling_views/scheme_observation_membership.csv",
        REPO_ROOT / "data/interim/evaluation_partitions/partition_schemes.csv",
        REPO_ROOT / "data/interim/evaluation_partitions/group_assignments.csv",
    ]
    if not all(path.exists() for path in required):
        return

    predictions, metrics, selection = prepare_experimental_structure_evaluation(
        REPO_ROOT
    )

    assert set(predictions["scheme_id"]) == set(CORE_SCHEME_IDS)
    assert set(predictions["model_id"]) == {
        MUTATION_CONDITION_MODEL,
        EXPERIMENTAL_STRUCTURE_MODEL,
    }
    assert predictions["y_true"].notna().all()
    assert predictions["y_pred"].notna().all()

    key = [
        "scheme_id",
        "evaluation_iteration",
        "evaluation_role",
        "observation_id",
        "model_id",
    ]
    assert not predictions.duplicated(key).any()

    counts = predictions.groupby(["scheme_id", "model_id"]).size().unstack()
    assert (
        counts[MUTATION_CONDITION_MODEL] == counts[EXPERIMENTAL_STRUCTURE_MODEL]
    ).all()

    selected = selection[selection["selection_scope"].eq("selected_setting")]
    assert selected["selected"].astype(bool).all()

    pooled = metrics[metrics["evaluation_iteration"].astype(str).eq("pooled")]
    assert len(pooled) == len(CORE_SCHEME_IDS) * 2

    cross_protein = predictions[
        predictions["scheme_id"].isin(
            [
                "ndm1_to_vim2_amp_source_cv5",
                "vim2_to_ndm1_amp_source_cv5",
            ]
        )
    ]
    assert set(cross_protein["evaluation_iteration"].astype(str)) == {"target"}
    assert set(cross_protein["evaluation_role"]) == {"target_test"}
