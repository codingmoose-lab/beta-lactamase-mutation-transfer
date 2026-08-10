from betalactamase_transfer.assay_definitions import (
    _condition_candidates,
    _score_registry,
    _stable_id,
)


def test_stable_id_is_deterministic() -> None:
    assert _stable_id("x", "a", 1) == _stable_id("x", "a", 1)
    assert _stable_id("x", "a", 1) != _stable_id("x", "a", 2)


def test_amp_is_the_only_direct_cross_protein_condition() -> None:
    rows = _condition_candidates()
    direct = [row for row in rows if row["direct_comparison_use"]]
    assert len(direct) == 1
    assert direct[0]["antibiotic"] == "AMP"
    assert direct[0]["ndm_concentration"] == 256.0
    assert direct[0]["vim_concentration"] == 128.0
    assert (
        direct[0]["comparability_status"] == "study_defined_similar_selection_pressure"
    )


def test_ctx_and_mem_are_condition_specific() -> None:
    rows = _condition_candidates()
    condition_specific = {
        row["antibiotic"]: row for row in rows if row["antibiotic"] != "AMP"
    }
    assert set(condition_specific) == {"CTX", "MEM"}
    for row in condition_specific.values():
        assert row["direct_comparison_use"] is False
        assert row["comparability_status"] == "condition_specific_only"


def test_assay_registry_uses_documented_selection_temperatures() -> None:
    assays = {row["study_id"]: row for row in _score_registry()}
    assert assays["chen_2024_natcomms"]["selection_temperature_c"] == 37
    assert (
        assays["chen_2020_elife"]["selection_temperature_c"]
        == "condition-specific: 25 or 37"
    )
    assert assays["chen_2022_natee"]["selection_temperature_c"] == 37
    assert assays["chen_2022_natee"]["host_system"] == "Escherichia coli E. cloni 10G"
    assert "Nature Ecology & Evolution" in assays["chen_2022_natee"]["source_support"]
