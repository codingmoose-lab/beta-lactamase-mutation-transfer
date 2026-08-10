from betalactamase_transfer.mutation_measurements import (
    _condition_specs_2020,
    _condition_specs_2022,
    _condition_specs_2024,
    _mutation_validation,
)


def test_condition_parsers_pair_scores_and_uncertainties():
    specs_2024 = _condition_specs_2024(["32.0_AMP_score", "32.0_AMP_score_sd"])
    assert [(item.antibiotic, item.concentration) for item in specs_2024] == [
        ("AMP", 32.0)
    ]
    assert specs_2024[0].temperature_c == 37.0
    specs_2020 = _condition_specs_2020(["128ug/mL_AMP_25C", "128ug/mL_AMP_25C_SD"])
    assert specs_2020[0].temperature_c == 25.0
    specs_2022 = _condition_specs_2022(["4.0ug/mL_fitness", "4.0ug/mL_sd"])
    assert specs_2022[0].antibiotic == "AMP"
    assert specs_2022[0].temperature_c == 37.0


def test_mutation_validation_distinguishes_controls_and_errors():
    sequence = "MGA"
    assert _mutation_validation(sequence, "M", 1, "A") == ("valid", "missense")
    assert _mutation_validation(sequence, "M", 1, "*") == ("valid", "stop_control")
    assert _mutation_validation(sequence, "M", 1, "M") == ("valid", "wild_type_control")
    assert _mutation_validation(sequence, "A", 1, "G")[0] == "wild_type_mismatch"
    assert _mutation_validation(sequence, "M", 9, "A")[0] == "invalid_position"
