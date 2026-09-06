from pathlib import Path

from betalactamase_transfer.model_performance_summary import (
    CORE_SCHEME_IDS,
    inspect_model_performance_summary_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_model_performance_summary_input_contract() -> None:
    summary = inspect_model_performance_summary_inputs(ROOT)

    assert summary["status"] == "PASS"

    assert summary["analysis_role"] == "post_evaluation_result_summary"

    assert summary["model_fitting_performed"] is False

    assert summary["parameter_selection_performed"] is False

    assert summary["prediction_values_modified"] is False

    assert summary["final_labels_used_for_model_selection"] is False

    assert summary["core_scheme_count"] == len(CORE_SCHEME_IDS) == 5

    assert summary["paired_comparison_count"] == 460

    assert summary["paired_intervals_excluding_zero"] == 299

    assert summary["paired_intervals_including_zero"] == 161

    assert summary["biological_summary_row_count"] == 4

    assert summary["training_seed_direction_change_count"] == 1
    assert summary["evaluation_population_count"] == 2

    assert summary["causal_interpretation_permitted"] is False
