from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )

from betalactamase_transfer.model_performance_summary import (  # noqa: E402
    inspect_model_performance_summary_inputs,
    run_model_performance_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize completed mutation-effect model results, "
            "paired uncertainty, biological result counts, "
            "and training-seed sensitivity."
        )
    )

    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help=("Review completed result inputs without writing summary tables."),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.validate_inputs:
        summary = inspect_model_performance_summary_inputs(ROOT)

        print("MODEL PERFORMANCE SUMMARY INPUT REVIEW PASSED")
    else:
        summary = run_model_performance_summary(ROOT)

        print("MODEL PERFORMANCE SUMMARY PASSED")

        print(f"Output directory: {ROOT / 'data/interim/model_performance_summary'}")

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
