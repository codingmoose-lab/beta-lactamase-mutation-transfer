from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.matched_homologue_model_analysis import (  # noqa: E402
    run_matched_homologue_model_analysis,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-inputs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_matched_homologue_model_analysis(
        ROOT, validate_inputs=args.validate_inputs
    )
    if args.validate_inputs:
        print("MATCHED HOMOLOGUE MODEL INPUT REVIEW PASSED")
    else:
        print("MATCHED HOMOLOGUE MODEL ANALYSIS PASSED")
        print(
            "Output directory: "
            f"{ROOT / 'data/interim/matched_homologue_model_analysis'}"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
