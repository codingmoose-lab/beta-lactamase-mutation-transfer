from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer import (  # noqa: E402
    biological_model_difference_analysis as model_difference_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse biological and structural patterns in matched prediction-error "
            "differences among sequence, residue-graph, geometric, and joint models."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        summary = model_difference_analysis.inspect_biological_model_difference_inputs(
            args.repo_root
        )
        print("BIOLOGICAL MODEL-DIFFERENCE INPUT REVIEW PASSED")
        print(json.dumps(summary, indent=2))
        return

    print(
        "Running biological model-difference analysis: matched error differences and "
        "grouped bootstrap intervals; no prediction model is being fitted.",
        flush=True,
    )
    summary = model_difference_analysis.run_biological_model_difference_analysis(
        args.repo_root
    )
    print("BIOLOGICAL MODEL-DIFFERENCE ANALYSIS PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
