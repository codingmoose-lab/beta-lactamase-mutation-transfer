from __future__ import annotations

import argparse
import json
from pathlib import Path

from betalactamase_transfer.condition_concentration_biological_analysis import (
    review_condition_concentration_inputs,
    run_condition_concentration_biological_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize VIM-2 antibiotic, temperature, and concentration-response "
            "results without fitting another model."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-inputs", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.validate_inputs:
        summary = review_condition_concentration_inputs(repo_root)
        print("CONDITION AND CONCENTRATION BIOLOGY INPUT REVIEW PASSED")
    else:
        summary = run_condition_concentration_biological_analysis(repo_root)
        print("CONDITION AND CONCENTRATION BIOLOGICAL ANALYSIS PASSED")
        print(
            "Output directory:",
            repo_root / "data/interim/condition_concentration_biological_analysis",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
