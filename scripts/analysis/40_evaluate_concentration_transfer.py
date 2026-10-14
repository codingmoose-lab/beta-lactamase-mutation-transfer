from __future__ import annotations

import argparse
import json
from pathlib import Path

from betalactamase_transfer.concentration_transfer_evaluation import (
    review_concentration_transfer_inputs,
    run_concentration_transfer_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VIM-2 transfer to held-out ampicillin concentrations using "
            "source-only model selection."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        summary = review_concentration_transfer_inputs(args.repo_root)
        print("CONCENTRATION TRANSFER INPUT REVIEW PASSED")
    else:
        summary = run_concentration_transfer_evaluation(args.repo_root)
        print("CONCENTRATION TRANSFER EVALUATION PASSED")
        print(
            "Output directory:",
            args.repo_root / "data/interim/concentration_response_evaluation",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
