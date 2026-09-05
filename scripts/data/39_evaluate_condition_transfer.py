from __future__ import annotations

import argparse
import json
from pathlib import Path

from betalactamase_transfer.condition_transfer_evaluation import (
    review_condition_transfer_inputs,
    run_condition_transfer_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VIM-2 antibiotic and temperature transfer from "
            "source-only model selection."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.check_only:
        result = review_condition_transfer_inputs(repo_root)
        print("CONDITION TRANSFER INPUT REVIEW PASSED")
    else:
        result = run_condition_transfer_evaluation(repo_root)
        print("CONDITION TRANSFER EVALUATION PASSED")
        print(
            "Output directory:",
            repo_root / "data/interim/condition_transfer_evaluation",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
