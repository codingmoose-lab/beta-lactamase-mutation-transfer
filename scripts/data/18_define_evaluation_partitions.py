#!/usr/bin/env python3
# Define grouped evaluation partitions for mutation-effect analyses.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from betalactamase_transfer.evaluation_partitions import (
        EvaluationPartitionError,
        prepare_evaluation_partitions,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Define grouped evaluation partitions from the fixed task definitions "
            "without creating modelling feature matrices."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_evaluation_partitions(args.repo_root)
    except EvaluationPartitionError as exc:
        print(f"EVALUATION PARTITION FAILED: {exc}", file=sys.stderr)
        return 1

    print("EVALUATION PARTITION PASSED")
    print(
        "Output directory: "
        f"{args.repo_root.resolve() / 'data/interim/evaluation_partitions'}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
