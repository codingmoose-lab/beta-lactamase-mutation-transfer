#!/usr/bin/env python3
# Evaluate direct residue-neighbourhood descriptors from experimental structures.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.residue_neighbourhood_evaluation import (  # noqa: E402
    run_residue_neighbourhood_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate direct residue-neighbourhood structural descriptors on the "
            "experimental-structure-supported mutation-effect population."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the repository containing this script.",
    )
    args = parser.parse_args()
    summary = run_residue_neighbourhood_evaluation(args.repo_root)
    output_dir = (
        args.repo_root.resolve() / "data/interim/residue_neighbourhood_evaluation"
    )
    print("RESIDUE-NEIGHBOURHOOD EVALUATION PASSED")
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
