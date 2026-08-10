#!/usr/bin/env python3
# Mutation Measurement Extraction for the beta-lactamase mutation study.

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
    from betalactamase_transfer.mutation_measurements import (
        MutationMeasurementError,
        extract_mutation_measurements,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Mutation Measurement Extraction for the beta-lactamase mutation study."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        summary = extract_mutation_measurements(args.repo_root)
    except MutationMeasurementError as exc:
        print(f"MUTATION MEASUREMENT EXTRACTION FAILED: {exc}", file=sys.stderr)
        return 1
    print("MUTATION MEASUREMENT EXTRACTION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/mutation_measurements"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
