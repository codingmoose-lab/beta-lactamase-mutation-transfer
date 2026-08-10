#!/usr/bin/env python3

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
    from betalactamase_transfer.baseline_uncertainty import (
        BaselineUncertaintyError,
        run_baseline_uncertainty,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Calculate grouped bootstrap intervals for completed baseline predictions."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = run_baseline_uncertainty(args.repo_root)
    except BaselineUncertaintyError as exc:
        print(f"BASELINE UNCERTAINTY CALCULATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("BASELINE UNCERTAINTY CALCULATION PASSED")
    print(
        "Output directory:",
        args.repo_root.resolve() / "data/interim/baseline_evaluation",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
