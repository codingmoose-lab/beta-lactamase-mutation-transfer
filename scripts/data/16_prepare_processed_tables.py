#!/usr/bin/env python3
# Prepare mutation-effect and residue-feature tables for analysis and modelling.

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
    from betalactamase_transfer.processed_tables import (
        ProcessedTableError,
        prepare_processed_tables,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare checked observation, residue-feature, EC50, homologue-comparison, "
            "and residue-pair tables under data/processed."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_processed_tables(args.repo_root)
    except ProcessedTableError as exc:
        print(f"PROCESSED TABLE PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("PROCESSED TABLE PREPARATION PASSED")
    print(f"Output directory: {args.repo_root.resolve() / 'data/processed'}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
