#!/usr/bin/env python3
# Protein Structure Preparation for the beta-lactamase mutation study.

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
    from betalactamase_transfer.protein_structures import (
        ProteinStructureError,
        prepare_protein_structures,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Protein Structure Preparation for the beta-lactamase mutation study."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        summary = prepare_protein_structures(args.repo_root)
    except ProteinStructureError as exc:
        print(f"PROTEIN STRUCTURE PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1
    print("PROTEIN STRUCTURE PREPARATION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/protein_structures"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
