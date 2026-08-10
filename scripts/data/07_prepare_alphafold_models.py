#!/usr/bin/env python3
# Prepare AlphaFold reference models for residue-level confidence and correspondence.

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
    from betalactamase_transfer.alphafold_preparation import (
        AlphaFoldPreparationError,
        prepare_alphafold_models,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare AlphaFold reference models for residue-level confidence and "
            "correspondence."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_alphafold_models(args.repo_root)
    except AlphaFoldPreparationError as exc:
        print(f"ALPHAFOLD PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("ALPHAFOLD PREPARATION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/alphafold_preparation"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
