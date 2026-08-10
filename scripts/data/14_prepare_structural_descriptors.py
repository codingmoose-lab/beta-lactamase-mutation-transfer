#!/usr/bin/env python3
# Generate selected-structure residue descriptors and pairwise geometry.

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
    from betalactamase_transfer.structural_descriptors import (
        StructuralDescriptorError,
        prepare_structural_descriptors,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate residue geometry, solvent accessibility, catalytic-site "
            "distances, and pairwise distances from the selected experimental "
            "structures."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_structural_descriptors(args.repo_root)
    except StructuralDescriptorError as exc:
        print(f"STRUCTURAL DESCRIPTOR PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("STRUCTURAL DESCRIPTOR PREPARATION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/structural_descriptors"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
