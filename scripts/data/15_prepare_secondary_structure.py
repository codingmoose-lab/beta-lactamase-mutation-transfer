#!/usr/bin/env python3
# Assign DSSP secondary structure to selected experimental residues.

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
    from betalactamase_transfer.secondary_structure import (
        SecondaryStructureError,
        prepare_secondary_structure,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Assign DSSP secondary structure to the selected experimental "
            "PDB chains and map the assignments to assay positions."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--mkdssp",
        type=Path,
        default=None,
        help="Path to mkdssp. If omitted, PATH and ~/.local/bin are checked.",
    )
    args = parser.parse_args()

    try:
        summary = prepare_secondary_structure(
            args.repo_root,
            mkdssp_executable=args.mkdssp,
        )
    except SecondaryStructureError as exc:
        print(f"SECONDARY STRUCTURE PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("SECONDARY STRUCTURE PREPARATION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/secondary_structure"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
