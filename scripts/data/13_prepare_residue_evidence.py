#!/usr/bin/env python3
# Prepare residue-level evidence for the selected NDM-1 and VIM-2 references.

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
    from betalactamase_transfer.residue_evidence import (
        ResidueEvidenceError,
        prepare_residue_evidence,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare residue-level assay, UniProt, selected-PDB, AlphaFold, and "
            "M-CSA evidence without collapsing source-specific meaning."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_residue_evidence(args.repo_root)
    except ResidueEvidenceError as exc:
        print(f"RESIDUE EVIDENCE PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("RESIDUE EVIDENCE PREPARATION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/residue_evidence"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
