#!/usr/bin/env python3
# Prepare residue graphs from the selected experimental structures.

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
    from betalactamase_transfer.residue_graphs import (
        ResidueGraphError,
        prepare_residue_graphs,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare outcome-blind residue graphs from the selected experimental "
            "structures and audited residue-pair geometry."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_residue_graphs(args.repo_root)
    except ResidueGraphError as exc:
        print(f"RESIDUE GRAPH PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("RESIDUE GRAPH PREPARATION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/residue_graphs"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
