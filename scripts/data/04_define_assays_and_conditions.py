#!/usr/bin/env python3
# Assay Definition for the beta-lactamase mutation study.

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
    from betalactamase_transfer.assay_definitions import (
        AssayDefinitionError,
        define_assays_and_conditions,
    )

    parser = argparse.ArgumentParser(
        description="Assay Definition for the beta-lactamase mutation study."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        summary = define_assays_and_conditions(args.repo_root)
    except AssayDefinitionError as exc:
        print(f"ASSAY DEFINITION FAILED: {exc}", file=sys.stderr)
        return 1
    print("ASSAY DEFINITION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/assay_definitions"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
