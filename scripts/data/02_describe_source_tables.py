#!/usr/bin/env python3
# Source Table Description for the beta-lactamase mutation study.

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
    from betalactamase_transfer.source_tables import (
        SourceTableError,
        describe_source_tables,
    )

    parser = argparse.ArgumentParser(
        description="Source Table Description for the beta-lactamase mutation study."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        summary = describe_source_tables(args.repo_root)
    except SourceTableError as exc:
        print(f"SOURCE TABLE DESCRIPTION FAILED: {exc}", file=sys.stderr)
        return 1
    print("SOURCE TABLE DESCRIPTION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/source_tables"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
