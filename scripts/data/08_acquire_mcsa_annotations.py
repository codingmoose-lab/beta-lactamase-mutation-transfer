#!/usr/bin/env python3
# Acquire targeted M-CSA Class B1 catalytic and mechanistic records.

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
    from betalactamase_transfer.mcsa_annotations import (
        DEFAULT_MCSA_IDS,
        MCSAAnnotationError,
        acquire_mcsa_annotations,
    )

    parser = argparse.ArgumentParser(
        description="Acquire targeted M-CSA Class B1 catalytic and mechanistic records."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--mcsa-id",
        action="append",
        dest="mcsa_ids",
        type=int,
        help="M-CSA entry identifier; repeat for multiple entries",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mcsa_ids = tuple(args.mcsa_ids or DEFAULT_MCSA_IDS)
    try:
        summary = acquire_mcsa_annotations(
            args.repo_root,
            mcsa_ids,
            timeout=args.timeout,
            overwrite=args.overwrite,
        )
    except MCSAAnnotationError as exc:
        print(f"M-CSA ACQUISITION FAILED: {exc}", file=sys.stderr)
        return 1

    print("M-CSA ACQUISITION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/mcsa_acquisition"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
