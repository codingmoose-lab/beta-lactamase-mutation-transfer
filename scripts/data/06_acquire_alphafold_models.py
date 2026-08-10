#!/usr/bin/env python3
# Acquire exact AlphaFold DB models for NDM-1 and VIM-2.

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
    from betalactamase_transfer.alphafold_models import (
        DEFAULT_ACCESSIONS,
        AlphaFoldAcquisitionError,
        acquire_alphafold_models,
    )

    parser = argparse.ArgumentParser(
        description="Acquire exact AlphaFold DB models for NDM-1 and VIM-2."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--accession",
        action="append",
        dest="accessions",
        help="Exact UniProt accession; repeat for multiple accessions",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    accessions = tuple(args.accessions or DEFAULT_ACCESSIONS)
    try:
        summary = acquire_alphafold_models(
            args.repo_root,
            accessions,
            timeout=args.timeout,
            overwrite=args.overwrite,
        )
    except AlphaFoldAcquisitionError as exc:
        print(f"ALPHAFOLD ACQUISITION FAILED: {exc}", file=sys.stderr)
        return 1

    print("ALPHAFOLD ACQUISITION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/alphafold_acquisition"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
