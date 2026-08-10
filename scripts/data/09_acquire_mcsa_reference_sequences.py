#!/usr/bin/env python3
# Acquire exact UniProt records for the selected M-CSA reference proteins.

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
    from betalactamase_transfer.mcsa_reference_sequences import (
        MCSAReferenceSequenceError,
        acquire_mcsa_reference_sequences,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Acquire exact UniProt records for the selected M-CSA reference proteins."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    try:
        summary = acquire_mcsa_reference_sequences(
            args.repo_root,
            timeout=args.timeout,
        )
    except MCSAReferenceSequenceError as exc:
        print(f"M-CSA REFERENCE SEQUENCE ACQUISITION FAILED: {exc}", file=sys.stderr)
        return 1

    print("M-CSA REFERENCE SEQUENCE ACQUISITION PASSED")
    output_dir = args.repo_root.resolve() / "data/interim/mcsa_reference_sequences"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
