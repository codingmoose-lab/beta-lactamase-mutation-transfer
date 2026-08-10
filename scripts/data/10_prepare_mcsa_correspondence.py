#!/usr/bin/env python3
# Prepare M-CSA catalytic-residue correspondence for NDM-1 and VIM-2.

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from betalactamase_transfer.mcsa_correspondence import (
        MCSACorrespondenceError,
        prepare_mcsa_correspondence,
    )

    try:
        summary = prepare_mcsa_correspondence(REPO_ROOT)
    except MCSACorrespondenceError as exc:
        print(f"M-CSA CORRESPONDENCE FAILED: {exc}")
        return 1

    print("M-CSA CORRESPONDENCE PASSED")
    print(f"Output directory: {REPO_ROOT / summary['output_directory']}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
