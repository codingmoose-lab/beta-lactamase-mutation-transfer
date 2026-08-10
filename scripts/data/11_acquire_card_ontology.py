#!/usr/bin/env python3
# Preserve and validate the frozen official CARD ontology 4.0.1 source.

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
    from betalactamase_transfer.card_context import (
        CARDContextError,
        acquire_card_ontology,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Preserve and validate the frozen official CARD ontology 4.0.1 source."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Official card-ontology.tar.bz2 downloaded from CARD",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        summary = acquire_card_ontology(
            args.repo_root,
            args.source_archive,
            overwrite=args.overwrite,
        )
    except CARDContextError as exc:
        print(f"CARD ONTOLOGY ACQUISITION FAILED: {exc}", file=sys.stderr)
        return 1

    print("CARD ONTOLOGY ACQUISITION PASSED")
    print(
        "Output directory: "
        f"{args.repo_root.resolve() / 'data/interim/card_acquisition'}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
