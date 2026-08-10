#!/usr/bin/env python3
# Prepare targeted CARD ontology context for NDM-1, VIM-2, and study antibiotics.

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
        prepare_card_context,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare targeted CARD ontology context for NDM-1, VIM-2, and study "
            "antibiotics."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_card_context(args.repo_root)
    except CARDContextError as exc:
        print(f"CARD CONTEXT PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("CARD CONTEXT PREPARATION PASSED")
    print(f"Output directory: {args.repo_root.resolve() / 'data/interim/card_context'}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
