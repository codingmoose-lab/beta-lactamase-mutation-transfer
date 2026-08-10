#!/usr/bin/env python3
# Prepare modelling-view contracts without generating model features.

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
    from betalactamase_transfer.modelling_views import (
        ModellingViewError,
        prepare_modelling_views,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prepare modelling-view contracts and observation membership without "
            "generating embeddings, graphs, images, or trained models."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_modelling_views(args.repo_root)
    except ModellingViewError as exc:
        print(f"MODELLING VIEW PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("MODELLING VIEW PREPARATION PASSED")
    print(
        f"Output directory: {args.repo_root.resolve() / 'data/interim/modelling_views'}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
