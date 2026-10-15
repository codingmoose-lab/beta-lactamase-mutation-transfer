"""Inspect manuscript figure sources without regenerating publication figures."""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from betalactamase_transfer.manuscript_figures import FIGURE_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report manuscript figure sources without changing figures/main."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    print("Publication figures are preserved; no figure generation was run.")
    for number, rel in FIGURE_SOURCES.items():
        path = root / rel
        state = (
            "PRESENT"
            if path.is_file()
            else ("PLANNED" if number in {5, 6} else "MISSING")
        )
        print(f"Figure {number}: {state:7} | {rel}")


if __name__ == "__main__":
    main()
