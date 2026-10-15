#!/usr/bin/env python3
"""List figure source files without generating or modifying graphics."""

from __future__ import annotations

import argparse
from pathlib import Path

FIGURE_SOURCES = {
    1: Path("src/betalactamase_transfer/manuscript_figure1.py"),
    2: Path("scripts/figures/generate_figure2_homologue_effects.py"),
    3: Path("scripts/figures/generate_figure3_representation_comparison.py"),
    4: Path("scripts/figures/generate_figure4_cross_homologue_prediction.py"),
    5: Path("scripts/figures/generate_figure5_model_differences.py"),
    6: Path("scripts/figures/generate_figure6_condition_and_concentration.py"),
}


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
        state = "PRESENT" if path.is_file() else "MISSING"
        print(f"Figure {number}: {state:7} | {rel}")


if __name__ == "__main__":
    main()
