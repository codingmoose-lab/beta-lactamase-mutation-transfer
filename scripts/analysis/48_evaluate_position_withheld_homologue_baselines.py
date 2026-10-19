"""Fit source-protein references excluding every held-out homologous position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from betalactamase_transfer.position_withheld_homologue_baselines import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(args.repo_root, args.output_dir, repetitions=args.repetitions),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
