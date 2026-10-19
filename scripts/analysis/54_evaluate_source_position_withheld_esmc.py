"""Evaluate source-only ESM-C Ridge with held-out homologous positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from betalactamase_transfer.source_position_withheld_esmc import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run(args.repo_root, args.output_dir, args.repetitions), indent=2))


if __name__ == "__main__":
    main()
