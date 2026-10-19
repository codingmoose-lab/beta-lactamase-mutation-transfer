"""Check source-protein fitting partitions and stored model-selection records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from betalactamase_transfer.source_selection_audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.repo_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed"]:
        raise SystemExit("Invalid source-protein partition assignments")


if __name__ == "__main__":
    main()
