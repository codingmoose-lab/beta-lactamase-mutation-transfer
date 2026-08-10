#!/usr/bin/env python3
# Define scientific prediction tasks and evaluation rules before partition generation.

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
    from betalactamase_transfer.task_definitions import (
        TaskDefinitionError,
        prepare_task_definitions,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Define mutation-effect tasks, exact data scopes, and evaluation "
            "rules without creating train/validation/test assignments."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        summary = prepare_task_definitions(args.repo_root)
    except TaskDefinitionError as exc:
        print(f"TASK DEFINITION FAILED: {exc}", file=sys.stderr)
        return 1

    print("TASK DEFINITION PASSED")
    print(
        "Output directory: "
        f"{args.repo_root.resolve() / 'data/interim/task_definitions'}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
