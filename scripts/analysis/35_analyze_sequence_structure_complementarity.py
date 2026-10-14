from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from betalactamase_transfer import (
    sequence_structure_complementarity as stage35,  # noqa: E402
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assess whether sequence information adds predictive value "
            "to residue-graph predictions using validation-set model selection."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--batch-size", type=int, default=stage35.graph_stage.BATCH_SIZE
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        summary = stage35.inspect_sequence_structure_complementarity_inputs(
            args.repo_root
        )
        print("SEQUENCE-STRUCTURE COMPLEMENTARITY INPUT REVIEW PASSED")
        print(json.dumps(summary, indent=2))
        return

    summary = stage35.run_sequence_structure_complementarity(
        args.repo_root,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print("SEQUENCE-STRUCTURE COMPLEMENTARITY ANALYSIS PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
