from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from betalactamase_transfer.atomic_local_voxel_evaluation import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    TRAINING_REPEATS,
    inspect_atomic_local_voxel_inputs,
    run_atomic_local_voxel_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a mutation-centred atom-level 3-D voxel CNN using selected "
            "wild-type experimental structures."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--repeats", type=int, default=TRAINING_REPEATS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help=(
            "Use completed context checkpoints and rerun only aggregation "
            "and comparisons."
        ),
    )
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help=(
            "Validate coordinate mapping, voxel construction, comparators, "
            "and evaluation populations without fitting the CNN."
        ),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.validate_inputs:
        summary = inspect_atomic_local_voxel_inputs(repo_root)
        print("ATOMIC LOCAL VOXEL INPUT REVIEW PASSED")
    else:
        summary = run_atomic_local_voxel_evaluation(
            repo_root=repo_root,
            device_name=args.device,
            repeats=args.repeats,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            postprocess_only=args.postprocess_only,
        )
        print("ATOMIC LOCAL VOXEL EVALUATION PASSED")
        print(
            "Output directory:",
            repo_root / "data/interim/atomic_local_voxel_evaluation",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
