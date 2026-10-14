from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from betalactamase_transfer.residue_geometry_neural_evaluation import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    TRAINING_REPEATS,
    inspect_residue_geometry_neural_inputs,
    run_residue_geometry_neural_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an E(n)-equivariant residue model for mutation-effect prediction."
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
            "Use completed context checkpoints and run only aggregation, metrics, "
            "and paired comparisons."
        ),
    )
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help=(
            "Validate source tables, residue-graph neural analysis comparator "
            "files, geometry, and evaluation populations without fitting a model."
        ),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    if args.validate_inputs:
        summary = inspect_residue_geometry_neural_inputs(repo_root)
        print("RESIDUE GEOMETRY NEURAL INPUT REVIEW PASSED")
    else:
        summary = run_residue_geometry_neural_evaluation(
            repo_root=repo_root,
            device_name=args.device,
            repeats=args.repeats,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            postprocess_only=args.postprocess_only,
        )
        print("RESIDUE GEOMETRY NEURAL EVALUATION PASSED")
        print(
            "Output directory:",
            repo_root / "data/interim/residue_geometry_neural_evaluation",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
