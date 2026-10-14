from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from betalactamase_transfer.residue_graph_neural_evaluation import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    TRAINING_REPEATS,
    inspect_residue_graph_neural_inputs,
    run_residue_graph_neural_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate centre-residue and residue-graph neural mutation-effect models."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=TRAINING_REPEATS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help=(
            "Validate source tables and evaluation populations without fitting a model."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_inputs:
        summary = inspect_residue_graph_neural_inputs(args.repo_root)
        print("RESIDUE-GRAPH NEURAL INPUT REVIEW PASSED")
    else:
        summary = run_residue_graph_neural_evaluation(
            args.repo_root,
            device_name=args.device,
            repeats=args.repeats,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        print("RESIDUE-GRAPH NEURAL EVALUATION PASSED")
        print(
            "Output directory:",
            args.repo_root / "data/interim/residue_graph_neural_evaluation",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
