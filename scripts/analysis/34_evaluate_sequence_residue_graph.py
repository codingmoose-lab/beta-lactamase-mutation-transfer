from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from betalactamase_transfer import (  # noqa: E402
    sequence_residue_graph_neural_evaluation as sequence_residue_graph_eval,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ESMC site-delta features together with the residue graph."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--repeats", type=int, default=sequence_residue_graph_eval.TRAINING_REPEATS
    )
    parser.add_argument(
        "--max-epochs", type=int, default=sequence_residue_graph_eval.MAX_EPOCHS
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=sequence_residue_graph_eval.EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--batch-size", type=int, default=sequence_residue_graph_eval.BATCH_SIZE
    )
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    if args.validate_inputs and args.postprocess_only:
        parser.error("--validate-inputs and --postprocess-only cannot be used together")

    if args.validate_inputs:
        summary = sequence_residue_graph_eval.inspect_sequence_residue_graph_inputs(
            args.repo_root
        )
        print("SEQUENCE + RESIDUE GRAPH INPUT REVIEW PASSED")
        print(json.dumps(summary, indent=2))
        return

    summary = sequence_residue_graph_eval.run_sequence_residue_graph_neural_evaluation(
        args.repo_root,
        device_name=args.device,
        repeats=args.repeats,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        postprocess_only=args.postprocess_only,
    )
    print("SEQUENCE + RESIDUE GRAPH EVALUATION PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
