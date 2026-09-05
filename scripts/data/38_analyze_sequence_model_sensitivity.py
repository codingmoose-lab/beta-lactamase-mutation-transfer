from __future__ import annotations

import argparse
import json
from pathlib import Path

from betalactamase_transfer.sequence_model_sensitivity import (
    extract_esm2_embeddings,
    review_sequence_model_sensitivity_inputs,
    write_sequence_model_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    if args.check_only:
        summary = review_sequence_model_sensitivity_inputs(
            root, batch_size=args.batch_size
        )
        print("SEQUENCE-MODEL SENSITIVITY INPUT REVIEW PASSED")
        print(json.dumps(summary, indent=2))
        print("No model was fitted.")
        print("No existing prediction was changed.")
        return

    if args.extract:
        summary = extract_esm2_embeddings(
            root,
            batch_size=args.batch_size,
            local_files_only=not args.allow_download,
        )
        print("ESM-2 EMBEDDING EXTRACTION PASSED")
        print(json.dumps(summary, indent=2))
        return

    summary = write_sequence_model_sensitivity(
        root,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print("SEQUENCE-MODEL SENSITIVITY ANALYSIS PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
