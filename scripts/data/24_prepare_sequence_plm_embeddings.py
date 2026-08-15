from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.sequence_plm_embeddings import (  # noqa: E402
    EmbeddingConfig,
    extract_sequence_plm_embeddings,
    prepare_sequence_plm_embedding_index,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract",
        action="store_true",
        help=(
            "Run GPU ESMC extraction after validating the "
            "outcome-blind embedding index."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Allow Hugging Face downloads instead of requiring the pinned "
            "checkpoint in cache."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PLM embedding outputs.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[9, 18, 27, 36],
        help="ESMC hidden-state indices to retain.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = EmbeddingConfig(
        hidden_state_indices=tuple(args.layers),
        batch_size=args.batch_size,
    )
    embedding_summary = prepare_sequence_plm_embedding_index(ROOT, config)
    print("SEQUENCE PLM EMBEDDING INDEX PASSED")
    print(json.dumps(embedding_summary, indent=2))

    if args.extract:
        extraction_summary = extract_sequence_plm_embeddings(
            ROOT,
            config,
            local_files_only=not args.allow_download,
            overwrite=args.overwrite,
        )
        print("SEQUENCE PLM EXTRACTION PASSED")
        print(json.dumps(extraction_summary, indent=2))
