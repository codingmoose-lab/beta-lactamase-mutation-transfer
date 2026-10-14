from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer import training_seed_stability as stage37  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure neural-model variation across ten fixed training seeds without "
            "repeating model selection."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=stage37.stage32.BATCH_SIZE)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    if args.check_only and args.postprocess_only:
        parser.error("--check-only and --postprocess-only cannot be used together")

    if args.check_only:
        summary = stage37.inspect_training_seed_stability_inputs(args.repo_root)
        print("TRAINING-SEED STABILITY INPUT REVIEW PASSED")
        print(json.dumps(summary, indent=2))
        return

    summary = stage37.run_training_seed_stability(
        args.repo_root,
        device_name=args.device,
        batch_size=args.batch_size,
        postprocess_only=args.postprocess_only,
    )
    print("TRAINING-SEED STABILITY ANALYSIS PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
