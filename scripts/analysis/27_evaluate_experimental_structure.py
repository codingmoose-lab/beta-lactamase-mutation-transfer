from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.experimental_structure_evaluation import (  # noqa: E402
    run_experimental_structure_evaluation,
)

if __name__ == "__main__":
    summary = run_experimental_structure_evaluation(ROOT)
    print("EXPERIMENTAL STRUCTURE EVALUATION PASSED")
    print(
        "Output directory:",
        ROOT / "data/interim/experimental_structure_evaluation",
    )
    print(json.dumps(summary, indent=2))
