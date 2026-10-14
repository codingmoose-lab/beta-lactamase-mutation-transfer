from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.sequence_model_comparison import (  # noqa: E402
    write_sequence_model_comparison,
)

if __name__ == "__main__":
    summary = write_sequence_model_comparison(ROOT)
    print("SEQUENCE MODEL COMPARISON PASSED")
    print("Output directory:", ROOT / "data/interim/sequence_model_comparison")
    print(json.dumps(summary, indent=2))
