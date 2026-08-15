from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.sequence_plm_evaluation import (  # noqa: E402
    write_sequence_plm_evaluation,
)

if __name__ == "__main__":
    summary = write_sequence_plm_evaluation(ROOT)

    print("ESMC MUTATION-EFFECT EVALUATION PASSED")
    print(json.dumps(summary, indent=2))
