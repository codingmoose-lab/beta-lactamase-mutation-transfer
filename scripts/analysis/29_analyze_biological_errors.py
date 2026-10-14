from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.biological_error_analysis import (  # noqa: E402
    write_biological_error_analysis,
)

if __name__ == "__main__":
    summary = write_biological_error_analysis(ROOT)
    print("BIOLOGICAL ERROR ANALYSIS PASSED")
    print(
        "Output directory:",
        ROOT / "data/interim/biological_error_analysis",
    )
    print(json.dumps(summary, indent=2))
