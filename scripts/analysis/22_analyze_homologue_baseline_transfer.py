from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.homologue_baseline_analysis import (  # noqa: E402
    run_homologue_baseline_analysis,
)

if __name__ == "__main__":
    summary = run_homologue_baseline_analysis(ROOT)
    print("HOMOLOGUE BASELINE ANALYSIS PASSED")
    output_dir = ROOT / "data/interim/baseline_evaluation/homologue_analysis"
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary, indent=2))
