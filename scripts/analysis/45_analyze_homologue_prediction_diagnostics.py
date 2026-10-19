#!/usr/bin/env python3
"""Write separate NDM-1/VIM-2 cross-protein prediction diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betalactamase_transfer.homologue_prediction_diagnostics import (  # noqa: E402
    HomologuePredictionDiagnosticError,
    analyze,
    load_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    try:
        if output == repo or repo in output.parents:
            raise HomologuePredictionDiagnosticError(
                "Diagnostic outputs must be outside the research repository"
            )
        if output.exists() and any(output.iterdir()):
            raise HomologuePredictionDiagnosticError(
                "Diagnostic output directory must be empty to prevent overwriting"
            )
        predictions, source_hashes = load_predictions(repo)
        report = analyze(predictions, repetitions=args.repetitions)
        output.mkdir(parents=True, exist_ok=True)
        metrics_path = output / "homologue_prediction_diagnostics.csv"
        summary_path = output / "homologue_prediction_diagnostics.txt"
        report.to_csv(metrics_path, index=False, lineterminator="\n")
        lines = [
            "NDM-1 AND VIM-2 MATCHED HOMOLOGUE PREDICTION DIAGNOSTICS",
            "UTC: " + datetime.now(timezone.utc).isoformat(),
            "Source file SHA-256:",
        ]
        lines.extend(f"  {path}: {digest}" for path, digest in source_hashes.items())
        lines.append(
            "\nPaired model comparisons (positive MAE improvement favours model):"
        )
        for _, item in report.iterrows():
            lines.append(
                f"{item['direction']} / {item['population']} / {item['model_id']}\n"
                f"  n={item['n_pairs']}; positions={item['n_homologue_positions']}; "
                f"model_rho={item['model_spearman']:.6f}; "
                f"direct_rho={item['direct_spearman']:.6f}\n"
                f"  model_MAE={item['model_mae']:.6f}; "
                f"direct_MAE={item['direct_mae']:.6f}; "
                f"paired_MAE_improvement={item['mae_improvement_over_direct']:.6f}\n"
                "  grouped_95%_MAE_difference=["
                f"{item['mae_improvement_ci_lower']:.6f}, "
                f"{item['mae_improvement_ci_upper']:.6f}]\n"
                "  grouped_95%_Spearman_difference=["
                f"{item['spearman_difference_ci_lower']:.6f}, "
                f"{item['spearman_difference_ci_upper']:.6f}]\n"
                "  Figure_4D_rho="
                f"{item['experimental_difference_vs_model_error_spearman']:.6f}; "
                "within_position_permutation_median="
                f"{item['within_position_null_median']:.6f}; "
                f"range_95%=[{item['within_position_null_lower']:.6f}, "
                f"{item['within_position_null_upper']:.6f}]"
            )
        lines.extend(
            [
                "",
                "SCIENTIFIC LIMITATIONS",
                "The null distribution is descriptive and retains the target score in "
                "both absolute differences. It is not evidence of a biochemical cause.",
                "The direct experimental estimate has access to "
                "the measured equivalent "
                "source substitution; fitted models are evaluated under a different "
                "information requirement.",
                "This command performs no model fitting and does not test performance "
                "when the equivalent source substitution was withheld from fitting.",
                "No main or supplementary figure was generated or modified.",
                "",
                "COMPLETE NUMERIC DATA",
                str(metrics_path),
            ]
        )
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("HOMOLOGUE PREDICTION DIAGNOSTICS COMPLETED")
        print(
            json.dumps(
                {
                    "report": str(summary_path),
                    "metrics": str(metrics_path),
                    "rows": len(report),
                },
                indent=2,
            )
        )
        return 0
    except (HomologuePredictionDiagnosticError, OSError, ValueError) as exc:
        print(f"HOMOLOGUE PREDICTION DIAGNOSTICS FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
