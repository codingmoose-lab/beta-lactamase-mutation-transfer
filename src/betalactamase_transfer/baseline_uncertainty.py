from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from betalactamase_transfer.baseline_evaluation import _metrics

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = "beta_lactamase_baseline_bootstrap"


class BaselineUncertaintyError(RuntimeError):
    pass


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _grouped_bootstrap(
    frame: pd.DataFrame,
    *,
    group_column: str,
    true_column: str,
    prediction_column: str,
    seed_parts: tuple,
) -> list:
    grouped = {
        key: subset[[true_column, prediction_column]].to_numpy(dtype=float)
        for key, subset in frame.groupby(group_column, sort=True)
    }
    if not grouped:
        raise BaselineUncertaintyError("Bootstrap population contains no groups")
    group_ids = sorted(grouped)
    rng = np.random.default_rng(_stable_seed(BOOTSTRAP_SEED, *seed_parts))
    distributions = {name: [] for name in ("spearman", "pearson", "mae", "rmse")}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled = np.concatenate([grouped[group_id] for group_id in sampled_ids])
        values = _metrics(sampled[:, 0], sampled[:, 1])
        for metric, value in values.items():
            if math.isfinite(value):
                distributions[metric].append(value)
    point = _metrics(
        frame[true_column].to_numpy(dtype=float),
        frame[prediction_column].to_numpy(dtype=float),
    )
    rows = []
    for metric, values in distributions.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower": float(np.quantile(array, 0.025)) if len(array) else np.nan,
                "ci_upper": float(np.quantile(array, 0.975)) if len(array) else np.nan,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_valid_replicates": len(array),
                "resampling_unit": group_column,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    return rows


def run_baseline_uncertainty(repo_root: Path) -> Mapping[str, object]:
    output_dir = repo_root / "data/interim/baseline_evaluation"
    prediction_path = output_dir / "baseline_predictions.csv"
    direct_path = output_dir / "homologue_direct_transfer_predictions.csv"
    if not prediction_path.is_file() or not direct_path.is_file():
        raise BaselineUncertaintyError(
            "Baseline prediction outputs must exist before uncertainty calculation"
        )

    predictions = pd.read_csv(
        prediction_path,
        dtype={"evaluation_iteration": "string"},
    )
    direct = pd.read_csv(direct_path)
    interval_rows = []
    for (scheme_id, baseline_id), group in predictions.groupby(
        ["scheme_id", "baseline_id"], sort=True
    ):
        rows = _grouped_bootstrap(
            group,
            group_column="evaluation_group_id",
            true_column="y_true",
            prediction_column="y_pred",
            seed_parts=(scheme_id, baseline_id),
        )
        for row in rows:
            interval_rows.append(
                {"scheme_id": scheme_id, "baseline_id": baseline_id, **row}
            )

    direct_rows = []
    for direction, group in direct.groupby("direction", sort=True):
        rows = _grouped_bootstrap(
            group,
            group_column="homologue_position_id",
            true_column="target_effect",
            prediction_column="predicted_target_effect",
            seed_parts=("direct_homologue_transfer", direction),
        )
        for row in rows:
            direct_rows.append({"direction": direction, **row})

    intervals = pd.DataFrame(interval_rows)
    direct_intervals = pd.DataFrame(direct_rows)
    _write_csv(intervals, output_dir / "baseline_bootstrap_intervals.csv")
    _write_csv(direct_intervals, output_dir / "homologue_direct_transfer_intervals.csv")
    summary = {
        "status": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "baseline_interval_row_count": len(intervals),
        "direct_homologue_interval_row_count": len(direct_intervals),
        "baseline_resampling_unit": "evaluation_group_id",
        "direct_homologue_resampling_unit": "homologue_position_id",
        "outputs": [
            "baseline_bootstrap_intervals.csv",
            "homologue_direct_transfer_intervals.csv",
            "baseline_uncertainty_summary.json",
        ],
    }
    (output_dir / "baseline_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
