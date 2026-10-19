"""Check source-protein fitting inputs for baseline, ESM-C, and structural Ridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from betalactamase_transfer import (
    baseline_evaluation as baseline,
)
from betalactamase_transfer import (
    experimental_structure_evaluation as structure,
)
from betalactamase_transfer import (
    sequence_plm_evaluation as plm,
)

SCHEMES = (
    ("ndm1_to_vim2_amp_source_cv5", "NDM-1", "VIM-2"),
    ("vim2_to_ndm1_amp_source_cv5", "VIM-2", "NDM-1"),
)
MODULES = (
    ("baseline_evaluation", baseline),
    ("sequence_plm_evaluation", plm),
    ("experimental_structure_evaluation", structure),
)


class SourceSelectionCheckError(RuntimeError):
    """Fitting rows or validation folds violate source-protein selection."""


def _sample(scheme_id: str, source_protein: str, target_protein: str):
    source = pd.DataFrame(
        {
            "observation_id": [f"source_{i:02d}" for i in range(10)],
            "group_id": [f"source_group_{i:02d}" for i in range(10)],
            "scope_role": ["source_model_selection"] * 10,
            "protein_id": [source_protein] * 10,
            "score_analysis": np.linspace(-2.0, 2.0, 10),
            "concentration": np.arange(2.0, 12.0),
            "temperature_c": [37.0] * 10,
            "antibiotic": ["AMP"] * 10,
            "assay_position": np.arange(1, 11),
            "embedding_row": np.arange(10),
        }
    )
    target = pd.DataFrame(
        {
            "observation_id": [f"target_{i:02d}" for i in range(3)],
            "group_id": [f"target_group_{i:02d}" for i in range(3)],
            "scope_role": ["target_test"] * 3,
            "protein_id": [target_protein] * 3,
            "score_analysis": [999001.0, 999002.0, 999003.0],
            "concentration": [8192.0] * 3,
            "temperature_c": [37.0] * 3,
            "antibiotic": ["AMP"] * 3,
            "assay_position": np.arange(1, 4),
            "embedding_row": np.arange(10, 13),
        }
    )
    frame = pd.concat([source, target], ignore_index=True)
    source_assignments = pd.DataFrame(
        {
            "scheme_id": scheme_id,
            "group_id": source.group_id,
            "split_role": "source_model_selection",
            "cv_fold": [i % 5 for i in range(10)],
        }
    )
    target_assignments = pd.DataFrame(
        {
            "scheme_id": scheme_id,
            "group_id": target.group_id,
            "split_role": "target_test",
            "cv_fold": [np.nan] * 3,
        }
    )
    assignments = pd.concat([source_assignments, target_assignments], ignore_index=True)
    scheme = pd.Series({"scheme_id": scheme_id, "task_id": "source_only_test"})
    return scheme, frame, assignments, source, target


def _ids(rows: pd.DataFrame) -> set[str]:
    if not isinstance(rows, pd.DataFrame) or "observation_id" not in rows:
        raise SourceSelectionCheckError("Fitting rows lack observation identifiers")
    values = list(rows.observation_id.astype(str))
    if not values or len(values) != len(set(values)):
        raise SourceSelectionCheckError("Empty or repeated fitting observation IDs")
    return set(values)


def check_source_folds(folds, source_ids: set[str], target_ids: set[str]) -> None:
    if len(folds) != 5:
        raise SourceSelectionCheckError("Expected five source-validation folds")
    held_out = []
    for train, validation in folds:
        train_ids = _ids(train)
        validation_ids = _ids(validation)
        if (train_ids | validation_ids) != source_ids:
            raise SourceSelectionCheckError("Folds omit or add source observations")
        if train_ids & validation_ids:
            raise SourceSelectionCheckError("Fitting and validation rows overlap")
        if target_ids & (train_ids | validation_ids):
            raise SourceSelectionCheckError("Target measurement in source selection")
        if len(train_ids) != 8 or len(validation_ids) != 2:
            raise SourceSelectionCheckError("Unexpected fitting/validation fold sizes")
        held_out.extend(validation_ids)
    if len(held_out) != 10 or len(set(held_out)) != 10:
        raise SourceSelectionCheckError("Source-validation folds are incomplete")


def check_final_fit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_ids: set[str],
    target_ids: set[str],
) -> None:
    if _ids(train) != source_ids:
        raise SourceSelectionCheckError("Final fitting differs from source population")
    if _ids(test) != target_ids:
        raise SourceSelectionCheckError("Final target prediction population differs")
    if set(train.protein_id) & set(test.protein_id):
        raise SourceSelectionCheckError("Source and target proteins are not separated")


def _run_case(name: str, module, scheme_id, source_protein, target_protein):
    scheme, frame, assignments, source, target = _sample(
        scheme_id, source_protein, target_protein
    )
    source_ids, target_ids = _ids(source), _ids(target)
    observed = {"selections": 0, "fittings": 0, "predictions": 0}

    def selection(folds, *args, **kwargs):
        check_source_folds(folds, source_ids, target_ids)
        observed["selections"] += 1
        if name == "baseline_evaluation":
            return 0.1, 0.2, [{"alpha": 0.1, "cv_fold": 0, "validation_mae": 0.2}]
        if name == "sequence_plm_evaluation":
            return (
                9,
                0.1,
                0.2,
                [
                    {
                        "hidden_state_index": 9,
                        "alpha": 0.1,
                        "cv_fold": 0,
                        "validation_mae": 0.2,
                    }
                ],
            )
        return 0.1, 0.2, [{"alpha": 0.1, "cv_fold": 0, "validation_mae": 0.2}]

    def fitting(*args, **kwargs):
        if name == "baseline_evaluation":
            # Signature: baseline_id, source rows, target rows, alpha.
            train, test = args[1], args[2]
        else:
            train, test = args[0], args[1]
        check_final_fit(train, test, source_ids, target_ids)
        observed["fittings"] += 1
        return np.full(len(test), -0.75)

    def predictions(test, y_pred, **kwargs):
        if _ids(test) != target_ids:
            raise SourceSelectionCheckError("Predictions did not use target rows")
        if not np.allclose(y_pred, -0.75):
            raise SourceSelectionCheckError("Unexpected test prediction values")
        observed["predictions"] += 1
        return [{"evaluation_role": kwargs["evaluation_role"]}]

    if name == "baseline_evaluation":
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    module,
                    "_scheme_baselines",
                    return_value=("ridge_mutation_identity",),
                )
            )
            stack.enter_context(
                patch.object(
                    module,
                    "_select_alpha_from_folds",
                    side_effect=lambda _id, f: selection(f),
                )
            )
            stack.enter_context(
                patch.object(module, "_predict_baseline", side_effect=fitting)
            )
            stack.enter_context(
                patch.object(module, "_prediction_records", side_effect=predictions)
            )
            stack.enter_context(
                patch.object(module, "_direct_condition_predictions", return_value=[])
            )
            rows, selections = module._source_cv_evaluation(scheme, frame, assignments)
        expected = 1
    elif name == "sequence_plm_evaluation":
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(module, "_select_model_setting", side_effect=selection)
            )
            stack.enter_context(
                patch.object(module, "_fit_predict", side_effect=fitting)
            )
            stack.enter_context(
                patch.object(module, "_prediction_records", side_effect=predictions)
            )
            rows, selections = module._source_cv_evaluation(
                scheme, frame, assignments, np.empty((0,)), np.empty((0,))
            )
        expected = 1
    else:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(module, "_select_setting", side_effect=selection)
            )
            stack.enter_context(
                patch.object(module, "_fit_predict", side_effect=fitting)
            )
            stack.enter_context(
                patch.object(module, "_prediction_records", side_effect=predictions)
            )
            rows, selections = module._source_cv_evaluation(scheme, frame, assignments)
        expected = len(module.MODEL_IDS)
    if observed != {
        "selections": expected,
        "fittings": expected,
        "predictions": expected,
    }:
        raise SourceSelectionCheckError(
            f"Unexpected calls: {observed}; expected {expected} each"
        )
    if len(rows) != expected or not selections:
        raise SourceSelectionCheckError("Selection/prediction records incomplete")
    if not all(row["evaluation_role"] == "target_test" for row in rows):
        raise SourceSelectionCheckError("Incorrect prediction evaluation role")
    return {
        "model_source": name,
        "scheme_id": scheme_id,
        "status": "PASS",
        "source_fold_count": 5,
        "final_source_rows": 10,
        "selection_calls": expected,
        "fitting_calls": expected,
    }


def check_transform_inputs():
    base = pd.DataFrame({"concentration": [2.0, 4.0, 8.0]})
    result = baseline._fit_ridge_transform(base, include_log2_concentration=True)
    if not np.isclose(result.concentration_mean, 2.0):
        raise SourceSelectionCheckError("Baseline concentration mean differs")

    features = pd.DataFrame(
        {"concentration": [2.0, 4.0, 8.0], "sasa_angstrom2": [1.0, 2.0, 3.0]}
    )
    for column in structure.STRUCTURAL_NUMERIC_COLUMNS:
        if column not in features:
            features[column] = [1.0, 2.0, 3.0]
    transform = structure._fit_transform(
        features, include_structure=True, condition_mode="log2_concentration"
    )
    if not np.isclose(transform.concentration_mean, 2.0):
        raise SourceSelectionCheckError("Structural concentration mean differs")
    if not np.allclose(transform.structural_means, 2.0):
        raise SourceSelectionCheckError("Structural feature means differ")

    # Embedding dimensions are small because _fit_transform only measures
    # training-site differences; no fitted model or pretrained weights are loaded.
    mutant = np.zeros((4, 4, 5), dtype=np.float32)
    mutant[:, 0, :] = np.asarray([1.0, 2.0, 3.0, 900.0])[:, None]
    wild_type = np.zeros((2, 4, 270, 5), dtype=np.float32)
    inputs = pd.DataFrame(
        {
            "protein_id": ["NDM-1"] * 3,
            "assay_position": [1, 2, 3],
            "embedding_row": [0, 1, 2],
            "concentration": [2.0, 4.0, 8.0],
        }
    )
    emb = plm._fit_transform(
        inputs,
        mutant,
        wild_type,
        condition_mode="log2_concentration",
        hidden_state_index=9,
    )
    if not np.allclose(emb.embedding_mean, 2.0):
        raise SourceSelectionCheckError("ESM-C site means include wrong rows")
    if not np.isclose(emb.concentration_mean, 2.0):
        raise SourceSelectionCheckError("ESM-C concentration mean differs")
    return {
        "baseline": "PASS",
        "experimental_structure": "PASS",
        "sequence_plm": "PASS",
    }


def run_checks(repo: Path):
    paths = {
        name: repo / "src/betalactamase_transfer" / f"{name}.py" for name, _ in MODULES
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SourceSelectionCheckError(f"Missing required sources: {missing}")
    sha = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths.values()
    }
    rows = []
    for name, module in MODULES:
        for scheme_id, source_protein, target_protein in SCHEMES:
            try:
                row = _run_case(name, module, scheme_id, source_protein, target_protein)
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "model_source": name,
                        "scheme_id": scheme_id,
                        "status": "FAIL",
                        "details": str(exc),
                    }
                )
    try:
        transform = check_transform_inputs()
    except Exception as exc:
        transform = {"status": "FAIL", "details": str(exc)}
    summary = {
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] == "FAIL" for row in rows),
        "transforms": transform,
        "source_sha256": sha,
        "simulated_fitting_only": True,
        "historic_model_training_verified": False,
        "limitation": (
            "These checks intercept fitting calls using simulated rows and "
            "exercise three feature-statistic calculations. They do not "
            "reconstruct historical fitting or model checkpoints."
        ),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if repo == output or repo in output.parents:
        parser.error("Output directory must be outside the repository")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty or absent")
    rows, summary = run_checks(repo)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "source_ridge_checks.csv", index=False)
    (output / "source_ridge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["SOURCE-PROTEIN RIDGE FITTING CHECKS", ""]
    lines.extend(
        f"{row['status']}: {row['model_source']} / {row['scheme_id']}" for row in rows
    )
    lines.extend(
        [
            "",
            "Feature-statistic calculations:",
            str(summary["transforms"]),
            "",
            "Limitations:",
            summary["limitation"],
        ]
    )
    (output / "source_ridge_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return (
        0
        if summary["failed"] == 0 and "FAIL" not in summary["transforms"].values()
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
