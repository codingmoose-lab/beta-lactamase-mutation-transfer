"""Inspect source-only fitting paths without loading models or fitting weights."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCHEMES = (
    ("ndm1_to_vim2_amp_source_cv5", "NDM-1", "VIM-2"),
    ("vim2_to_ndm1_amp_source_cv5", "VIM-2", "NDM-1"),
)
NEURAL_MODULES = (
    "residue_graph_neural_evaluation",
    "residue_geometry_neural_evaluation",
    "sequence_residue_graph_neural_evaluation",
    "atomic_local_voxel_evaluation",
)
RIDGE_MODULES = (
    "baseline_evaluation",
    "sequence_plm_evaluation",
    "experimental_structure_evaluation",
)


class SourceFittingBoundaryError(ValueError):
    """An inspected source-fitting contract is inconsistent."""


def _function(path: Path, name: str, namespace: dict):
    source = path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise SourceFittingBoundaryError(f"Expected one {name} in {path}")
    selected = matches[0]
    copied = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                selected,
            ],
            type_ignores=[],
        )
    )
    scope = dict(namespace)
    exec(compile(copied, filename=str(path), mode="exec"), scope)
    return scope[name], selected


def _records(scheme: str, source_protein: str, target_protein: str) -> dict:
    source = pd.DataFrame(
        {
            "observation_id": [f"source_{i:02d}" for i in range(10)],
            "protein_id": [source_protein] * 10,
            "score_analysis": np.linspace(-2, 2, 10),
        }
    )
    target = pd.DataFrame(
        {
            "observation_id": [f"target_{i:02d}" for i in range(3)],
            "protein_id": [target_protein] * 3,
            "score_analysis": [999001.0, 999002.0, 999003.0],
        }
    )
    folds = []
    for fold in range(5):
        validation = source.iloc[2 * fold : 2 * fold + 2].copy()
        fitting = source.drop(validation.index).copy()
        folds.append({"fold": fold, "train": fitting, "validation": validation})
    return {
        "scheme_id": scheme,
        "evaluation_iteration": "target",
        "source": source,
        "target": target,
        "reference": target[["observation_id"]].copy(),
        "source_folds": folds,
    }


def _assert_source_rows(rows: pd.DataFrame, source: pd.DataFrame, label: str):
    ids = set(rows.observation_id.astype(str))
    expected = set(source.observation_id.astype(str))
    if not ids.issubset(expected) or not ids:
        raise SourceFittingBoundaryError(
            f"{label} includes a target observation or has no source observations"
        )
    if len(ids) != len(rows):
        raise SourceFittingBoundaryError(f"{label} duplicates source observations")


def _selection_check(path: Path, module_name: str, record: dict) -> dict:
    calls = []
    expected = record["source_folds"]

    def pretend_fit(*args, **kwargs):
        first = 1 if module_name == "residue_graph_neural_evaluation" else 0
        fitting, validation = args[first : first + 2]
        fold = len(calls)
        if fold >= 5:
            raise SourceFittingBoundaryError("More than five source fitting folds")
        _assert_source_rows(fitting, record["source"], "fitting fold")
        _assert_source_rows(validation, record["source"], "validation fold")
        if set(fitting.observation_id).intersection(validation.observation_id):
            raise SourceFittingBoundaryError("Source fitting and validation overlap")
        if set(fitting.observation_id) != set(expected[fold]["train"].observation_id):
            raise SourceFittingBoundaryError(
                "Fitting fold differs from source assignment"
            )
        if set(validation.observation_id) != set(
            expected[fold]["validation"].observation_id
        ):
            raise SourceFittingBoundaryError(
                "Validation fold differs from source assignment"
            )
        calls.append(fold)
        return (
            2 + fold,
            0.25 + fold,
            [{"epoch": 2 + fold, "validation_mae": 0.25 + fold}],
        )

    lookup = {record["scheme_id"]: "none"}
    graph = SimpleNamespace(CONDITION_MODE_BY_SCHEME=lookup)
    namespace = {
        "np": np,
        "pd": pd,
        "_fit_with_validation": pretend_fit,
        "SELECTION_SEED": 23,
        "MODEL_ID": module_name,
        "CONDITION_MODE_BY_SCHEME": lookup,
        "graph_stage": graph,
        "graph_eval": graph,
    }
    function, selected = _function(path, "_selection_for_context", namespace)
    tables = {"nodes": None, "edges": None, "residue_features": None}
    if module_name == "residue_graph_neural_evaluation":
        args = (record, "residue_graph_neural_regressor", tables, None, 8, 2, 4)
    elif module_name == "residue_geometry_neural_evaluation":
        args = (record, tables, None, 8, 2, 4)
    elif module_name == "sequence_residue_graph_neural_evaluation":
        args = (record, tables, None, None, None, None, 8, 2, 4)
    else:
        args = (record, None, {}, None, 8, 2, 4)
    epoch, history = function(*args)
    if calls != list(range(5)) or epoch != 4:
        raise SourceFittingBoundaryError(
            f"Source fold selection differed for {module_name}"
        )
    if not any(row.get("fit_role") == "selected_epoch" for row in history):
        raise SourceFittingBoundaryError("Selected epoch not recorded")
    return {"fold_count": len(calls), "selected_epoch": epoch, "line": selected.lineno}


def _fitting_check(path: Path, module_name: str, record: dict) -> dict:
    if module_name == "residue_graph_neural_evaluation":
        return _graph_final_fit_check(path, module_name, record)
    name = (
        "_fitting_rows_for_context"
        if module_name == "sequence_residue_graph_neural_evaluation"
        else "_fitting_rows"
    )
    namespace = {"pd": pd}
    function, selected = _function(path, name, namespace)
    rows = function(record)
    _assert_source_rows(rows, record["source"], "final fitting")
    if set(rows.observation_id) != set(record["source"].observation_id):
        raise SourceFittingBoundaryError("Final fitting omits source observations")
    return {"final_source_rows": len(rows), "line": selected.lineno}


def _graph_final_fit_check(path: Path, module_name: str, record: dict) -> dict:
    captures = []

    def pretend_predict(*args):
        fitting, test = args[1], args[2]
        _assert_source_rows(fitting, record["source"], "final fitting")
        if set(fitting.observation_id) != set(record["source"].observation_id):
            raise SourceFittingBoundaryError(
                "Graph final fit omits source observations"
            )
        if set(test.observation_id) != set(record["target"].observation_id):
            raise SourceFittingBoundaryError(
                "Graph test observations differ from target"
            )
        captures.append(1)
        return np.zeros(len(test)), [{"epoch": 1, "training_loss": 0.0}]

    namespace = {
        "pd": pd,
        "np": np,
        "CONDITION_MODE_BY_SCHEME": {record["scheme_id"]: "none"},
        "TARGET_COLUMN": "score_analysis",
        "FINAL_REPEAT_SEEDS": (7, 11),
        "_selection_for_context": lambda *args: (4, [{"fit_role": "selected_epoch"}]),
        "_fit_fixed_epochs_and_predict": pretend_predict,
        "_prediction_records": lambda ref, model, preds: (
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    }
    function, selected = _function(path, "_fit_context", namespace)
    function(
        record,
        "residue_graph_neural_regressor",
        {"nodes": None, "edges": None, "residue_features": None},
        None,
        1,
        8,
        2,
        4,
    )
    if captures != [1]:
        raise SourceFittingBoundaryError("Graph final fitting was not reached")
    return {"final_source_rows": len(record["source"]), "line": selected.lineno}


def _transform_check(path: Path, module_name: str) -> int:
    count = 0
    for name in ("_fit_with_validation", "_fit_fixed_epochs_and_predict"):
        _, selected = _function(path, name, {})
        calls = [node for node in ast.walk(selected) if isinstance(node, ast.Call)]
        required = {"_target_transform", "_fit_condition_transform"}
        found = set()
        for call in calls:
            fname = (
                call.func.attr
                if isinstance(call.func, ast.Attribute)
                else (call.func.id if isinstance(call.func, ast.Name) else "")
            )
            if fname in required:
                if not call.args or not isinstance(call.args[0], ast.Name):
                    raise SourceFittingBoundaryError(f"Unexpected input to {fname}")
                if call.args[0].id != "fitting_rows":
                    raise SourceFittingBoundaryError(
                        f"{module_name}/{name}: {fname} uses non-fitting observations"
                    )
                found.add(fname)
        if found != required:
            raise SourceFittingBoundaryError(
                f"{module_name}/{name}: missing fitting-only transformations"
            )
        count += len(found)
    return count


def _ridge_check(path: Path, module_name: str) -> int:
    _, selected = _function(path, "_source_cv_evaluation", {})
    literals = {
        node.value
        for node in ast.walk(selected)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assigns = {
        target.id
        for node in ast.walk(selected)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if not {"source_model_selection", "target_test"}.issubset(literals):
        raise SourceFittingBoundaryError(
            f"{module_name}: missing source and target selection roles"
        )
    if not {"source", "target"}.issubset(assigns):
        raise SourceFittingBoundaryError(
            f"{module_name}: source or target selection missing"
        )
    return selected.lineno


def run_checks(repo_root: Path) -> tuple[list[dict], dict]:
    rows = []
    hashes = {}
    for module in (*NEURAL_MODULES, *RIDGE_MODULES):
        relative = f"src/betalactamase_transfer/{module}.py"
        path = repo_root / relative
        if not path.is_file():
            raise SourceFittingBoundaryError(f"Missing source module: {relative}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if module in NEURAL_MODULES:
            for scheme, source_protein, target_protein in SCHEMES:
                record = _records(scheme, source_protein, target_protein)
                try:
                    folds = _selection_check(path, module, record)
                    fitting = _fitting_check(path, module, record)
                    transforms = _transform_check(path, module)
                    rows.append(
                        {
                            "model_source": module,
                            "scheme_id": scheme,
                            "status": "PASS",
                            "source_fold_count": folds["fold_count"],
                            "final_source_rows": fitting["final_source_rows"],
                            "fitting_transform_call_count": transforms,
                            "selection_line": folds["line"],
                            "fitting_line": fitting["line"],
                            "details": (
                                "Selection and final fitting kept source and "
                                "target rows separate"
                            ),
                        }
                    )
                except Exception as error:
                    rows.append(
                        {
                            "model_source": module,
                            "scheme_id": scheme,
                            "status": "FAIL",
                            "details": str(error),
                        }
                    )
        else:
            try:
                line = _ridge_check(path, module)
                rows.append(
                    {
                        "model_source": module,
                        "scheme_id": "both",
                        "status": "REVIEW",
                        "selection_line": line,
                        "details": (
                            "Source and target selection statements found; "
                            "numerical fitting not simulated"
                        ),
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "model_source": module,
                        "scheme_id": "both",
                        "status": "FAIL",
                        "details": str(error),
                    }
                )
    summary = {
        "passed": sum(r["status"] == "PASS" for r in rows),
        "review": sum(r["status"] == "REVIEW" for r in rows),
        "failed": sum(r["status"] == "FAIL" for r in rows),
        "source_code_sha256": hashes,
        "selection_simulation_uses_synthetic_rows": True,
        "training_executed": False,
        "target_outcomes_used_for_model_selection_proved_absent": False,
        "limitation": (
            "These tests execute selected source-code functions with simulated fitting "
            "operations. They do not run neural training, verify historic checkpoints, "
            "or establish every call in the complete feature-preparation process."
        ),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        parser.error("The output directory must be outside the repository")
    if output.exists() and any(output.iterdir()):
        parser.error("The output directory must be new or empty")
    rows, summary = run_checks(repo)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output / "source_fitting_boundary_checks.csv", index=False
    )
    (output / "source_fitting_boundary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["SOURCE-PROTEIN FITTING BOUNDARY CHECKS", ""]
    for row in rows:
        lines.append(f"{row['status']}: {row['model_source']} / {row['scheme_id']}")
        lines.append(f"  {row['details']}")
    lines.extend(
        [
            "",
            (
                f"PASS={summary['passed']} REVIEW={summary['review']} "
                f"FAIL={summary['failed']}"
            ),
            "",
            "Limitations:",
            summary["limitation"],
        ]
    )
    (output / "source_fitting_boundary_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
