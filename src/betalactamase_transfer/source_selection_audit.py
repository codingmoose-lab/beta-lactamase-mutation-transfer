"""Read-only checks of source-protein model-selection records and partitions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

SCHEMES = ("ndm1_to_vim2_amp_source_cv5", "vim2_to_ndm1_amp_source_cv5")
MODEL_SELECTION_FILES = (
    "data/interim/baseline_evaluation/baseline_model_selection.csv",
    "data/interim/sequence_plm_evaluation/sequence_plm_model_selection.csv",
    "data/interim/experimental_structure_evaluation/"
    "experimental_structure_model_selection.csv",
)
TRAINING_FILES = (
    "data/interim/residue_graph_neural_evaluation/"
    "residue_graph_neural_training_history.csv",
    "data/interim/residue_geometry_neural_evaluation/"
    "residue_geometry_neural_training_history.csv",
    "data/interim/sequence_residue_graph_neural_evaluation/"
    "sequence_residue_graph_training_history.csv",
    "data/interim/atomic_local_voxel_evaluation/"
    "atomic_local_voxel_training_history.csv",
)
ASSIGNMENTS = "data/interim/evaluation_partitions/group_assignments.csv"


class SourceSelectionAuditError(ValueError):
    """A required source-selection record is invalid."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def audit_source_selection(repo_root: Path) -> tuple[pd.DataFrame, dict]:
    """Check recorded folds and selection roles; code-level review remains necessary."""
    path = repo_root / ASSIGNMENTS
    if not path.exists():
        raise SourceSelectionAuditError(f"Missing partition assignments: {path}")
    assignments = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {"scheme_id", "group_id", "split_role", "cv_fold"}
    if not expected.issubset(assignments.columns):
        raise SourceSelectionAuditError("Group assignments lack required columns")
    checks: list[dict] = []
    hashes = {ASSIGNMENTS: _hash(path)}
    for scheme in SCHEMES:
        selected = assignments.loc[assignments.scheme_id.eq(scheme)]
        if selected.empty:
            raise SourceSelectionAuditError(f"No assignment rows for {scheme}")
        source = selected.loc[selected.split_role.eq("source_model_selection")]
        target = selected.loc[selected.split_role.eq("target_test")]
        folds = set(source.cv_fold)
        overlap = set(source.group_id).intersection(target.group_id)
        other_roles = set(selected.split_role).difference(
            {"source_model_selection", "target_test"}
        )
        passes = (
            not overlap
            and not other_roles
            and len(folds) == 5
            and folds == {"0", "1", "2", "3", "4"}
            and not source.group_id.duplicated().any()
            and not target.group_id.duplicated().any()
            and target.cv_fold.eq("").all()
        )
        checks.append(
            {
                "check": "source_target_partition",
                "file": ASSIGNMENTS,
                "scheme_id": scheme,
                "status": "PASS" if passes else "FAIL",
                "source_groups": source.group_id.nunique(),
                "target_groups": target.group_id.nunique(),
                "source_fold_count": len(folds),
                "details": "No fit/test group overlap; source uses five folds"
                if passes
                else "Partition roles, group identity, or fold counts differ",
            }
        )
    for relative in MODEL_SELECTION_FILES + TRAINING_FILES:
        file_path = repo_root / relative
        if not file_path.is_file():
            checks.append(
                {
                    "check": "model_selection_record",
                    "file": relative,
                    "scheme_id": "both",
                    "status": "REVIEW",
                    "details": (
                        "Output file absent; cannot verify source selection records"
                    ),
                }
            )
            continue
        hashes[relative] = _hash(file_path)
        with file_path.open(encoding="utf-8", newline="") as handle:
            column_names = set(next(csv.reader(handle)))
        if "scheme_id" not in column_names:
            checks.append(
                {
                    "check": "model_selection_record",
                    "file": relative,
                    "scheme_id": "both",
                    "status": "REVIEW",
                    "details": "No scheme_id column; manual record review required",
                }
            )
            continue
        cols = [
            name
            for name in (
                "scheme_id",
                "selection_scope",
                "fit_role",
                "source_cv_fold",
                "cv_fold",
                "evaluation_iteration",
                "model_id",
                "selected",
                "epoch",
                "validation_mae",
            )
            if name in column_names
        ]
        selection = pd.read_csv(
            file_path, usecols=cols, dtype=str, keep_default_na=False
        )
        for scheme in SCHEMES:
            part = selection.loc[selection.scheme_id.eq(scheme)]
            if part.empty:
                checks.append(
                    {
                        "check": "model_selection_record",
                        "file": relative,
                        "scheme_id": scheme,
                        "status": "REVIEW",
                        "details": "No records for this cross-protein scheme",
                    }
                )
                continue
            if "selection_scope" in part:
                values = set(part.selection_scope)
                good = "source_cv" in values and "selected_setting" in values
                details = f"Recorded selection_scope values: {sorted(values)}"
            elif "fit_role" in part:
                values = set(part.fit_role)
                good = (
                    "source_cv_epoch_selection" in values and "selected_epoch" in values
                )
                details = f"Recorded fit_role values: {sorted(values)}"
            else:
                good = False
                details = "Neither selection_scope nor fit_role is recorded"
            if "evaluation_iteration" in part:
                good = good and set(part.evaluation_iteration) == {"target"}
            checks.append(
                {
                    "check": "model_selection_record",
                    "file": relative,
                    "scheme_id": scheme,
                    "status": "PASS" if good else "REVIEW",
                    "rows": len(part),
                    "details": details,
                }
            )
    report = pd.DataFrame(checks).fillna("")
    summary = {
        "input_sha256": hashes,
        "passed": int(report.status.eq("PASS").sum()),
        "failed": int(report.status.eq("FAIL").sum()),
        "review": int(report.status.eq("REVIEW").sum()),
        "source_only_model_selection_fully_proved": False,
        "limitation": (
            "Recorded selection roles and partition assignments are evidence, "
            "but they cannot alone prove how every fitted model used the records. "
            "Source fitting functions and checkpoint construction still require review."
        ),
    }
    return report, summary


def run_audit(repo_root: Path, output_dir: Path) -> dict:
    if output_dir.resolve().is_relative_to(repo_root.resolve()):
        raise SourceSelectionAuditError(
            "Reports must be written outside the repository"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SourceSelectionAuditError("Output directory is not empty")
    details, summary = audit_source_selection(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "source_selection_record_checks.csv", index=False)
    (output_dir / "source_selection_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
