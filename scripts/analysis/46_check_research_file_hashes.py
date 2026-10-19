#!/usr/bin/env python3
"""Compare research file SHA-256 values before and after validation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PATHS = (
    "data/raw",
    "data/interim",
    "data/processed",
    "figures",
    "src",
    "scripts",
    "tests",
    "docs",
    "README.md",
    "pyproject.toml",
)
EXCLUDE = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def snapshot(repo: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for name in PATHS:
        candidate = repo / name
        if candidate.is_file():
            items = [candidate]
        elif candidate.is_dir():
            items = candidate.rglob("*")
        else:
            continue
        for path in items:
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in EXCLUDE for part in path.relative_to(repo).parts):
                continue
            files[str(path.relative_to(repo))] = hash_file(path)
    return dict(sorted(files.items()))


def compare(left: dict[str, str], right: dict[str, str]) -> dict[str, list[str]]:
    a, b = set(left), set(right)
    return {
        "added": sorted(b - a),
        "removed": sorted(a - b),
        "changed_content": sorted(p for p in a & b if left[p] != right[p]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (args.snapshot is None) == (args.compare is None):
        parser.error("Use exactly one of --snapshot or --compare")
    repo = args.repo_root.resolve()
    try:
        if not repo.is_dir():
            raise ValueError(f"Research repository does not exist: {repo}")
        current = snapshot(repo)
        if args.snapshot:
            dest = args.snapshot.resolve()
            if repo == dest or repo in dest.parents or dest.exists():
                raise ValueError("Snapshot must be a new file outside the repository")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps({"root": str(repo), "sha256": current}, indent=2) + "\n"
            )
            print(f"SHA-256 snapshot: {len(current)} files; {dest}")
            return 0
        reference = json.loads(args.compare.read_text(encoding="utf-8"))
        if reference.get("root") != str(repo) or not isinstance(
            reference.get("sha256"), dict
        ):
            raise ValueError("Comparison snapshot belongs to a different repository")
        findings = compare(reference["sha256"], current)
        lines = [
            "RESEARCH FILE CONTENT COMPARISON",
            f"Repository: {repo}",
            f"Files checked: {len(current)}",
        ]
        for category, items in findings.items():
            lines.append(f"{category}: {len(items)}")
            lines.extend(f"  {item}" for item in items)
        report_text = "\n".join(lines) + "\n"
        print(report_text)
        if args.report:
            target = args.report.resolve()
            if repo == target or repo in target.parents:
                raise ValueError("Report must be outside the repository")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(report_text, encoding="utf-8")
        return 0 if all(not part for part in findings.values()) else 3
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FILE CONTENT COMPARISON FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
