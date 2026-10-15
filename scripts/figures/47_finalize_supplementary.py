from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Keep backend selection before importing pyplot.
# ruff: noqa: I001
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "figures/main"
SUPP = ROOT / "figures/supplementary"
AUDITS = ROOT / "audits"

MANIFEST = AUDITS / "main_figures_PROTECTED_before.sha256"

BLUE = "#4F86A6"
RUST = "#C76B4F"
GOLD = "#D6B23C"
DARK = "#303030"
GRID = "#E8E5DF"
WHITE = "#FFFFFF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def protected_files() -> dict[str, str]:
    if not MANIFEST.is_file():
        raise RuntimeError("Main-figure protection manifest is missing.")

    expected = {}

    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue

        digest, relative_path = line.split(None, 1)
        expected[relative_path.strip()] = digest

    if not expected:
        raise RuntimeError("Protection manifest is empty.")

    return expected


def verify_main_figures() -> None:
    expected = protected_files()

    actual_paths = {
        str(path.relative_to(ROOT)) for path in MAIN.rglob("*") if path.is_file()
    }

    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        additional = sorted(actual_paths - set(expected))

        raise RuntimeError(
            "Main-figure file inventory changed.\n"
            f"Missing: {missing}\n"
            f"Additional: {additional}"
        )

    for relative_path, expected_hash in expected.items():
        actual_hash = sha256(ROOT / relative_path)

        if actual_hash != expected_hash:
            raise RuntimeError(f"Protected main figure changed: {relative_path}")

    print(f"PASS: {len(expected)} protected main-figure files unchanged.")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#BDBDBD",
            "axes.linewidth": 0.8,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, stem: str) -> None:
    SUPP.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        SUPP / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor=WHITE,
    )

    fig.savefig(
        SUPP / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor=WHITE,
    )

    plt.close(fig)


def backup_existing(stem: str, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("pdf", "png", "svg"):
        source = SUPP / f"{stem}.{extension}"

        if source.is_file():
            shutil.copy2(
                source,
                backup_dir / source.name,
            )


def generate_s7() -> None:
    configure_plotting()

    original = np.array([0.658297, 0.680002])
    extended = np.array([0.664879, 0.669492])

    original_epochs = [195, 181]
    extended_epochs = [200, 240]

    x = np.arange(2)
    width = 0.31

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.18,
        top=0.78,
    )

    bars_original = ax.bar(
        x - width / 2,
        original,
        width,
        color=BLUE,
        edgecolor="none",
        label="Original maximum: 200 epochs",
    )

    bars_extended = ax.bar(
        x + width / 2,
        extended,
        width,
        color=RUST,
        edgecolor="none",
        label="Extended maximum: 400 epochs",
    )

    for bars, epochs in (
        (bars_original, original_epochs),
        (bars_extended, extended_epochs),
    ):
        for bar, epoch in zip(bars, epochs):
            ax.annotate(
                f"Epoch {epoch}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=DARK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "VIM-2 2022\nIteration 1",
            "VIM-2 2022\nIteration 2",
        ]
    )

    ax.set_ylabel("Best validation MAE")

    ax.set_ylim(0, 0.79)

    ax.yaxis.grid(
        True,
        color=GRID,
        linewidth=0.7,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.16),
        ncol=2,
        frameon=False,
        fontsize=9,
        columnspacing=1.5,
        borderaxespad=0,
    )

    save_figure(
        fig,
        "S7_extended_epoch",
    )

    print("PASS: regenerated Supplementary Figure S7.")


def audit_s10() -> None:
    source = (
        ROOT
        / "data/interim/biological_error_analysis"
        / "biological_reference_associations.csv"
    )

    if not source.is_file():
        raise FileNotFoundError(f"Missing biological-association data: {source}")

    frame = pd.read_csv(source)

    report_path = AUDITS / "supplementary_S10_association_audit.txt"

    with report_path.open("w") as handle:
        handle.write("SUPPLEMENTARY FIGURE S10 DATA AUDIT\n")

        handle.write("====================================\n\n")

        handle.write(f"Source: {source}\n")
        handle.write(f"Rows: {len(frame)}\n\n")

        handle.write("COLUMNS\n")
        handle.write("-------\n")

        for column in frame.columns:
            handle.write(f"{column}: {frame[column].dtype}\n")

        handle.write("\nFIRST 30 ROWS\n")
        handle.write("-------------\n")

        handle.write(frame.head(30).to_string(index=False))

        handle.write("\n\nUNIQUE VALUES\n")
        handle.write("-------------\n")

        for column in frame.columns:
            if frame[column].dtype == "object":
                values = frame[column].dropna().unique()

                handle.write(f"\n{column} ({len(values)} values)\n")

                for value in values[:100]:
                    handle.write(f"  {value}\n")

    print("PASS: S10 data inspected.")

    print(f"Report: {report_path.relative_to(ROOT)}")

    print(
        "S10 has not been overwritten. "
        "Its 18 associations require verification "
        "against the saved data."
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Regenerate S7 after backing up existing files.",
    )

    args = parser.parse_args()

    AUDITS.mkdir(parents=True, exist_ok=True)

    verify_main_figures()

    audit_s10()

    if not args.apply:
        print("DRY RUN: no figures modified. Use --apply to regenerate S7.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_dir = AUDITS / f"supplementary_final_backup_{timestamp}"

    backup_existing(
        "S7_extended_epoch",
        backup_dir,
    )

    generate_s7()

    verify_main_figures()

    print("FINAL RESULT: main figures unchanged.")
    print(f"Backup: {backup_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
