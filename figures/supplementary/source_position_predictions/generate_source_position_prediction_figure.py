"""Plot source-position-withheld ESM-C comparisons from saved numerical tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
INPUT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures/supplementary/S12_source_position_predictions.pdf"

MODELS = (
    ("ESM-C site-difference Ridge", "ESM-C Ridge"),
    ("source substitution mean", "Substitution mean"),
    ("source identity Ridge", "Identity Ridge"),
    ("source mutant mean", "Mutant mean"),
    ("source assay mean", "Source assay mean"),
)
DIRECTIONS = (
    ("NDM-1_to_VIM-2", "#4F86A6", "NDM-1 → VIM-2"),
    ("VIM-2_to_NDM-1", "#C76B4F", "VIM-2 → NDM-1"),
)
POPULATIONS = (
    ("all_shared_wild_type_pairs", "All substitutions"),
    ("target_above_minimum", "Target above −4"),
    ("both_above_minimum", "Both above −4"),
)


def _one_row(table: pd.DataFrame, condition: pd.Series) -> pd.Series:
    selected = table.loc[condition]
    if len(selected) != 1:
        raise ValueError(f"Expected one numerical row; found {len(selected)}")
    return selected.iloc[0]


def main() -> None:
    plt.switch_backend("Agg")
    auc = pd.read_csv(INPUT / "minimum_score_roc_auc.csv")
    groups = pd.read_csv(INPUT / "target_score_stratified_comparisons.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#818181",
            "xtick.color": "#4c5357",
            "ytick.color": "#4c5357",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 3.55), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0]
    for index, (direction, color, _) in enumerate(DIRECTIONS):
        for y, (method, _) in zip(np.arange(len(MODELS))[::-1], MODELS):
            row = _one_row(
                auc,
                auc.direction.eq(direction) & auc.method.eq(method),
            )
            x = float(row["roc_auc_minimum"])
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(
                    f"Invalid minimum-score ROC AUC: {direction}, {method}"
                )
            ax.scatter(
                x,
                y + (0.14 if index == 0 else -0.14),
                s=47,
                color=color,
                marker="o" if index == 0 else "s",
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
    ax.axvline(0.5, color="#b3b3b3", lw=1, ls="--")
    ax.set_yticks(np.arange(len(MODELS))[::-1], [m[1] for m in MODELS])
    ax.set_xlim(0.4, 0.96)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("ROC AUC (target analysis score = −4)")
    ax.grid(axis="x", color="#ececec", lw=0.6)
    ax.set_title(
        "A   Minimum-score discrimination",
        loc="left",
        weight="bold",
        fontsize=11,
        pad=12,
    )

    ax = axes[1]
    for index, (direction, color, _) in enumerate(DIRECTIONS):
        for y, (population, _) in zip(np.arange(len(POPULATIONS))[::-1], POPULATIONS):
            row = _one_row(
                groups,
                groups.direction.eq(direction)
                & groups.population.eq(population)
                & groups.comparator.eq("source_assay_substitution_mean"),
            )
            mean, lo, hi = (
                float(row[name])
                for name in (
                    "paired_mae_gain",
                    "position_ci_lower",
                    "position_ci_upper",
                )
            )
            if (
                not all(np.isfinite(value) for value in (mean, lo, hi))
                or not lo <= mean <= hi
            ):
                raise ValueError(
                    f"Invalid MAE difference or interval: {direction}, {population}"
                )
            yy = y + (0.13 if index == 0 else -0.13)
            ax.plot([lo, hi], [yy, yy], color=color, lw=2, solid_capstyle="round")
            for x in (lo, hi):
                ax.plot([x, x], [yy - 0.06, yy + 0.06], color=color, lw=1)
            ax.scatter(
                [mean],
                [yy],
                s=49,
                color=color,
                marker="o" if index == 0 else "s",
                edgecolors="white",
                linewidths=0.65,
                zorder=3,
            )
    ax.axvline(0, color="#666666", lw=1, ls="--")
    ax.set_yticks(np.arange(len(POPULATIONS))[::-1], [p[1] for p in POPULATIONS])
    ax.set_xlim(-0.085, 0.72)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="x", color="#ececec", lw=0.6)
    ax.set_xlabel("MAE improvement (reference − ESM-C)")
    ax.set_title(
        "B   Prediction error by score group",
        loc="left",
        weight="bold",
        fontsize=11,
        pad=12,
    )
    for ax in axes:
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="y", pad=7)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color=DIRECTIONS[0][1],
                label=DIRECTIONS[0][2],
                markersize=6,
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="None",
                color=DIRECTIONS[1][1],
                label=DIRECTIONS[1][2],
                markersize=6,
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.87, bottom=0.22, wspace=0.52)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.11)
    plt.close(fig)
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
