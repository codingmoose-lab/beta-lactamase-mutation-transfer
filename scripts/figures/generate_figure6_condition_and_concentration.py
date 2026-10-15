#!/usr/bin/env python3
"""Generate the final Figure 6 visual composition from audited plotted values.

The four audit CSVs are the numerical contract for this figure. This script
changes presentation only; it does not fit models or recalculate study results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
AUDITS = ROOT / "audits"
MAIN = ROOT / "figures" / "main"
COMP = ROOT / "figures" / "components" / "figure_6"

A_CSV = AUDITS / "figure_6_panel_A_values.csv"
B_CSV = AUDITS / "figure_6_panel_B_values.csv"
C_CSV = AUDITS / "figure_6_panel_C_values.csv"
D_CSV = AUDITS / "figure_6_panel_D_values.csv"

OUT_STEM = "figure_6_condition_and_concentration"

# Warm manuscript palette requested for Figure 6.
DEEP_RED = "#A94438"
RUST = "#C76B4F"
CORAL = "#E29A7F"
GOLD = "#D6B23C"
BLUE = "#4F86A6"
CHARCOAL = "#292929"
MID_GREY = "#777777"
LIGHT_GREY = "#D7DADD"
WARM_GREY = "#E9E7E2"
PALE_GOLD = "#F7F1D8"
WHITE = "#FFFFFF"

MODEL_ORDER = [
    "Experimental reference",
    "ESM-C",
    "ESM-2",
    "Structure descriptors",
]
MODEL_COLOR = {
    "Experimental reference": CHARCOAL,
    "ESM-C": RUST,
    "ESM-2": CORAL,
    "Structure descriptors": BLUE,
}
MODEL_MARKER = {
    "Experimental reference": "D",
    "ESM-C": "o",
    "ESM-2": "s",
    "Structure descriptors": "^",
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    }
)


def read_contract() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in (A_CSV, B_CSV, C_CSV, D_CSV) if not p.is_file()]
    if missing:
        names = "\n".join(str(p.relative_to(ROOT)) for p in missing)
        raise FileNotFoundError(f"Missing audited Figure 6 value files:\n{names}")

    a = pd.read_csv(A_CSV)
    b = pd.read_csv(B_CSV)
    c = pd.read_csv(C_CSV)
    d = pd.read_csv(D_CSV)

    assert set(a["condition"]) == {"AMP", "CTX", "MEM"}
    assert set(b["direction"]) == {"25°C → 37°C", "37°C → 25°C"}
    assert sorted(c["concentration"].unique().tolist()) == [
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        256.0,
    ]
    assert d["mutation"].nunique() == 4
    assert d.groupby("mutation")["concentration"].nunique().eq(7).all()
    return a, b, c, d


def clean_axes(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=WARM_GREY, linewidth=0.65, zorder=0)


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=CHARCOAL,
    )


def draw_panel_a(ax: plt.Axes, a: pd.DataFrame) -> None:
    """Warm MAE matrix: compact view of antibiotic holdout results."""
    conditions = ["AMP", "CTX", "MEM"]
    values = np.array(
        [
            [
                float(
                    a.loc[
                        (a["display_model"] == model) & (a["condition"] == condition),
                        "mae",
                    ].iloc[0]
                )
                for condition in conditions
            ]
            for model in MODEL_ORDER
        ]
    )

    cmap = LinearSegmentedColormap.from_list(
        "figure6_warm_mae",
        ["#FFF8DC", GOLD, CORAL, DEEP_RED],
    )
    # The matrix is descriptive: lower MAE is lighter, higher MAE is deeper red.
    image = ax.imshow(
        values,
        cmap=cmap,
        vmin=float(values.min()),
        vmax=float(values.max()),
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    midpoint = (float(values.min()) + float(values.max())) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = WHITE if value > midpoint + 0.12 else CHARCOAL
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=text_color,
            )

    cbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.075,
        pad=0.16,
        aspect=18,
    )
    cbar.set_label("MAE", labelpad=2)
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(length=2.2, width=0.55, labelsize=5.8)
    panel_letter(ax, "A", x=-0.28, y=1.07)


def draw_direction(
    ax: plt.Axes,
    y: float,
    start_x: float,
    end_x: float,
    color: str,
    label: str,
) -> None:
    arrow = FancyArrowPatch(
        (start_x, y),
        (end_x, y),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.6,
        color=color,
        shrinkA=11,
        shrinkB=11,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    ax.text(
        (start_x + end_x) / 2,
        y + 0.07,
        label,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=color,
    )


def draw_panel_b(ax: plt.Axes, b: pd.DataFrame) -> None:
    """Temperature-change schematic with exact MAE values beside each direction."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left = (0.18, 0.67)
    right = (0.82, 0.67)
    for x, label in [(left[0], "25°C"), (right[0], "37°C")]:
        circle = Circle(
            (x, left[1]),
            radius=0.075,
            facecolor=WHITE,
            edgecolor=CHARCOAL,
            linewidth=1.1,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            left[1],
            label,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=CHARCOAL,
        )

    draw_direction(ax, 0.76, left[0], right[0], RUST, "25°C → 37°C")
    draw_direction(ax, 0.58, right[0], left[0], GOLD, "37°C → 25°C")

    # Exact model MAEs in a compact two-column block.
    x_label = 0.08
    x_fwd = 0.60
    x_rev = 0.88
    ax.text(x_label, 0.38, "Model", fontsize=6.2, color=MID_GREY)
    ax.text(x_fwd, 0.38, "→37°C", fontsize=6.2, ha="center", color=RUST)
    ax.text(x_rev, 0.38, "→25°C", fontsize=6.2, ha="center", color=GOLD)

    y0 = 0.31
    dy = 0.075
    for idx, model in enumerate(MODEL_ORDER):
        y = y0 - idx * dy
        fwd = float(
            b.loc[
                (b["display_model"] == model) & (b["direction"] == "25°C → 37°C"),
                "mae",
            ].iloc[0]
        )
        rev = float(
            b.loc[
                (b["display_model"] == model) & (b["direction"] == "37°C → 25°C"),
                "mae",
            ].iloc[0]
        )
        ax.scatter(
            [x_label - 0.025],
            [y + 0.004],
            s=19,
            marker=MODEL_MARKER[model],
            color=MODEL_COLOR[model],
            edgecolor=WHITE,
            linewidth=0.35,
            zorder=3,
        )
        ax.text(x_label, y, model, fontsize=6.0, ha="left", va="center")
        ax.text(x_fwd, y, f"{fwd:.2f}", fontsize=6.0, ha="center", va="center")
        ax.text(x_rev, y, f"{rev:.2f}", fontsize=6.0, ha="center", va="center")

    ax.text(
        0.74,
        0.06,
        "MAE",
        fontsize=6.0,
        ha="center",
        va="center",
        color=MID_GREY,
    )
    panel_letter(ax, "B", x=-0.08, y=1.03)


def draw_panel_c(ax: plt.Axes, c: pd.DataFrame) -> None:
    """Concentration holdout MAE with endpoint regions and log2 dose spacing."""
    concentrations = [4, 8, 16, 32, 64, 128, 256]
    x = np.log2(np.asarray(concentrations, dtype=float))

    ax.axvspan(x[0] - 0.22, x[0] + 0.22, color=PALE_GOLD, zorder=0)
    ax.axvspan(x[-1] - 0.22, x[-1] + 0.22, color=PALE_GOLD, zorder=0)

    for model in MODEL_ORDER:
        subset = c.loc[c["display_model"] == model].sort_values("concentration").copy()
        ax.plot(
            np.log2(subset["concentration"].to_numpy(dtype=float)),
            subset["mae"].to_numpy(dtype=float),
            color=MODEL_COLOR[model],
            marker=MODEL_MARKER[model],
            markersize=4.5,
            linewidth=1.35,
            markeredgecolor=WHITE,
            markeredgewidth=0.35,
            label=model,
            zorder=3,
        )

    ax.set_xticks(x, [str(v) for v in concentrations])
    ax.set_xlabel("AMP concentration (µg/mL)")
    ax.set_ylabel("MAE")
    ax.set_ylim(bottom=0.15)
    clean_axes(ax, grid=True)

    ax.text(
        x[0],
        2.05,
        "Endpoint",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=GOLD,
    )
    ax.text(
        (x[1] + x[-2]) / 2,
        2.05,
        "Interpolation",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=MID_GREY,
    )
    ax.text(
        x[-1],
        2.05,
        "Endpoint",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=GOLD,
    )

    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0.2,
    )
    panel_letter(ax, "C", x=-0.07, y=1.07)


def draw_panel_d(
    fig: plt.Figure,
    slot,
    d: pd.DataFrame,
) -> list[plt.Axes]:
    """Four measured trajectories using a warm progression by effect-range quantile."""
    mutations = (
        d[["selection_quantile", "mutation", "effect_range"]]
        .drop_duplicates()
        .sort_values("selection_quantile")
    )
    mutation_order = mutations["mutation"].tolist()
    colors = [GOLD, CORAL, RUST, DEEP_RED]

    sub = slot.subgridspec(1, 4, wspace=0.18)
    axes: list[plt.Axes] = []

    y_min = float(d["y_true"].min())
    y_max = float(d["y_true"].max())
    y_pad = max(0.18, 0.05 * (y_max - y_min))
    x_ticks = np.log2(np.asarray([4, 16, 64, 256], dtype=float))

    for idx, (mutation, color) in enumerate(zip(mutation_order, colors, strict=True)):
        ax = fig.add_subplot(sub[0, idx])
        axes.append(ax)
        q = d.loc[d["mutation"] == mutation].sort_values("concentration").copy()
        x = np.log2(q["concentration"].to_numpy(dtype=float))
        y = q["y_true"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.45,
            marker="o",
            markersize=4.2,
            markeredgecolor=WHITE,
            markeredgewidth=0.35,
        )
        ax.axhline(0, color=LIGHT_GREY, linewidth=0.7, zorder=0)
        ax.set_title(mutation, fontsize=7.0, pad=5)
        ax.set_xticks(x_ticks, ["4", "16", "64", "256"])
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        clean_axes(ax, grid=False)

        if idx == 0:
            ax.set_ylabel("Measured mutation effect")
        else:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)

        ax.set_xlabel("AMP (µg/mL)")

    panel_letter(axes[0], "D", x=-0.28, y=1.09)
    return axes


def save_component(fig: plt.Figure, stem: str) -> None:
    COMP.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        path = COMP / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)


def make_components(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    d: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    draw_panel_a(ax, a)
    save_component(fig, "A_antibiotic_mae_matrix")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    draw_panel_b(ax, b)
    save_component(fig, "B_temperature_transfer")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 2.75))
    draw_panel_c(ax, c)
    save_component(fig, "C_concentration_transfer")
    plt.close(fig)

    fig = plt.figure(figsize=(6.8, 2.45))
    slot = fig.add_gridspec(1, 1)[0, 0]
    draw_panel_d(fig, slot, d)
    save_component(fig, "D_measured_trajectories")
    plt.close(fig)


def make_main(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    d: pd.DataFrame,
) -> plt.Figure:
    # 178 mm square: within the NAR double-column maximum width.
    width_in = 178 / 25.4
    fig = plt.figure(figsize=(width_in, width_in))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.92, 1.10, 0.96],
        width_ratios=[1.08, 0.92],
        hspace=0.48,
        wspace=0.38,
        left=0.105,
        right=0.985,
        bottom=0.075,
        top=0.975,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    draw_panel_a(ax_a, a)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_b(ax_b, b)

    ax_c = fig.add_subplot(gs[1, :])
    draw_panel_c(ax_c, c)

    draw_panel_d(fig, gs[2, :], d)
    return fig


def write_audit(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    d: pd.DataFrame,
) -> None:
    lines = [
        "FIGURE 6 FINAL VISUAL AUDIT",
        "=" * 72,
        "PASS: final figure reads audited plotted-value CSVs only",
        "PASS: no model fitting or result recalculation performed",
        f"PASS: panel A rows={len(a)}; conditions={sorted(a['condition'].unique())}",
        f"PASS: panel B rows={len(b)}; directions={sorted(b['direction'].unique())}",
        (
            "PASS: panel C concentrations="
            f"{sorted(c['concentration'].unique().tolist())}"
        ),
        (
            "PASS: panel D mutations="
            + str(
                d[["selection_quantile", "mutation"]]
                .drop_duplicates()
                .sort_values("selection_quantile")["mutation"]
                .tolist()
            )
        ),
        "PASS: panel A uses warm MAE matrix; lower MAE is lighter",
        "PASS: panel B uses circles/arrows only as temperature-direction notation",
        "PASS: panel C preserves interpolation and endpoint distinction",
        "PASS: panel D shows measured observations only",
        (
            "PASS: red/rust dominant, gold secondary, "
            "blue limited to structure descriptors"
        ),
        "PASS: Arial requested; panel letters 9 pt bold",
        "PASS: 178 mm main figure width",
        "",
        "OUTPUTS",
    ]
    for ext in ("svg", "pdf", "png"):
        path = MAIN / f"{OUT_STEM}.{ext}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(
            f"{path.relative_to(ROOT)} | "
            f"{path.stat().st_size:,} bytes | "
            f"sha256={digest}"
        )

    (AUDITS / "figure_6_final_visual_audit.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    COMP.mkdir(parents=True, exist_ok=True)

    a, b, c, d = read_contract()
    make_components(a, b, c, d)

    fig = make_main(a, b, c, d)
    for ext in ("svg", "pdf", "png"):
        path = MAIN / f"{OUT_STEM}.{ext}"
        kwargs = {}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        print(f"WROTE {path}")
    plt.close(fig)

    write_audit(a, b, c, d)
    print(f"WROTE {AUDITS / 'figure_6_final_visual_audit.txt'}")


if __name__ == "__main__":
    main()
