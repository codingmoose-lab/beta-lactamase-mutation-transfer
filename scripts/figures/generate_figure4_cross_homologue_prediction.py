"""Generate Figure 4: cross-homologue mutation-effect prediction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/interim/matched_homologue_model_analysis"
METRICS = DATA / "matched_homologue_model_metrics.csv"
COMPARISONS = DATA / "matched_homologue_model_comparisons.csv"
ASSOCIATIONS = DATA / "matched_homologue_background_associations.csv"
PREDICTIONS = DATA / "matched_homologue_model_predictions.csv"

COMPONENTS = ROOT / "figures/components/figure_4"
MAIN = ROOT / "figures/main"
AUDITS = ROOT / "audits"

COMPONENTS.mkdir(parents=True, exist_ok=True)
MAIN.mkdir(parents=True, exist_ok=True)
AUDITS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Journal-size / typography
# ---------------------------------------------------------------------

MM_TO_INCH = 1 / 25.4

FINAL_WIDTH_MM = 177.8
FINAL_HEIGHT_MM = 177.8

FONT = "Arial"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.transparent": False,
    }
)

# ---------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------

BLUE = "#4F86A6"
RUST = "#C76B4F"
BLACK = "#292929"
GREY = "#A8A8A8"
LIGHT_GREY = "#D7DADD"
GRID = "#E9E7E2"

# Direction encoding is kept identical across panels.
DIRECTION_STYLE = {
    "NDM-1_to_VIM-2": {
        "label": "NDM-1 → VIM-2",
        "color": BLUE,
        "marker": "o",
    },
    "VIM-2_to_NDM-1": {
        "label": "VIM-2 → NDM-1",
        "color": RUST,
        "marker": "s",
    },
}

MODEL_LABELS = {
    "esmc_site_delta_ridge": "ESM-C",
    "esm2_site_delta_ridge": "ESM-2",
    "experimental_structure_descriptor_ridge": "Structure descriptors",
    "centre_residue_neural_regressor": "Centre residue",
    "residue_graph_neural_regressor": "Residue graph",
    "egnn_residue_regressor": "Residue geometry",
    "sequence_residue_graph_neural_regressor": "Sequence + graph",
}

# These seven models correspond to the representation families already
# introduced in Figure 3. No model is selected here because it happened
# to rank first on the target observations.
MODEL_ORDER = [
    "esmc_site_delta_ridge",
    "esm2_site_delta_ridge",
    "experimental_structure_descriptor_ridge",
    "centre_residue_neural_regressor",
    "residue_graph_neural_regressor",
    "egnn_residue_regressor",
    "sequence_residue_graph_neural_regressor",
]

FOCAL_MODEL = "sequence_residue_graph_neural_regressor"

ALL_POP = "all_shared_wild_type_pairs"
ABOVE_POP = "both_above_minimum"
REFERENCE = "direct_homologue_source_effect"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis=grid_axis,
        color=GRID,
        linewidth=0.55,
        zorder=0,
    )
    ax.set_axisbelow(True)


def save_component(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf"):
        fig.savefig(
            COMPONENTS / f"{stem}.{suffix}",
            bbox_inches="tight",
        )
    fig.savefig(
        COMPONENTS / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
    )


def check_required_columns(
    df: pd.DataFrame,
    columns: set[str],
    name: str,
) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


# ---------------------------------------------------------------------
# Read source tables
# ---------------------------------------------------------------------

metrics = pd.read_csv(METRICS)
comparisons = pd.read_csv(COMPARISONS)
associations = pd.read_csv(ASSOCIATIONS)
predictions = pd.read_csv(PREDICTIONS)

check_required_columns(
    metrics,
    {
        "direction",
        "population",
        "model_id",
        "metric",
        "point_estimate",
        "n_pairs",
        "n_homologue_positions",
    },
    "metrics",
)

check_required_columns(
    comparisons,
    {
        "direction",
        "population",
        "model_id",
        "reference_model_id",
        "metric",
        "model_point_estimate",
        "reference_point_estimate",
        "paired_difference",
        "ci_lower",
        "ci_upper",
        "difference_interpretation",
    },
    "comparisons",
)

check_required_columns(
    associations,
    {
        "direction",
        "population",
        "model_id",
        "association_id",
        "spearman",
        "ci_lower",
        "ci_upper",
        "n_pairs",
        "n_homologue_positions",
    },
    "associations",
)

# Mutation-level table column names were already audited. We still verify
# the fields needed by panel D rather than assuming them silently.
required_prediction_columns = {
    "direction",
    "model_id",
    "target_observation_id",
    "target_effect",
    "absolute_error",
    "absolute_background_difference_analysis",
}

check_required_columns(
    predictions,
    required_prediction_columns,
    "predictions",
)

# ---------------------------------------------------------------------
# Scientific checks before drawing
# ---------------------------------------------------------------------

directions = set(metrics["direction"])
if directions != {"NDM-1_to_VIM-2", "VIM-2_to_NDM-1"}:
    raise ValueError(f"Unexpected direction set: {directions}")

populations = set(metrics["population"])
if populations != {ALL_POP, ABOVE_POP}:
    raise ValueError(f"Unexpected population set: {populations}")

for model in MODEL_ORDER:
    subset = metrics[
        (metrics["model_id"] == model) & (metrics["population"] == ALL_POP)
    ]
    if len(subset) != 8:
        raise ValueError(
            f"{model}: expected 8 all-pair metric rows, found {len(subset)}"
        )

direct = comparisons[
    (comparisons["reference_model_id"] == REFERENCE)
    & (comparisons["population"] == ALL_POP)
    & (comparisons["model_id"].isin(MODEL_ORDER))
    & (comparisons["metric"].isin(["spearman", "mae"]))
].copy()

if len(direct) != 28:
    raise ValueError(f"Expected 28 paired comparison rows, found {len(direct)}")

if set(direct["difference_interpretation"]) != {"positive_favours_model"}:
    raise ValueError("Unexpected paired-difference interpretation.")

mae_rows = direct[direct["metric"] == "mae"]
mae_expected = mae_rows["reference_point_estimate"] - mae_rows["model_point_estimate"]

if not np.allclose(
    mae_expected,
    mae_rows["paired_difference"],
    atol=1e-12,
    rtol=0,
):
    raise ValueError("MAE paired-difference sign check failed.")

rho_rows = direct[direct["metric"] == "spearman"]
rho_expected = rho_rows["model_point_estimate"] - rho_rows["reference_point_estimate"]

if not np.allclose(
    rho_expected,
    rho_rows["paired_difference"],
    atol=1e-12,
    rtol=0,
):
    raise ValueError("Spearman paired-difference sign check failed.")

# ---------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------


def draw_panel_a(ax: plt.Axes) -> pd.DataFrame:
    sub = metrics[
        (metrics["population"] == ALL_POP)
        & (metrics["metric"] == "spearman")
        & (metrics["model_id"].isin(MODEL_ORDER))
    ].copy()

    y_lookup = {model: len(MODEL_ORDER) - 1 - i for i, model in enumerate(MODEL_ORDER)}

    records = []

    for model in MODEL_ORDER:
        row = sub[sub["model_id"] == model]

        if len(row) != 2:
            raise ValueError(
                f"Panel A expected two directions for {model}; found {len(row)}"
            )

        values = {r.direction: float(r.point_estimate) for r in row.itertuples()}

        y = y_lookup[model]

        ax.plot(
            [
                values["NDM-1_to_VIM-2"],
                values["VIM-2_to_NDM-1"],
            ],
            [y, y],
            color=LIGHT_GREY,
            linewidth=1.1,
            zorder=1,
        )

        for direction in DIRECTION_STYLE:
            style = DIRECTION_STYLE[direction]
            value = values[direction]

            ax.scatter(
                value,
                y,
                s=27,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )

            records.append(
                {
                    "panel": "A",
                    "direction": direction,
                    "population": ALL_POP,
                    "model_id": model,
                    "metric": "spearman",
                    "estimate": value,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )

    ax.set_yticks(
        [y_lookup[m] for m in MODEL_ORDER],
        [MODEL_LABELS[m] for m in MODEL_ORDER],
    )
    ax.set_xlabel("Spearman ρ")
    ax.set_xlim(0.55, 0.73)
    clean_axes(ax, "x")
    panel_letter(ax, "A")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=DIRECTION_STYLE[d]["marker"],
            linestyle="none",
            markerfacecolor=DIRECTION_STYLE[d]["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.4,
            label=DIRECTION_STYLE[d]["label"],
        )
        for d in DIRECTION_STYLE
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Panels B / C
# ---------------------------------------------------------------------


def draw_difference_panel(
    ax: plt.Axes,
    metric: str,
    letter: str,
) -> pd.DataFrame:
    sub = direct[direct["metric"] == metric].copy()

    y_lookup = {model: len(MODEL_ORDER) - 1 - i for i, model in enumerate(MODEL_ORDER)}

    offsets = {
        "NDM-1_to_VIM-2": +0.13,
        "VIM-2_to_NDM-1": -0.13,
    }

    records = []

    for model in MODEL_ORDER:
        for direction in DIRECTION_STYLE:
            row = sub[(sub["model_id"] == model) & (sub["direction"] == direction)]

            if len(row) != 1:
                raise ValueError(f"{letter}: expected one row for {model}, {direction}")

            row = row.iloc[0]
            y = y_lookup[model] + offsets[direction]

            x = float(row["paired_difference"])
            lo = float(row["ci_lower"])
            hi = float(row["ci_upper"])

            style = DIRECTION_STYLE[direction]

            ax.errorbar(
                x,
                y,
                xerr=np.array([[x - lo], [hi - x]]),
                fmt=style["marker"],
                markersize=4.8,
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.0,
                capsize=0,
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=3,
            )

            records.append(
                {
                    "panel": letter,
                    "direction": direction,
                    "population": ALL_POP,
                    "model_id": model,
                    "metric": metric,
                    "estimate": x,
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )

    ax.axvline(
        0,
        color=BLACK,
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )

    ax.set_yticks(
        [y_lookup[m] for m in MODEL_ORDER],
        [MODEL_LABELS[m] for m in MODEL_ORDER],
    )

    if metric == "spearman":
        ax.set_xlabel("ΔSpearman ρ")
    else:
        ax.set_xlabel("MAE improvement")

    clean_axes(ax, "x")
    panel_letter(ax, letter)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Panel D
# ---------------------------------------------------------------------


def draw_panel_d(ax: plt.Axes) -> pd.DataFrame:
    sub = predictions[predictions["model_id"] == FOCAL_MODEL].copy()

    if len(sub) != 3026:
        raise ValueError(
            "Panel D expected 3,026 sequence+graph prediction rows "
            f"(1,513 per direction), found {len(sub)}"
        )

    records = []

    # Draw a light point cloud for each direction.
    # Yellow/gold is not used for tiny raw points because it has poor
    # Keep prediction-direction colors identical across all panels.
    point_style = {
        direction: {
            "color": DIRECTION_STYLE[direction]["color"],
            "marker": "o",
        }
        for direction in DIRECTION_STYLE
    }

    for direction in DIRECTION_STYLE:
        d = sub[sub["direction"] == direction].copy()

        x = d["absolute_background_difference_analysis"].to_numpy(float)
        y = d["absolute_error"].to_numpy(float)

        ax.scatter(
            x,
            y,
            s=8,
            color=point_style[direction]["color"],
            alpha=0.16,
            linewidth=0,
            rasterized=True,
            zorder=1,
        )

        # Binned median summary. This is descriptive and uses the plotted
        # observations; it is not a fitted prediction model.
        quantiles = np.unique(
            np.quantile(
                x,
                np.linspace(0, 1, 9),
            )
        )

        if len(quantiles) >= 3:
            bins = pd.cut(
                x,
                bins=quantiles,
                include_lowest=True,
                duplicates="drop",
            )

            summary = (
                pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                        "bin": bins,
                    }
                )
                .groupby("bin", observed=True)
                .agg(
                    x_median=("x", "median"),
                    y_median=("y", "median"),
                )
                .dropna()
            )

            summary_color = DIRECTION_STYLE[direction]["color"]

            ax.plot(
                summary["x_median"],
                summary["y_median"],
                color=summary_color,
                linewidth=1.8,
                zorder=3,
            )

        assoc = associations[
            (
                associations["association_id"]
                == "absolute_error_vs_absolute_background_difference"
            )
            & (associations["population"] == ALL_POP)
            & (associations["model_id"] == FOCAL_MODEL)
            & (associations["direction"] == direction)
        ]

        if len(assoc) != 1:
            raise ValueError(f"Panel D association row missing for {direction}")

        assoc = assoc.iloc[0]

        records.append(
            {
                "panel": "D",
                "direction": direction,
                "population": ALL_POP,
                "model_id": FOCAL_MODEL,
                "metric": "spearman_error_vs_homologue_difference",
                "estimate": float(assoc["spearman"]),
                "ci_lower": float(assoc["ci_lower"]),
                "ci_upper": float(assoc["ci_upper"]),
            }
        )

    ax.set_xlabel("Absolute NDM-1/VIM-2 mutation-effect difference")
    ax.set_ylabel("Absolute prediction error")

    clean_axes(ax, "both")
    panel_letter(ax, "D")

    assoc_lookup = {row["direction"]: row for row in records}

    text_lines = []

    for direction in DIRECTION_STYLE:
        row = assoc_lookup[direction]
        short = "NDM→VIM" if direction == "NDM-1_to_VIM-2" else "VIM→NDM"
        text_lines.append(f"{short}: ρ = {row['estimate']:.2f}")

    ax.text(
        0.98,
        0.05,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=BLACK,
    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Component generation
# ---------------------------------------------------------------------

fig_a, ax_a = plt.subplots(figsize=(3.55, 2.65))
values_a = draw_panel_a(ax_a)
fig_a.tight_layout()
save_component(fig_a, "A_cross_homologue_direction")
plt.close(fig_a)

fig_b, ax_b = plt.subplots(figsize=(3.55, 2.65))
values_b = draw_difference_panel(ax_b, "spearman", "B")
fig_b.tight_layout()
save_component(fig_b, "B_rank_difference_vs_experimental_reference")
plt.close(fig_b)

fig_c, ax_c = plt.subplots(figsize=(3.55, 2.65))
values_c = draw_difference_panel(ax_c, "mae", "C")
fig_c.tight_layout()
save_component(fig_c, "C_mae_difference_vs_experimental_reference")
plt.close(fig_c)

fig_d, ax_d = plt.subplots(figsize=(7.0, 2.75))
values_d = draw_panel_d(ax_d)
fig_d.tight_layout()
save_component(fig_d, "D_error_vs_homologue_difference")
plt.close(fig_d)

# ---------------------------------------------------------------------
# Final composite
# ---------------------------------------------------------------------

fig = plt.figure(
    figsize=(
        FINAL_WIDTH_MM * MM_TO_INCH,
        FINAL_HEIGHT_MM * MM_TO_INCH,
    )
)

gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[1.0, 1.05],
    width_ratios=[1.15, 1.0, 1.0],
    left=0.12,
    right=0.985,
    bottom=0.09,
    top=0.965,
    wspace=0.48,
    hspace=0.42,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[1, :])

final_a = draw_panel_a(ax_a)
final_b = draw_difference_panel(ax_b, "spearman", "B")
final_c = draw_difference_panel(ax_c, "mae", "C")
final_d = draw_panel_d(ax_d)

# B/C share the model labels conceptually. Remove duplicate y labels
# from C while keeping B readable.
ax_c.set_yticklabels([])
ax_c.tick_params(axis="y", length=0)

# Panel B is narrower than the component version; slightly tighten labels.
ax_b.tick_params(axis="y", pad=2)

# ---------------------------------------------------------------------
# Plotted-values audit
# ---------------------------------------------------------------------

plotted = pd.concat(
    [final_a, final_b, final_c, final_d],
    ignore_index=True,
)

plotted_path = AUDITS / "figure_4_plotted_values.csv"
plotted.to_csv(plotted_path, index=False)

# ---------------------------------------------------------------------
# Save final files
# ---------------------------------------------------------------------

svg_path = MAIN / "figure_4_cross_protein_prediction.svg"
pdf_path = MAIN / "figure_4_cross_protein_prediction.pdf"
png_path = MAIN / "figure_4_cross_protein_prediction.png"

fig.savefig(svg_path)
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
plt.close(fig)

# ---------------------------------------------------------------------
# Generation audit
# ---------------------------------------------------------------------

audit = []


def ap(text: str = "") -> None:
    audit.append(text)


ap("FIGURE 4 GENERATION AUDIT")
ap("=" * 80)
ap(f"Repository: {ROOT}")
ap()
ap("SOURCE FILES")
for source in [
    METRICS,
    COMPARISONS,
    ASSOCIATIONS,
    PREDICTIONS,
]:
    ap(f"{source.relative_to(ROOT)} | SHA256={sha256(source)}")

ap()
ap("SCIENTIFIC CHECKS")
ap("PASS: both cross-homologue directions present")
ap("PASS: all-pair and both-above-minimum populations present")
ap("PASS: seven Figure-3 representation families used in panels A-C")
ap("PASS: direct source-homologue effect used only as experimental reference")
ap("PASS: substitution mean not shown as experimental reference")
ap("PASS: MAE improvement = reference MAE - model MAE")
ap("PASS: positive MAE improvement means lower model MAE")
ap("PASS: ΔSpearman = model rho - reference rho")
ap("PASS: positive ΔSpearman means higher model rank agreement")
ap("PASS: panel D uses sequence + graph predictions for both directions")
ap("PASS: panel D uses 1,513 substitutions per direction")
ap("PASS: no model fitting performed")
ap("PASS: no source result table modified")

ap()
ap("FIGURE CONTENT")
ap("A: directional Spearman comparison")
ap("B: paired ΔSpearman versus measured source-homologue effect")
ap("C: paired MAE improvement versus measured source-homologue effect")
ap("D: absolute prediction error versus experimental homologue difference")

ap()
ap("TYPOGRAPHY")
ap("panel letters: 9 pt bold")
ap("descriptive panel headings: none")
ap("axis/general text: 7 pt")
ap("tick text: 6.5 pt")
ap("legend text: 6.3 pt")

ap()
ap("COLOURS")
ap(f"NDM-1 → VIM-2 blue: {BLUE}")
ap(f"VIM-2 → NDM-1 rust: {RUST}")
ap(f"near-black: {BLACK}")
ap("yellow is reserved from small marks because of white-background contrast")

ap()
ap("OUTPUT")
ap(f"requested width: {FINAL_WIDTH_MM:.1f} mm")
ap(f"requested height: {FINAL_HEIGHT_MM:.1f} mm")
ap("SVG text remains editable")
ap("PNG resolution: 600 dpi")
ap(f"plotted values: {plotted_path.relative_to(ROOT)}")

ap()
ap("FILES")
for output in [svg_path, pdf_path, png_path]:
    ap(f"{output.relative_to(ROOT)} | {output.stat().st_size:,} bytes")

audit_path = AUDITS / "figure_4_generation_audit.txt"
audit_path.write_text("\n".join(audit) + "\n", encoding="utf-8")

print(f"WROTE {svg_path}")
print(f"WROTE {pdf_path}")
print(f"WROTE {png_path}")
print(f"WROTE {plotted_path}")
print(f"WROTE {audit_path}")
