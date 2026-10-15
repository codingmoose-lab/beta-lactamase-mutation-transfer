"""Generate Figure 2: measured mutation effects in NDM-1 and VIM-2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]

PAIR_FILE = (
    ROOT / "data/interim/baseline_evaluation/homologue_analysis/"
    "matched_homologue_predictions.csv"
)
POSITION_FILE = (
    ROOT / "data/interim/matched_homologue_model_analysis/"
    "matched_homologue_position_summary.csv"
)
POSITION_MAP_FILE = (
    ROOT / "data/interim/residue_evidence/homologue_position_correspondence.csv"
)

COMPONENT_DIR = ROOT / "figures/components/figure_2"
MAIN_DIR = ROOT / "figures/main"

NDM_BLUE = "#4F86A6"
VIM_GOLD = "#D6B23C"
RUST = "#C76B4F"
GREY = "#A8A8A8"
DARK_GREY = "#777777"
LIGHT_GREY = "#D7DADD"
VERY_LIGHT_GREY = "#E9E7E2"
BLACK = "#292929"
WHITE = "#FFFFFF"

MINIMUM_SCORE = -4.0
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261005


def configure() -> None:
    """Set figure typography."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(direction="out")


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def save_component(fig: plt.Figure, stem: str) -> None:
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
        }
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(COMPONENT_DIR / f"{stem}.{suffix}", **kwargs)


def save_main(fig: plt.Figure) -> None:
    """Save the composite at its exact final physical dimensions."""
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    stem = MAIN_DIR / "figure_2_homologue_effects"

    # Do not use bbox_inches="tight" here. It changes the physical canvas
    # and makes manuscript-wide figure sizing less predictable.
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)


def load_pairs() -> pd.DataFrame:
    p = pd.read_csv(PAIR_FILE)

    required = {
        "direction",
        "homologue_position_id",
        "wild_type",
        "mutant",
        "source_effect",
        "target_effect",
        "minimum_score_class",
    }
    missing = required - set(p.columns)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")

    directions = list(p["direction"].drop_duplicates())
    if "NDM-1_to_VIM-2" in directions:
        p = p[p["direction"].eq("NDM-1_to_VIM-2")].copy()
    else:
        p = p[p["direction"].eq(directions[0])].copy()

    # The NDM-1 -> VIM-2 direction gives one copy of every measured pair.
    p = p.rename(
        columns={
            "source_effect": "ndm_effect",
            "target_effect": "vim_effect",
        }
    )

    return p.reset_index(drop=True)


def validate_pairs(p: pd.DataFrame) -> None:
    if len(p) != 1513:
        raise ValueError(f"Expected 1,513 pairs; found {len(p):,}.")

    n_positions = p["homologue_position_id"].nunique()
    if n_positions != 81:
        raise ValueError(f"Expected 81 positions; found {n_positions}.")

    counts = p["minimum_score_class"].value_counts()

    expected = {
        "both_above_minimum": 478,
        "both_at_minimum": 729,
    }
    for key, value in expected.items():
        observed = int(counts.get(key, 0))
        if observed != value:
            raise ValueError(f"Expected {value} rows for {key}; found {observed}.")

    one = int(counts.get("source_at_minimum", 0) + counts.get("target_at_minimum", 0))
    if one != 306:
        raise ValueError(f"Expected 306 one-minimum pairs; found {one}.")

    rho_all = p["ndm_effect"].corr(p["vim_effect"], method="spearman")
    q = p[p["minimum_score_class"].eq("both_above_minimum")]
    rho_above = q["ndm_effect"].corr(q["vim_effect"], method="spearman")

    if not np.isclose(rho_all, 0.6629440756414804, atol=1e-12):
        raise ValueError(f"Unexpected all-pair Spearman value: {rho_all}")

    if not np.isclose(rho_above, 0.310124296805946, atol=1e-12):
        raise ValueError(f"Unexpected above-minimum Spearman value: {rho_above}")


def position_bootstrap_spearman(
    frame: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, int]:
    """Return Spearman estimate and a position-resampled 95% interval."""
    frame = frame.copy()

    point = frame["ndm_effect"].corr(frame["vim_effect"], method="spearman")
    positions = frame["homologue_position_id"].drop_duplicates().to_numpy()

    groups = {
        position: frame[frame["homologue_position_id"].eq(position)]
        for position in positions
    }

    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(replicates):
        sampled = rng.choice(positions, size=len(positions), replace=True)

        # Duplicate sampled positions must remain separate bootstrap groups.
        pieces = []
        for bootstrap_group, position in enumerate(sampled):
            part = groups[position][["ndm_effect", "vim_effect"]].copy()
            part["_bootstrap_group"] = bootstrap_group
            pieces.append(part)

        sample = pd.concat(pieces, ignore_index=True)

        rho = sample["ndm_effect"].corr(
            sample["vim_effect"],
            method="spearman",
        )
        if np.isfinite(rho):
            estimates.append(float(rho))

    if not estimates:
        raise ValueError("No valid bootstrap estimates were produced.")

    lower, upper = np.quantile(estimates, [0.025, 0.975])

    return float(point), float(lower), float(upper), len(estimates)


def minimum_group(p: pd.DataFrame) -> pd.Series:
    result = pd.Series(index=p.index, dtype="object")
    result.loc[p["minimum_score_class"].eq("both_above_minimum")] = "Both above minimum"
    result.loc[p["minimum_score_class"].eq("both_at_minimum")] = "Both at minimum"
    result.loc[
        p["minimum_score_class"].isin(["source_at_minimum", "target_at_minimum"])
    ] = "One at minimum"

    if result.isna().any():
        bad = p.loc[result.isna(), "minimum_score_class"].value_counts()
        raise ValueError(f"Unexpected minimum-score classes:\n{bad}")

    return result


def load_position_values(p: pd.DataFrame) -> pd.DataFrame:
    summary = pd.read_csv(POSITION_FILE)
    mapping = pd.read_csv(POSITION_MAP_FILE)

    if "direction" in summary.columns:
        summary = summary[summary["direction"].eq("NDM-1_to_VIM-2")].copy()

    required_summary = {
        "homologue_position_id",
        "n_substitutions",
        "median_absolute_background_difference",
    }
    required_mapping = {
        "homologue_position_id",
        "ndm_assay_position",
        "vim_assay_position",
        "same_wild_type_residue",
    }

    missing_summary = required_summary - set(summary.columns)
    missing_mapping = required_mapping - set(mapping.columns)

    if missing_summary:
        raise ValueError(f"Missing position-summary columns: {sorted(missing_summary)}")
    if missing_mapping:
        raise ValueError(f"Missing position-map columns: {sorted(missing_mapping)}")

    mapping = mapping[mapping["same_wild_type_residue"].eq(True)].copy()

    d = summary.merge(
        mapping[
            [
                "homologue_position_id",
                "ndm_assay_position",
                "vim_assay_position",
            ]
        ],
        on="homologue_position_id",
        how="inner",
        validate="one_to_one",
    )

    expected_positions = set(p["homologue_position_id"].unique())
    observed_positions = set(d["homologue_position_id"].unique())

    if observed_positions != expected_positions:
        missing = sorted(expected_positions - observed_positions)
        extra = sorted(observed_positions - expected_positions)
        raise ValueError(
            "Position table does not match the 81 plotted positions. "
            f"Missing={missing}; extra={extra}"
        )

    if len(d) != 81:
        raise ValueError(f"Expected 81 position rows; found {len(d)}.")

    # Independently recalculate the plotted quantity from the 1,513 pairs.
    check = (
        p.assign(absolute_difference=(p["ndm_effect"] - p["vim_effect"]).abs())
        .groupby("homologue_position_id", as_index=False)
        .agg(
            recalculated_n=("absolute_difference", "size"),
            recalculated_median=(
                "absolute_difference",
                "median",
            ),
        )
    )

    d = d.merge(
        check,
        on="homologue_position_id",
        how="left",
        validate="one_to_one",
    )

    if not np.array_equal(
        d["n_substitutions"].to_numpy(dtype=int),
        d["recalculated_n"].to_numpy(dtype=int),
    ):
        raise ValueError("Position-level substitution counts do not match.")

    if not np.allclose(
        d["median_absolute_background_difference"].to_numpy(float),
        d["recalculated_median"].to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Stored and recalculated position-level medians do not match.")

    return d.sort_values("ndm_assay_position").reset_index(drop=True)


def draw_panel_a(ax: plt.Axes, p: pd.DataFrame) -> None:
    groups = [
        ("Both at minimum", GREY, 14, 0.72),
        ("One at minimum", RUST, 15, 0.72),
        ("Both above minimum", NDM_BLUE, 14, 0.46),
    ]

    labels = minimum_group(p)

    values = np.concatenate([p["ndm_effect"].to_numpy(), p["vim_effect"].to_numpy()])
    lo = float(np.floor(np.nanmin(values) * 2) / 2)
    hi = float(np.ceil(np.nanmax(values) * 2) / 2)
    pad = max(0.25, (hi - lo) * 0.025)
    lo -= pad
    hi += pad

    ax.plot(
        [lo, hi],
        [lo, hi],
        color=DARK_GREY,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    ax.axvline(
        MINIMUM_SCORE,
        color=LIGHT_GREY,
        linewidth=0.75,
        zorder=0,
    )
    ax.axhline(
        MINIMUM_SCORE,
        color=LIGHT_GREY,
        linewidth=0.75,
        zorder=0,
    )

    for label, color, size, alpha in groups:
        q = p[labels.eq(label)]
        ax.scatter(
            q["ndm_effect"],
            q["vim_effect"],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            rasterized=False,
            zorder=2,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("NDM-1 mutation effect")
    ax.set_ylabel("VIM-2 mutation effect")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=NDM_BLUE,
            markeredgecolor="none",
            label="Both above minimum",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=GREY,
            markeredgecolor="none",
            label="Both at minimum",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=RUST,
            markeredgecolor="none",
            label="One at minimum",
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.985),
        borderaxespad=0,
        handletextpad=0.4,
        labelspacing=0.35,
    )

    clean_axis(ax)


def draw_panel_b(ax: plt.Axes, p: pd.DataFrame) -> None:
    labels = minimum_group(p)
    order = [
        "Both above minimum",
        "Both at minimum",
        "One at minimum",
    ]
    colors = [NDM_BLUE, GREY, RUST]

    counts = [int(labels.eq(label).sum()) for label in order]
    total = sum(counts)

    left = 0
    for label, count, color in zip(order, counts, colors):
        fraction = count / total

        ax.barh(
            [0],
            [fraction],
            left=left,
            height=0.36,
            color=color,
            edgecolor=WHITE,
            linewidth=0.8,
        )

        x = left + fraction / 2
        text_color = WHITE if label != "Both at minimum" else BLACK

        ax.text(
            x,
            0,
            f"{count:,}\n({100 * fraction:.1f}%)",
            ha="center",
            va="center",
            fontsize=6.5,
            color=text_color,
        )

        left += fraction

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.62, 0.62)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of equivalent substitutions")
    ax.set_xticks(
        [0, 0.25, 0.50, 0.75, 1.0],
        ["0", "25", "50", "75", "100%"],
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor="none",
            label=label,
        )
        for label, color in zip(order, colors)
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.30, 0.96),
        ncol=1,
        borderaxespad=0,
        handletextpad=0.55,
        labelspacing=0.30,
    )

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="y", length=0)


def draw_panel_c(
    ax: plt.Axes,
    interval_table: pd.DataFrame,
) -> None:
    d = interval_table.copy()

    y = np.array([1, 0], dtype=float)

    ax.axvline(
        0,
        color=LIGHT_GREY,
        linewidth=0.7,
        zorder=0,
    )

    for yy, row in zip(y, d.itertuples(index=False)):
        xerr = np.array(
            [
                [row.spearman - row.ci_lower],
                [row.ci_upper - row.spearman],
            ]
        )

        ax.errorbar(
            row.spearman,
            yy,
            xerr=xerr,
            fmt="o",
            markersize=5.0,
            color=NDM_BLUE,
            ecolor=DARK_GREY,
            elinewidth=1.0,
            capsize=2.2,
            capthick=0.8,
            zorder=3,
        )

        ax.text(
            row.ci_upper + 0.025,
            yy,
            f"{row.spearman:.2f}",
            va="center",
            ha="left",
            fontsize=7,
        )

    labels = [
        (
            f"All pairs\n"
            f"(n={int(d.iloc[0]['n_pairs']):,}; "
            f"{int(d.iloc[0]['n_positions'])} positions)"
        ),
        (
            f"Both above minimum\n"
            f"(n={int(d.iloc[1]['n_pairs']):,}; "
            f"{int(d.iloc[1]['n_positions'])} positions)"
        ),
    ]

    ax.set_yticks(y, labels)
    ax.set_xlabel("Spearman ρ")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(
        axis="x",
        color=VERY_LIGHT_GREY,
        linewidth=0.6,
        zorder=0,
    )

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="y", length=0)


def draw_panel_d(ax: plt.Axes, d: pd.DataFrame) -> None:
    x = d["ndm_assay_position"].to_numpy(float)
    y = d["median_absolute_background_difference"].to_numpy(float)

    ax.vlines(
        x,
        0,
        y,
        color=LIGHT_GREY,
        linewidth=0.55,
        zorder=1,
    )
    ax.scatter(
        x,
        y,
        s=17,
        color=NDM_BLUE,
        edgecolor=WHITE,
        linewidth=0.35,
        zorder=2,
    )

    ax.set_xlabel("NDM-1 residue position")
    ax.set_ylabel("Median absolute\nmutation-effect difference")
    ax.set_ylim(bottom=0)
    ax.grid(
        axis="y",
        color=VERY_LIGHT_GREY,
        linewidth=0.6,
        zorder=0,
    )

    clean_axis(ax)


def build_value_tables(
    p: pd.DataFrame,
    position_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    above = p[p["minimum_score_class"].eq("both_above_minimum")].copy()

    all_estimate = position_bootstrap_spearman(
        p,
        seed=BOOTSTRAP_SEED,
    )
    above_estimate = position_bootstrap_spearman(
        above,
        seed=BOOTSTRAP_SEED + 1,
    )

    intervals = pd.DataFrame(
        [
            {
                "population": "all_pairs",
                "spearman": all_estimate[0],
                "ci_lower": all_estimate[1],
                "ci_upper": all_estimate[2],
                "n_pairs": len(p),
                "n_positions": p["homologue_position_id"].nunique(),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_valid_replicates": all_estimate[3],
                "resampling_unit": "homologue_position_id",
            },
            {
                "population": "both_above_minimum",
                "spearman": above_estimate[0],
                "ci_lower": above_estimate[1],
                "ci_upper": above_estimate[2],
                "n_pairs": len(above),
                "n_positions": above["homologue_position_id"].nunique(),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_valid_replicates": above_estimate[3],
                "resampling_unit": "homologue_position_id",
            },
        ]
    )

    group_labels = minimum_group(p)
    composition = pd.DataFrame(
        {
            "category": [
                "both_above_minimum",
                "both_at_minimum",
                "one_at_minimum",
            ],
            "n_pairs": [
                int(group_labels.eq("Both above minimum").sum()),
                int(group_labels.eq("Both at minimum").sum()),
                int(group_labels.eq("One at minimum").sum()),
            ],
        }
    )
    composition["fraction"] = composition["n_pairs"] / len(p)

    source_min = int(p["minimum_score_class"].eq("source_at_minimum").sum())
    target_min = int(p["minimum_score_class"].eq("target_at_minimum").sum())

    summary = pd.DataFrame(
        [
            {
                "quantity": "all_pairs",
                "value": len(p),
            },
            {
                "quantity": "all_homologue_positions",
                "value": p["homologue_position_id"].nunique(),
            },
            {
                "quantity": "both_above_minimum_pairs",
                "value": len(above),
            },
            {
                "quantity": "both_above_minimum_positions",
                "value": above["homologue_position_id"].nunique(),
            },
            {
                "quantity": "both_at_minimum_pairs",
                "value": int(group_labels.eq("Both at minimum").sum()),
            },
            {
                "quantity": "one_at_minimum_pairs",
                "value": int(group_labels.eq("One at minimum").sum()),
            },
            {
                "quantity": "ndm_at_minimum_vim_above",
                "value": source_min,
            },
            {
                "quantity": "ndm_above_vim_at_minimum",
                "value": target_min,
            },
            {
                "quantity": "position_rows",
                "value": len(position_values),
            },
        ]
    )

    return intervals, pd.concat(
        [
            summary.assign(table="summary"),
            composition.rename(
                columns={
                    "category": "quantity",
                    "n_pairs": "value",
                }
            )[["quantity", "value"]].assign(table="composition"),
        ],
        ignore_index=True,
    )


def make_components(
    p: pd.DataFrame,
    intervals: pd.DataFrame,
    positions: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 3.05))
    draw_panel_a(ax, p)
    panel_letter(ax, "A")
    save_component(fig, "A_equivalent_substitution_effects")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    draw_panel_b(ax, p)
    panel_letter(ax, "B")
    save_component(fig, "B_minimum_score_composition")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    draw_panel_c(ax, intervals)
    panel_letter(ax, "C")
    save_component(fig, "C_agreement_comparison")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    draw_panel_d(ax, positions)
    panel_letter(ax, "D")
    save_component(fig, "D_position_level_difference")
    plt.close(fig)


def make_composite(
    p: pd.DataFrame,
    intervals: pd.DataFrame,
    positions: pd.DataFrame,
) -> None:
    """Assemble the final 178-mm-wide Figure 2 composite."""

    # Final manuscript canvas.
    width_mm = 178.0
    height_mm = 132.0

    width_inches = width_mm / 25.4
    height_inches = height_mm / 25.4

    fig = plt.figure(
        figsize=(width_inches, height_inches),
        facecolor=WHITE,
    )

    # Fixed manuscript grid.
    #
    # The axes are positioned directly rather than allowing label lengths
    # to change the apparent panel alignment. This gives both columns and
    # both rows common visual boundaries.
    #
    # [left, bottom, width, height] in figure coordinates.
    positions_axes = {
        "A": [0.105, 0.515, 0.350, 0.350],
        "B": [0.590, 0.570, 0.365, 0.245],
        "C": [0.175, 0.125, 0.280, 0.275],
        "D": [0.590, 0.125, 0.365, 0.275],
    }

    ax_a = fig.add_axes(positions_axes["A"])
    ax_b = fig.add_axes(positions_axes["B"])
    ax_c = fig.add_axes(positions_axes["C"])
    ax_d = fig.add_axes(positions_axes["D"])

    draw_panel_a(ax_a, p)
    draw_panel_b(ax_b, p)
    draw_panel_c(ax_c, intervals)
    draw_panel_d(ax_d, positions)

    # Keep the legends created by the panel drawing functions so the
    # composite uses the same legend appearance as the component files.

    heading_fontsize = 8.0
    letter_fontsize = 9.0

    # Common heading anchors.
    left_x = 0.055
    right_x = 0.535
    top_y = 0.965
    bottom_y = 0.475

    heading_specs = [
        (
            "A",
            "Equivalent substitution effects",
            left_x,
            top_y,
        ),
        (
            "B",
            "Minimum-score composition",
            right_x,
            top_y,
        ),
        (
            "C",
            "Agreement with and without minimum-score pairs",
            left_x,
            bottom_y,
        ),
        (
            "D",
            "Position-level mutation-effect differences",
            right_x,
            bottom_y,
        ),
    ]

    for letter, heading, x, y in heading_specs:
        fig.text(
            x,
            y,
            letter,
            ha="left",
            va="top",
            fontsize=letter_fontsize,
            fontweight="bold",
            color=BLACK,
        )
        fig.text(
            x + 0.035,
            y,
            heading,
            ha="left",
            va="top",
            fontsize=heading_fontsize,
            fontweight="normal",
            color=BLACK,
        )

    save_main(fig)
    plt.close(fig)


def main() -> None:
    configure()

    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_DIR.mkdir(parents=True, exist_ok=True)

    p = load_pairs()
    validate_pairs(p)

    positions = load_position_values(p)
    intervals, summary = build_value_tables(p, positions)

    intervals.to_csv(
        COMPONENT_DIR / "figure_2_correlation_intervals.csv",
        index=False,
    )

    summary.to_csv(
        COMPONENT_DIR / "figure_2_values.csv",
        index=False,
    )

    positions[
        [
            "homologue_position_id",
            "ndm_assay_position",
            "vim_assay_position",
            "n_substitutions",
            "median_absolute_background_difference",
        ]
    ].to_csv(
        COMPONENT_DIR / "figure_2_position_values.csv",
        index=False,
    )

    make_components(p, intervals, positions)
    make_composite(p, intervals, positions)

    print("Figure 2 generated.")
    print()
    print("Correlation estimates:")
    print(intervals.to_string(index=False))
    print()
    print("Minimum-score counts:")
    print(
        minimum_group(p)
        .value_counts()
        .reindex(
            [
                "Both above minimum",
                "Both at minimum",
                "One at minimum",
            ]
        )
        .to_string()
    )
    print()
    print(
        "Position-level median absolute difference range: "
        f"{positions['median_absolute_background_difference'].min():.3f}"
        " to "
        f"{positions['median_absolute_background_difference'].max():.3f}"
    )


if __name__ == "__main__":
    main()
