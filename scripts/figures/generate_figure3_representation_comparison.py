from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SUMMARY = ROOT / "data/interim/model_performance_summary/model_performance_metrics.csv"

GRAPH_PAIRED = (
    ROOT / "data/interim/residue_graph_neural_evaluation/"
    "residue_graph_neural_paired_comparisons.csv"
)

SEQ_GRAPH_PAIRED = (
    ROOT / "data/interim/sequence_residue_graph_neural_evaluation/"
    "sequence_residue_graph_paired_comparisons.csv"
)

COMPONENT_DIR = ROOT / "figures/components/figure_3"
MAIN_DIR = ROOT / "figures/main"
AUDIT_DIR = ROOT / "audits"

WIDTH_MM = 177.8
HEIGHT_MM = 178.0
MM_PER_INCH = 25.4

SCHEMES = [
    "ndm1_position_nested5",
    "vim2_2020_position_nested5",
    "vim2_2022_mutation_nested5",
]

SCHEME_LABELS = {
    "ndm1_position_nested5": "NDM-1\nunseen position",
    "vim2_2020_position_nested5": "VIM-2 2020\nunseen position",
    "vim2_2022_mutation_nested5": "VIM-2 2022\nunseen mutation",
}

SHORT_SCHEME_LABELS = {
    "ndm1_position_nested5": "NDM-1",
    "vim2_2020_position_nested5": "VIM-2 2020",
    "vim2_2022_mutation_nested5": "VIM-2 2022",
}

MODELS = [
    "esmc_site_delta_ridge",
    "experimental_structure_descriptor_ridge",
    "centre_residue_neural_regressor",
    "residue_graph_neural_regressor",
    "egnn_residue_regressor",
    "atomic_local_voxel_cnn",
    "sequence_residue_graph_neural_regressor",
]

MODEL_LABELS = {
    "esmc_site_delta_ridge": "ESM-C",
    "experimental_structure_descriptor_ridge": "Structure descriptors",
    "centre_residue_neural_regressor": "Centre residue",
    "residue_graph_neural_regressor": "Residue graph",
    "egnn_residue_regressor": "Residue geometry",
    "atomic_local_voxel_cnn": "Atomic voxel",
    "sequence_residue_graph_neural_regressor": "Sequence + graph",
}

D_MODELS = [
    "residue_graph_neural_regressor",
    "egnn_residue_regressor",
    "atomic_local_voxel_cnn",
    "sequence_residue_graph_neural_regressor",
]

D_LABELS = {
    "residue_graph_neural_regressor": "Residue graph",
    "egnn_residue_regressor": "Residue geometry",
    "atomic_local_voxel_cnn": "Atomic voxel",
    "sequence_residue_graph_neural_regressor": "Sequence + graph",
}

# Muted, distinguishable task colors.
TASK_COLORS = {
    "ndm1_position_nested5": "#4F86A6",
    "vim2_2020_position_nested5": "#D6B23C",
    "vim2_2022_mutation_nested5": "#C76B4F",
}

TASK_MARKERS = {
    "ndm1_position_nested5": "o",
    "vim2_2020_position_nested5": "s",
    "vim2_2022_mutation_nested5": "^",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linewidth=0.45, alpha=0.20)
    ax.set_axisbelow(True)


def panel_letter(fig: plt.Figure, x: float, y: float, letter: str) -> None:
    fig.text(
        x,
        y,
        letter,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="top",
    )


def panel_heading(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
) -> None:
    fig.text(
        x,
        y,
        text,
        fontsize=8.0,
        fontweight="normal",
        ha="left",
        va="top",
    )


def save_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    tight: bool,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if tight:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}

    fig.savefig(stem.with_suffix(".svg"), **kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=600, **kwargs)


def require_files() -> None:
    for path in [SUMMARY, GRAPH_PAIRED, SEQ_GRAPH_PAIRED]:
        if not path.exists():
            raise FileNotFoundError(path)


def load_performance() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)

    required = {
        "scheme_id",
        "model_id",
        "spearman",
        "mae",
        "evaluation_iteration",
        "n_observations",
        "n_groups",
        "evaluation_population",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SUMMARY}: {sorted(missing)}")

    df = df[
        df["scheme_id"].isin(SCHEMES)
        & df["model_id"].isin(MODELS)
        & (df["evaluation_iteration"] == "pooled")
        & (df["evaluation_population"] == "experimental_structure_matched_population")
    ].copy()

    expected = len(SCHEMES) * len(MODELS)
    if len(df) != expected:
        counts = (
            df.groupby(["scheme_id", "model_id"]).size().rename("rows").reset_index()
        )
        raise ValueError(
            f"Expected {expected} matched performance rows, found {len(df)}.\n"
            f"{counts.to_string(index=False)}"
        )

    duplicates = df.duplicated(["scheme_id", "model_id"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate scheme/model rows:\n"
            + df.loc[
                duplicates,
                ["scheme_id", "model_id", "evaluation_population"],
            ].to_string(index=False)
        )

    return df


def load_pair_table(
    path: Path,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "scheme_id",
        "model_a",
        "model_b",
        "metric",
        "model_a_value",
        "model_b_value",
        "paired_difference",
        "ci_lower",
        "ci_upper",
        "bootstrap_replicates",
        "bootstrap_valid_replicates",
        "resampling_unit",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out = df[
        df["scheme_id"].isin(SCHEMES)
        & (df["model_a"] == model_a)
        & (df["model_b"] == model_b)
        & df["metric"].isin(["spearman", "mae"])
    ].copy()

    expected = len(SCHEMES) * 2
    if len(out) != expected:
        raise ValueError(
            f"Expected {expected} paired rows in {path}, found {len(out)}."
        )

    if out.duplicated(["scheme_id", "metric"]).any():
        raise ValueError(f"Duplicate paired rows in {path}")

    # The stored paired_difference is already oriented so that positive
    # values favour model_a for both metrics. For MAE this corresponds to
    # model_b MAE - model_a MAE.
    return out


def performance_lookup(
    performance: pd.DataFrame,
    scheme: str,
    model: str,
    metric: str,
) -> float:
    row = performance[
        (performance["scheme_id"] == scheme) & (performance["model_id"] == model)
    ]
    if len(row) != 1:
        raise ValueError(f"Missing unique row for {scheme}, {model}")
    return float(row.iloc[0][metric])


def draw_task_legend(ax: plt.Axes) -> None:
    handles = []
    for scheme in SCHEMES:
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                marker=TASK_MARKERS[scheme],
                linestyle="none",
                markerfacecolor=TASK_COLORS[scheme],
                markeredgecolor=TASK_COLORS[scheme],
                markersize=4.5,
                label=SCHEME_LABELS[scheme].replace("\n", " "),
            )
        )

    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.43, 1.105),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.15,
        borderaxespad=0.0,
    )


def draw_performance_metric(
    ax: plt.Axes,
    performance: pd.DataFrame,
    metric: str,
    show_ylabels: bool,
) -> None:
    clean_axis(ax)

    y = np.arange(len(MODELS))
    offsets = {
        "ndm1_position_nested5": -0.18,
        "vim2_2020_position_nested5": 0.0,
        "vim2_2022_mutation_nested5": 0.18,
    }

    for scheme in SCHEMES:
        values = [
            performance_lookup(performance, scheme, model, metric) for model in MODELS
        ]

        ax.scatter(
            values,
            y + offsets[scheme],
            s=23,
            marker=TASK_MARKERS[scheme],
            facecolor=TASK_COLORS[scheme],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    else:
        ax.set_yticklabels([])

    ax.invert_yaxis()

    if metric == "spearman":
        ax.set_xlabel("Spearman ρ")
        ax.set_xlim(0.48, 0.86)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    elif metric == "mae":
        ax.set_xlabel("MAE")
        ax.set_xlim(0.50, 1.38)
        ax.set_xticks([0.6, 0.8, 1.0, 1.2])
    else:
        raise ValueError(metric)

    ax.tick_params(axis="y", length=0)


def draw_panel_a(
    fig: plt.Figure,
    ax_rho: plt.Axes,
    ax_mae: plt.Axes,
    performance: pd.DataFrame,
) -> None:
    draw_performance_metric(
        ax_rho,
        performance,
        metric="spearman",
        show_ylabels=True,
    )
    draw_performance_metric(
        ax_mae,
        performance,
        metric="mae",
        show_ylabels=False,
    )

    ax_rho.set_title(
        "Rank agreement",
        fontsize=7.0,
        fontweight="normal",
        pad=3,
    )
    ax_mae.set_title(
        "Absolute error",
        fontsize=7.0,
        fontweight="normal",
        pad=3,
    )

    draw_task_legend(ax_rho)


def draw_paired_metric(
    ax: plt.Axes,
    paired: pd.DataFrame,
    metric: str,
    xlabel: str,
) -> None:
    clean_axis(ax)

    y = np.arange(len(SCHEMES))

    for i, scheme in enumerate(SCHEMES):
        row = paired[(paired["scheme_id"] == scheme) & (paired["metric"] == metric)]

        if len(row) != 1:
            raise ValueError(f"Missing paired row: {scheme}, {metric}")

        row = row.iloc[0]

        value = float(row["paired_difference"])
        low = float(row["ci_lower"])
        high = float(row["ci_upper"])

        ax.hlines(
            i,
            low,
            high,
            color=TASK_COLORS[scheme],
            linewidth=1.5,
            zorder=2,
        )
        ax.scatter(
            [value],
            [i],
            s=28,
            marker=TASK_MARKERS[scheme],
            facecolor=TASK_COLORS[scheme],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

    ax.axvline(
        0,
        color="0.35",
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_SCHEME_LABELS[s] for s in SCHEMES])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)


def draw_panel_b(
    ax_rho: plt.Axes,
    ax_mae: plt.Axes,
    paired: pd.DataFrame,
) -> None:
    draw_paired_metric(
        ax_rho,
        paired,
        metric="spearman",
        xlabel="ΔSpearman ρ",
    )
    draw_paired_metric(
        ax_mae,
        paired,
        metric="mae",
        xlabel="MAE improvement",
    )

    ax_rho.set_xlim(-0.015, 0.075)
    ax_mae.set_xlim(-0.035, 0.070)

    ax_rho.set_title(
        "Rank agreement",
        fontsize=7.0,
        fontweight="normal",
        pad=3,
    )
    ax_mae.set_title(
        "Absolute error",
        fontsize=7.0,
        fontweight="normal",
        pad=3,
    )

    ax_mae.set_yticklabels([])


def draw_panel_c(
    ax_rho: plt.Axes,
    ax_mae: plt.Axes,
    paired: pd.DataFrame,
) -> None:
    draw_paired_metric(
        ax_rho,
        paired,
        metric="spearman",
        xlabel="ΔSpearman ρ",
    )
    draw_paired_metric(
        ax_mae,
        paired,
        metric="mae",
        xlabel="MAE improvement",
    )

    ax_rho.set_xlim(-0.105, 0.035)
    ax_mae.set_xlim(-0.175, 0.090)

    ax_rho.set_title(
        "Rank agreement",
        fontsize=7.0,
        fontweight="normal",
        pad=3,
    )
    ax_mae.set_title(
        "Absolute error",
        fontsize=7.0,
        fontweight="normal",
        pad=3,
    )

    ax_mae.set_yticklabels([])


def normalize_for_glyph(
    performance: pd.DataFrame,
    scheme: str,
    model: str,
) -> float:
    values = np.array(
        [performance_lookup(performance, scheme, m, "spearman") for m in D_MODELS],
        dtype=float,
    )

    value = performance_lookup(
        performance,
        scheme,
        model,
        "spearman",
    )

    lo = float(values.min())
    hi = float(values.max())

    if np.isclose(lo, hi):
        return 0.5

    # Keep the centre away from zero so small differences remain visible
    # without implying that the centre is Spearman rho = 0.
    scaled = (value - lo) / (hi - lo)
    return 0.34 + 0.66 * scaled


def draw_triangle_glyph(
    ax: plt.Axes,
    performance: pd.DataFrame,
    model: str,
) -> None:
    angles = np.deg2rad([90, 210, 330])
    unit = np.column_stack([np.cos(angles), np.sin(angles)])

    # Reference triangle.
    ref = np.vstack([unit, unit[0]])
    ax.plot(
        ref[:, 0],
        ref[:, 1],
        color="0.78",
        linewidth=0.75,
        zorder=1,
    )

    for fraction in [0.34, 0.67]:
        ring = np.vstack([unit * fraction, unit[0] * fraction])
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color="0.90",
            linewidth=0.45,
            zorder=1,
        )

    points = []
    for scheme, direction in zip(SCHEMES, unit, strict=True):
        radius = normalize_for_glyph(
            performance,
            scheme,
            model,
        )
        point = direction * radius
        points.append(point)

        ax.scatter(
            [point[0]],
            [point[1]],
            s=19,
            marker=TASK_MARKERS[scheme],
            facecolor=TASK_COLORS[scheme],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    points = np.asarray(points)
    closed = np.vstack([points, points[0]])
    ax.fill(
        closed[:, 0],
        closed[:, 1],
        color="0.45",
        alpha=0.055,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color="0.28",
        linewidth=0.9,
        zorder=2,
    )

    labels = [
        "NDM-1",
        "VIM-2\n2020",
        "VIM-2\n2022",
    ]
    label_positions = [
        (0.0, 1.16, "center", "bottom"),
        (-1.02, -0.68, "right", "center"),
        (1.02, -0.68, "left", "center"),
    ]

    for text, (x, y, ha, va) in zip(
        labels,
        label_positions,
        strict=True,
    ):
        ax.text(
            x,
            y,
            text,
            fontsize=6.5,
            ha=ha,
            va=va,
        )

    ax.set_title(
        D_LABELS[model],
        fontsize=7.0,
        fontweight="normal",
        pad=1,
    )
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.10, 1.32)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_profile_panel(
    ax: plt.Axes,
    performance: pd.DataFrame,
) -> None:
    clean_axis(ax)

    x = np.arange(len(SCHEMES))

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    for model, linestyle, marker in zip(
        D_MODELS,
        line_styles,
        markers,
        strict=True,
    ):
        values = [
            performance_lookup(
                performance,
                scheme,
                model,
                "spearman",
            )
            for scheme in SCHEMES
        ]

        ax.plot(
            x,
            values,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.05,
            markersize=4.0,
            label=D_LABELS[model],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "NDM-1\nunseen position",
            "VIM-2 2020\nunseen position",
            "VIM-2 2022\nunseen mutation",
        ]
    )
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(0.62, 0.84)

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
        columnspacing=1.1,
        handlelength=2.2,
    )


def make_component_a(
    performance: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(6.7, 3.15))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.25,
        right=0.98,
        bottom=0.18,
        top=0.82,
        wspace=0.20,
    )

    ax_rho = fig.add_subplot(gs[0, 0])
    ax_mae = fig.add_subplot(gs[0, 1])

    draw_panel_a(fig, ax_rho, ax_mae, performance)

    save_figure(
        fig,
        COMPONENT_DIR / "figure_3A_representation_performance",
        tight=True,
    )
    plt.close(fig)


def make_component_b(
    paired: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(6.2, 2.55))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.19,
        right=0.98,
        bottom=0.23,
        top=0.88,
        wspace=0.24,
    )

    ax_rho = fig.add_subplot(gs[0, 0])
    ax_mae = fig.add_subplot(gs[0, 1])

    draw_panel_b(ax_rho, ax_mae, paired)

    save_figure(
        fig,
        COMPONENT_DIR / "figure_3B_residue_neighbour_information",
        tight=True,
    )
    plt.close(fig)


def make_component_c(
    paired: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(6.2, 2.55))
    gs = fig.add_gridspec(
        1,
        2,
        left=0.19,
        right=0.98,
        bottom=0.23,
        top=0.88,
        wspace=0.24,
    )

    ax_rho = fig.add_subplot(gs[0, 0])
    ax_mae = fig.add_subplot(gs[0, 1])

    draw_panel_c(ax_rho, ax_mae, paired)

    save_figure(
        fig,
        COMPONENT_DIR / "figure_3C_sequence_plus_graph",
        tight=True,
    )
    plt.close(fig)


def make_component_d_triangle(
    performance: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(6.7, 2.2))
    gs = fig.add_gridspec(
        1,
        4,
        left=0.03,
        right=0.97,
        bottom=0.06,
        top=0.90,
        wspace=0.20,
    )

    for i, model in enumerate(D_MODELS):
        ax = fig.add_subplot(gs[0, i])
        draw_triangle_glyph(ax, performance, model)

    save_figure(
        fig,
        COMPONENT_DIR / "figure_3D_representation_signatures",
        tight=True,
    )
    plt.close(fig)


def make_component_d_profile(
    performance: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 2.55))

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.23,
        top=0.82,
    )

    draw_profile_panel(ax, performance)

    save_figure(
        fig,
        COMPONENT_DIR / "figure_3D_alternative_profile",
        tight=True,
    )
    plt.close(fig)


def make_composite(
    performance: pd.DataFrame,
    graph_paired: pd.DataFrame,
    seq_graph_paired: pd.DataFrame,
) -> None:
    fig = plt.figure(
        figsize=(
            WIDTH_MM / MM_PER_INCH,
            HEIGHT_MM / MM_PER_INCH,
        )
    )

    # A: full-width quantitative overview.
    gs_a = fig.add_gridspec(
        1,
        2,
        left=0.205,
        right=0.965,
        bottom=0.670,
        top=0.855,
        wspace=0.18,
    )
    ax_a_rho = fig.add_subplot(gs_a[0, 0])
    ax_a_mae = fig.add_subplot(gs_a[0, 1])

    # B and C: paired comparisons.
    gs_b = fig.add_gridspec(
        1,
        2,
        left=0.105,
        right=0.475,
        bottom=0.390,
        top=0.555,
        wspace=0.25,
    )
    ax_b_rho = fig.add_subplot(gs_b[0, 0])
    ax_b_mae = fig.add_subplot(gs_b[0, 1])

    gs_c = fig.add_gridspec(
        1,
        2,
        left=0.595,
        right=0.965,
        bottom=0.390,
        top=0.555,
        wspace=0.25,
    )
    ax_c_rho = fig.add_subplot(gs_c[0, 0])
    ax_c_mae = fig.add_subplot(gs_c[0, 1])

    # D: four compact representation signatures.
    gs_d = fig.add_gridspec(
        1,
        4,
        left=0.060,
        right=0.970,
        bottom=0.055,
        top=0.275,
        wspace=0.15,
    )

    draw_panel_a(
        fig,
        ax_a_rho,
        ax_a_mae,
        performance,
    )
    draw_panel_b(
        ax_b_rho,
        ax_b_mae,
        graph_paired,
    )
    draw_panel_c(
        ax_c_rho,
        ax_c_mae,
        seq_graph_paired,
    )

    for i, model in enumerate(D_MODELS):
        ax = fig.add_subplot(gs_d[0, i])
        draw_triangle_glyph(
            ax,
            performance,
            model,
        )

    panel_letter(fig, 0.035, 0.965, "A")
    panel_heading(
        fig,
        0.070,
        0.965,
        "Performance across representation types",
    )

    panel_letter(fig, 0.035, 0.625, "B")
    panel_heading(
        fig,
        0.070,
        0.625,
        "Effect of residue-neighbour information",
    )

    panel_letter(fig, 0.525, 0.625, "C")
    panel_heading(
        fig,
        0.560,
        0.625,
        "Sequence information added to the residue graph",
    )

    panel_letter(fig, 0.035, 0.335, "D")
    panel_heading(
        fig,
        0.070,
        0.335,
        "Relative Spearman performance across prediction tests",
    )

    save_figure(
        fig,
        MAIN_DIR / "figure_3_representation_comparison",
        tight=False,
    )
    plt.close(fig)


def make_composite_profile_alternative(
    performance: pd.DataFrame,
    graph_paired: pd.DataFrame,
    seq_graph_paired: pd.DataFrame,
) -> None:
    fig = plt.figure(
        figsize=(
            WIDTH_MM / MM_PER_INCH,
            HEIGHT_MM / MM_PER_INCH,
        )
    )

    gs_a = fig.add_gridspec(
        1,
        2,
        left=0.205,
        right=0.965,
        bottom=0.665,
        top=0.885,
        wspace=0.18,
    )
    ax_a_rho = fig.add_subplot(gs_a[0, 0])
    ax_a_mae = fig.add_subplot(gs_a[0, 1])

    gs_b = fig.add_gridspec(
        1,
        2,
        left=0.105,
        right=0.475,
        bottom=0.365,
        top=0.535,
        wspace=0.25,
    )
    ax_b_rho = fig.add_subplot(gs_b[0, 0])
    ax_b_mae = fig.add_subplot(gs_b[0, 1])

    gs_c = fig.add_gridspec(
        1,
        2,
        left=0.595,
        right=0.965,
        bottom=0.365,
        top=0.535,
        wspace=0.25,
    )
    ax_c_rho = fig.add_subplot(gs_c[0, 0])
    ax_c_mae = fig.add_subplot(gs_c[0, 1])

    ax_d = fig.add_axes([0.115, 0.065, 0.82, 0.175])

    draw_panel_a(
        fig,
        ax_a_rho,
        ax_a_mae,
        performance,
    )
    draw_panel_b(
        ax_b_rho,
        ax_b_mae,
        graph_paired,
    )
    draw_panel_c(
        ax_c_rho,
        ax_c_mae,
        seq_graph_paired,
    )
    draw_profile_panel(
        ax_d,
        performance,
    )

    panel_letter(fig, 0.035, 0.965, "A")
    panel_heading(
        fig,
        0.070,
        0.965,
        "Performance across representation types",
    )

    panel_letter(fig, 0.035, 0.615, "B")
    panel_heading(
        fig,
        0.070,
        0.615,
        "Effect of residue-neighbour information",
    )

    panel_letter(fig, 0.525, 0.615, "C")
    panel_heading(
        fig,
        0.560,
        0.615,
        "Effect of sequence information on residue-graph prediction",
    )

    panel_letter(fig, 0.035, 0.295, "D")
    panel_heading(
        fig,
        0.070,
        0.295,
        "Relative Spearman performance across prediction tests",
    )

    save_figure(
        fig,
        MAIN_DIR / "figure_3_representation_comparison_profile_alternative",
        tight=False,
    )
    plt.close(fig)


def write_values(
    performance: pd.DataFrame,
    graph_paired: pd.DataFrame,
    seq_graph_paired: pd.DataFrame,
) -> None:
    rows = []

    for _, row in performance.iterrows():
        rows.append(
            {
                "panel": "A/D",
                "scheme_id": row["scheme_id"],
                "model_id": row["model_id"],
                "metric": "spearman",
                "value": row["spearman"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_observations": row["n_observations"],
                "n_groups": row["n_groups"],
                "evaluation_population": row["evaluation_population"],
            }
        )
        rows.append(
            {
                "panel": "A",
                "scheme_id": row["scheme_id"],
                "model_id": row["model_id"],
                "metric": "mae",
                "value": row["mae"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_observations": row["n_observations"],
                "n_groups": row["n_groups"],
                "evaluation_population": row["evaluation_population"],
            }
        )

    for panel, table in [
        ("B", graph_paired),
        ("C", seq_graph_paired),
    ]:
        for _, row in table.iterrows():
            rows.append(
                {
                    "panel": panel,
                    "scheme_id": row["scheme_id"],
                    "model_id": (f"{row['model_a']} vs {row['model_b']}"),
                    "metric": row["metric"],
                    "value": row["paired_difference"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                    "n_observations": np.nan,
                    "n_groups": np.nan,
                    "evaluation_population": (
                        "experimental_structure_matched_population"
                    ),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(
        AUDIT_DIR / "figure_3_plotted_values.csv",
        index=False,
    )


def audit(
    performance: pd.DataFrame,
    graph_paired: pd.DataFrame,
    seq_graph_paired: pd.DataFrame,
) -> None:
    lines = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("FIGURE 3 — FINAL NUMERICAL AND DESIGN AUDIT")
    add("=" * 78)
    add()

    add("A. PERFORMANCE TABLE")
    add("-" * 78)
    add(f"Rows: {len(performance)} (expected {len(SCHEMES) * len(MODELS)})")
    add(
        "Evaluation populations: "
        + ", ".join(
            sorted(performance["evaluation_population"].dropna().unique().tolist())
        )
    )
    add()

    for scheme in SCHEMES:
        add(SCHEME_LABELS[scheme].replace("\n", " "))
        subset = performance[performance["scheme_id"] == scheme]
        for model in MODELS:
            row = subset[subset["model_id"] == model].iloc[0]
            add(
                f"  {MODEL_LABELS[model]:24s} "
                f"rho={row['spearman']:.6f}  "
                f"MAE={row['mae']:.6f}  "
                f"n={int(row['n_observations'])}  "
                f"groups={int(row['n_groups'])}"
            )
        add()

    add("B. RESIDUE GRAPH VS CENTRE RESIDUE")
    add("-" * 78)
    for scheme in SCHEMES:
        for metric in ["spearman", "mae"]:
            row = graph_paired[
                (graph_paired["scheme_id"] == scheme)
                & (graph_paired["metric"] == metric)
            ].iloc[0]
            add(
                f"{SHORT_SCHEME_LABELS[scheme]:12s} "
                f"{metric:8s} "
                f"improvement={row['paired_difference']:+.6f} "
                f"95%=[{row['ci_lower']:+.6f}, "
                f"{row['ci_upper']:+.6f}] "
                f"bootstrap="
                f"{int(row['bootstrap_valid_replicates'])}/"
                f"{int(row['bootstrap_replicates'])}"
            )
    add()

    add("C. SEQUENCE + GRAPH VS RESIDUE GRAPH")
    add("-" * 78)
    for scheme in SCHEMES:
        for metric in ["spearman", "mae"]:
            row = seq_graph_paired[
                (seq_graph_paired["scheme_id"] == scheme)
                & (seq_graph_paired["metric"] == metric)
            ].iloc[0]
            add(
                f"{SHORT_SCHEME_LABELS[scheme]:12s} "
                f"{metric:8s} "
                f"improvement={row['paired_difference']:+.6f} "
                f"95%=[{row['ci_lower']:+.6f}, "
                f"{row['ci_upper']:+.6f}] "
                f"bootstrap="
                f"{int(row['bootstrap_valid_replicates'])}/"
                f"{int(row['bootstrap_replicates'])}"
            )
    add()

    add("CHECKS")
    add("-" * 78)

    checks = []

    checks.append(
        (
            "21 matched model/task rows",
            len(performance) == 21,
        )
    )

    checks.append(
        (
            "single evaluation population in Panel A",
            performance["evaluation_population"].nunique() == 1
            and performance["evaluation_population"].iloc[0]
            == "experimental_structure_matched_population",
        )
    )

    checks.append(
        (
            "six graph-vs-centre paired rows",
            len(graph_paired) == 6,
        )
    )

    checks.append(
        (
            "six sequence+graph-vs-graph paired rows",
            len(seq_graph_paired) == 6,
        )
    )

    checks.append(
        (
            "all paired intervals finite",
            np.isfinite(
                pd.concat(
                    [
                        graph_paired[["ci_lower", "ci_upper"]],
                        seq_graph_paired[["ci_lower", "ci_upper"]],
                    ]
                ).to_numpy(dtype=float)
            ).all(),
        )
    )

    graph_low = graph_paired[["ci_lower", "ci_upper"]].min(axis=1)
    graph_high = graph_paired[["ci_lower", "ci_upper"]].max(axis=1)
    seq_graph_low = seq_graph_paired[["ci_lower", "ci_upper"]].min(axis=1)
    seq_graph_high = seq_graph_paired[["ci_lower", "ci_upper"]].max(axis=1)

    graph_inside = graph_paired["paired_difference"].between(
        graph_low,
        graph_high,
    )
    seq_graph_inside = seq_graph_paired["paired_difference"].between(
        seq_graph_low,
        seq_graph_high,
    )

    checks.append(
        (
            "all paired estimates inside stored intervals",
            bool(graph_inside.all() and seq_graph_inside.all()),
        )
    )

    checks.append(
        (
            "all paired analyses resample evaluation groups",
            set(graph_paired["resampling_unit"]) == {"evaluation_group_id"}
            and set(seq_graph_paired["resampling_unit"]) == {"evaluation_group_id"},
        )
    )

    for label, passed in checks:
        add(f"{'PASS' if passed else 'FAIL'}  {label}")

    add()
    add("OUTPUT GEOMETRY")
    add("-" * 78)
    add(f"Requested width : {WIDTH_MM:.1f} mm")
    add(f"Requested height: {HEIGHT_MM:.1f} mm")
    add("Main composite uses fixed figure dimensions.")
    add("Main SVG text remains editable.")
    add("PNG output requested at 600 dpi.")
    add()
    add("DESIGN")
    add("-" * 78)
    add("Panel A: aligned Spearman-rho and MAE dot plots.")
    add("Panel B: paired graph-versus-centre improvement intervals.")
    add("Panel C: paired sequence+graph-versus-graph improvement intervals.")
    add("Panel D: four triangular Spearman performance signatures.")
    add("Alternative D: connected Spearman profiles across the three prediction tests.")
    add("No full-population baseline is mixed into the matched-population Panel A.")
    add("ESM-2 is not added as another Panel-A row.")
    add("No model training is performed by this script.")
    add("No source result table is modified by this script.")

    report = "\n".join(lines) + "\n"
    path = AUDIT_DIR / "figure_3_generation_audit.txt"
    path.write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    configure()
    require_files()

    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    performance = load_performance()

    graph_paired = load_pair_table(
        GRAPH_PAIRED,
        model_a="residue_graph_neural_regressor",
        model_b="centre_residue_neural_regressor",
    )

    seq_graph_paired = load_pair_table(
        SEQ_GRAPH_PAIRED,
        model_a="sequence_residue_graph_neural_regressor",
        model_b="residue_graph_neural_regressor",
    )

    make_component_a(performance)
    make_component_b(graph_paired)
    make_component_c(seq_graph_paired)
    make_component_d_triangle(performance)
    make_component_d_profile(performance)

    make_composite(
        performance,
        graph_paired,
        seq_graph_paired,
    )

    make_composite_profile_alternative(
        performance,
        graph_paired,
        seq_graph_paired,
    )

    write_values(
        performance,
        graph_paired,
        seq_graph_paired,
    )

    audit(
        performance,
        graph_paired,
        seq_graph_paired,
    )


if __name__ == "__main__":
    main()
