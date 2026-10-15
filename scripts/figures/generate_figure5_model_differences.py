#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path.cwd().resolve()
DATA = (
    ROOT
    / "data/interim/biological_model_difference_analysis"
    / "biological_model_difference_position_summary.csv"
)
COMP = ROOT / "figures/components/figure_5"
MAIN = ROOT / "figures/main"
AUD = ROOT / "audits"
COMP.mkdir(parents=True, exist_ok=True)
MAIN.mkdir(parents=True, exist_ok=True)
AUD.mkdir(parents=True, exist_ok=True)

SCHEMES = {
    "NDM-1": "ndm1_position_nested5",
    "VIM-2": "vim2_2020_position_nested5",
}
PDB = {
    "NDM-1": ("3SPU", ROOT / "data/raw/pdb/3SPU/3SPU.cif"),
    "VIM-2": ("5YD7", ROOT / "data/raw/pdb/5YD7/5YD7.cif"),
}
VALUE = "median_graph_minus_centre_absolute_error"

BLUE = "#D6B23C"
MAGENTA = "#A94438"
DARK = "#292929"
LIGHT = "#D7DADD"
MID = "#F7F1D8"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    df = pd.read_csv(DATA, low_memory=False)
    required = {
        "scheme_id",
        "protein_id",
        "assay_position",
        "n_observations",
        VALUE,
        "selected_pdb_id",
        "selected_pdb_chain",
        "pdb_author_position",
        "pdb_coordinate_present",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    selected = {}
    for protein, scheme in SCHEMES.items():
        x = df[(df["protein_id"] == protein) & (df["scheme_id"] == scheme)].copy()
        x = x[np.isfinite(pd.to_numeric(x[VALUE], errors="coerce"))]
        x["assay_position"] = pd.to_numeric(x["assay_position"], errors="raise").astype(
            int
        )
        x[VALUE] = pd.to_numeric(x[VALUE], errors="raise")
        if x["assay_position"].duplicated().any():
            d = x.loc[x["assay_position"].duplicated(), "assay_position"].tolist()
            raise RuntimeError(f"Duplicate {protein} positions: {d[:10]}")
        selected[protein] = x.sort_values("assay_position").reset_index(drop=True)

    return df, selected


def symmetric_limit(selected: dict[str, pd.DataFrame]) -> float:
    vals = np.concatenate([x[VALUE].to_numpy(float) for x in selected.values()])
    lim = float(np.nanmax(np.abs(vals)))
    if not np.isfinite(lim) or lim <= 0:
        raise RuntimeError("Invalid common error-difference range")
    return lim


def sequence_panel(ax, x: pd.DataFrame, protein: str, lim: float, letter: str):
    pos = x["assay_position"].to_numpy(float)
    val = x[VALUE].to_numpy(float)
    neg = val < 0
    posi = val > 0

    ax.vlines(pos[neg], 0, val[neg], color=BLUE, alpha=0.38, linewidth=0.55, zorder=1)
    ax.vlines(
        pos[posi], 0, val[posi], color=MAGENTA, alpha=0.38, linewidth=0.55, zorder=1
    )
    ax.scatter(
        pos[neg],
        val[neg],
        s=9,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    ax.scatter(
        pos[posi],
        val[posi],
        s=9,
        facecolor=MAGENTA,
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    zero = ~(neg | posi)
    if zero.any():
        ax.scatter(
            pos[zero],
            val[zero],
            s=8,
            facecolor=DARK,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )

    ax.axhline(0, color=DARK, linewidth=0.65, zorder=0)
    ax.set_ylim(-lim * 1.08, lim * 1.08)
    ax.set_xlim(max(0, pos.min() - 5), pos.max() + 5)
    ax.set_xlabel(f"{protein} residue position")
    ax.set_ylabel("Graph − centre\nabsolute error")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.18)
    ax.text(
        -0.12,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_sequence_panels(selected, lim):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), constrained_layout=True)
    sequence_panel(axes[0], selected["NDM-1"], "NDM-1", lim, "A")
    sequence_panel(axes[1], selected["VIM-2"], "VIM-2", lim, "B")
    for ext in ("svg", "pdf", "png"):
        kw = {"dpi": 600} if ext == "png" else {}
        fig.savefig(
            COMP / f"figure_5_AB_residue_error_maps.{ext}", bbox_inches="tight", **kw
        )
    plt.close(fig)


def interpolate_hex(t: float) -> str:
    # Blue -> near-white -> magenta.
    def rgb(h):
        return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))

    def hx(v):
        return "#" + "".join(f"{max(0, min(255, round(q))):02X}" for q in v)

    a, m, b = rgb(BLUE), rgb(MID), rgb(MAGENTA)
    if t <= 0.5:
        u = t / 0.5
        v = tuple(a[i] * (1 - u) + m[i] * u for i in range(3))
    else:
        u = (t - 0.5) / 0.5
        v = tuple(m[i] * (1 - u) + b[i] * u for i in range(3))
    return hx(v)


def structure_rows(x: pd.DataFrame, protein: str) -> pd.DataFrame:
    z = x.copy()
    z = z[truthy(z["pdb_coordinate_present"])]
    z = z[z["selected_pdb_id"].astype(str).str.upper().eq(PDB[protein][0])]
    z = z[z["pdb_author_position"].notna() & z["selected_pdb_chain"].notna()]
    z["pdb_author_position"] = (
        z["pdb_author_position"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    return z


def write_structure_csv_and_pml(selected, lim):
    allmap = []
    for protein in ("NDM-1", "VIM-2"):
        pdb_id, cif = PDB[protein]
        if not cif.is_file():
            raise FileNotFoundError(cif)
        z = structure_rows(selected[protein], protein)
        if z.empty:
            raise RuntimeError(f"No mapped structure rows for {protein}")
        out = z[
            [
                "protein_id",
                "scheme_id",
                "assay_position",
                "n_observations",
                VALUE,
                "selected_pdb_id",
                "selected_pdb_chain",
                "pdb_author_position",
            ]
        ].copy()
        out["display_value"] = out[VALUE].clip(-lim, lim)
        out.to_csv(
            COMP / f"{protein.lower().replace('-', '')}_structure_values.csv",
            index=False,
        )
        allmap.append(out)

        obj = protein.lower().replace("-", "")

        mapped_chains = sorted(
            {
                str(v).strip()
                for v in z["selected_pdb_chain"].dropna().astype(str)
                if str(v).strip()
            }
        )
        if len(mapped_chains) != 1:
            raise RuntimeError(
                f"{protein}: expected exactly one mapped PDB chain, "
                f"found {mapped_chains}"
            )
        mapped_chain = mapped_chains[0]

        pml = []
        pml += [
            "reinitialize",
            f"load {cif.as_posix()}, {obj}",
            "bg_color white",
            f"hide everything, {obj}",
            f"select mapped_chain, {obj} and chain {mapped_chain} and polymer.protein",
            "show cartoon, mapped_chain",
            "set cartoon_transparency, 0.08",
            "set cartoon_smooth_loops, 1",
            "set antialias, 2",
            "set ray_opaque_background, off",
            "set orthoscopic, on",
            "set depth_cue, 0",
            "set ray_shadows, 0",
            "set specular, 0.15",
            "set ambient, 0.55",
            "color gray80, mapped_chain",
        ]
        # Use discrete per-residue colors computed from the same common numeric range.
        for _idx, row in z.iterrows():
            val = float(row[VALUE])
            t = (max(-lim, min(lim, val)) + lim) / (2 * lim)
            color = interpolate_hex(t)
            cname = (
                f"f5_{protein.replace('-', '').lower()}_{int(row['assay_position'])}"
            )
            chain = str(row["selected_pdb_chain"])
            resi = str(row["pdb_author_position"])
            rgb = (
                f"{int(color[1:3], 16) / 255:.6f}, "
                f"{int(color[3:5], 16) / 255:.6f}, "
                f"{int(color[5:7], 16) / 255:.6f}"
            )
            pml.append(f"set_color {cname}, [{rgb}]")
            pml.append(f"color {cname}, mapped_chain and chain {chain} and resi {resi}")
        pml += [
            f"select f5_zn_candidates, ({obj} and elem Zn) within 6.0 of mapped_chain",
            "python",
            "from pymol import cmd",
            "import math",
            "center = cmd.centerofmass('mapped_chain')",
            "atoms = []",
            "cmd.iterate_state(1, 'f5_zn_candidates', "
            "'atoms.append((model, index, x, y, z))', space={'atoms': atoms})",
            "atoms.sort(key=lambda a: "
            "(a[2]-center[0])**2 + (a[3]-center[1])**2 + (a[4]-center[2])**2)",
            "cmd.select('f5_zn', 'none')",
            "for model, index, x, y, z in atoms[:2]: "
            "cmd.select('f5_zn', 'f5_zn or (%s and index %d)' % (model, index))",
            "python end",
            "show spheres, f5_zn",
            "set sphere_scale, 0.45, f5_zn",
            "set_color f5_zinc_gold, [0.85, 0.64, 0.13]",
            "color f5_zinc_gold, f5_zn",
            "select displayed_structure, mapped_chain or f5_zn",
            "orient displayed_structure",
            "zoom displayed_structure, 1.0",
            (
                "png "
                + str(
                    (
                        COMP
                        / (
                            "figure_5_"
                            + protein.lower().replace("-", "")
                            + "_structure.png"
                        )
                    ).resolve()
                )
                + ", 2400, 1900, dpi=600, ray=1"
            ),
            "quit",
        ]
        (COMP / f"render_figure5_{protein.lower().replace('-', '')}.pml").write_text(
            "\n".join(pml) + "\n", encoding="utf-8"
        )

    pd.concat(allmap, ignore_index=True).to_csv(
        AUD / "figure_5_structure_plotted_values.csv", index=False
    )


def try_render():
    candidates = [
        shutil.which("pymol"),
        "/Applications/PyMOL.app/Contents/MacOS/PyMOL",
        "/Applications/PyMOL.app/Contents/MacOS/MacPyMOL",
    ]
    exe = next((x for x in candidates if x and Path(x).exists()), None)
    if not exe:
        print("PYMOL: not found; PML scripts were written for manual rendering.")
        return False
    for protein in ("ndm1", "vim2"):
        pml = COMP / f"render_figure5_{protein}.pml"
        print("PYMOL:", exe, pml)
        subprocess.run([exe, "-cq", str(pml)], check=True)
    return True


def add_structure_panel(ax, path: Path, letter: str, extra_crop: float = 0.0):
    img = plt.imread(path)

    # Crop empty PyMOL canvas while preserving all visible molecular content.
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]
        mask = alpha > 0.01
    else:
        rgb = img[..., :3]
        mask = np.min(rgb, axis=2) < 0.985

    ys, xs = np.where(mask)
    if len(xs) and len(ys):
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1

        # Small padding prevents clipping of antialiased molecular edges.
        pad_x = max(8, int(0.035 * (x1 - x0)))
        pad_y = max(8, int(0.035 * (y1 - y0)))

        x0 = max(0, x0 - pad_x)
        x1 = min(img.shape[1], x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(img.shape[0], y1 + pad_y)

        img = img[y0:y1, x0:x1]

    # NDM-1 is rendered with more surrounding space than VIM-2.
    # A modest symmetric crop gives the two structures comparable
    # visual weight without changing molecular geometry.
    if extra_crop > 0:
        h, w = img.shape[:2]
        dx = int(w * extra_crop)
        dy = int(h * extra_crop)
        if 2 * dx < w and 2 * dy < h:
            img = img[dy : h - dy, dx : w - dx]

    ax.imshow(img)
    ax.set_axis_off()
    ax.text(
        0.0,
        1.01,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def assemble_if_rendered(selected, lim):
    ndm = COMP / "figure_5_ndm1_structure.png"
    vim = COMP / "figure_5_vim2_structure.png"
    if not (ndm.is_file() and vim.is_file()):
        print("COMPOSITE: waiting for PyMOL structure PNGs.")
        return False

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18], hspace=0.18, wspace=0.34)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    sequence_panel(axA, selected["NDM-1"], "NDM-1", lim, "A")
    sequence_panel(axB, selected["VIM-2"], "VIM-2", lim, "B")
    add_structure_panel(axC, ndm, "C", extra_crop=0.075)
    add_structure_panel(axD, vim, "D")

    cmap = LinearSegmentedColormap.from_list("f5", [BLUE, MID, MAGENTA])
    sm = plt.cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=cmap)
    cbar = fig.colorbar(
        sm,
        ax=[axC, axD],
        orientation="horizontal",
        fraction=0.035,
        pad=0.035,
        aspect=40,
    )
    cbar.set_label("Graph − centre absolute error", fontsize=7)
    cbar.ax.text(
        0.0,
        1.85,
        "Graph lower error",
        transform=cbar.ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
    )
    cbar.ax.text(
        1.0,
        1.85,
        "Centre lower error",
        transform=cbar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
    )
    cbar.ax.tick_params(labelsize=6.5)

    for ext in ("svg", "pdf", "png"):
        kw = {"dpi": 600} if ext == "png" else {}
        fig.savefig(
            MAIN / f"figure_5_model_differences.{ext}", bbox_inches="tight", **kw
        )
    plt.close(fig)
    return True


def audit(selected, lim, rendered, assembled):
    rows = []
    for protein, x in selected.items():
        z = structure_rows(x, protein)
        rows.append(
            {
                "protein": protein,
                "scheme_id": SCHEMES[protein],
                "n_positions": len(x),
                "n_structure_mapped_positions": len(z),
                "min_value": float(x[VALUE].min()),
                "median_value": float(x[VALUE].median()),
                "max_value": float(x[VALUE].max()),
                "n_graph_lower": int((x[VALUE] < 0).sum()),
                "n_centre_lower": int((x[VALUE] > 0).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(AUD / "figure_5_generation_summary.csv", index=False)
    vals = pd.concat(
        [
            x.assign(figure5_protein=p)[
                [
                    "figure5_protein",
                    "scheme_id",
                    "assay_position",
                    "n_observations",
                    VALUE,
                    "selected_pdb_id",
                    "selected_pdb_chain",
                    "pdb_author_position",
                    "pdb_coordinate_present",
                ]
            ]
            for p, x in selected.items()
        ],
        ignore_index=True,
    )
    vals.to_csv(AUD / "figure_5_plotted_values.csv", index=False)

    txt = [
        "FIGURE 5 GENERATION AUDIT",
        "=" * 72,
        f"Input: {DATA.relative_to(ROOT)}",
        f"Position value: {VALUE}",
        (
            "Interpretation: negative = residue graph lower absolute error; "
            "positive = centre-residue model lower absolute error."
        ),
        f"Common symmetric limit: {-lim:.12g} to {lim:.12g}",
        "NDM-1 scheme: ndm1_position_nested5",
        "VIM-2 scheme: vim2_2020_position_nested5",
        f"PyMOL rendered automatically: {rendered}",
        f"Composite assembled: {assembled}",
        "No model fitting or model selection was performed.",
        "No Figures 1-4 were read, modified, or regenerated.",
        "",
    ]
    for r in rows:
        txt += [
            f"{r['protein']}: {r['n_positions']} plotted positions; "
            f"{r['n_structure_mapped_positions']} mapped to experimental structure; "
            f"{r['n_graph_lower']} graph-lower; {r['n_centre_lower']} centre-lower."
        ]
    (AUD / "figure_5_generation_audit.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8"
    )


def main():
    _, selected = load()
    lim = symmetric_limit(selected)
    save_sequence_panels(selected, lim)
    write_structure_csv_and_pml(selected, lim)

    # Remove previous renders before invoking PyMOL. This prevents an old
    # structure image from being assembled if the current render fails.
    for protein in ("ndm1", "vim2"):
        old_render = COMP / f"figure_5_{protein}_structure.png"
        if old_render.exists():
            old_render.unlink()

    rendered = try_render()
    assembled = assemble_if_rendered(selected, lim)
    audit(selected, lim, rendered, assembled)
    print("WROTE:", COMP / "figure_5_AB_residue_error_maps.svg")
    print("WROTE:", AUD / "figure_5_plotted_values.csv")
    print("WROTE:", AUD / "figure_5_generation_audit.txt")
    if assembled:
        print("WROTE:", MAIN / "figure_5_model_differences.svg")
        print("WROTE:", MAIN / "figure_5_model_differences.pdf")
        print("WROTE:", MAIN / "figure_5_model_differences.png")
    else:
        print(
            "NEXT: render the two PML files, then rerun this script; "
            "it will assemble the composite."
        )


if __name__ == "__main__":
    main()
