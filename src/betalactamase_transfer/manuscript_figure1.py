from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#4F86A6"
CORAL = "#E29A7F"
PALE_CREAM = "#F7F1D8"
WARM_GREY = "#E9E7E2"
RUST = "#C76B4F"
DEEP_RUST = "#A94438"
GOLD = "#D6B23C"
SOFT_GOLD = "#EEDC9A"
GREY = "#777777"
LIGHT_GREY = "#D7DADD"
DARK = "#292929"


def _save_vector(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def _clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7, width=0.6, length=3)
    ax.grid(False)


def _copy_structure_components(root: Path, out: Path) -> list[dict[str, str]]:
    src_dir = root / "figures/components/pymol"
    rows = []
    for src_name, dst_name in [
        ("NDM1_3SPU_chainB_overview.png", "A_NDM1_3SPU_overview.png"),
        ("VIM2_5YD7_chainB_overview.png", "A_VIM2_5YD7_overview.png"),
    ]:
        src = src_dir / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out / dst_name
        shutil.copy2(src, dst)
        rows.append(
            {
                "component": dst.stem,
                "source": str(src.relative_to(root)),
                "type": "raster",
            }
        )
    return rows


def _condition_map(root: Path, out: Path) -> dict[str, str]:
    path = root / "data/processed/missense_observations.csv"
    cols = ["study_id", "protein_id", "antibiotic", "concentration", "temperature_c"]
    frame = pd.read_csv(path, usecols=cols).drop_duplicates()
    order = ["chen_2024_natcomms", "chen_2020_elife", "chen_2022_natee"]
    labels = {
        "chen_2024_natcomms": "Chen 2024 · NDM-1",
        "chen_2020_elife": "Chen 2020 · VIM-2",
        "chen_2022_natee": "Chen 2022 · VIM-2",
    }
    drugs = ["AMP", "CTX", "MEM"]
    fig, ax = plt.subplots(figsize=(4.6, 2.45))
    for yi, study in enumerate(order):
        q = frame[frame.study_id.eq(study)]
        for xi, drug in enumerate(drugs):
            d = q[q.antibiotic.eq(drug)]
            if d.empty:
                continue
            concs = sorted(
                pd.to_numeric(d.concentration, errors="coerce").dropna().unique()
            )
            temps = sorted(
                pd.to_numeric(d.temperature_c, errors="coerce").dropna().unique()
            )
            size = 34 + 13 * min(len(concs), 7)
            color = GOLD if study == "chen_2024_natcomms" else CORAL
            ax.scatter(
                xi,
                yi,
                s=size,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            if study == "chen_2022_natee":
                txt = "4–256\n7 levels"
            elif len(concs) > 1:
                txt = f"{len(concs)} levels"
            else:
                txt = f"{concs[0]:g}" if concs else ""
            ax.text(
                xi, yi + 0.29, txt, ha="center", va="bottom", fontsize=6.2, color=GREY
            )
            if len(temps) > 1:
                ax.text(
                    xi,
                    yi - 0.29,
                    "25/37 °C",
                    ha="center",
                    va="top",
                    fontsize=6.2,
                    color=GREY,
                )
    ax.set_xticks(range(3), drugs, fontsize=7.5)
    ax.set_yticks(range(3), [labels[x] for x in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(2.55, -0.55)
    ax.set_xlabel("Antibiotic", fontsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    _save_vector(fig, out / "B_experimental_condition_map")
    return {
        "component": "B_experimental_condition_map",
        "source": str(path.relative_to(root)),
        "type": "vector",
    }


def _choose_example(root: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(root / "data/interim/residue_graphs/residue_graph_nodes.csv")
    edges = pd.read_csv(root / "data/interim/residue_graphs/residue_graph_edges.csv")
    features = pd.read_csv(root / "data/processed/residue_features.csv")
    obs = pd.read_csv(
        root / "data/processed/missense_observations.csv",
        usecols=[
            "protein_id",
            "assay_position",
            "wild_type",
            "mutant",
            "mutation",
            "experimental_structure_task_eligible",
        ],
    )
    n = nodes[nodes.protein_id.eq("NDM-1")].copy()
    e = edges[edges.protein_id.eq("NDM-1")].copy()
    degree = pd.concat(
        [e.source_assay_position, e.target_assay_position]
    ).value_counts()
    n["degree"] = n.assay_position.map(degree).fillna(0)
    f = features[
        (features.protein_id.eq("NDM-1")) & features.pdb_coordinate_present.astype(bool)
    ].copy()
    candidates = n.merge(
        f[["assay_position", "pdb_author_position"]], on="assay_position", how="inner"
    )
    candidates = candidates[candidates.assay_position.between(40, 250)]
    candidates = candidates.sort_values(
        ["degree", "assay_position"], ascending=[False, True]
    )
    if candidates.empty:
        raise RuntimeError(
            "No coordinate-supported NDM-1 residue is available for Figure 1C."
        )
    centre = candidates.iloc[0]
    oq = obs[
        (obs.protein_id.eq("NDM-1")) & obs.assay_position.eq(int(centre.assay_position))
    ]
    oq = oq[oq.experimental_structure_task_eligible.astype(bool)]
    if oq.empty:
        raise RuntimeError(
            "Selected Figure 1C residue has no eligible missense observation."
        )
    mutation = oq.sort_values("mutant").iloc[0]
    centre = centre.copy()
    centre["mutation"] = mutation.mutation
    centre["wild_type"] = mutation.wild_type
    centre["mutant"] = mutation.mutant
    return centre, n, e


def _sequence_component(centre: pd.Series, out: Path) -> None:
    pos = int(centre.assay_position)
    wt, mut = str(centre.wild_type), str(centre.mutant)

    # The selected NDM-1 example is displayed as an 11-residue sequence window.
    # These residues correspond to assay positions 87-97 around position 92.
    window_start = pos - 5
    positions = np.arange(window_start, pos + 6)
    wild_type_sequence = list("LVVDTAWTDDQ")

    if pos != 92 or wt != "A":
        raise ValueError(
            "The Figure 1C sequence window is verified for the selected "
            "NDM-1 A92 example only."
        )

    mutant_sequence = wild_type_sequence.copy()
    mutant_sequence[5] = mut

    fig, ax = plt.subplots(figsize=(6.6, 2.15))
    ax.set_axis_off()

    x_positions = np.arange(len(positions), dtype=float)
    box_width = 0.78
    box_height = 0.58
    wild_y = 0.55
    mutant_y = -0.25

    # Position numbers.
    for x, residue_position in zip(x_positions, positions, strict=True):
        ax.text(
            x,
            1.13,
            str(residue_position),
            ha="center",
            va="center",
            fontsize=7,
            color=GREY,
            fontweight="bold" if residue_position == pos else "normal",
        )

    # Wild-type and mutant rows.
    for row_y, residues, row_name in [
        (wild_y, wild_type_sequence, "Wild type"),
        (mutant_y, mutant_sequence, "Mutant"),
    ]:
        ax.text(
            -1.05,
            row_y,
            row_name,
            ha="right",
            va="center",
            fontsize=8,
            color="black",
        )

        for x, residue, residue_position in zip(
            x_positions, residues, positions, strict=True
        ):
            is_centre = residue_position == pos

            face = "#F7E9E4" if is_centre else "white"
            edge = RUST if is_centre else LIGHT_GREY
            text_color = RUST if is_centre else "black"

            patch = plt.Rectangle(
                (x - box_width / 2, row_y - box_height / 2),
                box_width,
                box_height,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.2 if is_centre else 0.9,
                joinstyle="round",
            )
            ax.add_patch(patch)

            ax.text(
                x,
                row_y,
                residue,
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold" if is_centre else "normal",
            )

    # Mutation identifier.
    centre_x = x_positions[5]
    ax.plot(
        [centre_x, centre_x],
        [-0.60, -0.78],
        color=RUST,
        lw=1.0,
    )
    ax.text(
        centre_x,
        -0.98,
        f"{wt}{pos}{mut}",
        ha="center",
        va="center",
        fontsize=9,
        color=RUST,
        fontweight="bold",
    )

    ax.set_xlim(-1.75, len(positions) - 0.35)
    ax.set_ylim(-1.25, 1.42)
    _save_vector(fig, out / "C_sequence_mutation")


def _graph_component(
    centre: pd.Series, nodes: pd.DataFrame, edges: pd.DataFrame, out: Path
) -> None:
    pos = int(centre.assay_position)
    e = edges[
        (edges.source_assay_position.eq(pos)) | (edges.target_assay_position.eq(pos))
    ].copy()
    neigh = sorted(set(e.source_assay_position).union(e.target_assay_position))
    n = nodes[nodes.assay_position.isin(neigh)].copy()
    if len(n) < 3:
        raise RuntimeError(
            "Figure 1C graph neighbourhood contains fewer than three nodes."
        )
    xyz = n[["ca_x", "ca_y", "ca_z"]].to_numpy(float)
    xyz -= xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(xyz, full_matrices=False)
    xy = xyz @ vh[:2].T
    scale = np.max(np.abs(xy)) or 1.0
    xy /= scale
    coords = dict(zip(n.assay_position.astype(int), xy, strict=True))
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    for row in e.itertuples():
        a, b = int(row.source_assay_position), int(row.target_assay_position)
        if a in coords and b in coords:
            ax.plot(
                [coords[a][0], coords[b][0]],
                [coords[a][1], coords[b][1]],
                color=LIGHT_GREY,
                lw=0.8,
                zorder=1,
            )
    for row in n.itertuples():
        p = int(row.assay_position)
        x, y = coords[p]
        centre_node = p == pos
        ax.scatter(
            x,
            y,
            s=230 if centre_node else 38,
            facecolor=RUST if centre_node else CORAL,
            edgecolor="white",
            linewidth=0.8 if centre_node else 0.6,
            zorder=3,
        )
        if centre_node:
            ax.text(
                x,
                y,
                f"{row.assay_residue}{p}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=4,
            )
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save_vector(fig, out / "C_residue_graph")


def _write_pymol_component_script(root: Path, out: Path, centre: pd.Series) -> None:
    pos = int(centre.assay_position)
    author = str(centre.pdb_author_position)
    script = f"""# Generated from repository tables for Figure 1 components.\n# Run from repository root in PyMOL: @figures/components/figure_1/render_figure1_components.pml\nreinitialize\nload data/raw/pdb/3SPU/3SPU.cif, NDM1\nload data/raw/pdb/5YD7/5YD7.cif, VIM2\ncreate NDM1B, NDM1 and chain B\ncreate VIM2B, VIM2 and chain B\ndelete NDM1\ndelete VIM2\nbg_color white\nset ray_opaque_background, off\nset opaque_background, off\nset antialias, 2\nset depth_cue, 0\nset fog, 0\nset specular, 0.35\nset cartoon_fancy_helices, 1\nset cartoon_flat_sheets, 1\nset_color ndm_blue, [0.310,0.525,0.651]\nset_color vim_coral, [0.886,0.604,0.498]\nset_color zinc_gold, [0.94,0.64,0.18]\nset_color mutation_rust, [0.72,0.30,0.20]\n\n# Comparative superposition; visualization only.\nalign VIM2B and polymer and name CA, NDM1B and polymer and name CA, cycles=5\nhide everything, all\nshow cartoon, NDM1B and polymer\nshow cartoon, VIM2B and polymer\ncolor ndm_blue, NDM1B and polymer\ncolor vim_coral, VIM2B and polymer\nshow spheres, (NDM1B or VIM2B) and resn ZN and resi 301+302\ncolor zinc_gold, (NDM1B or VIM2B) and resn ZN and resi 301+302\norient NDM1B or VIM2B\nzoom NDM1B or VIM2B, 4\nray 3600, 3000\npng figures/components/figure_1/A_NDM1_VIM2_superposition.png, dpi=600\n\n# Actual local atomic neighbourhood around selected NDM-1 assay position {pos}.\nhide everything, all\nselect centre, NDM1B and polymer and resi {author}\nselect local_atoms, byres (NDM1B and polymer within 5 of centre)\nshow cartoon, NDM1B and polymer\nset cartoon_transparency, 0.78, NDM1B and polymer\nshow sticks, local_atoms\nshow sticks, centre\ncolor grey70, local_atoms and elem C\ncolor mutation_rust, centre and elem C\ncolor nitrogen, local_atoms and elem N\ncolor oxygen, local_atoms and elem O\ncolor sulfur, local_atoms and elem S\nshow spheres, NDM1B and resn ZN and resi 301+302\ncolor zinc_gold, NDM1B and resn ZN and resi 301+302\norient local_atoms\nzoom local_atoms, 3\nray 3000, 3000\npng figures/components/figure_1/C_local_atoms_NDM1_pos{pos}.png, dpi=600\nquit\n"""  # noqa: E501
    (out / "render_figure1_components.pml").write_text(script, encoding="utf-8")


def _evaluation_components(root: Path, out: Path) -> list[dict[str, str]]:
    path = root / "data/interim/evaluation_partitions/partition_schemes.csv"
    schemes = pd.read_csv(path)
    if len(schemes) != 17:
        raise ValueError(f"Expected 17 evaluation schemes, found {len(schemes)}")
    if schemes.test_labels_used_for_model_selection.astype(bool).any():
        raise ValueError("At least one scheme uses test labels for model selection.")
    groups = {
        "D_unseen_position": schemes[
            schemes.scheme_id.isin(
                ["ndm1_position_nested5", "vim2_2020_position_nested5"]
            )
        ],
        "D_unseen_mutation": schemes[
            schemes.scheme_id.eq("vim2_2022_mutation_nested5")
        ],
        "D_cross_protein": schemes[schemes.test_category.eq("cross_protein")],
        "D_antibiotic_transfer": schemes[
            schemes.test_category.astype(str).str.contains("antibiotic", na=False)
        ],
        "D_temperature_transfer": schemes[
            schemes.test_category.astype(str).str.contains("temperature", na=False)
        ],
        "D_concentration_transfer": schemes[schemes.test_dimension.eq("concentration")],
    }
    rows = []
    for name, q in groups.items():
        if q.empty:
            raise ValueError(f"No schemes found for {name}")
        fig, ax = plt.subplots(figsize=(3.2, 1.25))
        ax.set_axis_off()
        if name == "D_unseen_position":
            x = np.arange(12)
            test = np.array([2, 7, 10])
            ax.scatter(
                x,
                np.zeros_like(x),
                s=58,
                c=[RUST if i in test else PALE_CREAM for i in x],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(0, -0.45, "residue positions", fontsize=7, color=GREY, ha="left")
        elif name == "D_unseen_mutation":
            x = np.arange(12)
            test = np.array([1, 5, 9])
            ax.scatter(
                x,
                np.zeros_like(x),
                s=58,
                c=[RUST if i in test else PALE_CREAM for i in x],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(0, -0.45, "mutation identities", fontsize=7, color=GREY, ha="left")
        elif name == "D_cross_protein":
            ax.plot(
                [0.08, 0.42],
                [0.62, 0.62],
                lw=5,
                color=SOFT_GOLD,
                solid_capstyle="round",
            )
            ax.annotate(
                "",
                xy=(0.88, 0.62),
                xytext=(0.48, 0.62),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": GREY},
            )
            ax.plot(
                [0.08, 0.42], [0.25, 0.25], lw=5, color=CORAL, solid_capstyle="round"
            )
            ax.annotate(
                "",
                xy=(0.88, 0.25),
                xytext=(0.48, 0.25),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": GREY},
            )
            ax.text(0.25, 0.76, "NDM-1", ha="center", fontsize=7)
            ax.text(0.78, 0.76, "VIM-2", ha="center", fontsize=7)
            ax.text(0.25, 0.06, "VIM-2", ha="center", fontsize=7)
            ax.text(0.78, 0.06, "NDM-1", ha="center", fontsize=7)
        elif name == "D_antibiotic_transfer":
            for i, (drug, c) in enumerate(
                zip(["AMP", "CTX", "MEM"], [GOLD, CORAL, RUST], strict=True)
            ):
                ax.scatter(i, 0.5, s=105, facecolor=c, edgecolor="white", linewidth=0.7)
                ax.text(i, 0.12, drug, ha="center", fontsize=7)
            ax.plot([0, 2], [0.5, 0.5], color=LIGHT_GREY, lw=0.8, zorder=0)
        elif name == "D_temperature_transfer":
            ax.text(0.12, 0.5, "25 °C", fontsize=9, ha="center", va="center")
            ax.annotate(
                "",
                xy=(0.82, 0.5),
                xytext=(0.30, 0.5),
                arrowprops={"arrowstyle": "<->", "lw": 1.1, "color": GREY},
            )
            ax.text(0.90, 0.5, "37 °C", fontsize=9, ha="center", va="center")
        else:
            vals = (
                pd.to_numeric(q.test_value, errors="coerce")
                .dropna()
                .sort_values()
                .unique()
            )
            if len(vals) != 7 or set(vals) != {4, 8, 16, 32, 64, 128, 256}:
                raise ValueError(f"Unexpected concentration targets: {vals.tolist()}")
            x = np.log2(vals)
            cats = q.set_index(
                pd.to_numeric(q.test_value, errors="coerce")
            ).test_category.to_dict()
            for xx, v in zip(x, vals, strict=True):
                edge = "black" if "extrapolation" in str(cats.get(v, "")) else "white"
                ax.scatter(
                    xx, 0.5, s=80, facecolor=CORAL, edgecolor=edge, linewidth=1.0
                )
                ax.text(xx, 0.12, f"{v:g}", ha="center", fontsize=6.5)
            ax.plot([x.min(), x.max()], [0.5, 0.5], color=LIGHT_GREY, lw=0.8, zorder=0)
            ax.text(
                x.mean(),
                -0.18,
                "AMP concentration (µg/mL)",
                ha="center",
                fontsize=7,
                color=GREY,
            )
        ax.set_xlim(-0.3, max(2.3, ax.get_xlim()[1]))
        ax.set_ylim(-0.35, 1.05)
        _save_vector(fig, out / name)
        rows.append(
            {"component": name, "source": str(path.relative_to(root)), "type": "vector"}
        )
    return rows


def figure1(root: Path, out: Path) -> None:
    """Generate Figure 1 scientific components for manual journal assembly."""
    root = Path(root)
    component_dir = root / "figures/components/figure_1"
    component_dir.mkdir(parents=True, exist_ok=True)
    manifest = _copy_structure_components(root, component_dir)
    manifest.append(_condition_map(root, component_dir))
    centre, nodes, edges = _choose_example(root)
    _sequence_component(centre, component_dir)
    _graph_component(centre, nodes, edges, component_dir)
    manifest.extend(
        [
            {
                "component": "C_sequence_mutation",
                "source": "data/processed/missense_observations.csv",
                "type": "vector",
            },
            {
                "component": "C_residue_graph",
                "source": (
                    "data/interim/residue_graphs/residue_graph_nodes.csv; "
                    "data/interim/residue_graphs/residue_graph_edges.csv"
                ),
                "type": "vector",
            },
        ]
    )
    _write_pymol_component_script(root, component_dir, centre)
    manifest.append(
        {
            "component": f"C_local_atoms_NDM1_pos{int(centre.assay_position)}",
            "source": "data/raw/pdb/3SPU/3SPU.cif",
            "type": "PyMOL render script",
        }
    )
    manifest.extend(_evaluation_components(root, component_dir))
    pd.DataFrame(manifest).to_csv(
        component_dir / "figure_1_component_manifest.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "protein_id": "NDM-1",
                "assay_position": int(centre.assay_position),
                "wild_type": centre.wild_type,
                "mutant": centre.mutant,
                "mutation": centre.mutation,
                "pdb_author_position": centre.pdb_author_position,
                "selection_rule": (
                    "coordinate-supported NDM-1 residue with highest "
                    "residue-graph degree; lowest assay position breaks ties"
                ),
            }
        ]
    ).to_csv(component_dir / "figure_1_example_residue.csv", index=False)
    print(f"Figure 1 components written to: {component_dir}")
    print("Final multipanel assembly is intentionally not generated by Python.")
