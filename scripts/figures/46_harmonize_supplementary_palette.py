"""Harmonize supplementary figure colors for publication.

This script changes graphical styling only.

It does not recalculate study results, alter plotted coordinates,
change labels, or rerun any analysis stage.

Vector PDFs are converted to SVG with Inkscape, publication colors are
substituted consistently across graphical objects, and the final PDF and
PNG files are exported from the updated SVG.

For sequential blue heatmaps, the complete blue scale is mapped to the
study's light-cream -> gold -> coral -> deep-rust sequential palette while
preserving the original ordering of color intensity.
"""

from __future__ import annotations

import argparse
import colorsys
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

PALE_CREAM = "#F7F1D8"
SOFT_GOLD = "#EEDC9A"
GOLD = "#D6B23C"
CORAL = "#E29A7F"
RUST = "#C76B4F"
DEEP_RUST = "#A94438"
BLUE = "#4F86A6"
CHARCOAL = "#292929"
GREY = "#A7A7A7"


# Standard Matplotlib categorical colors -> publication palette.
#
# These substitutions apply to the graphical representation only.
# The same source category always receives the same destination color,
# including line, marker, legend, and error-bar elements.
CATEGORICAL = {
    "#1f77b4": BLUE,
    "#ff7f0e": RUST,
    "#2ca02c": GOLD,
    "#d62728": DEEP_RUST,
    "#9467bd": CORAL,
    "#8c564b": "#8E6B57",
    "#e377c2": "#C78383",
    "#7f7f7f": GREY,
    "#bcbd22": "#B8A33B",
    "#17becf": "#6F9E9A",
    # Common Matplotlib / existing green used in S06.
    "#4c8b4b": GOLD,
    "#4c8b4a": GOLD,
    "#4b8a4a": GOLD,
}


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_hex(rgb: tuple[float, float, float]) -> str:
    vals = [int(round(max(0, min(255, v)))) for v in rgb]
    return "#" + "".join(f"{v:02x}" for v in vals)


def interpolate(
    stops: list[str],
    t: float,
) -> str:
    t = max(0.0, min(1.0, t))

    if len(stops) == 1:
        return stops[0]

    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    frac = pos - i

    a = hex_rgb(stops[i])
    b = hex_rgb(stops[i + 1])

    rgb = tuple(a[j] + frac * (b[j] - a[j]) for j in range(3))

    return rgb_hex(rgb)


WARM_SEQUENTIAL = [
    PALE_CREAM,
    SOFT_GOLD,
    GOLD,
    CORAL,
    RUST,
    DEEP_RUST,
]


def looks_like_blue_sequential(
    color: str,
) -> tuple[bool, float]:
    """Identify colors belonging to a blue sequential scale.

    Returns (is_blue_scale, darkness_fraction).

    darkness_fraction is monotonic:
    pale blue -> 0
    dark blue -> 1
    """

    r, g, b = hex_rgb(color)
    rf, gf, bf = (
        r / 255,
        g / 255,
        b / 255,
    )

    h, s, v = colorsys.rgb_to_hsv(
        rf,
        gf,
        bf,
    )

    # Matplotlib Blues family occupies roughly cyan-blue hues.
    blue_hue = 0.52 <= h <= 0.67

    if not blue_hue or s < 0.08:
        return False, 0.0

    # Darker and more saturated blue represents larger values.
    darkness = 0.68 * (1.0 - v) + 0.32 * s

    # Normalize approximate Blues range.
    darkness = np.clip(
        darkness / 0.72,
        0.0,
        1.0,
    )

    return True, float(darkness)


HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")
RGB_PERCENT_RE = re.compile(
    r"rgb\(\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*\)",
    re.IGNORECASE,
)


def restyle_svg(
    source: Path,
    destination: Path,
    sequential_blue: bool,
) -> None:
    text = source.read_text(
        encoding="utf-8",
    )

    def repl(match: re.Match[str]) -> str:
        original = match.group(0)
        low = original.lower()

        if low in CATEGORICAL:
            return CATEGORICAL[low]

        if sequential_blue:
            is_blue, amount = looks_like_blue_sequential(low)

            if is_blue:
                return interpolate(
                    WARM_SEQUENTIAL,
                    amount,
                )

        return original

    updated = HEX_RE.sub(
        repl,
        text,
    )

    def repl_rgb_percent(match: re.Match[str]) -> str:
        rgb = tuple(
            int(round(float(match.group(i)) * 255.0 / 100.0)) for i in (1, 2, 3)
        )
        best_key = min(
            CATEGORICAL,
            key=lambda key: sum(
                (a - b) ** 2 for a, b in zip(hex_rgb(key), rgb, strict=True)
            ),
        )
        distance = (
            sum((a - b) ** 2 for a, b in zip(hex_rgb(best_key), rgb, strict=True))
            ** 0.5
        )
        if distance > 3.0:
            return match.group(0)
        r, g, b = hex_rgb(CATEGORICAL[best_key])
        return f"rgb({r / 255 * 100:.6f}%, {g / 255 * 100:.6f}%, {b / 255 * 100:.6f}%)"

    updated = RGB_PERCENT_RE.sub(repl_rgb_percent, updated)

    destination.write_text(
        updated,
        encoding="utf-8",
    )


def run(
    args: list[str],
) -> None:
    subprocess.run(
        args,
        check=True,
    )


def process(
    pdf: Path,
    inkscape: str,
    sequential_blue: bool,
) -> None:
    temp = pdf.with_name(f".{pdf.stem}.palette_input.svg")

    final_svg = pdf.with_name(f".{pdf.stem}.palette_output.svg")

    run(
        [
            inkscape,
            str(pdf),
            "--pdf-poppler",
            f"--export-filename={temp}",
        ]
    )

    restyle_svg(
        temp,
        final_svg,
        sequential_blue=sequential_blue,
    )

    run(
        [
            inkscape,
            str(final_svg),
            "--export-type=pdf",
            f"--export-filename={pdf}",
        ]
    )

    png = pdf.with_suffix(".png")

    run(
        [
            inkscape,
            str(final_svg),
            "--export-type=png",
            "--export-dpi=300",
            f"--export-filename={png}",
        ]
    )

    temp.unlink(missing_ok=True)
    final_svg.unlink(missing_ok=True)

    print("UPDATED", pdf.name)


def regenerate_s07(root: Path) -> None:
    source = (
        root
        / "data"
        / "interim"
        / "baseline_evaluation"
        / "baseline_model_selection.csv"
    )
    frame = pd.read_csv(source)
    selected = frame[frame["selected"]].copy()

    scheme_order = list(dict.fromkeys(frame["scheme_id"]))
    alpha_order = sorted(selected["alpha"].dropna().unique())
    counts = (
        selected.groupby(["scheme_id", "alpha"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=scheme_order, columns=alpha_order, fill_value=0)
    )

    labels = {
        "ndm1_position_nested5": "NDM-1\
unseen position",
        "ndm1_to_vim2_amp_source_cv5": "NDM-1 -> VIM-2",
        "vim2_2020_loo_antibiotic_amp": "loo antibiotic amp",
        "vim2_2020_loo_antibiotic_ctx": "loo antibiotic ctx",
        "vim2_2020_loo_antibiotic_mem": "loo antibiotic mem",
        "vim2_2020_position_nested5": "VIM-2 2020\
unseen position",
        "vim2_2020_temperature_25_to_37": "temperature 25 to 37",
        "vim2_2020_temperature_37_to_25": "temperature 37 to 25",
        "vim2_2022_loo_concentration_128": "loo concentration 128",
        "vim2_2022_loo_concentration_16": "loo concentration 16",
        "vim2_2022_loo_concentration_256": "loo concentration 256",
        "vim2_2022_loo_concentration_32": "loo concentration 32",
        "vim2_2022_loo_concentration_4": "loo concentration 4",
        "vim2_2022_loo_concentration_64": "loo concentration 64",
        "vim2_2022_loo_concentration_8": "loo concentration 8",
        "vim2_2022_mutation_nested5": "VIM-2 2022\
unseen mutation",
        "vim2_to_ndm1_amp_source_cv5": "VIM-2 -> NDM-1",
    }

    values = counts.to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("warm_selection_count", WARM_SEQUENTIAL)
    norm = Normalize(vmin=0.0, vmax=float(values.max()))

    fig, ax = plt.subplots(figsize=(10.2, 6.5))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(
        range(len(scheme_order)),
        [labels.get(item, item.replace("_", " ")) for item in scheme_order],
        fontsize=8,
    )
    ax.set_xticks(
        range(len(alpha_order)),
        [f"{value:g}" for value in alpha_order],
        fontsize=8,
    )
    ax.set_xlabel("Selected ridge alpha", fontsize=10)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = int(values[row, column])
            if value:
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#292929",
                )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Selection count", fontsize=9)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8)

    fig.tight_layout()
    out = root / "figures" / "supplementary"
    fig.savefig(out / "S3_baseline_selection.pdf", bbox_inches="tight")
    fig.savefig(
        out / "S3_baseline_selection.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("UPDATED S3_baseline_selection.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--inkscape",
        default=("/Applications/Inkscape.app/Contents/MacOS/inkscape"),
    )

    args = parser.parse_args()

    supp = args.root / "figures" / "supplementary"

    # S07 is the only supplementary figure in the current set
    # using a continuous blue heatmap/colorbar.
    #
    # It therefore receives a monotonic sequential transformation.
    #
    # The other figures use categorical/default plotting colors.
    sequential: set[str] = set()

    stems = [
        "S1_condition_score_distributions",
        "S2_partition_sizes",
        "S4_sequence_model_comparison",
        "S5_graph_training",
        "S6_seed_stability",
        "S7_extended_epoch",
        "S8_minimum_score_sensitivity",
        "S9_sequence_structure",
        "S10_biological_associations",
        "S11_condition_concentration",
    ]

    regenerate_s07(args.root)

    for stem in stems:
        pdf = supp / f"{stem}.pdf"

        if not pdf.exists():
            raise FileNotFoundError(pdf)

        process(
            pdf,
            args.inkscape,
            sequential_blue=(stem in sequential),
        )

    print()
    print("PASS: supplementary PDF/PNG palette harmonization complete")


if __name__ == "__main__":
    main()
