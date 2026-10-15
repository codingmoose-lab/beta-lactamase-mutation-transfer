"""Registry for manuscript figure sources.

Publication-ready files in figures/main may contain manual Inkscape edits and
are authoritative. Figure generators are not run automatically.
"""

from __future__ import annotations

from pathlib import Path

FIGURE_SOURCES = {
    1: Path("src/betalactamase_transfer/manuscript_figure1.py"),
    2: Path("scripts/figures/generate_figure2_homologue_effects.py"),
    3: Path("scripts/figures/generate_figure3_representation_comparison.py"),
    4: Path("scripts/figures/generate_figure4_cross_homologue_prediction.py"),
    5: Path("scripts/figures/generate_figure5_model_differences.py"),
    6: Path("scripts/figures/generate_figure6_condition_and_concentration.py"),
}


def source_for(figure_number: int) -> Path:
    if figure_number not in FIGURE_SOURCES:
        raise ValueError(f"Unknown manuscript figure: {figure_number}")
    return FIGURE_SOURCES[figure_number]
