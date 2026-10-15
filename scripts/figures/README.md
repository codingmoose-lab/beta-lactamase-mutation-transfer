# Publication figure sources

The figure-generation scripts are listed below. The final PDFs in `figures/main/` and `figures/supplementary/` are the versions selected for publication. Some figures received manual editing in Inkscape; rerunning Python scripts alone may not reproduce those PDFs exactly.

| Figure | Source code |
| --- | --- |
| 1 | `src/betalactamase_transfer/manuscript_figure1.py` |
| 2 | `scripts/figures/generate_figure2_homologue_effects.py` |
| 3 | `scripts/figures/generate_figure3_representation_comparison.py` |
| 4 | `scripts/figures/generate_figure4_cross_homologue_prediction.py` |
| 5 | `scripts/figures/generate_figure5_model_differences.py` |
| 6 | `scripts/figures/generate_figure6_condition_and_concentration.py` |

`45_generate_manuscript_figures.py` lists figure sources; it does **not** generate or replace figures. `46_harmonize_supplementary_palette.py` and `47_finalize_supplementary.py` modify supplementary graphics when invoked with their applicable options; do not run them when inspecting the published figures. `pymol/render_beta_lactamase_structures.pml` specifies structural selections and rendering commands using PDB entries 3SPU and 5YD7; these input structures are not included in the repository.

The scripts read processed research tables and, for some panels, structural renderings. Their input paths are specified in the individual source files. The source datasets and preparation steps are described in the repository documentation. Figure component files are local working outputs and are not part of the public repository.

To inspect source locations without generating graphics, run from the repository root after installing the package:

```bash
.venv/bin/python scripts/figures/45_generate_manuscript_figures.py
```
