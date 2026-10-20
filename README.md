# Beta-lactamase mutation transfer

Computational analysis of mutation-effect prediction and transfer between the B1 metallo-β-lactamases NDM-1 and VIM-2. Published deep mutational scanning measurements are compared across homologous residue positions, antibiotic conditions, temperatures, and ampicillin concentrations. The study examines amino-acid representations, ESM-C and ESM-2, experimental structural descriptors, residue graphs, residue geometry, local atomic voxels, and sequence–graph models.

## Repository organization

- `scripts/data/`: source acquisition, assay and residue-number mapping, structure preparation, and evaluation definitions (commands 01–19).
- `scripts/analysis/`: baseline, sequence, structural and neural evaluations; transfer studies; source-position holdouts; and numerical verification (commands 20–55).
- `scripts/figures/`: scripts for main figures 2–6, Supplementary Figure S12, and figure-source inspection. Figure 1 is assembled separately from recorded scientific components.
- `src/betalactamase_transfer/`: reusable Python implementations.
- `tests/`: unit and integration tests.
- `docs/`: methodological descriptions, scientific checks, and [repository layout](docs/repository_layout.md).
- `figures/main/`: six main figure PDFs; `figures/supplementary/`: supplementary figure PDFs and their source tables.
- `data/raw/`, `data/interim/`, and `data/processed/`: local scientific inputs and outputs; see [data/README.md](data/README.md) and [RIGHTS.md](RIGHTS.md) before sharing any files.

## Execution

Use the Python environment specified by `requirements.txt`, `requirements-plm.txt`, `requirements-graph.txt`, and `requirements-dev.txt`, as applicable. See [docs/execution.md](docs/execution.md) for dependencies and execution requirements. In particular, GPU model training, sequence embedding extraction, external data acquisition, and structure preparation are not interchangeable with analyses of previously saved results.

### External secondary-structure dependency

Secondary-structure regeneration requires a working `mkdssp` 4.x installation in addition to the Python dependencies. The study environment recorded `mkdssp` 4.6.1. The executable may be on `PATH`, at `~/.local/bin/mkdssp`, or provided with `python scripts/data/15_prepare_secondary_structure.py --mkdssp /absolute/path/to/mkdssp`. The libcifpp Chemical Component Dictionary used by `mkdssp` must also be available. Regeneration fails on an unsupported DSSP major version, missing output, unexpected residue counts, coordinate-identity mismatches, or `Unknown compound:` warnings.


The executable commands are located at:

### Data acquisition and preparation

```
scripts/data/01_map_reference_sequences.py
scripts/data/02_describe_source_tables.py
scripts/data/03_extract_mutation_measurements.py
scripts/data/04_define_assays_and_conditions.py
scripts/data/05_prepare_protein_structures.py
scripts/data/06_acquire_alphafold_models.py
scripts/data/07_prepare_alphafold_models.py
scripts/data/08_acquire_mcsa_annotations.py
scripts/data/09_acquire_mcsa_reference_sequences.py
scripts/data/10_prepare_mcsa_correspondence.py
scripts/data/11_acquire_card_ontology.py
scripts/data/12_prepare_card_context.py
scripts/data/13_prepare_residue_evidence.py
scripts/data/14_prepare_structural_descriptors.py
scripts/data/15_prepare_secondary_structure.py
scripts/data/16_prepare_processed_tables.py
scripts/data/17_define_modelling_tasks.py
scripts/data/18_define_evaluation_partitions.py
scripts/data/19_prepare_modelling_views.py
```

### Analyses and numerical checks

```
scripts/analysis/20_run_baseline_evaluation.py
scripts/analysis/21_calculate_baseline_uncertainty.py
scripts/analysis/22_analyze_homologue_baseline_transfer.py
scripts/analysis/23_evaluate_sequence_neighbourhood.py
scripts/analysis/24_prepare_sequence_plm_embeddings.py
scripts/analysis/25_evaluate_sequence_plm.py
scripts/analysis/26_compare_sequence_models.py
scripts/analysis/27_evaluate_experimental_structure.py
scripts/analysis/28_compare_structure_matched_models.py
scripts/analysis/29_analyze_biological_errors.py
scripts/analysis/30_prepare_residue_graphs.py
scripts/analysis/31_evaluate_residue_neighbourhood.py
scripts/analysis/32_evaluate_residue_graph_neural.py
scripts/analysis/33_evaluate_residue_geometry_neural.py
scripts/analysis/34_evaluate_sequence_residue_graph.py
scripts/analysis/35_analyze_sequence_structure_complementarity.py
scripts/analysis/36_analyze_biological_model_differences.py
scripts/analysis/37_analyze_training_seed_stability.py
scripts/analysis/38_analyze_sequence_model_sensitivity.py
scripts/analysis/39_evaluate_condition_transfer.py
scripts/analysis/40_evaluate_concentration_transfer.py
scripts/analysis/41_analyze_condition_concentration_biology.py
scripts/analysis/42_analyze_matched_homologue_models.py
scripts/analysis/43_evaluate_atomic_local_voxel.py
scripts/analysis/44_summarize_model_performance.py
scripts/analysis/45_analyze_homologue_prediction_diagnostics.py
scripts/analysis/46_check_research_file_hashes.py
scripts/analysis/47_audit_source_selection_records.py
scripts/analysis/48_evaluate_position_withheld_homologue_baselines.py
scripts/analysis/49_verify_source_fitting_boundaries.py
scripts/analysis/50_verify_source_ridge_selection.py
scripts/analysis/51_validate_mutation_effect_results.py
scripts/analysis/52_compare_source_fitting_populations.py
scripts/analysis/53_check_structural_associations.py
scripts/analysis/54_evaluate_source_position_withheld_esmc.py
scripts/analysis/55_summarize_source_position_predictions.py
```


These are command locations, not a command to execute every analysis consecutively. Individual commands have different data, device, and resource requirements. For example, sequence embeddings and neural-model outputs should not be regenerated when analysing saved results. The script-specific documentation and available command-line options determine how each command should be run.

## Scientific conventions

- Raw source files are not modified by the preparation programs.
- Original experimental scores are preserved unchanged. Primary DMS regression uses a separate `score_analysis` field that applies the reported minimum fitness score; missing scores are not imputed and scores are not globally standardised.
- Experimental conditions describe the documented assay, even when one condition field is supplied by the publication Methods rather than encoded in a spreadsheet column name.
- Chen et al. (2020) VIM-2 observations retain their condition-specific 25 °C or 37 °C temperatures.
- Chen et al. (2022) VIM-2 concentration observations are recorded at 37 °C, as reported in the final Nature Ecology & Evolution Methods.
- Chen et al. (2024) NDM-1 observations are recorded at 37 °C, as reported in the Nature Communications Methods.
- VIM-2 numbering retains the artificial Gly2 in the assay construct.
- Values reproduced in later comparison tables are not counted as independent measurements.
- NDM-1 and VIM-2 are compared directly for ampicillin only at the study-defined concentrations chosen to provide similar selection pressure.
- Cefotaxime and meropenem measurements remain condition-specific and are not treated as directly comparable between proteins solely because the antibiotic name matches.
- Experimental PDB structures remain the primary structural source. AlphaFold is used for coverage, confidence, correspondence, and later sensitivity analyses; predicted mutant structures are not used.
- CARD supplies AMR family, drug-class, and mechanism context rather than mutation-effect labels or residue numbering.
- M-CSA supplies mechanistic reference information and is kept separate from CARD and from experimental mutation labels.
- Residue-level M-CSA fields describe correspondence to curated M-CSA reference residues; they are not relabelled as direct experimental catalytic annotations of NDM-1 or VIM-2.
- The complete Chen et al. (2024) aligned-position source yields 217 one-to-one NDM-1/VIM-2 position pairs after validated assay/reference mapping: 81 shared-WT and 136 differing-WT positions. The separate 132 WT-reversion comparison rows remain outcome-based supporting evidence and are not used as the alignment.
- Sequence-to-PDB correspondence is accepted only where all equally scoring optimal alignments agree. The known NDM-1/3SPU tie is confined to unmodelled PDB entity position 1; no coordinate-bearing residue is resolved by arbitrary alignment order.
- Processed observation columns have declared roles so outcomes, grouping fields, evaluation metadata, and source-record fields cannot silently become default model inputs.
- Task-definition preparation sets task meaning before partitions are generated: cross-protein model selection uses source-protein validation only, VIM-2 concentration and condition transfer remain distinct from unseen-mutation prediction, and direct paired equivalent-mutation reporting uses only the 81 shared-WT aligned positions without consulting mutation-effect values.
- Evaluation-partition preparation assigns residue-position or mutation groups according to the scientific claim. Group assignment does not use mutation-effect scores and keeps held-out labels unavailable during model selection.
- Modelling-view preparation fixes modelling input roles and exact scheme membership before feature generation. Biological reference fields reserved for later validation do not become default predictors, and modality comparisons use matched eligible rows.


## Figure generation

Figures 2–6 have source programs in `scripts/figures/`. Supplementary Figure S12 is generated by `scripts/figures/generate_source_position_prediction_figure.py` using the saved ROC AUC table in its directory and the 10,000-resample position-bootstrap comparison table in `data/interim/source_position_prediction_tables/`. Figure 1 is assembled separately from scientific components. The saved numerical inputs should be checked before replacing a published figure PDF.

## Research data and external references

Experimental mutation measurements were obtained from the studies by Chen and colleagues (2020, 2022 and 2024), and structural and annotation records were obtained from their cited databases. The original source data and downloaded reference files are retained separately from author-generated tables, figures and predictions. Third-party data are not automatically covered by the repository's software licence. Refer to [RIGHTS.md](RIGHTS.md), [data/README.md](data/README.md), and the original publications for the applicable terms.

The scientific outputs include numerical comparison tables in `data/interim/` and the generated figures in `figures/`. Files containing original experimental measurements or predictions copied directly from source assays require independent redistribution review.

## Verification

Run from the repository root with the project environment active:

```bash
ruff check .
ruff format --check .
python -m compileall -q src scripts tests
pytest -ra
```

For detailed task definitions, structure selection, source-only fitting restrictions, and evaluation results, see [docs/](docs/).

## Citation and licence

See [CITATION.cff](CITATION.cff) for software citation information, [LICENSE](LICENSE) for the code licence, and [RIGHTS.md](RIGHTS.md) for authorship and third-party material restrictions.