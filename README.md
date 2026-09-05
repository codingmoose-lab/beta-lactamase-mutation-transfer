# Beta-lactamase mutation transfer

Computational study of mutation effects across NDM-1 and VIM-2 under antibiotic, concentration, and temperature conditions. The repository prepares published deep-mutational-scanning measurements, maps residue numbering between assay and reference sources, prepares experimental and AlphaFold structures, and records catalytic and AMR annotations for sequence- and structure-informed mutation-effect analyses.

## Reproducible research workflow

Run the repository-controlled commands from the repository root in numerical order where their required local inputs are available:

```bash
python scripts/data/01_map_reference_sequences.py
python scripts/data/02_describe_source_tables.py
python scripts/data/03_extract_mutation_measurements.py
python scripts/data/04_define_assays_and_conditions.py
python scripts/data/05_prepare_protein_structures.py
python scripts/data/06_acquire_alphafold_models.py
python scripts/data/07_prepare_alphafold_models.py
python scripts/data/08_acquire_mcsa_annotations.py
python scripts/data/09_acquire_mcsa_reference_sequences.py
python scripts/data/10_prepare_mcsa_correspondence.py
python scripts/data/11_acquire_card_ontology.py --source-archive /path/to/card-ontology.tar.bz2
python scripts/data/12_prepare_card_context.py
python scripts/data/13_prepare_residue_evidence.py
python scripts/data/14_prepare_structural_descriptors.py
python scripts/data/15_prepare_secondary_structure.py
python scripts/data/16_prepare_processed_tables.py
python scripts/data/17_define_modelling_tasks.py
python scripts/data/18_define_evaluation_partitions.py
python scripts/data/19_prepare_modelling_views.py
python scripts/data/20_run_baseline_evaluation.py
python scripts/data/21_calculate_baseline_uncertainty.py
python scripts/data/22_analyze_homologue_baseline_transfer.py
python scripts/data/23_evaluate_sequence_neighbourhood.py
python scripts/data/24_prepare_sequence_plm_embeddings.py
# GPU extraction is performed separately with --extract; see docs/execution.md.
python scripts/data/25_evaluate_sequence_plm.py
python scripts/data/26_compare_sequence_models.py
python scripts/data/27_evaluate_experimental_structure.py
python scripts/data/28_compare_structure_matched_models.py
python scripts/data/29_analyze_biological_errors.py
python scripts/data/30_prepare_residue_graphs.py
python scripts/data/31_evaluate_residue_neighbourhood.py
python scripts/data/32_evaluate_residue_graph_neural.py
python scripts/data/33_evaluate_residue_geometry_neural.py
python scripts/data/34_evaluate_sequence_residue_graph.py
python scripts/data/35_analyze_sequence_structure_complementarity.py --check-only
python scripts/data/36_analyze_biological_model_differences.py --check-only
python scripts/data/37_analyze_training_seed_stability.py --check-only
python scripts/data/38_analyze_sequence_model_sensitivity.py --check-only
python scripts/data/39_evaluate_condition_transfer.py --check-only
python scripts/data/40_evaluate_concentration_transfer.py --check-only
python scripts/data/41_analyze_condition_concentration_biology.py --check-only
python scripts/data/42_analyze_matched_homologue_models.py --check-only
```

Acquisition commands that contact external services or import a separately downloaded archive should only be rerun when the corresponding source is intentionally being refreshed. Existing local raw sources are otherwise left unchanged.

### External secondary-structure dependency

Secondary-structure regeneration requires a working `mkdssp` 4.x installation in addition to the Python dependencies. The study environment recorded `mkdssp` 4.6.1. The executable may be on `PATH`, at `~/.local/bin/mkdssp`, or provided with `python scripts/data/15_prepare_secondary_structure.py --mkdssp /absolute/path/to/mkdssp`. The libcifpp Chemical Component Dictionary used by `mkdssp` must also be available. Regeneration fails on an unsupported DSSP major version, missing output, unexpected residue counts, coordinate-identity mismatches, or `Unknown compound:` warnings.

The current programs produce linked tables under:

- `data/interim/sequence_mapping/`: assay, UniProt, PDB, and residue-number correspondence;
- `data/interim/source_tables/`: workbook, worksheet, and column inventories;
- `data/interim/mutation_measurements/`: long-form DMS, EC50, comparison, validation, and structure-mapping tables;
- `data/interim/assay_definitions/`: assay metadata, condition summaries, grouping identifiers, and cross-protein comparison records;
- `data/interim/protein_structures/`: experimental-chain selection, measured-position coverage, metal sites, ligand sites, and coordinate checks;
- `data/interim/alphafold_acquisition/` and `alphafold_preparation/`: exact AlphaFold models, confidence, PAE, assay mapping, and PDB correspondence;
- `data/interim/mcsa_acquisition/`, `mcsa_reference_sequences/`, and `mcsa_correspondence/`: selected M-CSA records and catalytic-residue correspondence;
- `data/interim/card_acquisition/` and `card_context/`: selected CARD ontology context, drug-class mapping, and resistance-mechanism annotations;
- `data/interim/residue_evidence/`: one selected-reference row per assay residue, the complete publication-reported NDM-1/VIM-2 aligned-position correspondence, and the separate Chen et al. (2024) reported WT-reversion comparison positions;
- `data/interim/structural_descriptors/`: experimental-coordinate residue descriptors and pair geometry;
- `data/interim/secondary_structure/`: retained DSSP assignments for selected experimental chains;
- `data/processed/`: checked research tables plus declared observation-column roles for later modelling;
- `data/interim/task_definitions/`: task scopes and evaluation rules;
- `data/interim/evaluation_partitions/`: grouped evaluation assignments and partition audits;
- `data/interim/modelling_views/`: modelling-view observation membership, input contracts, and matched modality-eligible populations;
- `data/interim/baseline_evaluation/`: baseline predictions, grouped uncertainty, and matched homologue analyses;
- `data/interim/sequence_neighbourhood_evaluation/`: mutation/condition and amino-acid-neighbourhood comparisons on the predefined core schemes;
- `data/interim/sequence_plm_embeddings/`: outcome-blind ESMC mutation-site and wild-type residue embeddings plus extraction provenance;
- `data/interim/sequence_plm_evaluation/`: ESMC site-delta predictions, metrics, and model-selection records.
- `data/interim/sequence_structure_complementarity/`: validation-selected sequence/graph combinations, matched final comparisons, and predictive-complementarity diagnostics.
- `data/interim/training_seed_stability/`: fixed-setting training-seed metrics and model-comparison stability records.
- `data/interim/biological_model_difference_analysis/`: position-level structural associations, condition-stratified error differences, homologue-pair comparisons, and task concordance for the evaluated model families.
- `data/interim/condition_transfer_evaluation/`: held-out-antibiotic and temperature-transfer predictions, source-only model-choice records, paired comparisons, and condition-shift summaries.
- `data/interim/concentration_response_evaluation/`: held-out-concentration predictions, model-choice records, paired comparisons, and interpolation/extrapolation summaries.
- `data/interim/condition_concentration_biological_analysis/`: condition and concentration transfer summaries at mutation and position levels.
- `data/interim/matched_homologue_model_analysis/`: matched NDM-1/VIM-2 model predictions, grouped comparisons, background-difference associations, and position summaries.

## Scientific conventions

- Raw source files are not modified by the preparation programs.
- Original experimental scores are preserved unchanged. Primary DMS regression uses a separate `score_analysis` field that applies only the publication-defined lower fitness floor; missing scores are not imputed and scores are not globally standardised.
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
- Processed observation columns have declared roles so outcomes, grouping fields, evaluation metadata, and provenance cannot silently become default model inputs.
- Task-definition preparation sets task meaning before partitions are generated: cross-protein model selection uses source-protein validation only, VIM-2 concentration and condition transfer remain distinct from unseen-mutation prediction, and direct paired equivalent-mutation reporting uses only the 81 shared-WT aligned positions without consulting mutation-effect values.
- Evaluation-partition preparation assigns residue-position or mutation groups according to the scientific claim. Group assignment does not use mutation-effect scores and keeps held-out labels unavailable during model selection.
- Modelling-view preparation fixes modelling input roles and exact scheme membership before feature generation. Biological reference fields reserved for later validation do not become default predictors, and modality comparisons use matched eligible rows.

See `docs/` for assay definitions, residue numbering, experimental-structure selection, evaluation design, sequence-model analyses, repository layout, and execution details.

## Quality checks

```bash
ruff check .
ruff format --check .
python -m compileall -q src scripts tests
pytest -ra
```
