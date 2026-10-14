# Reproducible execution

Run commands from the repository root. The numbered files under `scripts/data/` are research commands for source inspection, data curation, sequence or structure mapping, and biological annotation. Reusable implementation code lives under `src/betalactamase_transfer/`.

## Experimental measurements and reference mapping

```bash
python scripts/data/01_map_reference_sequences.py
python scripts/data/02_describe_source_tables.py
python scripts/data/03_extract_mutation_measurements.py
python scripts/data/04_define_assays_and_conditions.py
python scripts/data/05_prepare_protein_structures.py
```

The mutation-measurement parser combines condition information encoded in workbook headers with fixed study-level conditions reported in the corresponding publication Methods. In particular, the 2022 VIM-2 concentration experiment and the 2024 NDM-1 DMS were conducted at 37 °C; their spreadsheet score headers encode concentration and/or antibiotic but not temperature. The observation tables therefore record the documented 37 °C assay temperature rather than leaving it missing.

For the 2022 VIM-2 concentration study, the published Nature Ecology & Evolution article is the Methods reference. The retained supplementary workbook remains the measurement source.

## AlphaFold

```bash
python scripts/data/06_acquire_alphafold_models.py
python scripts/data/07_prepare_alphafold_models.py
```

The acquisition command retrieves only the selected UniProt accessions. Preparation records confidence, PAE summaries, assay correspondence, and correspondence to the selected experimental structures.

## M-CSA

```bash
python scripts/data/08_acquire_mcsa_annotations.py
python scripts/data/09_acquire_mcsa_reference_sequences.py
python scripts/data/10_prepare_mcsa_correspondence.py
```

M-CSA residue roles are mapped through the selected reference proteins and are not treated as direct experimental annotations of NDM-1 or VIM-2 where residue identity or evidence differs.

## CARD

CARD ontology acquisition imports the selected CARD ontology 4.0.1 archive:

```bash
python scripts/data/11_acquire_card_ontology.py \
  --source-archive /path/to/card-ontology.tar.bz2
python scripts/data/12_prepare_card_context.py
```

CARD is used for AMR family, drug-class, and mechanism context. Direct exact-term assertions remain distinguishable from family context or context inherited through ontology ancestry.

## Residue evidence

```bash
python scripts/data/13_prepare_residue_evidence.py
```

This command prepares one row for every assay residue using the selected PDB chain, UniProt correspondence, AlphaFold mapping/confidence, coordinate-quality fields, and M-CSA reference correspondence. M-CSA correspondence is retained as reference correspondence and is not converted into a direct experimental catalytic label.

The command also prepares the complete publication-reported NDM-1/VIM-2 DMS position correspondence from the Chen et al. (2024) source data. Publication VIM-2 positions are converted through the validated reference-to-assay mapping. The resulting 217 one-to-one pairs comprise 81 shared-WT and 136 differing-WT positions. The separate 132 WT-reversion comparison rows remain outcome-based supporting evidence and are not used to construct the correspondence.

## Experimental structural descriptors

```bash
python scripts/data/14_prepare_structural_descriptors.py
```

This command derives residue-level experimental-coordinate descriptors and the complete pair geometry for the selected modeled residues. Cross-environment regeneration is checked numerically; machine-scale floating-point differences are not removed by arbitrary rounding.

## Secondary structure

```bash
python scripts/data/15_prepare_secondary_structure.py
```

This command runs `mkdssp` on the deposited 3SPU and 5YD7 coordinate files and maps DSSP assignments to the selected chain-B assay positions. The original DSSP class and a reduced helix/strand/coil class are both retained. Residues without an experimental coordinate remain in the output with a missing-coordinate status.

The command requires `mkdssp` 4.x and rejects another detected major version. The validated study installation was `mkdssp` 4.6.1, built from DSSP source revision `526448b48e05e30de4912a186faace700050860d`. The Chemical Component Dictionary must be available to libcifpp; a working installation should not emit `Unknown compound:` messages for the selected structures.

DSSP solvent-accessibility values are not used as a replacement for the Shrake-Rupley/Wilke accessibility descriptors prepared during structural-descriptor preparation. Secondary-structure preparation uses DSSP for secondary-structure assignment only.

## Processed research tables

```bash
python scripts/data/16_prepare_processed_tables.py
```

Processed-table preparation preserves `score_original` unchanged and creates `score_analysis`, which applies the reported minimum fitness score recorded for each assay. It also creates the checked observation, residue, pair-geometry, reported-comparison, EC50, and assay tables. It also writes `observation_column_roles.csv`. Only fields declared as default mutation or condition inputs are permitted by default; outcomes, grouping identifiers, evaluation-only fields, and source and quality fields remain excluded unless a task definition permits a different use. No train/validation/test assignment is created here.

## Task definitions

```bash
python scripts/data/17_define_modelling_tasks.py
```

task-definition preparation reads the checked processed-table preparation tables and records the exact scientific task scopes and evaluation rules. It does not create train/validation/test assignments. The two headline transfer tasks use the study-defined AMP comparison settings in both directions, with source-protein validation required before the target protein is evaluated.

The task definitions distinguish unseen-position, unseen-mutation, antibiotic-transfer, temperature-transfer, and concentration-transfer claims. They also define aligned shared-WT reporting using the complete 217-position correspondence while keeping cross-protein model selection restricted to the source protein. The 132 Chen 2024 WT-reversion rows remain separate supporting evidence. See `docs/task_definitions.md`.

## Evaluation partitions

```bash
python scripts/data/18_define_evaluation_partitions.py
```

Evaluation-partition preparation converts the task definitions into group-level partitions. Within-protein evaluation is grouped by residue position; unseen-mutation concentration evaluation is grouped by exact mutation; cross-protein and condition-transfer model selection uses source data only, followed by one test on the target population. Assignment never uses mutation-effect scores. See `docs/evaluation_partitions.md`.

No modelling feature matrix is created by this command.

## Modelling views and input contracts

```bash
python scripts/data/19_prepare_modelling_views.py
```

Modelling-view preparation joins the task and scheme definitions to the exact observation IDs
that belong to each evaluation scope and records the observation, residue, and
pair-geometry fields available to prediction models. The prediction target is not
written into the scheme-membership table. M-CSA/mechanistic reference fields and
the Chen-2024 reported-pair fields are excluded from the primary predictor set
and retained for biological validation. Representation comparisons
use matched eligible rows. See `docs/modelling_views.md`.

No embeddings, graph tensors, structural images, or trained models are created
by this command.

## Validation

Before accepting a change, run:

```bash
ruff check .
ruff format --check .
python -m compileall -q src scripts tests
pytest -ra
```

Scientific acceptance also requires inspection of the generated tables affected by the change; a program-level PASS message alone is not treated as sufficient evidence.

## Baseline evaluation

After the preparation, task, partition, and modelling-view checks are closed, run:

```bash
python scripts/analysis/20_run_baseline_evaluation.py
```

This command evaluates simple baselines under the specified evaluation schemes, selects ridge regularisation only from permitted validation/source rows, and prepares the direct 1,513-pair homologue transfer analysis. It does not create sequence embeddings, residue graphs, structural images, or neural models.

After the baseline predictions are written, calculate grouped-bootstrap uncertainty with:

```bash
python scripts/analysis/21_calculate_baseline_uncertainty.py
```

## Matched homologue baseline analysis

After baseline predictions and grouped uncertainty are available, run:

```bash
python scripts/analysis/22_analyze_homologue_baseline_transfer.py
```

This command performs the outcome-independent 1,513-pair shared-wild-type homologue
analysis, minimum-score sensitivity checks, the 478-pair both-above-minimum analysis, and the
306 discordant minimum-score analysis. It is post-hoc analysis of saved baseline predictions;
it does not refit prediction models.

## Sequence-neighbourhood evaluation

Run:

```bash
python scripts/analysis/23_evaluate_sequence_neighbourhood.py
```

Here, `neighbourhood` means the biological amino-acid neighbourhood around the mutated
residue. The command reuses the specified evaluation partitions and compares the same
mutation/condition model with and without neighbouring amino-acid states.

The VIM-2 2022 unseen-mutation experiment must not be described as unseen-position
performance on unseen mutations. In these grouped folds, every test mutation occurs at a residue
position represented in fitting. At radius 16, all 267 VIM-2 assay positions have
unique sequence-neighbourhood signatures, and every test signature is already present
in the corresponding fitting population. Consequently, the strong VIM-2 2022 result
shows that raw sequence identity is useful for unseen mutations at familiar positions;
it does not test completely unseen positions.

The selected radius reaches the upper radius boundary in several settings, including
all five VIM-2 2022 folds. The reported analysis retains the tested radii 4, 8, 12,
and 16 because test results have already been inspected. A broader-context sequence
experiment requires a separately specified model with selection restricted to
training and validation rows.


## Protein-language-model embeddings

ESMC extraction is performed in a separate CUDA-enabled environment because it
requires GPU-enabled PyTorch and substantially more compute than the preceding
data preparation and statistical modelling commands.

Prepare and validate the mutation-to-embedding index constructed without mutation-effect outcomes with:

    python scripts/analysis/24_prepare_sequence_plm_embeddings.py

Run GPU extraction with the pinned checkpoint already available in the local
model cache using:

    python scripts/analysis/24_prepare_sequence_plm_embeddings.py --extract

`--allow-download` is used only when an intentional checkpoint download is
required. `--overwrite` is used only when replacing existing embedding outputs
is scientifically intended.

The experimentally verified assay constructs are supplied directly to the
protein language model. Generated embeddings and extraction records are written under
`data/interim/sequence_plm_embeddings/`. Model weights remain in the machine
model cache and are not added to the research repository.

The ESMC extraction environment used Python 3.11.15, PyTorch runtime
`2.13.0+cu130`, CUDA 13.0, Transformers 4.57.6, and an NVIDIA GeForce RTX
4060 Ti. `requirements-plm.txt` records package versions known from the verified
environment. No unverified historical PyTorch installation URL is asserted.

## ESMC mutation-effect evaluation

After validating the extracted arrays and their extraction records, run:

    python scripts/analysis/25_evaluate_sequence_plm.py

The ESMC model uses the mutation-site hidden-state difference relative to the
corresponding wild-type residue, together with ordered substitution identity and
the task-appropriate condition information. The hidden-state index and ESMC ridge
regularisation are selected as a pair from candidate sets defined before evaluation,
using only the validation population permitted by each evaluation scheme. Test labels
are not used for model selection.

See `docs/sequence_plm_evaluation.md`.

## Sequence model comparison

After the mutation/condition, sequence-neighbourhood, and ESMC predictions are
available, run:

```bash
python scripts/analysis/26_compare_sequence_models.py
```

The command compares the three models on identical test observations and calculates
paired bootstrap intervals using `evaluation_group_id` as the resampling unit. It does
not refit models or use test labels for model selection.

See `docs/sequence_model_comparison.md`.

## Experimental-structure descriptor evaluation

After the sequence-model comparison is available, run:

```bash
python scripts/analysis/27_evaluate_experimental_structure.py
```

This command restricts fitting and evaluation to observations marked
`experimental_structure_task_eligible` and compares a structure-matched
mutation/condition ridge model with a ridge model that adds generic experimental
structure descriptors. Catalytic-zinc distances, M-CSA correspondence,
reported-homologue fields, raw coordinates, protein identity, and assay position are
not direct prediction inputs. They remain available for independent biological
interpretation.

See `docs/experimental_structure_evaluation.md`.

## Structure-matched model comparison

After Script 27 has produced the verified matched structural predictions, run:

```bash
python scripts/analysis/28_compare_structure_matched_models.py
```

This command compares mutation/condition, sequence-neighbourhood, ESMC, and
experimental-structure descriptor models on the same experimental-structure-eligible
fitting and test populations. Script 27 predictions are reused for the two models it
already evaluated; the sequence-neighbourhood and ESMC models are reselected and
refitted on the same matched training populations. All six model contrasts use paired
`evaluation_group_id` bootstrap resampling. Target-test labels remain outside model
selection.

This command is not the structural-image, residue-graph, or multimodal-fusion analysis.

See `docs/structure_matched_model_comparison.md`.

## Biological error analysis

After the structure-matched model comparison is available, run:

```bash
python scripts/analysis/29_analyze_biological_errors.py
```

This command analyzes the verified Script 28 prediction errors against catalytic-Zn
distance, M-CSA correspondence, and outcome-independent homologue correspondence.
SASA, residue-neighbour counts, contact counts, and secondary structure are reported
separately because they were inputs to the experimental-structure model. Prediction
errors retain the evaluation groups used by each scheme, while biological associations
are summarized and resampled by residue position so repeated mutations at one residue
do not count as independent structural observations. No prediction model is fitted or
selected in this analysis.

See `docs/biological_error_analysis.md`.

## Experimental-structure residue-graph preparation

After the structure-matched analyses and biological error analysis have been
checked, prepare the experimental-structure residue graphs constructed without mutation-effect outcomes with:

```bash
python scripts/analysis/30_prepare_residue_graphs.py
```

The command builds one undirected graph for each selected experimental
structure using the audited C-alpha distance threshold of 8 Å. It does not fit
a graph model or perform model selection. Biological-reference fields,
mutation-effect values, predictions, and evaluation assignments do not enter
graph construction. Core evaluation membership is used only to verify graph
coverage after topology has been created.

See `docs/residue_graphs.md`.

## Residue-neighbourhood evaluation

Evaluate whether local three-dimensional residue neighbours add information relative to the mutated residue's own structural descriptors with:

```bash
python scripts/analysis/31_evaluate_residue_neighbourhood.py
```

The analysis uses the fixed experimental-structure residue graph and the same structure-supported observation populations used by the descriptor analysis.

See `docs/residue_neighbourhood_evaluation.md`.

## Residue-graph neural evaluation

Inspect the graph-model inputs with:

```bash
python scripts/analysis/32_evaluate_residue_graph_neural.py --validate-inputs
```

Then fit and evaluate the centre-residue control and residue-graph neural model on the intended compute device:

```bash
python scripts/analysis/32_evaluate_residue_graph_neural.py --device cuda
```

Model selection uses permitted validation rows for within-protein tasks and source-protein folds for cross-protein tasks. Final evaluation outcomes and cross-protein target outcomes are not used to select the training epoch.

See `docs/residue_graph_neural_evaluation.md`.

## Residue-geometry neural evaluation

Inspect the geometry-model inputs with:

```bash
python scripts/analysis/33_evaluate_residue_geometry_neural.py --validate-inputs
```

Then fit the E(n)-equivariant residue-geometry model on the intended compute device:

```bash
python scripts/analysis/33_evaluate_residue_geometry_neural.py --device cuda
```

The model uses the same residue graph, mutation encoding, assay-condition inputs, evaluation populations, and selection rules as the residue-graph neural comparison while adding three-dimensional residue geometry.

See `docs/residue_geometry_neural_evaluation.md`.

## Sequence-plus-residue-graph neural evaluation

Inspect the combined sequence/graph inputs with:

```bash
python scripts/analysis/34_evaluate_sequence_residue_graph.py --validate-inputs
```

Then fit the predefined ESMC-plus-residue-graph model on the intended compute device:

```bash
python scripts/analysis/34_evaluate_sequence_residue_graph.py --device cuda
```

The ESMC site-difference representation is combined with the established residue-graph model without reopening hidden-state selection after final evaluation results are available.

See `docs/sequence_residue_graph_neural_evaluation.md`.

## Sequence-structure complementarity

After the residue-graph and sequence-plus-residue-graph evaluations are available, inspect the sequence/graph combination inputs with:

```bash
python scripts/analysis/35_analyze_sequence_structure_complementarity.py --validate-inputs
```

Then run:

```bash
python scripts/analysis/35_analyze_sequence_structure_complementarity.py
```

The analysis selects a convex combination of the established ESMC and residue-graph predictions using permitted validation rows or source-protein cross-validation. Final evaluation outcomes and cross-protein target outcomes are not used to select the weight. The post-evaluation sequence-versus-graph oracle is diagnostic only and is never used for model selection.

See `docs/sequence_structure_complementarity.md`.

## Biological model-difference analysis

Inspect the matched prediction and biological-annotation inputs with:

```bash
python scripts/analysis/36_analyze_biological_model_differences.py --validate-inputs
```

Then run:

```bash
python scripts/analysis/36_analyze_biological_model_differences.py
```

This analysis does not fit or select a prediction model. It compares matched absolute-error differences for the centre-residue control, residue graph, residue geometry, ESMC sequence model, sequence-plus-residue-graph model, and validation-selected sequence/graph prediction. Structural associations use protein positions as the statistical unit. Condition summaries resample `evaluation_group_id`. The reported associations are descriptive and are not interpreted as evidence of a molecular cause.

See `docs/biological_model_difference_analysis.md`.

## Training-seed stability

Inspect the fixed model settings and saved three-seed predictions with:

```bash
python scripts/analysis/37_analyze_training_seed_stability.py --validate-inputs
```

A complete rebuild of the ten-seed stability analysis fits seven additional seeds per neural model and evaluation context and therefore requires the appropriate compute device:

```bash
python scripts/analysis/37_analyze_training_seed_stability.py --device cuda
```

When the completed additional fits are already present, regenerate only the summaries without fitting again:

```bash
python scripts/analysis/37_analyze_training_seed_stability.py --postprocess-only --device cpu
```

See `docs/training_seed_stability.md`.

## Sequence-model sensitivity

Inspect the ESMC results and the saved ESM-2 extraction records with:

```bash
python scripts/analysis/38_analyze_sequence_model_sensitivity.py --validate-inputs
```

With the ESM-2 embeddings already present, run the comparison with:

```bash
python scripts/analysis/38_analyze_sequence_model_sensitivity.py
```

Embedding extraction is a separate GPU operation and should only be repeated when the saved ESM-2 arrays are intentionally being rebuilt. See `docs/sequence_model_sensitivity.md`.

## Condition-transfer evaluation

Inspect and then evaluate the held-out-antibiotic and AMP temperature-transfer tests with:

```bash
python scripts/analysis/39_evaluate_condition_transfer.py --validate-inputs
python scripts/analysis/39_evaluate_condition_transfer.py
```

Settings are selected from source-condition rows only. Held-out condition outcomes are not used for model choice.

See `docs/condition_transfer_evaluation.md`.

## Concentration-transfer evaluation

Inspect and then evaluate the leave-one-concentration-out AMP tests with:

```bash
python scripts/analysis/40_evaluate_concentration_transfer.py --validate-inputs
python scripts/analysis/40_evaluate_concentration_transfer.py
```

The direct same-mutation concentration references are defined by interpolation or boundary extrapolation rules before target performance is examined.

See `docs/concentration_transfer_evaluation.md`.

## Condition and concentration biological analysis

Inspect and summarize the completed condition- and concentration-transfer results with:

```bash
python scripts/analysis/41_analyze_condition_concentration_biology.py --validate-inputs
python scripts/analysis/41_analyze_condition_concentration_biology.py
```

This analysis summarizes already evaluated transfer results and mutation/concentration trajectories. It does not fit another prediction model.

See `docs/condition_concentration_biological_analysis.md`.

## Matched homologue model analysis

Inspect and analyze model behavior on the 1,513 directly comparable substitutions at 81 aligned shared-WT positions with:

```bash
python scripts/analysis/42_analyze_matched_homologue_models.py --validate-inputs
python scripts/analysis/42_analyze_matched_homologue_models.py
```

The paired substitution population is defined without mutation-effect values. The 132 publication-reported WT-reversion rows remain separate supporting evidence.

See `docs/matched_homologue_model_analysis.md`.

## Atom-level local structural evaluation

Inspect coordinate mapping, voxel construction, comparator coverage, and evaluation populations with:

```bash
python scripts/analysis/43_evaluate_atomic_local_voxel.py --validate-inputs --device cpu
```

A complete rebuild fits the 3-D CNN on the selected experimental wild-type structures and requires the intended compute device:

```bash
python scripts/analysis/43_evaluate_atomic_local_voxel.py --device cuda
```

When all 17 completed context checkpoints are already present, regenerate only the final aggregation, metrics, and comparisons with:

```bash
python scripts/analysis/43_evaluate_atomic_local_voxel.py --postprocess-only --device cpu
```

The atom-level representation uses protein heavy atoms from the selected experimental PDB chains. Zinc ions, other ligands, mutation-effect outcomes, biological-reference annotations, and predicted mutant structures are excluded from voxel construction.

See `docs/atomic_local_voxel_evaluation.md`.
