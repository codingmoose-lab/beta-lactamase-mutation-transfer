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

For the 2022 VIM-2 concentration study, the final Nature Ecology & Evolution article is the Methods reference. The retained supplementary workbook remains the measurement source.

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

CARD ontology acquisition imports the selected frozen official ontology archive supplied by the researcher:

```bash
python scripts/data/11_acquire_card_ontology.py \
  --source-archive /path/to/card-ontology.tar.bz2
python scripts/data/12_prepare_card_context.py
```

CARD is used for AMR family, drug-class, and mechanism context. Direct exact-term assertions remain distinguishable from family- or ancestry-derived context.

## Residue evidence

```bash
python scripts/data/13_prepare_residue_evidence.py
```

This command prepares one row for every assay residue using the selected PDB chain, UniProt correspondence, AlphaFold mapping/confidence, coordinate-quality fields, and M-CSA reference correspondence. Source meanings remain explicit: M-CSA correspondence is not converted into a direct experimental catalytic label.

The command also prepares the complete publication-derived NDM-1/VIM-2 DMS position correspondence from the Chen et al. (2024) source data. Publication VIM-2 positions are converted through the validated reference-to-assay mapping. The resulting 217 one-to-one pairs comprise 81 shared-WT and 136 differing-WT positions. The separate 132 WT-reversion comparison rows remain outcome-derived supporting evidence and are not used to construct the correspondence.

## Experimental structural descriptors

```bash
python scripts/data/14_prepare_structural_descriptors.py
```

This command derives residue-level experimental-coordinate descriptors and the complete pair geometry for the selected modeled residues. Cross-environment regeneration is checked numerically; machine-scale floating-point differences are not removed by arbitrary rounding.

## Secondary structure

```bash
python scripts/data/15_prepare_secondary_structure.py
```

This command runs `mkdssp` on the deposited 3SPU and 5YD7 coordinate files and maps DSSP assignments to the selected chain-B assay positions. The original DSSP class and a reduced helix/strand/coil class are both retained. Residues without an experimental coordinate remain in the output with an explicit missing-coordinate status.

The command requires `mkdssp` 4.x and now fails explicitly when the detected executable reports another major version. The accepted local installation used for the study was `mkdssp` 4.6.1, built from DSSP source revision `526448b48e05e30de4912a186faace700050860d`. The Chemical Component Dictionary must be available to libcifpp; a working installation should not emit `Unknown compound:` messages for the selected structures.

DSSP solvent-accessibility values are not used as a replacement for the Shrake-Rupley/Wilke accessibility descriptors prepared during structural-descriptor preparation. Secondary-structure preparation uses DSSP for secondary-structure assignment only.

## Processed research tables

```bash
python scripts/data/16_prepare_processed_tables.py
```

Processed-table preparation preserves `score_original` unchanged and creates `score_analysis`, which applies only the publication-defined lower DMS fitness floor recorded for each assay. It also creates the checked observation, residue, pair-geometry, reported-comparison, EC50, and assay tables. It also writes `observation_column_roles.csv`. Only fields declared as default mutation or condition inputs are permitted by default; outcomes, grouping identifiers, evaluation-only fields, and provenance/quality fields remain excluded unless a later task definition explicitly justifies a different use. No train/validation/test assignment is created here.

## Task definitions

```bash
python scripts/data/17_define_modelling_tasks.py
```

task-definition preparation reads the checked processed-table preparation tables and records the exact scientific task scopes and evaluation rules. It does not create train/validation/test assignments. The two headline transfer tasks use the study-defined AMP comparison settings in both directions, with source-protein validation required before the target protein is evaluated.

The task definitions distinguish unseen-position, unseen-mutation, antibiotic-transfer, temperature-transfer, and concentration-transfer claims. They also define held-out aligned shared-WT reporting using the complete 217-position correspondence while keeping source-only cross-protein model selection unchanged. The 132 Chen 2024 WT-reversion rows remain separate supporting evidence. See `docs/task_definitions.md`.

## Evaluation partitions

```bash
python scripts/data/18_define_evaluation_partitions.py
```

evaluation-partition preparation converts the frozen task definitions into deterministic group-level partitions. Within-protein evaluation is grouped by residue position; unseen-mutation concentration evaluation is grouped by exact mutation; cross-protein and condition-transfer model selection uses source data only before one final held-out evaluation. Assignment never uses mutation-effect scores. See `docs/evaluation_partitions.md`.

No modelling feature matrix is created by this command.

## Modelling views and input contracts

```bash
python scripts/data/19_prepare_modelling_views.py
```

modelling-view preparation joins the frozen task/scheme definitions to the exact observation IDs
that belong to each evaluation scope and records which observation, residue, and
pair-geometry fields may be used by later models. The prediction target is not
written into the scheme-membership table. M-CSA/mechanistic reference fields and
the Chen-2024 reported-pair fields are reserved from the primary predictor set
so later biological validation remains independent. Representation comparisons
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
python scripts/data/20_run_baseline_evaluation.py
```

This command evaluates deterministic simple baselines under the frozen evaluation schemes, selects ridge regularisation only from permitted validation/source rows, and prepares the direct 1,513-pair homologue transfer analysis. It does not create sequence embeddings, residue graphs, structural images, or neural models.

After those predictions are frozen, calculate grouped-bootstrap uncertainty with:

```bash
python scripts/data/21_calculate_baseline_uncertainty.py
```

## Matched homologue baseline analysis

After baseline predictions and grouped uncertainty are frozen, run:

```bash
python scripts/data/22_analyze_homologue_baseline_transfer.py
```

This command performs the outcome-independent 1,513-pair shared-wild-type homologue
analysis, floor sensitivity checks, the 478-pair neither-at-floor analysis, and the
306 discordant-floor analysis. It is post-hoc analysis of frozen baseline predictions;
it does not refit predictive models.

## Sequence-neighbourhood evaluation

Run:

```bash
python scripts/data/23_evaluate_sequence_neighbourhood.py
```

Here, `neighbourhood` means the biological amino-acid neighbourhood around the mutated
residue. The command reuses the frozen evaluation partitions and compares the same
mutation/condition model with and without explicit neighbouring amino-acid states.

The VIM-2 2022 unseen-mutation experiment must not be described as unseen-position
generalisation. In the frozen folds, every held-out mutation occurs at a residue
position represented in fitting. At radius 16, all 267 VIM-2 assay positions have
unique sequence-neighbourhood signatures, and every test signature is already present
in the corresponding fitting population. Consequently, the strong VIM-2 2022 result
shows that raw sequence identity is useful for unseen mutations at familiar positions;
it does not demonstrate generalisation to completely unseen positions.

The selected radius reaches the upper radius boundary in several settings, including
all five VIM-2 2022 folds. Because final evaluation results have now been inspected,
the Step-23 radius grid is frozen and must not be expanded merely to improve reported
performance. Any broader-context sequence experiment belongs to a separately specified
model family with model selection restricted to permitted training/validation rows.


## Protein-language-model embeddings

ESMC extraction is performed in a separate CUDA-enabled environment because it
requires GPU-enabled PyTorch and substantially more compute than the preceding
data preparation and statistical modelling commands.

Prepare and validate the outcome-blind mutation-to-embedding index with:

    python scripts/data/24_prepare_sequence_plm_embeddings.py

Run GPU extraction with the pinned checkpoint already available in the local
model cache using:

    python scripts/data/24_prepare_sequence_plm_embeddings.py --extract

`--allow-download` is used only when an intentional checkpoint download is
required. `--overwrite` is used only when replacing existing embedding outputs
is scientifically intended.

The experimentally verified assay constructs are supplied directly to the
protein language model. Generated embeddings and provenance are written under
`data/interim/sequence_plm_embeddings/`. Model weights remain in the machine
model cache and are not added to the research repository.

The successful production environment used Python 3.11.15, PyTorch runtime
`2.13.0+cu130`, CUDA 13.0, Transformers 4.57.6, and an NVIDIA GeForce RTX
4060 Ti. `requirements-plm.txt` records package versions known from the verified
environment. No unverified historical PyTorch installation URL is asserted.

## ESMC mutation-effect evaluation

After validating the extracted arrays and their provenance, run:

    python scripts/data/25_evaluate_sequence_plm.py

The ESMC model uses the mutation-site hidden-state difference relative to the
corresponding wild-type residue, together with ordered substitution identity and
the task-appropriate condition information. Hidden-state index and ESMC ridge
regularisation are selected jointly from the predeclared candidate sets using
only the validation population permitted by each frozen evaluation scheme.
Final held-out labels are not used for model selection.

See `docs/sequence_plm_evaluation.md`.

## Two-workstation compute policy

The macOS workstation is the canonical working copy for the research repository
and the final validated analysis products. The WSL/CUDA workstation is used for
computations that genuinely require the NVIDIA GPU.

Repository-controlled source code, tests, and documentation are synchronized
through Git. The ignored `data/raw/`, `data/interim/`, and `data/processed/`
trees are not synchronized through Git. Required scientific inputs and generated
GPU products are transferred explicitly between workstations and verified by
checksum when identity matters.

A code synchronization alone does not require regeneration of an existing
scientific product. A computation is repeated only when a change affects its
inputs, model definition, evaluation design, or another assumption on which the
product depends.
