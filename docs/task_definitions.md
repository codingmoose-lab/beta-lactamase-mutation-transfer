# Task definitions

The task definitions specify the prediction targets, evaluation populations, and transfer settings used in the NDM-1 and VIM-2 analyses.

## Preparation

```bash
python scripts/data/17_define_modelling_tasks.py
```

The script writes:

- `task_definitions.csv`
- `task_scopes.csv`
- `evaluation_regimes.csv`
- `available_conditions.csv`
- `aligned_shared_wild_type_mutation_pairs.csv`
- `task_definition_summary.json`

## Primary prediction tasks

The main cross-protein tasks evaluate mutation-effect transfer in both directions:

- NDM-1 → VIM-2
- VIM-2 → NDM-1

The matched ampicillin comparison uses the Chen et al. (2024) conditions selected to give similar selection pressure:

- NDM-1: 256 µg/mL ampicillin at 37 °C
- VIM-2: 128 µg/mL ampicillin at 37 °C

These concentrations are assay-specific and are not treated as interchangeable with other ampicillin conditions.

Within-protein regression is evaluated with position-grouped splits for NDM-1 and for the Chen et al. (2020) VIM-2 measurements.

The Chen et al. (2020) VIM-2 data also support antibiotic and temperature transfer analyses.

The Chen et al. (2022) concentration series supports separate evaluation of:

- unseen mutations
- concentration interpolation
- concentration-edge extrapolation

## Cross-protein model selection

For NDM-1 → VIM-2 transfer, model selection uses NDM-1 observations only.

For VIM-2 → NDM-1 transfer, model selection uses VIM-2 observations only.

Mutation-effect labels from the target protein are not used for hyperparameter selection, early stopping, feature selection, threshold selection, or other model-selection decisions.

The target protein sequence or structure may be supplied during inference when required by the tested representation.

## Mutation-effect scores

`score_original` retains the experimental value reported by the source study.

The primary DMS regression target is `score_analysis`. It applies the lower measurement limit associated with each assay. The current DMS datasets use a lower limit of -4, and values above that limit remain unchanged.

Scores are not normalised, rescaled, or harmonised across studies or experimental conditions.

For the matched NDM-1/VIM-2 ampicillin transfer analysis, each protein retains its original assay scale.

Spearman correlation is the primary transfer metric. Pearson correlation, MAE, and RMSE provide additional descriptions of predictive agreement.

Raw unfloored scores may be examined in sensitivity analyses.

EC50 measurements and control variants are not included in the primary missense DMS regression target.

## NDM-1/VIM-2 residue correspondence

The NDM-1/VIM-2 correspondence contains 217 one-to-one aligned residue positions.

Of these:

- 81 positions have the same wild-type amino acid in both proteins.
- 136 positions have different wild-type amino acids.

VIM-2 positions are converted through the validated reference-to-assay mapping, including the assay-specific Gly2 insertion and the known assay/reference sequence difference.

The correspondence is defined from sequence and residue-position mapping only. Mutation-effect values are not used to determine aligned positions.

## Equivalent-mutation pairs

Direct equivalent-mutation comparisons are restricted to the 81 aligned positions with the same wild-type amino acid in NDM-1 and VIM-2.

At the matched ampicillin conditions, 1,513 measured substitutions can be paired across these positions.

Pairs are identified from:

- aligned residue position
- wild-type amino-acid identity
- mutant amino-acid identity
- observation identity

Mutation-effect values are not used when defining the paired population.

The separate 132-row WT-reversion comparison reported by Chen et al. (2024) is retained as supporting experimental evidence. It is not used to define the NDM-1/VIM-2 residue correspondence or the model-selection population.

Those 132 positions are contained within the 136 aligned positions whose wild-type residues differ between NDM-1 and VIM-2.

## Representation eligibility

Sequence-based analyses may use all sequence-eligible missense observations.

Analyses requiring experimental coordinates use observations marked `experimental_structure_task_eligible`.

AlphaFold-supported analyses use observations marked `alphafold_structure_task_eligible` and are reported separately from analyses based on experimental structures.

Comparisons between representations use the same eligible observation population. Sequence results on the full missense population are reported separately from matched-population representation comparisons.

## Repeated VIM-2 mutations across studies

The Chen et al. (2020) and Chen et al. (2022) datasets share 5,005 VIM-2 mutation groups.

The two studies remain separate experimental sources.

Analyses that combine observations from both studies account for repeated mutation identities and study membership rather than treating all observations as independent measurements.

## Evaluation partitions

Task definitions do not assign observations to train, validation, or test sets.

Partition assignments are generated separately by `scripts/data/18_define_evaluation_partitions.py` using the grouping rules associated with each task.
