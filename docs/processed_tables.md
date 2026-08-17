# Processed tables

The processed tables contain the mutation-effect measurements, assay metadata, residue annotations, and residue-pair geometry used in the NDM-1 and VIM-2 analyses.

## Preparation

```bash
python scripts/data/16_prepare_processed_tables.py
```

The script writes the following files to `data/processed/`:

- `observations.csv`: validated mutation-effect observations retained from the source datasets.
- `missense_observations.csv`: single amino-acid substitutions used in the main mutation-effect analyses.
- `control_observations.csv`: retained synonymous, stop, and other control observations.
- `ec50_observations.csv`: EC50 measurements extracted from the VIM-2 studies.
- `assay_metadata.csv`: antibiotic, concentration, temperature, score definition, and assay identifiers.
- `residue_features.csv`: residue annotations and coordinate availability for the study proteins.
- `residue_pair_geometry.csv`: geometric measurements between residue pairs in the selected experimental structures.
- `reported_cross_protein_comparisons.csv`: NDM-1/VIM-2 comparisons reported by Chen et al. (2024).
- `observation_column_roles.csv`: modelling roles assigned to columns in `observations.csv`.
- `processed_summary.json`: row counts and validation results for the processed tables.

## Mutation-effect scores

The experimental value reported by the source study is retained in `score_original`.

For DMS regression, `score_analysis` applies the lower measurement limit associated with the corresponding assay. Values above that limit are unchanged.

Scores are not standardised, rescaled, averaged, or imputed across antibiotics, concentrations, temperatures, or studies. Each observation retains its assay identity and experimental condition.

## Missense observations

`missense_observations.csv` contains single amino-acid substitutions with valid assay-position and amino-acid identities.

Synonymous substitutions, stop variants, and other controls are retained separately in `control_observations.csv` rather than mixed with the principal missense prediction set.

## Residue features

`residue_features.csv` records residue-level information assembled from the sequence, experimental-structure, AlphaFold, M-CSA, CARD, structural-descriptor, and secondary-structure preparation steps.

Experimental and AlphaFold coordinate availability are recorded separately.

Coordinate availability does not imply sequence agreement between the assay construct and the corresponding reference sequence. The known VIM-2 assay/reference difference at position 186 is retained and is not considered eligible for analyses requiring a reference-matched experimental coordinate.

The assay-specific VIM-2 Gly2 insertion has no corresponding reference coordinate.

## NDM-1/VIM-2 position correspondence

The NDM-1/VIM-2 residue correspondence contains 217 one-to-one aligned positions and is stored in:

`data/interim/residue_evidence/homologue_position_correspondence.csv`

Of these aligned positions, 81 have the same wild-type amino acid in NDM-1 and VIM-2 and 136 have different wild-type amino acids.

The 132 WT-reversion comparisons reported by Chen et al. (2024) are retained separately in `reported_cross_protein_comparisons.csv`. These experimental comparisons are not used to define the sequence correspondence.

## Observation-column roles

Every column in `observations.csv` is assigned one of the following roles in `observation_column_roles.csv`:

- `outcome`
- `mutation_input`
- `condition_input`
- `grouping`
- `evaluation_only`
- `provenance_quality`

Wild-type and mutant amino-acid identity are mutation inputs.

Antibiotic, concentration, and temperature are condition inputs.

Protein identity and assay position are retained for representation selection, transfer analyses, and grouped evaluation rather than used as default scalar predictors.

Outcome, evaluation, grouping, and source-record fields are not included automatically as model inputs.

## Dataset splitting

The processed tables do not contain train, validation, or test assignments.

Mutation, residue-position, experiment, protein, and condition identifiers required for grouped evaluation are retained in the observation tables. Dataset partitions are stored separately under `data/interim/evaluation_partitions/`.
