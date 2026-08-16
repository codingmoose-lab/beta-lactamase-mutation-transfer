# Experimental structure descriptor evaluation

## Purpose

This analysis tests whether generic descriptors from the selected experimental
PDB structures improve mutation-effect prediction beyond mutation and assay
condition information on the same structure-eligible observations.

The five core evaluation schemes and their existing partition assignments are
reused. The analysis does not redefine train, validation, source-model-selection,
outer-test, or target-test populations.

## Models

The two matched models are:

- `structure_matched_mutation_condition_ridge`;
- `experimental_structure_descriptor_ridge`.

Both use the same predefined 380-state missense-substitution vocabulary and the
same task-specific condition treatment. The structural model additionally uses
generic experimental-structure descriptors.

## Structural inputs

The structural model uses:

- solvent-accessible surface area;
- relative solvent accessibility;
- C-alpha neighbour counts within 8, 10, and 12 Å;
- heavy-atom contact count within 4.5 Å;
- reduced three-state secondary structure.

Numerical structural descriptors are standardised using the fitting population
only.

The following fields are deliberately excluded from direct prediction:

- protein identity and assay residue position;
- raw Cartesian coordinates;
- M-CSA correspondence fields;
- catalytic-zinc distance fields;
- reported homologue-pair fields;
- structure-mapping and coordinate-quality metadata.

This preserves mechanistic reference information for interpretation and avoids
protein/position shortcuts. Residue-pair geometry remains available for later
graph or structural-image construction.

## Evaluation

Within-protein schemes use the existing grouped outer-test and inner-validation
assignments. Cross-protein schemes choose the ridge penalty using source-protein
cross-validation only and evaluate the target protein after the setting is
fixed.

The mutation/condition comparator is refitted on the same
experimental-structure-eligible population as the structural model. This makes
the paired comparison a test of the added structural descriptors rather than a
comparison confounded by different observation coverage.

Paired uncertainty resamples `evaluation_group_id` with replacement using 1,000
bootstrap replicates and the specified random seed. Positive paired differences
favour the experimental-structure model.

## Outputs

The command writes:

- `experimental_structure_predictions.csv`;
- `experimental_structure_metrics.csv`;
- `experimental_structure_model_selection.csv`;
- `experimental_structure_paired_comparisons.csv`;
- `experimental_structure_summary.json`.

Run:

```bash
python scripts/data/27_evaluate_experimental_structure.py
```
