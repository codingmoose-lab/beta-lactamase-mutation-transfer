# Sequence model comparison

## Purpose

This analysis compares the completed mutation/condition,
sequence-neighbourhood, and ESMC site-delta models on the same test
observations.

The comparison is performed after model fitting and model selection. It does
not refit any model, change any evaluation partition, or use test labels to
choose model settings.

## Models

The compared models are:

- `mutation_condition_ridge`;
- `sequence_neighbourhood_ridge`;
- `esmc_site_delta_ridge`.

The three paired contrasts are:

1. `sequence_neighbourhood_ridge` versus `mutation_condition_ridge`;
2. `esmc_site_delta_ridge` versus `mutation_condition_ridge`;
3. `esmc_site_delta_ridge` versus `sequence_neighbourhood_ridge`.

## Prediction matching

The analysis reads the saved sequence-neighbourhood and ESMC prediction tables.

Prediction rows are matched using `scheme_id`, `evaluation_iteration`,
`evaluation_role`, and `observation_id`. The analysis also requires agreement
in task identity, evaluation group, protein, residue position, wild-type
residue, mutant residue, antibiotic, concentration, temperature, and observed
mutation-effect value. Any population mismatch stops the analysis rather than
dropping unmatched rows.

The five evaluation schemes contain 135,745 matched test predictions per model.
Within each scheme, an `evaluation_group_id` must occur in only one test
iteration. This allows pooled out-of-fold predictions to be compared while
retaining the evaluation group as the bootstrap resampling unit.

Input-file SHA-256 digests are written to the summary to record exact input identity. They are
not used as hard-coded acceptance criteria; validity is checked from model
identity, row matching, evaluation roles, group membership, and finite values.

## Metrics and paired differences

The reported metrics are Spearman correlation, Pearson correlation, MAE, and
RMSE. Correlation differences are `model_a - model_b`; MAE and RMSE differences
are `model_b - model_a`. Positive paired differences therefore favour
`model_a`.

Metrics are calculated once per evaluation scheme and model using the complete
matched test population for that scheme. Fold-level metric values are not
averaged.

## Uncertainty

Paired uncertainty is calculated separately for each evaluation scheme by
resampling `evaluation_group_id` with replacement. Each sampled group
contributes all of its observations to both models in the contrast. The
analysis uses 1,000 bootstrap replicates with the specified random seed and
reports percentile 95% intervals from the 0.025 and 0.975 bootstrap quantiles.

These intervals describe uncertainty in the paired metric differences. They are
not used to alter model settings or choose additional models.

## Outputs

The command writes:

- `sequence_model_metrics.csv`;
- `sequence_model_paired_comparisons.csv`;
- `sequence_model_comparison_summary.json`.

The paired-comparison table records both model metric values, their signed
difference, the bootstrap interval, observation and group counts, and the
resampling unit. The summary records the input-file digests and comparison
settings.

Run:

```bash
python scripts/data/26_compare_sequence_models.py
```
