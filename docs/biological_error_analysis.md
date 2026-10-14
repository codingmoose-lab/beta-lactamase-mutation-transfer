# Biological error analysis

## Purpose

This analysis examines prediction errors from the structure-matched comparison using
residue-level biological information that was not supplied to the prediction models.
It asks whether model error changes with catalytic-Zn proximity, M-CSA reference
proximity, M-CSA residue correspondence, catalytic-Zn contact, and NDM-1/VIM-2
homologue correspondence.

The analysis does not fit or select another prediction model. It reads the verified
Script 28 predictions and calculates post-model error summaries.

## Analysis populations

All four Script 28 models are retained:

- `structure_matched_mutation_condition_ridge`;
- `structure_matched_sequence_neighbourhood_ridge`;
- `structure_matched_esmc_site_delta_ridge`;
- `experimental_structure_descriptor_ridge`.

The prediction table contains 118,204 test observations per model. Biological
annotations are joined by `protein_id` and `assay_position`. The wild-type residue is
checked against the processed residue table before error analysis proceeds.

Three levels are retained.

Prediction-level errors preserve every Script 28 test observation.

Evaluation-group summaries preserve the dependency groups used by the prediction
evaluation. Position-grouped schemes therefore remain grouped by residue position,
while the VIM-2 concentration experiment retains its mutation-trajectory groups.

Biological association and category analyses are then calculated at residue-position
level. When several evaluation groups occur at the same residue, as in the VIM-2
concentration experiment, their group mean absolute errors are averaged within that
residue before association analysis. This avoids treating several mutations at one
residue as independent observations of the same residue-level structural quantity.

## Biological reference analyses

The primary biological reference variables were not prediction inputs to the
experimental-structure model:

- minimum atom distance to the nearest selected catalytic Zn²⁺;
- Cα distance to the selected catalytic-Zn centroid;
- minimum atom distance to the nearest M-CSA reference correspondence;
- minimum atom distance to the nearest M-CSA metal-ligand correspondence;
- direct M-CSA reference correspondence;
- M-CSA metal-ligand correspondence;
- selected catalytic-Zn contact;
- outcome-independent NDM-1/VIM-2 homologue position class.

Continuous associations use Spearman correlation between the biological quantity and
residue-position mean absolute error. Confidence intervals use 1,000 residue-position
resamples with the specified random seed.

Categorical summaries report the mean and median residue-position value for each
category, with percentile 95% intervals for the mean.

## Structural-input error stratification

SASA, relative SASA, Cα neighbour counts, heavy-atom contact count, and reduced
secondary structure were inputs to the experimental-structure model. They are
therefore reported separately from the withheld biological reference analyses.

These summaries describe where model errors occur across structural input
characteristics. They are not treated as biological reference evidence independent of
the structural model inputs.

## ESMC and experimental structure

For each residue position, the analysis calculates:

`ESMC position mean absolute error - experimental-structure position mean absolute error`

Positive values indicate a smaller error for the experimental-structure descriptor
model. Negative values indicate a smaller error for ESMC.

This comparison is carried through the biological-reference and structural-input
summaries so that the analysis can identify residue environments in which the relative
performance of the two models changes.

## Homologue correspondence

The primary homologue mapping is
`data/interim/residue_evidence/homologue_position_correspondence.csv`. The mapping is
accepted only when `mapping_uses_outcome_values == False`.

Positions are classified as:

- `shared_wild_type`;
- `differing_wild_type`;
- `unmapped`.

Reported comparison positions from the source study are not used to define the primary
homologue categories in this analysis.

## Outputs

The command writes:

- `biological_error_predictions.csv`;
- `biological_error_group_summary.csv`;
- `biological_error_position_summary.csv`;
- `esmc_structure_position_error_differences.csv`;
- `biological_reference_associations.csv`;
- `biological_reference_category_summaries.csv`;
- `structural_input_error_associations.csv`;
- `structural_input_error_category_summaries.csv`;
- `biological_error_analysis_summary.json`.

Run:

```bash
python scripts/analysis/29_analyze_biological_errors.py
```
