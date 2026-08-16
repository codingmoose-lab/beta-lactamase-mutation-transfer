# Structure-matched model comparison

## Purpose

This analysis compares four models on the same observations with eligible experimental-structure descriptors and under the same evaluation partitions. The comparison asks how predictive performance differs among mutation/condition, local sequence, ESMC, and experimental-structure inputs when model fitting and test evaluation use the same eligible population.

The compared models are:

- `structure_matched_mutation_condition_ridge`;
- `structure_matched_sequence_neighbourhood_ridge`;
- `structure_matched_esmc_site_delta_ridge`;
- `experimental_structure_descriptor_ridge`.

Script 27 already fitted and evaluated the mutation/condition and experimental-structure descriptor models on this population, so those predictions and model-selection records are reused. The sequence-neighbourhood and ESMC models are selected and fitted again using the same eligible fitting populations.

## Population matching

The comparison is not produced by filtering full-population test predictions after model fitting. For the two sequence models fitted in this analysis, training, validation, source-CV, and test rows are first restricted to observations with `experimental_structure_task_eligible == True`. Existing group assignments and evaluation partitions are retained.

For nested evaluation, model settings are selected from the eligible inner training and validation rows. The selected model is then fitted to the eligible training and validation rows and evaluated on the eligible outer test rows. For cross-protein transfer, model settings are selected using eligible source-protein CV rows only. The selected model is then fitted to all eligible source rows and evaluated once on the eligible `target_test` rows.

## Model selection

The sequence-neighbourhood model selects radius and ridge alpha from the candidate sets used by Script 23. The ESMC model selects the hidden-state index and ridge alpha from the candidate combinations used by Script 25. Selection uses only validation data or source-CV data allowed by the evaluation scheme. `target_test` labels are not used for model selection.

Existing ESMC embeddings are reused. ESMC inference is not repeated. The selected experimental PDB structures and existing structural descriptors are also reused; structure acquisition and coordinate preparation are not repeated in this analysis.

## Biological annotations reserved for interpretation

Catalytic-zinc distances, M-CSA correspondence fields, reported homologue-pair fields, and raw coordinates are not added as predictors. These fields remain available for later biological interpretation and error analysis.

## Metrics and uncertainty

Spearman correlation, Pearson correlation, MAE, and RMSE are calculated on the complete matched test population for each scheme and model. All six pairwise model contrasts are reported. Correlation differences are `model_a - model_b`; MAE and RMSE differences are `model_b - model_a`, so a positive paired difference favours `model_a`.

Uncertainty is estimated separately for each evaluation scheme by resampling `evaluation_group_id` with replacement. The same sampled groups are used for all four models within each bootstrap replicate, preserving paired comparisons across the six contrasts. The analysis uses 1,000 bootstrap replicates with the specified random seed and percentile 95% intervals.

## Scope

This analysis compares tabular mutation/condition, sequence-neighbourhood, ESMC, and experimental-structure descriptor models. It is not the structural-image, residue-graph, or multimodal analysis. Those analyses remain separate.

## Outputs

The command writes:

- `structure_matched_predictions.csv`;
- `structure_matched_metrics.csv`;
- `structure_matched_model_selection.csv`;
- `structure_matched_paired_comparisons.csv`;
- `structure_matched_summary.json`.

Run:

```bash
python scripts/data/28_compare_structure_matched_models.py
```
