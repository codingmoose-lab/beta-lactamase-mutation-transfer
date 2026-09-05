# Sequence–structure predictive complementarity

Stage 35 is a secondary analysis initiated after the Stage-34 results. It asks whether the ESMC mutation-site model contributes useful predictive information beyond the residue-graph model, without changing the established mutation-effect tasks or final evaluation populations.

The analysis uses the existing Stage-25 ESMC model definition and selected setting, the existing Stage-32 residue-graph model definition and selected epoch count, and the final Stage-25, Stage-32, and Stage-34 predictions. It does not alter those result files.

For the three nested grouped evaluations, the Stage-25 and Stage-32 model definitions are refitted on the existing training rows and used to predict the existing validation rows. For each cross-protein evaluation, the models are fitted on four source-protein folds and predict the remaining source-protein fold. Target-protein outcomes are not used to determine the combination.

The primary Stage-35 model is a convex combination

`prediction = graph_weight * graph_prediction + (1 - graph_weight) * sequence_prediction`

with graph weights `0, 0.25, 0.5, 0.75, 1`. The selected weight minimises mean absolute error in the permitted validation population. Exact ties favour the graph reference, so sequence contribution is not introduced when the validation criterion is exactly unchanged.

After the weight is fixed, it is applied to the pre-existing final ESMC and residue-graph predictions on the exact Stage-32 structure-supported evaluation population. Final metrics are reported for the selected combination, ESMC, residue graph, and the Stage-34 sequence+graph neural model. Paired 95% intervals use 1,000 bootstrap resamples of `evaluation_group_id`.

A sequence-versus-graph oracle is reported only as a post-evaluation diagnostic of error complementarity. It consults observed evaluation outcomes and therefore is not a fitted, selected, or deployable predictor and is never used to select the Stage-35 weight.

The analysis writes validation predictions, weight-selection records, final predictions, metrics, paired comparisons, complementarity diagnostics, and a JSON summary under `data/interim/sequence_structure_complementarity/`. Validation prediction checkpoints are retained so interrupted GPU work can resume without refitting completed validation models.
