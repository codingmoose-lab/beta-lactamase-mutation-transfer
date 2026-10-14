# Residue geometry neural evaluation

## Scientific question

The residue-geometry evaluation tests whether three-dimensional residue geometry improves mutation-effect prediction relative to the residue-graph neural model.

The primary comparison is:

- `egnn_residue_regressor`
- `residue_graph_neural_regressor`

The centre-residue neural model from the residue-graph evaluation and the three residue-neighbourhood structure reference models are retained as additional comparators.

## Evaluation population

The residue-geometry evaluation uses the same 17 evaluation contexts and the same 109,408 observations with matching experimental-structure residues used in the residue-neighbourhood and residue-graph analyses. The complete set contributes 118,204 test predictions because observations can occur in more than one evaluation context.

Outer-test outcomes and cross-protein target-test outcomes are not used for epoch selection.

## Model

The residue-geometry model follows the E(n)-equivariant graph neural network principle. Residue node features and edge features match the residue-graph neural model. C-alpha coordinates enter the message layers through relative residue displacements and squared inter-residue distances. The scalar prediction therefore does not depend on where the protein is placed or how it is rotated in the coordinate system.

The implementation contains a numerical rotation-and-translation test before model fitting.

The model keeps the residue-graph neural model hidden width, number of message layers, dropout, learning rate, weight decay, gradient clipping, batch size, early-stopping patience, and 400-epoch limit. The residue-geometry selection and repeat seeds are different from those used for the residue-graph neural model.

## Inputs deliberately excluded

The principal the residue-geometry evaluation model does not use these fields as direct prediction inputs:

- protein identifier;
- assay position;
- catalytic-site labels;
- zinc-distance annotations;
- reported NDM-1/VIM-2 residue correspondence;
- outer-test outcomes;
- cross-protein target-test outcomes.

These exclusions preserve the same biological test used in the residue-graph neural evaluation.

## Result files

The run writes:

- `residue_geometry_neural_repeat_predictions.csv`
- `residue_geometry_neural_predictions.csv`
- `residue_geometry_neural_training_history.csv`
- `residue_geometry_neural_metrics.csv`
- `residue_geometry_neural_paired_comparisons.csv`
- `residue_geometry_neural_summary.json`

The paired comparison resamples `evaluation_group_id` and reports percentile 95% intervals for Spearman correlation, Pearson correlation, MAE, and RMSE.

## Required checks before fitting

The input review verifies:

- the final residue-graph neural result hashes;
- required residue-graph neural source functions;
- 17 evaluation contexts;
- 109,408 structure-supported observations;
- 118,204 evaluation rows;
- 469 graph nodes;
- 2,416 graph edges;
- complete C-alpha coordinates;
- rotation/translation invariance of the scalar model output.

Run the input review first:

```bash
PYTHONPATH=src python scripts/analysis/33_evaluate_residue_geometry_neural.py \
  --repo-root . \
  --device cuda \
  --validate-inputs
```

Only after that passes should the model be fitted.

## Recovery from interrupted execution

The residue-geometry evaluation records each completed evaluation context under
`data/interim/residue_geometry_neural_evaluation/context_checkpoints/`. A context is
reused only when its completion record, file hashes, evaluation observations, outcomes,
and execution settings match the current run.

After all 17 contexts are available, the repeat predictions, averaged predictions, and
training history are written before the grouped bootstrap starts. Therefore an
interruption during the statistical comparison does not require the GPU fits to be
repeated.

The normal command resumes from verified context checkpoints automatically. If all
contexts are already present and only the final statistics need to be completed, use:

```bash
PYTHONPATH=src python scripts/analysis/33_evaluate_residue_geometry_neural.py \
  --repo-root . \
  --device cuda \
  --postprocess-only
```

The grouped bootstrap still uses 1,000 replicates and `evaluation_group_id` as the
resampling unit. Progress is shown during this calculation.
