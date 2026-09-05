# Residue geometry neural evaluation

## Scientific question

Stage 33 tests whether three-dimensional residue geometry improves mutation-effect prediction beyond the Stage-32 residue graph model.

The primary comparison is:

- `egnn_residue_regressor`
- `residue_graph_neural_regressor`

The Stage-32 centre-residue neural model and the three Stage-31 structure reference models are retained as additional comparators.

## Evaluation population

Stage 33 uses the same 17 evaluation contexts and the same 109,408 observations with matching experimental-structure residues used in Stages 31 and 32. The complete set contributes 118,204 test predictions because observations can occur in more than one evaluation context.

Outer-test outcomes and cross-protein target-test outcomes are not used for epoch selection.

## Model

The Stage-33 model follows the E(n)-equivariant graph neural network principle. Residue node features and edge features match Stage 32. C-alpha coordinates enter the message layers through relative residue displacements and squared inter-residue distances. The scalar prediction therefore does not depend on where the protein is placed or how it is rotated in the coordinate system.

The implementation contains a numerical rotation-and-translation test before model fitting.

The model keeps the Stage-32 hidden width, number of message layers, dropout, learning rate, weight decay, gradient clipping, batch size, early-stopping patience, and 400-epoch limit. Stage-33 selection and repeat seeds are different from Stage 32.

## Inputs deliberately excluded

The principal Stage-33 model does not use these fields as direct predictive inputs:

- protein identifier;
- assay position;
- catalytic-site labels;
- zinc-distance annotations;
- reported NDM-1/VIM-2 residue correspondence;
- outer-test outcomes;
- cross-protein target-test outcomes.

These exclusions preserve the same biological test used in Stage 32.

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

- the final Stage-32 result hashes;
- required Stage-32 source functions;
- 17 evaluation contexts;
- 109,408 structure-supported observations;
- 118,204 evaluation rows;
- 469 graph nodes;
- 2,416 graph edges;
- complete C-alpha coordinates;
- rotation/translation invariance of the scalar model output.

Run the input review first:

```bash
PYTHONPATH=src python scripts/data/33_evaluate_residue_geometry_neural.py \
  --repo-root . \
  --device cuda \
  --check-only
```

Only after that passes should the model be fitted.

## Recovery from interrupted execution

Stage 33 records each completed evaluation context under
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
PYTHONPATH=src python scripts/data/33_evaluate_residue_geometry_neural.py \
  --repo-root . \
  --device cuda \
  --postprocess-only
```

The grouped bootstrap still uses 1,000 replicates and `evaluation_group_id` as the
resampling unit. Progress is shown during this calculation.
