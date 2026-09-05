# Residue-graph neural evaluation

## Purpose

Script 32 tests whether graph communication among experimentally observed residue environments improves mutation-effect modelling relative to a matched neural model that uses only the mutated residue. The comparison is designed to separate information gained from neighbouring residues from gains attributable to nonlinear neural modelling alone.

The analysis uses the five core evaluation schemes already used by the sequence and experimental-structure analyses. Existing test populations are reused without changing their membership.

## Models

`centre_residue_neural_regressor` receives the mutated residue representation, directed amino-acid substitution, and assay-condition representation. It does not perform graph communication.

`residue_graph_neural_regressor` receives the same observation-level inputs and the same residue representation, then applies two GINEConv layers over the experimental residue graph before reading the mutated-residue representation.

The residue graph comes from the selected experimental PDB structures. Graph construction does not use mutation-effect outcomes, evaluation assignments, mechanistic-reference annotations, or reported homologue-effect pairs.

## Residue and edge inputs

Residues are represented by the 20-state amino-acid identity, three-state secondary structure, heavy-atom count, solvent-accessible surface area, relative solvent-accessible surface area, three C-alpha neighbour counts, and the 4.5-Angstrom heavy-atom contact count.

Edges use C-alpha distance, minimum heavy-atom distance, log-transformed sequence separation, peptide adjacency, and the 4.5-Angstrom heavy-atom-contact indicator. Absolute coordinates are not direct model inputs.

The mutation is represented as one of the 380 directed missense substitutions. Protein identity and assay position are used only to locate records and graph nodes; neither is supplied to the neural readout as a scalar or categorical model input.

## Model fitting

Training uses AdamW and mean squared error after standardising the fitting outcomes. Validation MAE determines the number of training epochs. Nested within-protein analyses determine the epoch count from their validation rows, then refit on training plus validation rows before evaluating the corresponding outer test rows.

For cross-protein transfer, epoch counts are obtained from five grouped source-protein CV folds. The model is then fitted to all source-protein rows for the selected number of epochs and evaluated once on the target-protein test population. Target-protein outcomes do not determine the epoch count.

Three independent final fits are run for each analysis context and model. Their test predictions are retained separately and averaged for the primary neural-model result table.

Long fitting loops use tqdm progress displays reporting training loss, validation MAE, and the best validation MAE observed during epoch selection.

## Comparisons

The primary neural comparison is residue-graph neural versus centre-residue neural. Additional paired comparisons use the Script-31 residue-neighbourhood ridge model and the experimental-structure ridge model. Uncertainty intervals resample evaluation groups with 1,000 bootstrap replicates.

## Outputs

The analysis writes:

- `residue_graph_neural_repeat_predictions.csv`
- `residue_graph_neural_predictions.csv`
- `residue_graph_neural_training_history.csv`
- `residue_graph_neural_metrics.csv`
- `residue_graph_neural_paired_comparisons.csv`
- `residue_graph_neural_summary.json`

## Neural-computation reproducibility

CUDA model fitting uses PyTorch deterministic algorithms and a fixed cuBLAS
workspace configuration. Random-number generators are seeded for Python,
NumPy, PyTorch, and CUDA. The training ceiling is 400 epochs with validation
MAE selection and early-stopping patience of 20 epochs. The higher ceiling
allows the stopping rule to complete without changing the model architecture,
evaluation partitions, or test-set restrictions.
