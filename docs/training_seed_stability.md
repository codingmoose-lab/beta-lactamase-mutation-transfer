# Training-seed stability analysis

The training-seed stability analysis measures how much the neural-model results vary across training initialization while keeping every previously selected model setting unchanged.

The primary neural analyses used three independently initialized fits. The training-seed stability analysis keeps those records and adds seven new fits for each model and evaluation context, giving ten training seeds per model. It does not repeat epoch selection, change an architecture, alter a predictor set, or use evaluation labels to choose settings.

The four neural models are the centre-residue model, residue-graph model, residue-geometry model, and sequence-plus-residue-graph model. The centre-residue, residue-graph, and sequence-plus-residue-graph models use seed numbers 32101 through 32110. The residue-geometry model uses 33101 through 33110. The first three values in each set are the seeds already used by the residue-graph, residue-geometry, and sequence-plus-graph analyses; only seed positions 4 through 10 require new fitting.

For each model and evaluation scheme, The training-seed stability analysis reports MAE, RMSE, Spearman correlation, and Pearson correlation for every training seed, together with their mean, standard deviation, median, minimum, and maximum. It also compares the mean across the original three seeds with the mean across all ten seeds.

Three model contrasts are examined: residue graph versus centre residue, residue geometry versus residue graph, and sequence-plus-residue graph versus residue graph. Models that share the same seed numbers are compared at matching seed numbers. Residue geometry and residue graph use different seed-number series, so their comparison uses every combination of the ten geometry seeds and ten graph seeds rather than treating unlike seed numbers as matched pairs.

A positive reported metric difference favours the first named model. For MAE and RMSE this is calculated as the second model minus the first model. For Spearman and Pearson correlation it is the first model minus the second model.

Additional seed predictions are written as gzip-compressed CSV to limit storage. During fitting, completed model/context records are saved so an interrupted run can continue. After all additional fits finish and the combined file is verified, the temporary checkpoint directory is removed.

The training-seed stability analysis is a secondary stability analysis. It does not replace the established three-seed predictions from the residue-graph, residue-geometry, and sequence-plus-graph analyses.
