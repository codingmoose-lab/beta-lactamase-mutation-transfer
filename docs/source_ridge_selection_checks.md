# Source-protein fitting checks for Ridge and ESM-C

These checks execute the cross-protein evaluation functions for three existing model groups with simulated source and target observation identifiers. For both NDM-1 → VIM-2 and VIM-2 → NDM-1, they verify that all five model-selection folds contain only source-protein rows, that the final fitting input contains only source observations, and that the target observations are used for prediction.

Additional checks execute the actual fitting-statistic functions for mutation-condition Ridge, experimental-structure Ridge, and ESM-C site differences. Their numeric inputs are small simulated arrays, with a large value excluded from the fitted source population. The checks do not fit the original models or modify recorded predictions.

The results constitute code-level verification of these paths. They do not independently reconstruct the original RTX training process, prove the accuracy of historical checkpoints, or establish that every preprocessing input was prepared without experimental target scores. Those questions require a separate review of preparation code and fitting records.

The report files are written to a new directory outside the repository. The existing output files, Figures 1–6, and model checkpoints are not rewritten. No Git commit is made.
