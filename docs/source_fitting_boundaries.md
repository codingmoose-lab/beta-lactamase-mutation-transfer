# Source-protein fitting checks

The script `49_verify_source_fitting_boundaries.py` executes the source-fold
selection functions extracted from the current repository Python files. It
replaces numerical fitting operations with functions that record the
observation identities received by fitting and validation calls.

The checks use ten synthetic source observations and three synthetic target
observations for each cross-protein direction. The target outcome values are
deliberately separated from the source outcome range. Each of five folds must
receive only its assigned source observations.

The residue-graph final-fit function is executed with a substituted fitting
operation. The residue-geometry, sequence–residue-graph, and local-voxel
models' fitting-population helpers are also executed. Model fitting is not
performed. Fitting-side target and condition transformations are inspected
from the source code for the four neural model families.

The baseline, ESM-C ridge, and experimental-structure ridge code is inspected
for source and target role selection. These are **review**, not pass, findings;
calling model selection on source folds is visible in code, but fitting is not
simulated for these models by this script.

This analysis does not establish historical checkpoint correctness, training
weight reproducibility, all feature-input restrictions, or the scientific
meaning of model performance. It does not change any existing table, model
prediction, figure, or source module.

All generated reports should be written outside the repository. The file hashes
in the JSON report identify the exact source code used in the inspection.
