# Atom-level local structural evaluation

This analysis tests whether atom-level local information from the selected wild-type experimental structures contains mutation-effect information that is not captured by the residue-level structural models already evaluated.

The model input is not a rendered protein image. Each experimentally supported residue is converted to a numerical 3-D tensor centred on its C-alpha atom. A right-handed residue-local coordinate frame is fixed from the backbone N, C-alpha, and C atoms so rigid translation or rotation of the deposited structure does not change the representation. The cube spans 20 A on each side at 1 A spacing. Protein heavy atoms are deposited by trilinear interpolation into carbon, nitrogen, oxygen, sulfur, other-element, and centre-residue channels.

Only `ATOM` records from the selected primary experimental PDB chain are used. Hydrogen/deuterium atoms, `HETATM` records, zinc ions, other ligands, curated catalytic annotations, mutation-effect outcomes, and predicted mutant structures do not enter voxel construction. Mutant identity and permitted assay-condition fields are supplied in the same manner as the established residue neural evaluations.

The model is a deliberately small 3-D CNN. Architecture hyperparameters are fixed before test evaluation. Epoch count is chosen from permitted validation records; cross-protein target labels are not used for epoch selection. Three fixed training seeds are averaged for final predictions. Evaluation uses the same five structure-supported schemes and the same observation populations used by the residue-level structural neural analyses.

Primary comparisons are paired against the centre-residue neural model, residue graph model, E(n)-equivariant residue geometry model, and experimental-structure descriptor ridge model. Paired uncertainty uses grouped bootstrap resampling by `evaluation_group_id`.

A positive result would indicate that atom-level local organization contains useful mutation-effect information beyond the residue-level encodings tested here. A null or negative result would indicate that increasing structural resolution to atom-level WT coordinates does not improve these mutation-effect tasks under the present data and evaluation design. Neither outcome is interpreted as observing mutation-induced structural relaxation, because no mutant structures are used.
## CUDA maximum reduction

The CNN retains non-overlapping 2 x 2 x 2 maximum reduction. Native CUDA
`MaxPool3d` backward did not execute with the PyTorch reproducibility setting
used by this study environment. The reduction is therefore implemented by
reshaping each non-overlapping 2 x 2 x 2 neighbourhood and applying
dimension-wise maxima. Before model fitting, this implementation was checked
against native `MaxPool3d` on CPU and produced identical forward values and
gradients for random, zero-rich, and tied inputs at both grid sizes used by
the CNN. CUDA forward and backward execution was also checked with the study
reproducibility settings enabled.
