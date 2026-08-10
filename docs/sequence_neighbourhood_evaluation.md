# Sequence-neighbourhood evaluation

Here, **neighbourhood** means the biological amino-acid neighbourhood surrounding the
mutated residue. It does not refer to the local computer or execution environment.

The analysis compares two matched regularised linear models on identical evaluation
rows. `mutation_condition_ridge` contains the complete predefined 380-class ordered
missense substitution encoding plus task-appropriate condition terms.
`sequence_neighbourhood_ridge` contains the same terms plus relative-position amino-acid
states around the mutated residue.

Candidate radii are 4, 8, 12, and 16 residues on each side. The centre is excluded
because wild-type and mutant identities are already represented by the ordered
substitution. Sequence termini use a `PAD` state; terminal observations are not removed.

All neighbourhoods come from the experimentally verified assay constructs:
`NDM1_2024_assay_construct` and `VIM2_2020_assay_construct`. Observation wild-type
residues are checked against those sequences before fitting.

The analysis is limited to five core schemes: NDM-1 unseen-position generalisation,
VIM-2 Chen-2020 unseen-position generalisation, both headline cross-protein directions,
and VIM-2 Chen-2022 unseen-mutation concentration-response generalisation. Existing
partition assignments are reused without modification.

Within-protein position tasks use categorical assay-condition terms. The VIM-2
concentration-response task uses training-standardised `log2(concentration)`. Headline
cross-protein transfer excludes condition predictors because source and target protein
backgrounds are tied to different study-defined ampicillin concentrations; encoding
that numerical difference would create a protein-background shortcut.

Model selection uses validation MAE. Ridge alpha is selected from the existing fixed
logarithmic grid. The sequence model additionally selects radius; ties prefer smaller
radius and then smaller alpha. Final cross-protein target labels never participate in
model selection.

The implementation uses sparse matrices and SciPy LSQR to avoid constructing a dense
observation-by-feature matrix.