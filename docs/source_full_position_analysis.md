# Source-assay-wide prediction with homologous positions withheld

## Biological question

When an equivalent source-protein substitution has not been measured, can measurements at other source-protein positions help estimate the target mutation effect?

The existing source-position-withheld reference experiment fitted its models using only the 81 positions shared by NDM-1 and VIM-2. This additional analysis uses all available missense observations from the **same source ampicillin assay**. It still excludes every source-protein observation at the homologous positions held out for target evaluation.

## Population and fitting

- The same 1,513 matched target substitutions, 81 homologous positions and five outer position folds are used in each direction. The fold assignment comes from `position_withheld_homologue_baselines.fold_for_positions`.
- Source observations come from the unique source assay associated with the matched-substitution records. Protein, antibiotic, concentration and temperature must be uniform within the selected assay.
- Every held-out source assay position is excluded from the entire source fitting table, including other mutations at that position. Observations at other source positions are retained.
- Model input is wild-type and mutant residue identity. Residue index, mutation-effect outcomes from the target protein, original target assay measurements and target model errors are not predictors.
- The fixed shrinkage amount for empirical means is 5 observations. Ridge penalties 0.1, 1, 10 and 100 are compared using three inner source-position folds; target effects never select a penalty.
- The output includes fitting population counts and assay IDs, position folds, Ridge selection records, target predictions and grouped-bootstrap uncertainty for MAE differences against the source-assay-wide mean.

## Limits

This is an exploratory reference analysis, not a new neural representation and not an unbiased estimate of model selection over all attempted methods. It uses the same 81 aligned shared-wild-type positions for testing and therefore does not establish performance on different-wild-type pairs or unrelated β-lactamases.

Its results may be paired directly with the prior 81-position-only references because the test substitutions and position folds are identical. They cannot be directly compared with the existing full-source neural predictions when those networks have been fitted using source effects at the target's homologous position. A matched neural refit would be a different experiment.

Do not automatically add new results to the abstract or figures. Compare the actual results, confirm the assay selection and review paired uncertainty first.
