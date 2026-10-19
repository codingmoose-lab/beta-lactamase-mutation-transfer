# Position-withheld source-protein reference estimates

This supplementary computational check predicts ampicillin mutation effects in NDM-1 or VIM-2 using **only source-protein measurements at other aligned positions**. For each direction, all observations for one of the five held-out sets of homologous positions are withheld from source fitting. Prediction of the target mutations at those positions is then completed using source measurements from the remaining aligned positions.

This check uses the 1,513 substitutions at 81 homologous positions having identical wild-type amino acids. The original pair population was selected without reference to effect magnitude. It does not include all available mutations of the source protein. Therefore, its estimates **cannot be directly compared** to published neural models fitted on the much larger full source dataset.

Three fixed source-only reference estimates are tested: source global mean, mutant-amino-acid mean shrunk toward the global mean, and a wild-type/mutant substitution mean shrunk toward the source mutant mean. The pseudo-count is five for both shrinkage calculations. It is specified in code and is not selected against target-protein observations. The five folds are allocated from stable hashes of homologous position identifiers. Folds are identical between transfer directions.

Prediction at a held-out position never reads the source effect at that position, and target experimental values are used only for evaluation. The model comparison is against the source global-mean estimate under the same fitting information. A direct experimental substitution measurement is deliberately **unavailable** to this prediction task; comparing to it as if it were an available fitted baseline would be misleading.

Results are reported for all 1,513 pairs and as an outcome-defined secondary comparison for 478 pairs above the −4 analysis level in both proteins. Confidence intervals resample 81 or 63 homologous positions rather than treating every substitution as independent. This experiment uses no GPU, trains no neural network, and does not modify existing predictions or figures.

Model selection in previously fitted sequence and structural models requires additional code inspection. The accompanying `source_selection_record_checks.csv` checks saved fitting roles and partition folds; these checks alone do not prove how each model consumed those records.
