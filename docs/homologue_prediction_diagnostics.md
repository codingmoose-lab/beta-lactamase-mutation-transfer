# Matched NDM-1 and VIM-2 mutation-effect prediction comparisons

The existing matched population contains 1,513 equivalent substitutions at 81 homologous positions with the same wild-type amino acid. These measurements are selected independently of their scores. The fitted cross-protein model results and the direct source-substitution estimate answer related questions with different information requirements.

## Existing cross-protein results

The diagnostic analysis reuses the files already produced by the research software. It does not fit new models, select hyperparameters, change score treatment, or substitute scores from an alternative experimental condition. The two model identifiers are `sequence_residue_graph_neural_regressor` and `residue_graph_neural_regressor`. Model identifiers and numerical data remain unchanged.

The direct estimate is the measured effect of the equivalent source-protein substitution. Paired differences use identical target observations. For MAE, a positive difference means the fitted model has lower absolute error than direct experimental transfer. For Spearman correlation, a positive difference means the fitted model has higher rank correlation. Confidence intervals are obtained by resampling homologous positions; substitutions at the same position are sampled together.

For the 478 pairs with effects above −4 in both proteins, the population is selected by observed measurements and therefore describes a different set of substitutions. Results for this subset cannot determine the isolated effect of the score threshold.

## Figure 4D

The originally reported association compares `abs(target_effect − source_effect)` with `abs(target_effect − y_pred)`. Both calculations contain `target_effect`, so the association is not independent evidence of a biochemical explanation.

The diagnostic analysis permutes the target effect among substitutions at the same homologous position, keeping the source and fitted estimates unchanged. The resulting distribution illustrates how associations may persist despite disrupting each substitution's original target effect. This procedure is descriptive and is **not** a formal biological null hypothesis test. It does not isolate the contribution of shared target values from other features of the data.

The triangle inequality also shows that the difference between the two absolute errors cannot exceed the difference between the fitted and direct predictions:

`abs(abs(y_pred − target_effect) − abs(source_effect − target_effect)) <= abs(y_pred − source_effect)`.

The optional Figure 4D candidate displays a paired absolute-error difference, not a structural explanation. It uses a dashed zero line; positive points favour the fitted model. The candidate is a *single SVG* written outside the repository and does not change the main figure. Its caption and any manuscript text would need to be revised if the candidate is adopted.

## New cross-protein experiment not supplied by these diagnostics

Prediction when the equivalent source substitution is unavailable requires new model fitting. Before running it, all measurements of that equivalent source substitution must be excluded from both fitting and model selection, while a stricter analysis could exclude all source substitutions at the homologous position. The target measurements must not be used to select models or tune their settings. Every reference predictor must follow the same source-data restrictions.

Existing fitted predictions cannot establish performance for that task. This new experiment should be run only after confirming the exact input population, fitting rules, baseline methods, and required hardware. MacOS CPU execution may be possible for ridge models; CUDA and MPS training are not interchangeable without tests of the required operations and numerical results.

## Saved results

The diagnostic command requires an empty output directory outside the repository. It produces a CSV table and a plain-text summary with the SHA-256 values of the two input prediction tables. Existing research files are not overwritten. The separate file-hash checker compares file **contents** before and after running the test suite, rather than treating a changed modification time as scientific data loss.
