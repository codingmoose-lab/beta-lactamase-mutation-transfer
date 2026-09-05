# Matched homologue model analysis

Stage 42 asks how already fitted cross-protein models behave on the 1,513 directly comparable NDM-1/VIM-2 substitutions at the 81 aligned positions that share the same wild-type residue. The pair population was defined without mutation-effect values. NDM-1 to VIM-2 and VIM-2 to NDM-1 remain separate prediction directions.

The analysis does not fit or retune a model. It reuses the target-protein predictions already produced for ESMC, ESM-2, the experimental-structure descriptor model and its mutation/condition comparator, the centre-residue neural control, the residue graph model, the residue-geometry model, and the sequence-plus-graph model. Every included model must cover all 1,513 target substitutions in each direction and its stored target values must agree with the matched homologue table.

Two biological references are retained. `direct_homologue_source_effect` predicts the target effect with the measured effect of the identical substitution in the other homologue. `substitution_mean` is the established training-data substitution baseline. Comparisons against both references are post-test summaries and do not choose model settings.

Results are reported for all 1,513 pairs and separately for the 478 pairs at 63 homologue positions where neither protein is at the study fitness floor. This separation matters because a large both-at-floor population can make cross-background agreement look stronger than the non-floor signal alone.

Spearman correlation, Pearson correlation, MAE, and RMSE are calculated in each direction and population. Paired model-versus-reference intervals use 1,000 reproducible bootstrap resamples of `homologue_position_id`. Substitutions at the same aligned position therefore move together during resampling rather than being treated as independent observations.

The analysis also measures whether model error changes with the observed NDM-1/VIM-2 background difference. It reports Spearman association between absolute prediction error and absolute background difference, and between signed prediction error and signed background difference. These associations are descriptive evidence about transfer failure; they do not establish a molecular mechanism.

A position summary reports the magnitude of background differences, floor-state composition, and the range of median errors across the already established models. Positions with large values are candidates for later biological interpretation rather than automatically being labelled mechanistic resistance sites.

The 132 publication-reported WT-reversion comparison rows remain separate supporting evidence. They are not used to define the 1,513-pair population, choose model settings, or select models in Stage 42.

Run the input check first:

```bash
python scripts/data/42_analyze_matched_homologue_models.py --check-only
```

Then run the analysis:

```bash
python scripts/data/42_analyze_matched_homologue_models.py
```
