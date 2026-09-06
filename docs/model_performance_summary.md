# Model performance summary

This analysis assembles the completed mutation-effect prediction results into study-level tables. It does not fit a prediction model, choose hyperparameters, alter saved predictions, or use final evaluation labels for model selection.

The summary covers five central evaluation schemes:

- NDM-1 unseen-position regression
- VIM-2 2020 unseen-position regression
- NDM-1 to VIM-2 ampicillin transfer
- VIM-2 to NDM-1 ampicillin transfer
- VIM-2 2022 unseen-mutation regression

Repeated model/scheme metric rows that occur in more than one completed analysis must agree numerically before one row is retained. A disagreement larger than `1e-10` stops the analysis.

Paired comparisons are not converted into a model-selection rule. Their confidence intervals are summarized only as intervals that exclude zero or include zero. The direction conventions recorded by the source analyses remain unchanged.

Biological associations remain descriptive rather than causal. The summary retains the reviewed counts of structural associations, primary structural associations, category summaries, and position-concordance rows.

The training-seed summary retains the single recorded mean-direction change.

Run the input review with:

```bash
python scripts/data/44_summarize_model_performance.py --check-only
```

Write the summary tables with:

```bash
python scripts/data/44_summarize_model_performance.py
```

Outputs are written under `data/interim/model_performance_summary/`:

- `model_performance_metrics.csv`
- `model_performance_rankings.csv`
- `model_pair_uncertainty_summary.csv`
- `biological_result_counts.csv`
- `training_seed_direction_change.csv`
- `model_performance_summary.json`

The ranking table is a descriptive ordering of completed point estimates. Lower MAE and RMSE rank first; higher Spearman and Pearson correlation rank first. Scientific claims should use paired uncertainty where a direct paired comparison exists.
