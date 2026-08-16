# Evaluation partitions

The evaluation partitions define train, validation, and test populations for the mutation-effect prediction and transfer analyses.

## Preparation

```bash
python scripts/data/18_define_evaluation_partitions.py
```

The script writes:

- `partition_schemes.csv`
- `partition_scopes.csv`
- `group_assignments.csv`
- `partition_audit.csv`
- `evaluation_partition_summary.json`

## Assignment rules

Partition assignments are based on stable group identifiers and the fixed seed `beta_lactamase_evaluation_partitions`.

Groups are ordered using SHA-256 hashes before fold assignment. Mutation-effect scores are not used when assigning observations to folds.

The grouping unit follows the evaluation question:

- residue position for unseen-position evaluation;
- mutation identity for unseen-mutation evaluation;
- experimental condition for the specified transfer analyses.

Random observation-level splitting is not used for the principal evaluations.

## Within-protein unseen-position evaluation

NDM-1 and Chen et al. (2020) VIM-2 use five outer folds grouped by `position_group_id`.

All mutations and measured conditions associated with the same residue position remain in the same outer fold.

Validation positions are selected from the remaining training positions. Each eligible residue position appears in the outer test set once across the five folds.

## NDM-1/VIM-2 transfer

The primary cross-protein transfer conditions are:

- NDM-1: 256 µg/mL ampicillin at 37 °C
- VIM-2: 128 µg/mL ampicillin at 37 °C

Model selection uses only observations from the source protein, with five position-grouped source folds.

After model selection, the selected model is fitted to the complete source population and evaluated on the target-protein population.

Target-protein mutation-effect values and target-protein performance are not used for feature selection, model selection, early stopping, threshold selection, or hyperparameter tuning.

The 217 aligned NDM-1/VIM-2 residue positions are not used to construct source folds.

The 81 shared-wild-type positions are used separately for equivalent-substitution reporting after target predictions have been generated.

The 132 WT-reversion comparisons reported by Chen et al. (2024) are retained as supporting experimental comparisons and are not used to define partition assignments.

## VIM-2 antibiotic transfer

Leave-one-antibiotic-out evaluation uses the Chen et al. (2020) high-screening conditions at 37 °C:

- ampicillin: 128 µg/mL
- cefotaxime: 4 µg/mL
- meropenem: 0.031 µg/mL

The evaluated mutation population is restricted to mutations observed under all three conditions.

For each analysis, one antibiotic is reserved for testing. Model selection uses mutation-grouped folds from the remaining two antibiotics.

The same mutation may therefore occur under both source and test conditions because this analysis measures transfer across antibiotics rather than prediction of unseen mutations.

## VIM-2 temperature transfer

Temperature transfer is evaluated in both directions:

- 25 °C → 37 °C
- 37 °C → 25 °C

The analysis uses ampicillin measurements at 2, 16, and 128 µg/mL and is restricted to the 5,016 mutation groups measured at both temperatures.

Twelve mutation groups observed only at 25 °C and seven observed only at 37 °C are excluded from the paired temperature-transfer population.

## VIM-2 concentration response

Two evaluation settings are retained for the Chen et al. (2022) concentration series.

### Unseen mutations

All available concentrations for the same mutation group remain in the same outer partition.

### Concentration transfer

Leave-one-concentration-out evaluation tests prediction at a concentration withheld from model fitting.

The test concentrations are classified as:

- edge extrapolation: 4 and 256 µg/mL;
- interpolation: 8, 16, 32, 64, and 128 µg/mL.

Fifteen mutation groups do not have a validated measurement at 16 µg/mL. These mutations remain eligible for unseen-mutation evaluation but are absent from the 16 µg/mL concentration-transfer test population.

## Partition checks

`partition_audit.csv` records group counts, observation counts, and assignment checks for each evaluation scheme.

Partition construction verifies that grouped observations do not cross incompatible train, validation, and test assignments and that mutation-effect values do not influence fold construction.
