# Modelling views and input roles

The modelling views specify which observations belong to each evaluation scheme and which fields may be used as model inputs.

## Preparation

```bash
python scripts/data/19_prepare_modelling_views.py
```

The script writes:

- `observation_column_contracts.csv`
- `residue_feature_contracts.csv`
- `residue_pair_feature_contracts.csv`
- `representation_contracts.csv`
- `scheme_observation_membership.csv`
- `comparison_populations.csv`
- `representation_scope_counts.csv`
- `modelling_view_summary.json`

The existing file names `representation_contracts.csv` and `representation_scope_counts.csv` are retained as internal data products.

## Observation-level predictors

The default mutation inputs are:

- wild-type amino-acid identity
- mutant amino-acid identity

The default condition inputs are:

- antibiotic
- concentration
- temperature

`protein_id` and `assay_position` are used to select the appropriate protein sequence, residue features, and structure-derived data. They are not default scalar predictors.

`score_analysis` is the primary DMS prediction target and is not stored in `scheme_observation_membership.csv`.

`score_original` remains the source measurement and is not used as a predictor.

`score_uncertainty` is retained for uncertainty-related analyses and is not part of the default predictor set.

## Structure-derived inputs

Experimental structure descriptors include solvent accessibility, local residue-neighbour counts, contact counts, and reduced secondary-structure assignments.

Raw coordinates and residue-pair geometry are used to construct graph-based or mutation-centred structure inputs rather than being passed directly as observation-level scalar variables.

Coordinate quality fields, occupancy, B factors, alternate-location status, and AlphaFold confidence are retained for eligibility checks and sensitivity analyses.

## Biological reference annotations

M-CSA correspondence, distances to M-CSA reference residues, catalytic-zinc reference distances, and the Chen et al. (2024) reported comparison fields are excluded from the primary predictor set.

These fields are reserved for biological interpretation and secondary analyses so that agreement with known catalytic or experimentally reported positions can be evaluated independently of model inputs.

The 217-position NDM-1/VIM-2 correspondence is also excluded from the default predictor set. The 81 aligned positions with the same wild-type amino acid are used for equivalent-substitution reporting after prediction.

## Analysis populations

Sequence-based analyses may use all sequence-eligible missense observations.

Models requiring experimental coordinates use observations marked `experimental_structure_task_eligible`.

AlphaFold-supported analyses use observations marked `alphafold_structure_task_eligible` and are reported separately from analyses based on experimental structures.

Direct comparisons between model input types use the same eligible observations so that performance differences are not caused by different sample populations.

## Evaluation membership

`scheme_observation_membership.csv` links observations to evaluation schemes and grouping identifiers without storing the prediction target.

Train, validation, and test assignments remain unchanged from the evaluation-partition tables.