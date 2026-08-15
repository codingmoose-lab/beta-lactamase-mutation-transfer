# ESMC mutation-effect evaluation

## Scientific question

The protein-language-model analysis tests whether pretrained protein sequence
information improves mutation-effect prediction beyond mutation identity,
assay condition, and explicit local amino-acid neighbourhood.

The analysis reuses the frozen task definitions and leakage-resistant
evaluation partitions used by the preceding baseline and sequence-neighbourhood
comparisons.

## Protein sequences

ESMC inputs use the experimentally verified NDM-1 and VIM-2 assay constructs.
This preserves assay residue numbering and wild-type identity.

The evaluator checks assay-sequence lengths and SHA-256 digests against the
provenance stored with the extracted embeddings.

## Pretrained model

The frozen source model is `biohub/ESMC-600M` at revision
`a7e82012c83126b9eedb055fea9fa84b6c02f094`.

The extracted hidden-state indices are 9, 18, 27, and 36, with hidden size
1152. Model weights are not fitted to mutation-effect measurements.

## Mutation-site embeddings

For every unique measured missense substitution, the assay construct is
changed at exactly the requested residue and ESMC hidden states are extracted
at the corresponding mutant-residue token. Wild-type residue hidden states are
extracted once for each assay construct.

The PLM feature used by the current evaluator is:

`delta_h = h_mutant_site - h_wild_type_site`

The mutation-to-embedding index is deterministic and records protein identity,
assay mutation, assay position, wild-type residue, mutant residue, and
mutant-sequence SHA-256.

## Extraction provenance

The production extraction used:

- Python 3.11.15;
- PyTorch package metadata 2.13.0;
- PyTorch runtime `2.13.0+cu130`;
- CUDA build 13.0;
- Transformers 4.57.6;
- Hugging Face Hub 0.36.0;
- Tokenizers 0.22.2;
- Safetensors 0.8.0;
- NumPy 2.2.6;
- SciPy 1.13.1;
- pandas 2.3.3;
- tqdm 4.68.2;
- NVIDIA GeForce RTX 4060 Ti.

The extraction summary remains the authoritative record of the successful
runtime. No historical PyTorch package index or direct installation URL is
asserted because that information was not retained by the environment.

The extracted arrays contain 10,086 unique protein-specific missense
substitutions: 5,041 for NDM-1 and 5,045 for VIM-2. The mutant array has shape
`(10086, 4, 1152)`. The wild-type array has shape `(2, 4, 270, 1152)`.

NDM-1 contains 270 assay residues and VIM-2 contains 267. The three trailing
VIM-2 positions in the rectangular wild-type array are storage padding outside
the biological assay sequence and are never used for evaluation. All embeddings
over biological residue ranges are required to be finite.

## Predictive model

The primary model is `esmc_site_delta_ridge`.

Its predictors are:

- an intercept;
- ordered wild-type-to-mutant substitution identity;
- the task-appropriate assay-condition term;
- standardized ESMC site-delta features.

Categorical-condition tasks retain antibiotic, concentration, and temperature
in the same condition definition used by the matched sequence comparator. The
VIM-2 concentration-response task uses log2 concentration. The headline
cross-protein AMP transfer tasks do not add a separate condition predictor
because source and target populations are already restricted to the
study-defined comparison condition.

ESMC features are standardized using fitting data only. Means and standard
deviations are recomputed within each permitted model-selection or final-fit
population.

## Evaluation design

The current core evaluation schemes are:

- `ndm1_position_nested5`;
- `vim2_2020_position_nested5`;
- `ndm1_to_vim2_amp_source_cv5`;
- `vim2_to_ndm1_amp_source_cv5`;
- `vim2_2022_mutation_nested5`.

Within-protein position-grouped experiments use grouped outer evaluation with
inner validation. The VIM-2 2022 experiment remains mutation-grouped and tests
generalisation to unseen mutations, not unseen positions.

Cross-protein transfer selects model settings using source-protein grouped
cross-validation and evaluates the target protein only after source-side model
selection is complete. Final held-out labels are not used for model selection.

The evaluator jointly selects the hidden-state index and ridge regularisation
from permitted validation data only. The hidden-state candidates are the four
predeclared extracted states 9, 18, 27, and 36. The ESMC-specific ridge grid is
`1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6`. Selection minimizes
mean validation MAE; exact ties are resolved deterministically by lower
hidden-state index and then lower alpha.

For nested within-protein evaluation, each outer iteration selects its own
hidden-state/alpha pair using only that iteration's inner validation data. For
cross-protein transfer, one pair is selected using source-protein grouped
cross-validation before the target protein is evaluated. Held-out labels do not
contribute to either selection criterion. If nested outer iterations select
different hidden states, the pooled metric row is labelled `mixed` rather than
being attributed to a single state.

## Metrics and matched comparisons

Primary regression reporting uses MAE, RMSE, Pearson correlation, and Spearman
correlation.

Comparisons with `mutation_condition_ridge` and
`sequence_neighbourhood_ridge` use identical held-out observations and the
grouping unit defined by the frozen evaluation scheme.

## Validation requirements

Before ESMC evaluation is accepted, the implementation verifies the frozen ESMC
model identity and revision, hidden-state inventory, embedding dimensions,
embedding-index digest, canonical embedding rows, protein identities,
assay-sequence provenance, finite biological embedding ranges, absence of
outcomes from modelling-membership tables, and availability of all core
evaluation schemes.

Repository-level acceptance additionally requires formatting, lint, compilation,
the full test suite, and inspection of affected scientific outputs.
