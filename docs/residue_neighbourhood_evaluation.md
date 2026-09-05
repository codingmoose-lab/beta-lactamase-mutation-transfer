# Residue-neighbourhood structural evaluation

## Scientific question

This analysis tests whether the experimentally resolved residues directly surrounding a mutated site contain mutation-effect information in addition to descriptors of the mutated residue itself.

The analysis uses the residue graph prepared from the selected experimental structures. Direct neighbours are residues connected to the mutated residue by the reviewed C-alpha distance threshold of 8 angstrom. The graph definition is not changed during this analysis.

## Comparison population

The analysis uses the same experimental-structure-supported observations evaluated by Script 27. Three models are compared on identical test observations:

- `structure_matched_mutation_condition_ridge`
- `experimental_structure_descriptor_ridge`
- `residue_neighbourhood_descriptor_ridge`

The primary comparison is the residue-neighbourhood model against the experimental-structure descriptor model. This comparison asks whether the local three-dimensional residue environment adds information after the mutated residue's own structural descriptors are already included.

## Residue-neighbourhood descriptors

The mutated residue is not included in its own neighbourhood summary. The direct-neighbour block contains 36 continuous descriptors:

- direct-neighbour count;
- fractions of the 20 standard amino-acid identities;
- helix, strand, and coil fractions;
- mean and population standard deviation of neighbour heavy-atom count;
- mean and population standard deviation of neighbour solvent-accessible surface area;
- mean and population standard deviation of neighbour relative solvent accessibility;
- mean and population standard deviation of centre-to-neighbour C-alpha distance;
- mean and population standard deviation of minimum heavy-atom distance;
- peptide-adjacent fraction;
- 4.5-angstrom heavy-atom-contact fraction.

Absolute Cartesian coordinates are not direct ridge inputs. Mechanistic-reference annotations, catalytic-zinc distance annotations, and reported homologue-pair annotations are not model inputs.

## Model fitting and evaluation

The residue-neighbourhood model retains the missense-substitution and assay-condition terms used in the experimental-structure analysis. It also retains the mutated residue's structural numeric descriptors and three-state secondary structure.

Continuous inputs are standardised using means and population standard deviations calculated from the fitting rows for each validation or source-CV fit. Ridge alpha is chosen from the existing logarithmic alpha grid using validation MAE. For within-protein analyses, alpha is chosen from the inner training/validation rows for each outer evaluation fold. For directional protein transfer, alpha is chosen using source-protein grouped CV only, then the model is fitted to all eligible source rows and evaluated once on the target-protein population.

Target-test mutation-effect values do not choose alpha.

## Statistical comparison

Spearman correlation, Pearson correlation, MAE, and RMSE are reported. Paired differences compare the residue-neighbourhood model with both the mutated-residue structure model and the mutation/condition model.

Uncertainty is estimated with 1,000 paired bootstrap resamples of `evaluation_group_id`. Positive paired differences favour the residue-neighbourhood model for every reported metric.

## Interpretation boundary

For within-protein evaluation, the wild-type experimental structure of the protein is available for both fitting and test positions. The analysis therefore evaluates mutation effects at held-out residue groups in a protein with an available reference structure. It does not represent transfer to a protein whose structure was unavailable during model development.

The directional NDM-1 to VIM-2 and VIM-2 to NDM-1 analyses provide the stronger protein-background transfer tests.

## Command

```bash
python scripts/data/31_evaluate_residue_neighbourhood.py
```

Outputs are written to `data/interim/residue_neighbourhood_evaluation/`.
