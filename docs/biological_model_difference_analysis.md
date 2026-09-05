# Biological model-difference analysis

Stage 36 examines where predictive information from residue structure, geometric message passing, sequence representation, and their joint model changes prediction error. It is a post-evaluation analysis: it does not tune a model and it does not use final evaluation outcomes to select settings.

The matched population is the 118,204 observations used by the residue-graph, geometric, sequence-plus-residue-graph, and validation-selected sequence/graph analyses. The ESMC prediction table contains additional sequence-eligible observations; Stage 36 retains only the observations that are present in the matched structural population.

Five error contrasts are reported. Each contrast is the absolute error of the first named model minus the absolute error of the reference model, so a negative value favours the first named model:

- residue graph minus centre-residue neural control;
- geometric residue model minus residue graph;
- ESMC sequence model minus residue graph;
- sequence-plus-residue-graph model minus residue graph;
- validation-selected prediction combination minus residue graph.

Structural interpretation is performed at the protein-position level because solvent exposure, packing, secondary structure, and functional-site distance are properties of residue positions rather than independent properties of every assay row. The primary continuous variables are relative solvent accessibility, heavy-atom contact count, and distance to the catalytic-zinc centroid. A 10 Å neighbour count and the distance to the nearest M-CSA metal-ligand correspondence are retained as sensitivity analyses rather than entered alongside strongly correlated structural variables in a single coefficient model.

Categorical summaries cover secondary-structure class, M-CSA metal-ligand correspondence, publication-reported homologue-pair membership, and a catalytic-zinc-contact sensitivity category. Category intervals are not interpreted as causal effects. Very small categories are retained for biological completeness but are marked when fewer than five positions are available.

Condition summaries are produced only where a condition varies within a scheme. Confidence intervals resample the scheme's evaluation-group identifier, preserving the grouping unit used by the evaluation design. Cross-protein target tasks have one antibiotic, concentration, and temperature and therefore do not receive artificial within-task condition comparisons.

The analysis also measures whether position-level model differences recur across the same protein positions in different tasks and across publication-reported NDM-1/VIM-2 homologous positions. These concordance results describe whether position-level error patterns recur; they do not establish a causal structural mechanism.

No multiple-testing p-value screen is used. The analysis reports estimates and grouped bootstrap intervals, and its structural associations are treated as biological interpretation of already evaluated models rather than a new model-selection step.
