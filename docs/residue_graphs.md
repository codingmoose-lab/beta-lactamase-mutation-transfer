# Experimental-structure residue graphs

## Purpose

Residue-graph preparation converts the selected experimental
structures into protein residue graphs without fitting a prediction model.

The graph is constructed from the same experimentally supported structural
population used for the structure-matched analyses. Graph preparation is
separate from subsequent graph-model design so that topology, structural
provenance, and evaluation coverage can be audited before any graph model is
selected or trained.

## Structural source

The primary structures remain:

- NDM-1: PDB 3SPU, chain B;
- VIM-2: PDB 5YD7, chain B.

Only residues with experimental coordinates in the selected chain become graph
nodes. The graph therefore contains 238 NDM-1 nodes and 231 VIM-2 nodes under
the current checked structural inputs.

The VIM-2 graph contains one structural context node, assay position 186, that
does not occur in the structure-matched tested residue population. It remains
part of the experimentally observed structural scaffold but is not converted
into a prediction target.

## Primary topology

The primary undirected graph connects two residues when their C-alpha distance
is at most 8 Å.

This choice is fixed after auditing the available 8 Å, 10 Å, 12 Å, heavy-atom
contact, and peptide-adjacency definitions. The 8 Å graphs are connected for
both proteins, contain no isolated nodes, retain substantial non-local
structural connectivity, and have similar graph density across NDM-1 and
VIM-2.

Peptide-adjacent residues are not added as a second duplicate edge set because
all peptide-adjacent pairs are already contained within the 8 Å graph for the
selected structures. Peptide adjacency remains available as edge metadata.

Each undirected edge is stored once in ascending node-index order. Self-loops are not
stored.

## Node information

The graph-node table retains identifiers and outcome-independent structural
information permitted by the modelling-view contracts, including:

- assay residue identity;
- selected experimental-structure identity;
- C-alpha coordinates;
- side-chain-centroid coordinates;
- heavy-atom count;
- solvent-accessibility values;
- three-state secondary structure.

Graph preparation does not define the final neural-model node-feature
vector. Each model that uses these graphs must state which permitted
node fields are supplied to a model.

## Edge information

For every primary 8 Å edge, the graph retains:

- C-alpha distance;
- minimum heavy-atom distance;
- sequence separation;
- peptide-adjacency status;
- 4.5 Å heavy-atom-contact status.

Retaining these measurements in the graph representation does not imply that
every field must be used as a direct edge predictor in a later model.

## Information boundary

Graph construction does not use mutation-effect values, model predictions,
test labels, fold assignments, selected hyperparameters, M-CSA correspondence,
catalytic-Zn distances, or reported homologue-pair annotations.

Modelling-view membership is consulted only after graph construction to verify
that all positions in the core experimental-structure evaluation population
have corresponding graph nodes. Evaluation membership therefore audits
coverage; it does not determine graph topology.

## Outputs

Running

```bash
python scripts/data/30_prepare_residue_graphs.py
```

writes:

- `data/interim/residue_graphs/residue_graph_nodes.csv`;
- `data/interim/residue_graphs/residue_graph_edges.csv`;
- `data/interim/residue_graphs/residue_graph_summary.json`.

The summary records graph counts, topology checks, context-only positions,
source-file SHA-256 values, graph-output SHA-256 values, and the information
boundary used during construction.
