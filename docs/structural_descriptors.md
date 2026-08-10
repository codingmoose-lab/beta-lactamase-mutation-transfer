# Structural residue descriptors

Structural-descriptor preparation derives geometry from the selected experimental references, 3SPU chain B for
NDM-1 and 5YD7 chain B for VIM-2. It retains all 537 assay residues so that residues
without experimental coordinates remain explicit rather than disappearing from later
joins.

## Residue-level quantities

For residues with experimental coordinates, the command records C-alpha coordinates,
side-chain centroid coordinates, heavy-atom count, solvent-accessible surface area,
local neighbour counts, physical-contact counts, and distances to the two selected
catalytic zinc atoms and to M-CSA-reference-corresponding residues.

Solvent-accessible surface area is calculated with Biopython's Shrake-Rupley
implementation using a 1.4 A probe and 100 sphere points. The calculation uses only
standard amino-acid residues from the selected protein chain, excluding waters,
metals, and other hetero residues so that crystal solvent and additives do not become
part of the protein-exposure definition. Relative accessibility is the Shrake-Rupley
area divided by the Wilke/Tien residue-specific maximum accessibility supplied by
Biopython.

## Pairwise geometry

`residue_pair_geometry.csv` contains every unordered pair of experimentally modeled
residues within each selected chain. It records C-alpha distance, minimum heavy-atom
distance, assay-sequence separation, peptide adjacency, three C-alpha radius flags,
and a 4.5 A heavy-atom contact flag. Raw distances are retained so later graph and
structural-image experiments can choose edge or neighbourhood definitions without
recomputing coordinates or changing the source geometry.

The pair table does not itself define the final graph. Edge relations and image
neighbourhood rules are frozen later with the model-input specification and evaluation
splits.

## Deliberately deferred quantities

Secondary structure is not assigned by this command because the preferred DSSP route
requires an external DSSP executable and should be checked explicitly in the working
environment before becoming a required dependency. Residue depth is also deferred
because Biopython's residue-depth implementation requires MSMS. Solvent accessibility
already provides a direct exposure measure without introducing that external program.

The selected 5YD7 structure contains formate additives. They are not treated as an
antibiotic or substrate feature. A bound-state ligand analysis, if later justified,
must use an explicitly selected biologically relevant structure rather than relabeling
formate as a drug-contact signal.
