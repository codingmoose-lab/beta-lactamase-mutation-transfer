# Residue numbering

NDM-1 mutation positions use the 270-residue assay construct and correspond directly to UniProt C7C422.

VIM-2 measurements use the 267-residue experimental construct. The artificial Gly2 is retained in the assay numbering: publication position 1 maps to assay position 1, Gly2 maps to assay position 2, and publication positions 2–266 map to assay positions 3–267.

All mutation records are checked against the assay sequence before structural mapping.


## Multiple highest-scoring sequence-to-structure alignments

Residue correspondence does not depend on the first alignment returned by Biopython. When several alignments have the same highest score, only residue correspondence shared by all of them is accepted. For NDM-1 versus the deposited 3SPU polymer sequence, the two highest-scoring alignments differ at UniProt positions 1 and 27 and PDB entity position 1. Entity position 1 is unmodelled in the selected experimental chains, so no coordinate-bearing residue mapping is affected; the ambiguous unmodelled correspondence is recorded and left unresolved.
