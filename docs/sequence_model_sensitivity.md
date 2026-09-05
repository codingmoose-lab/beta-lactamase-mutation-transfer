# Sequence-model sensitivity analysis

The sequence-model sensitivity analysis asks whether the sequence-model findings depend strongly on the ESMC-600M representation used in the main sequence analysis.

The added comparison uses `facebook/esm2_t33_650M_UR50D` as a second protein language model. It has 33 transformer layers and approximately 650 million parameters. Candidate hidden states are fixed before evaluation at layers 8, 16, 24, and 33. This gives four depth points comparable in purpose to the four ESMC states already evaluated.

The supervised prediction model is intentionally kept the same in scientific role: mutant-site hidden state minus the corresponding wild-type residue hidden state, mutation identity and assay-condition variables, ridge regression, the established grouped partitions, and the same ridge-penalty candidates. Cross-protein setting selection uses only source-protein validation rows.

This analysis is a sensitivity check. It does not replace the established ESMC result and does not use final evaluation labels to choose between ESMC and ESM-2.

ESM-2 embedding extraction is constructed without mutation-effect outcomes. The mutation inventory is built without mutation-effect scores. The exact downloaded model snapshot is recorded. Interrupted extraction resumes after the last completed mutation batch; an interrupted batch is calculated again.

Final ESM-2 and ESMC predictions are compared only after confirming exact equality of evaluation identities and targets. Confidence intervals are calculated by resampling the established evaluation groups rather than individual observation rows. Positive paired differences always mean that ESM-2 performed better than ESMC.
