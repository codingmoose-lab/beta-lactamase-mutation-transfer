# Condition-transfer evaluation

Stage 39 evaluates five VIM-2 Chen et al. (2020) transfer schemes already defined before model fitting: three held-out-antibiotic tests at the study-defined high screening condition and both AMP temperature directions.

Model settings are selected only from the source-condition rows using the existing five mutation-group folds. Target-condition scores are not used to choose hidden states, ridge penalties, model families, or any other setting.

For the held-out-antibiotic tests, the fitted sequence and structure models do not receive antibiotic identity. The target antibiotic is absent from source fitting, so assigning it an untrained categorical coefficient would not represent learned antibiotic transfer. ESMC and ESM-2 site-delta ridge models therefore use mutation information only. The experimental-structure models use mutation and WT structural descriptors without an antibiotic indicator.

For temperature transfer, the target and source populations contain matched AMP mutation/concentration measurements. Fitted sequence and structure models use log2 concentration so dose is represented while temperature remains the held-out condition. The direct reference copies the matched source-temperature mutation/concentration score.

Two evaluation populations are reported. `sequence_eligible` contains the complete target population. `experimental_structure_matched` contains only target observations with the selected experimental-structure descriptors and is used for fair sequence-versus-structure comparisons.

The primary reference is `source_antibiotic_mean` for held-out-antibiotic tests and `source_temperature_matched` for temperature transfer. Paired metric differences use mutation-group bootstrap resampling. Positive paired differences always mean that the fitted model performs better than the direct source-condition reference.

Stage 39 does not fit residue-graph, EGNN, or sequence-plus-graph neural models. Those models would require either omitting the held-out antibiotic identity, which duplicates the information question already tested by the sequence and structure ridge models, or introducing an unlearned antibiotic category. This stage keeps the comparison focused on whether sequence or WT experimental-structure information improves transfer beyond directly observed effects of the same mutations under source conditions.
