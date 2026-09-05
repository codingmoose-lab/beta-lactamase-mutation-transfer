# Sequence and residue-graph neural evaluation

The sequence-plus-residue-graph evaluation asks whether the ESMC mutation signal adds prediction information to the residue graph evaluated in the residue-graph neural evaluation.

The analysis uses the ESMC site delta at hidden state 36. This is the existing primary ESMC hidden state in the repository; the sequence-plus-residue-graph evaluation does not reopen hidden-state selection after later test results have been observed. The site delta is the mutant-site ESMC hidden state minus the wild-type residue hidden state for the same assay position.

The graph branch retains the residue-graph GINE graph, node measurements, edge measurements, mutation encoding, assay-condition encoding, optimisation settings, epoch-selection rule, and training seeds. The 1,152-dimensional ESMC site delta is standardised using fitting rows only and projected to 128 dimensions before it is combined with the graph residue state, mutation encoding, and assay-condition values.

The primary comparison is `sequence_residue_graph_neural_regressor` versus `residue_graph_neural_regressor`. Both are evaluated on the same 118,204 residue-graph evaluation rows, representing 109,408 distinct experimental observations across the predefined evaluation contexts. Model selection uses validation rows for within-protein analyses and source-protein cross-validation for cross-protein analyses. Outer-test and target-test outcomes are not used to choose the number of training epochs.

The sequence-plus-residue-graph evaluation does not search several fusion architectures. A positive result supports the narrower conclusion that the existing ESMC mutation signal contributes information in addition to the residue-graph model under this analysis. A null or negative result does not establish that sequence information is generally unhelpful; it applies to this tested combination of ESMC site-delta features and the the residue-graph neural evaluation residue graph.

Three training repeats are retained. Grouped bootstrap intervals resample `evaluation_group_id`. Context training files and bootstrap comparisons are saved as they finish so an interrupted run can continue without repeating completed work.
