# Source-condition selection and structural association review

The 1,513 paired substitutions per direction identify the source measurement condition. An `assay_id` may include several antibiotic, concentration, or temperature measurements. Fitting rows are restricted to the same assay ID, protein, ampicillin concentration, and temperature as the paired source observations. All paired observations must agree on these values. A missing or inconsistent paired condition stops the analysis. All substitutions at an excluded homologous source position remain excluded, including any other mutations measured at that position under the selected condition.

For cross-protein ampicillin comparison, the permitted source concentrations are 256 µg/mL NDM-1 or 128 µg/mL VIM-2 at 37 °C. These values are chosen by the experimental study to represent similar selection pressures. The model input remains wild-type and mutant amino-acid identity, with Ridge selection performed on source-position folds. The Ridge calculation uses regularized least squares, with non-finite input and output checks.

Structural associations are reviewed without rewriting existing calculations. An undefined correlation with zero valid resampled estimates may be mathematically unavoidable if the paired position-level model difference is constant. Such rows must not be displayed as zero correlations or positive evidence for an association. Other undefined rows remain marked for individual review.

None of these repairs changes stored original experimental scores, pre-existing neural predictions, or original figures. Manuscript conclusions remain contingent on real-data results, independent scientific checks and coauthor review.
