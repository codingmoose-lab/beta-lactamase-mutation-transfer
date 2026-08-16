# Assay definitions

Each DMS row represents one mutation measured under one antibiotic, concentration, temperature, and assay. Original scores are preserved and missing values are not imputed.

Condition metadata are taken from the most direct available source. Some workbooks encode temperature in the score-column name, while others encode only antibiotic and concentration. When the workbook omits temperature from the column name but the publication Methods specify one fixed selection temperature for the assay, the documented assay temperature is recorded in the observation table.

The study-level temperature rules currently retained are:

- Chen et al. (2020), VIM-2: condition-specific 25 °C or 37 °C, as encoded in the DMS condition columns and described in the article;
- Chen et al. (2022), VIM-2 AMP concentration landscape: 37 °C, as reported in the published Nature Ecology & Evolution article;
- Chen et al. (2024), NDM-1: 37 °C, as reported in the Nature Communications Methods.

NDM-1 and VIM-2 are compared directly for ampicillin at study-defined concentrations chosen to provide similar selection pressure: 256 µg/mL for NDM-1 and 128 µg/mL for VIM-2 at 37 °C. Cefotaxime and meropenem measurements remain condition-specific and are not treated as directly comparable between proteins solely because the antibiotic name matches.

The 2024 cross-protein comparison table is treated as a derived comparison table and is not counted as an additional set of DMS measurements.
