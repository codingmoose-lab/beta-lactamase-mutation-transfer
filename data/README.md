# Data

The repository does not redistribute source publications, supplementary
workbooks, downloaded database records, or generated intermediate tables
through ordinary Git history.

Local data are organized as follows:

- `raw/`: unchanged source files obtained from the cited publications and
  biological databases;
- `interim/`: outputs produced by the data-preparation programs;
- `processed/`: checked research tables used by the frozen task definitions and modelling views.

Source accessions, download locations, file checksums, inclusion decisions,
and generation commands are recorded in the repository documentation and generated provenance outputs.

Run the programs in `scripts/data/` in numerical order when intentionally rebuilding
outputs from locally acquired source files. Acquisition steps should not be rerun merely
to reproduce modelling results that use the prepared data when the frozen raw sources are already present.
