# Repository layout

The repository separates executable research commands, reusable Python code, tests, documentation, and local scientific data.


```
beta-lactamase-mutation-transfer
├─ README.md
├─ data
│  ├─ README.md
│  ├─ interim
│  │  ├─ alphafold_acquisition
│  │  │  ├─ acquisition_summary.json
│  │  │  ├─ downloaded_files.csv
│  │  │  └─ models.csv
│  │  ├─ alphafold_preparation
│  │  │  ├─ assay_alphafold_mapping.csv
│  │  │  ├─ local_confidence.csv
│  │  │  ├─ model_metadata.csv
│  │  │  ├─ pdb_alphafold_correspondence.csv
│  │  │  ├─ pdb_alphafold_summary.csv
│  │  │  ├─ preparation_summary.json
│  │  │  ├─ residue_confidence.csv
│  │  │  └─ residue_pae_summary.csv
│  │  ├─ assay_definitions
│  │  │  ├─ assay_summary.json
│  │  │  ├─ assays.csv
│  │  │  ├─ condition_score_statistics.csv
│  │  │  ├─ cross_protein_conditions.csv
│  │  │  ├─ comparison_value_relationships.csv
│  │  │  └─ measurement_groups.csv
│  │  ├─ baseline_evaluation
│  │  │  ├─ baseline_bootstrap_intervals.csv
│  │  │  ├─ baseline_metrics.csv
│  │  │  ├─ baseline_model_selection.csv
│  │  │  ├─ baseline_predictions.csv
│  │  │  ├─ baseline_summary.json
│  │  │  ├─ baseline_uncertainty_summary.json
│  │  │  ├─ homologue_analysis
│  │  │  │  ├─ discordant_floor_pairs.csv
│  │  │  │  ├─ discordant_floor_summary.csv
│  │  │  │  ├─ homologue_analysis_summary.json
│  │  │  │  ├─ homologue_floor_counts.csv
│  │  │  │  ├─ homologue_floor_sensitivity.csv
│  │  │  │  ├─ matched_homologue_metrics.csv
│  │  │  │  ├─ matched_homologue_predictions.csv
│  │  │  │  ├─ nonfloor_matched_metrics.csv
│  │  │  │  ├─ nonfloor_paired_comparisons.csv
│  │  │  │  └─ paired_baseline_comparisons.csv
│  │  │  ├─ homologue_direct_transfer_intervals.csv
│  │  │  ├─ homologue_direct_transfer_metrics.csv
│  │  │  └─ homologue_direct_transfer_predictions.csv
│  │  ├─ card_acquisition
│  │  │  ├─ acquisition_summary.json
│  │  │  ├─ extracted_files.csv
│  │  │  ├─ source_inventory.csv
│  │  │  └─ target_presence.csv
│  │  ├─ card_context
│  │  │  ├─ assay_drug_class_crosswalk.csv
│  │  │  ├─ context_summary.json
│  │  │  ├─ direct_antibiotic_annotations.csv
│  │  │  ├─ drug_class_annotations.csv
│  │  │  ├─ mechanism_annotations.csv
│  │  │  ├─ ontology_ancestry.csv
│  │  │  └─ target_terms.csv
│  │  ├─ evaluation_partitions
│  │  │  ├─ evaluation_partition_summary.json
│  │  │  ├─ group_assignments.csv
│  │  │  ├─ partition_audit.csv
│  │  │  ├─ partition_schemes.csv
│  │  │  └─ partition_scopes.csv
│  │  ├─ mcsa_acquisition
│  │  │  ├─ acquisition_summary.json
│  │  │  ├─ downloaded_files.csv
│  │  │  └─ entries.csv
│  │  ├─ mcsa_correspondence
│  │  │  ├─ alignment_summary.csv
│  │  │  ├─ catalytic_residue_correspondence.csv
│  │  │  ├─ catalytic_role_annotations.csv
│  │  │  └─ correspondence_summary.json
│  │  ├─ mcsa_reference_sequences
│  │  │  ├─ acquisition_summary.json
│  │  │  ├─ downloaded_files.csv
│  │  │  └─ reference_sequences.csv
│  │  ├─ modelling_views
│  │  │  ├─ comparison_populations.csv
│  │  │  ├─ modelling_view_summary.json
│  │  │  ├─ observation_column_contracts.csv
│  │  │  ├─ representation_contracts.csv
│  │  │  ├─ representation_scope_counts.csv
│  │  │  ├─ residue_feature_contracts.csv
│  │  │  ├─ residue_pair_feature_contracts.csv
│  │  │  └─ scheme_observation_membership.csv
│  │  ├─ mutation_measurements
│  │  │  ├─ cross_protein_comparisons.csv
│  │  │  ├─ ec50_measurements.csv
│  │  │  ├─ excluded_measurements.csv
│  │  │  ├─ measurement_summary.json
│  │  │  ├─ mutation_measurements.csv
│  │  │  ├─ mutation_measurements_raw.csv
│  │  │  ├─ mutation_structure_mapping.csv
│  │  │  └─ validation_summary.json
│  │  ├─ protein_structures
│  │  │  ├─ chain_coverage.csv
│  │  │  ├─ chain_metadata.csv
│  │  │  ├─ ligand_sites.csv
│  │  │  ├─ local_environment_completeness.csv
│  │  │  ├─ measured_position_coverage.csv
│  │  │  ├─ metal_contacts.csv
│  │  │  ├─ metal_sites.csv
│  │  │  ├─ residue_coordinate_status.csv
│  │  │  ├─ selected_structures.json
│  │  │  ├─ structure_metadata.csv
│  │  │  └─ structure_summary.json
│  │  ├─ residue_evidence
│  │  │  ├─ homologue_position_correspondence.csv
│  │  │  ├─ reported_homologue_position_pairs.csv
│  │  │  ├─ residue_evidence.csv
│  │  │  └─ residue_evidence_summary.json
│  │  ├─ secondary_structure
│  │  │  ├─ secondary_structure.csv
│  │  │  └─ secondary_structure_summary.json
│  │  ├─ sequence_mapping
│  │  │  ├─ reference_sequences.fasta
│  │  │  ├─ residue_number_mapping.csv
│  │  │  ├─ sequence_correspondence.csv
│  │  │  ├─ sequence_mapping_summary.json
│  │  │  ├─ sequence_sources.csv
│  │  │  └─ structure_residue_status.csv
│  │  ├─ sequence_neighbourhood_evaluation
│  │  │  ├─ sequence_neighbourhood_metrics.csv
│  │  │  ├─ sequence_neighbourhood_model_selection.csv
│  │  │  ├─ sequence_neighbourhood_paired_comparisons.csv
│  │  │  ├─ sequence_neighbourhood_predictions.csv
│  │  │  └─ sequence_neighbourhood_summary.json
│  │  ├─ sequence_plm_embeddings
│  │  │  ├─ embedding_summary.json
│  │  │  ├─ extraction_summary.json
│  │  │  ├─ mutant_site_embeddings.npy
│  │  │  ├─ mutation_embedding_index.csv
│  │  │  └─ wild_type_residue_embeddings.npy
│  │  ├─ sequence_plm_evaluation
│  │  │  ├─ sequence_plm_metrics.csv
│  │  │  ├─ sequence_plm_model_selection.csv
│  │  │  ├─ sequence_plm_predictions.csv
│  │  │  └─ sequence_plm_summary.json
│  │  ├─ source_tables
│  │  │  ├─ column_definitions.csv
│  │  │  ├─ mutation_field_candidates.csv
│  │  │  ├─ mutation_field_examples.csv
│  │  │  ├─ source_table_relationships.csv
│  │  │  ├─ source_table_summary.json
│  │  │  ├─ workbook_metadata.csv
│  │  │  └─ worksheet_metadata.csv
│  │  ├─ structural_descriptors
│  │  │  ├─ residue_pair_geometry.csv
│  │  │  ├─ residue_structural_descriptors.csv
│  │  │  └─ structural_descriptor_summary.json
│  │  └─ task_definitions
│  │     ├─ aligned_shared_wild_type_mutation_pairs.csv
│  │     ├─ available_conditions.csv
│  │     ├─ evaluation_regimes.csv
│  │     ├─ task_definition_summary.json
│  │     ├─ task_definitions.csv
│  │     └─ task_scopes.csv
│  ├─ processed
│  │  ├─ assay_metadata.csv
│  │  ├─ control_observations.csv
│  │  ├─ ec50_observations.csv
│  │  ├─ missense_observations.csv
│  │  ├─ observation_column_roles.csv
│  │  ├─ observations.csv
│  │  ├─ processed_summary.json
│  │  ├─ reported_cross_protein_comparisons.csv
│  │  ├─ residue_features.csv
│  │  └─ residue_pair_geometry.csv
│  └─ raw
│     ├─ alphafold
│     │  ├─ C7C422
│     │  │  ├─ SHA256SUMS.txt
│     │  │  ├─ confidence.json
│     │  │  ├─ metadata.json
│     │  │  ├─ model.cif
│     │  │  └─ pae.json
│     │  └─ Q9K2N0
│     │     ├─ SHA256SUMS.txt
│     │     ├─ confidence.json
│     │     ├─ metadata.json
│     │     ├─ model.cif
│     │     └─ pae.json
│     ├─ card
│     │  └─ ontology
│     │     └─ 4.0.1
│     │        ├─ Ontology-Download-README.txt
│     │        ├─ SHA256SUMS.txt
│     │        ├─ aro.json
│     │        ├─ aro.obo
│     │        ├─ aro.owl
│     │        ├─ aro.tsv
│     │        ├─ card-ontology.tar.bz2
│     │        ├─ mo.json
│     │        ├─ mo.obo
│     │        ├─ mo.tsv
│     │        ├─ ncbi_taxonomy.json
│     │        ├─ ncbi_taxonomy.obo
│     │        ├─ ncbi_taxonomy.tsv
│     │        ├─ ro.json
│     │        ├─ ro.obo
│     │        ├─ ro.tsv
│     │        ├─ viro.obo
│     │        └─ viro.owl
│     ├─ mcsa
│     │  ├─ SHA256SUMS.txt
│     │  ├─ entries_15_16.json
│     │  └─ residues_15_16.json
│     ├─ ndm_vim_2024
│     │  ├─ chen_2024_nature_communications_article.pdf
│     │  ├─ chen_2024_source_data.xlsx
│     │  ├─ chen_2024_supplementary_data_1_b1_sequences.txt
│     │  ├─ chen_2024_supplementary_data_3_ndm1_dms.xlsx
│     │  ├─ chen_2024_supplementary_data_4_ndm1_vim2_comparison.xlsx
│     │  ├─ chen_2024_supplementary_data_6_ndm1_plasmid.gb
│     │  ├─ chen_2024_supplementary_files_description.pdf
│     │  └─ chen_2024_supplementary_information.pdf
│     ├─ pdb
│     │  ├─ 3SPU
│     │  │  ├─ 3SPU-assembly1.cif
│     │  │  ├─ 3SPU-assembly2.cif
│     │  │  ├─ 3SPU-assembly3.cif
│     │  │  ├─ 3SPU.cif
│     │  │  └─ 3SPU_full_validation.pdf
│     │  └─ 5YD7
│     │     ├─ 5YD7-assembly1.cif
│     │     ├─ 5YD7-assembly2.cif
│     │     ├─ 5YD7.cif
│     │     └─ 5YD7_full_validation.pdf
│     ├─ uniprot
│     │  ├─ A4GRB6
│     │  │  └─ A4GRB6.json
│     │  ├─ C7C422
│     │  │  ├─ C7C422.fasta
│     │  │  ├─ C7C422.json
│     │  │  └─ C7C422.txt
│     │  ├─ P04190
│     │  │  ├─ P04190.fasta
│     │  │  ├─ P04190.json
│     │  │  └─ P04190.txt
│     │  ├─ P25910
│     │  │  ├─ P25910.fasta
│     │  │  ├─ P25910.json
│     │  │  └─ P25910.txt
│     │  ├─ Q9K2N0
│     │  │  ├─ Q9K2N0.fasta
│     │  │  ├─ Q9K2N0.json
│     │  │  └─ Q9K2N0.txt
│     │  └─ SHA256SUMS.txt
│     ├─ vim2_2020
│     │  ├─ chen_2020_elife_article.pdf
│     │  ├─ chen_2020_supplementary_file_1_vim2_plasmid.gb
│     │  ├─ chen_2020_supplementary_file_2_vim2_dms_and_annotations.xlsx
│     │  └─ chen_2020_supplementary_file_3_ctx_mem_linear_models.xlsx
│     └─ vim2_2022
│        ├─ chen_2021_biorxiv_preprint_v2.pdf
│        └─ chen_2022_supplementary_data_vim2_concentration_landscape.xlsx
├─ docs
│  ├─ assay_definitions.md
│  ├─ baseline_evaluation.md
│  ├─ evaluation_partitions.md
│  ├─ execution.md
│  ├─ modelling_views.md
│  ├─ processed_tables.md
│  ├─ repository_layout.md
│  ├─ residue_numbering.md
│  ├─ sequence_neighbourhood_evaluation.md
│  ├─ structural_descriptors.md
│  ├─ structure_selection.md
│  └─ task_definitions.md
├─ project codes.zip
├─ pyproject.toml
├─ requirements-dev.txt
├─ requirements.txt
├─ scripts
│  └─ data
│     ├─ 01_map_reference_sequences.py
│     ├─ 02_describe_source_tables.py
│     ├─ 03_extract_mutation_measurements.py
│     ├─ 04_define_assays_and_conditions.py
│     ├─ 05_prepare_protein_structures.py
│     ├─ 06_acquire_alphafold_models.py
│     ├─ 07_prepare_alphafold_models.py
│     ├─ 08_acquire_mcsa_annotations.py
│     ├─ 09_acquire_mcsa_reference_sequences.py
│     ├─ 10_prepare_mcsa_correspondence.py
│     ├─ 11_acquire_card_ontology.py
│     ├─ 12_prepare_card_context.py
│     ├─ 13_prepare_residue_evidence.py
│     ├─ 14_prepare_structural_descriptors.py
│     ├─ 15_prepare_secondary_structure.py
│     ├─ 16_prepare_processed_tables.py
│     ├─ 17_define_modelling_tasks.py
│     ├─ 18_define_evaluation_partitions.py
│     ├─ 19_prepare_modelling_views.py
│     ├─ 20_run_baseline_evaluation.py
│     ├─ 21_calculate_baseline_uncertainty.py
│     ├─ 22_analyze_homologue_baseline_transfer.py
│     ├─ 23_evaluate_sequence_neighbourhood.py
│     ├─ 24_prepare_sequence_plm_embeddings.py
│     ├─ 25_evaluate_sequence_plm.py
│     ├─ 26_compare_sequence_models.py
│     ├─ 27_evaluate_experimental_structure.py
│     ├─ 28_compare_structure_matched_models.py
│     ├─ 29_analyze_biological_errors.py
│     ├─ 30_prepare_residue_graphs.py
│     ├─ 31_evaluate_residue_neighbourhood.py
│     ├─ 32_evaluate_residue_graph_neural.py
│     ├─ 33_evaluate_residue_geometry_neural.py
│     ├─ 34_evaluate_sequence_residue_graph.py
│     ├─ 35_analyze_sequence_structure_complementarity.py
│     └─ 36_analyze_biological_model_differences.py
├─ src
│  └─ betalactamase_transfer
│     ├─ __init__.py
│     ├─ alphafold_models.py
│     ├─ alphafold_preparation.py
│     ├─ assay_definitions.py
│     ├─ baseline_evaluation.py
│     ├─ baseline_uncertainty.py
│     ├─ biological_error_analysis.py
│     ├─ biological_model_difference_analysis.py
│     ├─ card_context.py
│     ├─ evaluation_partitions.py
│     ├─ experimental_structure_evaluation.py
│     ├─ homologue_baseline_analysis.py
│     ├─ mcsa_annotations.py
│     ├─ mcsa_correspondence.py
│     ├─ mcsa_reference_sequences.py
│     ├─ modelling_views.py
│     ├─ structure_matched_model_comparison.py
│     ├─ mutation_measurements.py
│     ├─ processed_tables.py
│     ├─ protein_structures.py
│     ├─ residue_evidence.py
│     ├─ residue_graphs.py
│     ├─ secondary_structure.py
│     ├─ sequence_mapping.py
│     ├─ sequence_model_comparison.py
│     ├─ sequence_neighbourhood_evaluation.py
│     ├─ sequence_plm_embeddings.py
│     ├─ sequence_plm_evaluation.py
│     ├─ source_tables.py
│     ├─ structural_descriptors.py
│     └─ task_definitions.py
└─ tests
   ├─ __init__.py
   ├─ conftest.py
   ├─ test_alphafold_models.py
   ├─ test_alphafold_preparation.py
   ├─ test_alphafold_preparation_integration.py
   ├─ test_assay_definitions.py
   ├─ test_assay_definitions_integration.py
   ├─ test_baseline_evaluation.py
   ├─ test_baseline_evaluation_integration.py
   ├─ test_baseline_uncertainty.py
   ├─ test_biological_error_analysis.py
   ├─ test_biological_error_analysis_integration.py
   ├─ test_biological_model_difference_analysis.py
   ├─ test_biological_model_difference_analysis_integration.py
   ├─ test_card_context.py
   ├─ test_card_context_integration.py
   ├─ test_evaluation_partitions.py
   ├─ test_evaluation_partitions_integration.py
   ├─ test_experimental_structure_evaluation.py
   ├─ test_experimental_structure_evaluation_integration.py
   ├─ test_homologue_baseline_analysis.py
   ├─ test_homologue_baseline_analysis_integration.py
   ├─ test_mcsa_annotations.py
   ├─ test_mcsa_correspondence.py
   ├─ test_mcsa_correspondence_integration.py
   ├─ test_mcsa_reference_sequences.py
   ├─ test_modelling_views.py
   ├─ test_modelling_views_integration.py
   ├─ test_structure_matched_model_comparison.py
   ├─ test_structure_matched_model_comparison_integration.py
   ├─ test_mutation_measurements.py
   ├─ test_mutation_measurements_integration.py
   ├─ test_processed_tables.py
   ├─ test_processed_tables_integration.py
   ├─ test_protein_structures.py
   ├─ test_protein_structures_integration.py
   ├─ test_residue_evidence.py
   ├─ test_residue_evidence_integration.py
   ├─ test_residue_graphs.py
   ├─ test_residue_graphs_integration.py
   ├─ test_secondary_structure.py
   ├─ test_secondary_structure_integration.py
   ├─ test_sequence_mapping.py
   ├─ test_sequence_mapping_integration.py
   ├─ test_sequence_model_comparison.py
   ├─ test_sequence_model_comparison_integration.py
   ├─ test_sequence_neighbourhood_evaluation.py
   ├─ test_sequence_neighbourhood_evaluation_integration.py
   ├─ test_sequence_plm_embeddings.py
   ├─ test_sequence_plm_embeddings_integration.py
   ├─ test_sequence_plm_evaluation.py
   ├─ test_sequence_plm_evaluation_integration.py
   ├─ test_source_tables.py
   ├─ test_source_tables_integration.py
   ├─ test_structural_descriptors.py
   ├─ test_structural_descriptors_integration.py
   ├─ test_task_definitions.py
   └─ test_task_definitions_integration.py

```


## `scripts/data/`

The `data` name is intentional. These files are executable commands whose purpose is acquisition, source inspection, curation, mapping, preparation, biological annotation, pre-modelling task definition, evaluation-partition preparation, or modelling-view preparation. They are kept separate from reusable implementation code under `src/`.


## `src/betalactamase_transfer/`

This package contains reusable scientific and data-processing functions. The package is kept relatively flat while the number of modules remains manageable. Subpackages should be introduced only when a clear group of modules develops around a distinct responsibility.

## `data/`

- `raw/` contains locally retained source files from publications and biological databases;
- `interim/` contains generated preparation tables, task definitions, evaluation partitions, modelling-view contracts, sequence embeddings, predictions, metrics, and model-comparison outputs;
- `processed/` contains the checked observation, assay, residue, and residue-pair tables used by the modelling analyses.

The large local data directories are excluded from ordinary Git history. Code, tests, documentation, and small configuration files are version controlled; source publications, database downloads, and generated scientific tables remain local.
