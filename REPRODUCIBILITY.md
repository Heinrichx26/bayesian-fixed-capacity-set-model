# Reproducibility notes

## Software

The analysis scripts use Python 3.10 or later. Third-party Python packages are listed in `requirements.txt`.

## Recommended workflow

1. Run `python src/analysis/verify_replication_package.py` to check the repository contents.
2. Download and place raw public data files according to `data/DATA_SOURCES.md`.
3. Run the data-dependent smoke checks listed in `README.md`.
4. Run the full analysis scripts.
5. Compare generated CSV outputs with the summary CSV files in `results/`.

## Bayesian set model result files

- `results/experiments/bayesian_set_model/full/set_model_selection_aggregate.csv`
- `results/experiments/bayesian_set_model/full/set_model_selection_yearly.csv`
- `results/experiments/bayesian_set_model/full/set_model_probability_calibration.csv`
- `results/experiments/bayesian_set_model/full/set_model_certificate_cells_2025.csv`
- `results/experiments/bayesian_set_model/method_evidence/full/certificate_group_aggregate.csv`
- `results/experiments/bayesian_set_model/method_evidence/full/boundary_candidate_audit_aggregate.csv`
- `results/experiments/bayesian_set_model/method_evidence/full/hierarchy_ablation_aggregate.csv`
- `results/experiments/bayesian_set_model/method_evidence/full/year_bootstrap_intervals.csv`
- `results/experiments/bayesian_set_model/method_evidence/partial_2026/partial_2026_update.csv`
- `results/experiments/bayesian_set_model/method_evidence/partial_2026/partial_2026_certificate_groups.csv`
- `results/experiments/bayesian_set_model/decision_validation/full/bayesian_decision_rule_aggregate.csv`
- `results/experiments/bayesian_set_model/decision_validation/full/capacity_net_utility.csv`
- `results/experiments/bayesian_set_model/decision_validation/full/certificate_interval_coverage_aggregate.csv`
- `results/experiments/bayesian_set_model/decision_validation/full/certificate_bootstrap_stability_aggregate.csv`
- `results/experiments/bayesian_set_model/decision_validation/full/certificate_expansion_validity_aggregate.csv`
- `results/experiments/bayesian_set_model/prospective_windows/prospective_window_validation.csv`
- `results/experiments/bayesian_set_model/feature_comparison/full/feature_comparison_aggregate.csv`
- `results/experiments/bayesian_set_model/feature_comparison/full/feature_comparison_calibration.csv`
- `results/experiments/bayesian_set_model/feature_comparison/full/feature_comparison_yearly.csv`
## Additional result files

- `results/experiments/atads_exposure/atads_exposure_aggregate.csv`
- `results/experiments/field_validation/atads_exposure_strata_aggregate.csv`
- `results/experiments/field_validation/sdr_component_enrichment.csv`
- `results/experiments/field_validation/sdr_component_profile.csv`
- `results/experiments/field_reliability/budget_frontier_aggregate.csv`
- `results/experiments/field_reliability/asos_weather_aggregate.csv`
- `results/experiments/field_reliability/gbif_ecological_proxy_aggregate.csv`
- `results/experiments/field_reliability/ntsb_nonwildlife_stress_check.csv`
- `results/experiments/field_reliability/reporting_bias_strata_aggregate.csv`
- `results/experiments/transparency_checks/ntsb_external_enrichment_sets.csv`
- `results/experiments/transparency_checks/ntsb_stratified_audit_summary.csv`
- `results/experiments/posterior_burden/full/posterior_burden_aggregate.csv`


