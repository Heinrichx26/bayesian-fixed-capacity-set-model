# Bayesian capacity-constrained selected-set inference

This repository contains replication materials for Bayesian capacity-constrained selected-set inference for sparse field reliability inspection: code, public-data instructions, extraction dictionaries, certificate tables, and summary results. Article files are excluded.

The repository is maintained as a single current replication package. The current release archive is `v1.0`.

## Repository structure

- `src/analysis/`: analysis scripts for Bayesian selected-set rolling validation, comparison rules, certificate validation, capacity planning, exposure checks, external consistency checks, controlled simulations, permutation diagnostics, and posterior burden extensions.
- `src/data/`: public data download helpers where automated access is available.
- `src/figures/`: plotting and display-table generation scripts that read summary files.
- `results/`: summary CSV files used to reproduce reported numeric findings.
- `data/`: data-source notes and expected input file layout. Raw public data files are excluded from this repository.

## Data

All data sources are public:

- Federal Aviation Administration National Wildlife Strike Database.
- Federal Aviation Administration Air Traffic Activity Data System airport operations.
- Federal Aviation Administration Service Difficulty Reporting system.
- National Transportation Safety Board aviation accident data system.
- ASOS/METAR public aviation weather observations.
- Global Biodiversity Information Facility occurrence records.

Large raw exports are excluded from the repository. See `data/DATA_SOURCES.md` for source links, expected input folders, and download notes.

## Package smoke check

The package smoke check does not require raw data. It verifies that the repository contains the expected code, data-source instructions, audit files, and summary results, and that publication files are absent.

```powershell
python src/analysis/verify_replication_package.py
$files = Get-ChildItem -Recurse -Path src -Filter *.py | ForEach-Object { $_.FullName }
python -m py_compile @files
```

## Full reproduction sequence

After placing the raw public data files in the expected local folders, run the following from the repository root:

```powershell
python src/analysis/amm_synthetic_validation.py --mode smoke
python src/analysis/amm_synthetic_validation.py --mode full
python src/analysis/bayesian_set_experiments.py --mode smoke
python src/analysis/bayesian_set_experiments.py --mode full
python src/analysis/bayesian_set_method_evidence.py --mode smoke --draws 120
python src/analysis/bayesian_set_method_evidence.py --mode full --draws 500 --bootstrap 1000
python src/analysis/bayesian_set_method_evidence.py --mode full --draws 500 --candidate-only
python src/analysis/bayesian_set_method_evidence.py --mode full --draws 500 --partial-2026
python src/analysis/amm_decision_validation.py --mode smoke
python src/analysis/amm_decision_validation.py --mode full
python src/analysis/bayesian_set_feature_comparison.py --mode smoke
python src/analysis/bayesian_set_feature_comparison.py --mode full
python src/analysis/screening_comparison_rules.py
python src/analysis/permutation_diagnostics.py --full
python src/analysis/atads_exposure_validation.py
python src/analysis/field_atads_strata.py
python src/analysis/field_sdr_validation.py
python src/analysis/field_reliability_experiments.py
python src/analysis/field_asos_weather_full.py
python src/analysis/external_validation_transparency_checks.py --reps 500
python src/analysis/posterior_burden_allocation.py
```

For data-dependent smoke checks after placing the raw files:

```powershell
python src/analysis/bayesian_set_experiments.py --mode smoke
python src/analysis/permutation_diagnostics.py
python src/analysis/field_reliability_experiments.py --smoke
python src/analysis/external_validation_transparency_checks.py --smoke --reps 50
python src/analysis/posterior_burden_allocation.py --smoke
```

## Key result files

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
- `results/experiments/amm_set_model/full_synthetic_method_summary.csv`
- `results/experiments/amm_set_model/full_synthetic_scenario_summary.csv`
- `results/experiments/amm_set_model/full_synthetic_certificate_summary.csv`
- `results/experiments/amm_set_model/full_alpha_sensitivity.csv`
- `results/experiments/amm_set_model/full_draw_sensitivity.csv`
- `results/experiments/transparency_checks/ntsb_dictionary.csv`
- `results/experiments/transparency_checks/ntsb_component_mapping.csv`
- `results/experiments/transparency_checks/ntsb_stratified_audit_records.csv`
- `results/experiments/transparency_checks/ntsb_stratified_audit_summary.csv`
- `results/experiments/field_validation/sdr_component_enrichment.csv`
- `results/experiments/field_reliability/asos_weather_aggregate.csv`
- `results/experiments/field_reliability/gbif_ecological_proxy_aggregate.csv`

The main summary files report Bayesian selected-set inference, tree, forest, linear, and tabular-neural comparison rules, feature-aligned comparisons, certificate-derived ranking variants, certificate validation, capacity net-utility diagnostics, controlled sparse-cell simulations, probability calibration, annual stability, late-period prospective windows, and selected-cell certificate examples. In `prospective_window_validation.csv`, `partial_update=0` marks full-year prospective windows and `partial_update=1` marks the partial-year update. Display-table files are generated locally from the summary CSV files by the scripts in `src/figures/` and are not stored in this repository.


