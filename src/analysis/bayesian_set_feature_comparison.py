from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from bayesian_set_experiments import (
    COMPACT_FEATURES,
    MAIN_CAPACITIES,
    TARGET,
    aggregate_selection,
    calibration_row,
    evaluate_scored,
    score_catboost,
    score_bayesian_species_child,
    score_hierarchical_mean,
    score_lightgbm,
    write_csv,
)
from bayesian_set_core import load_parts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "feature_comparison"


def run_feature_comparison(start_year: int, end_year: int) -> None:
    parts = load_parts()
    yearly: list[dict] = []
    calibration: dict[str, dict] = defaultdict(lambda: {"y": [], "p": [], "years": set()})
    out_dir = RESULT_ROOT / ("smoke" if start_year == end_year else "full")

    for test_year in range(start_year, end_year + 1):
        train_start = test_year - 5
        train_end = test_year - 1
        train = [row for row in parts if train_start <= int(row["year"]) <= train_end]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue

        y_test = [int(bool(row[TARGET])) for row in test]
        scored_rules = {
            "hierarchical_eb_execution": (score_hierarchical_mean(train, test), None),
            "bayesian_species_child": score_bayesian_species_child(train, test),
            "compact_lightgbm": score_lightgbm(train, test, include_species=False),
            "compact_catboost": score_catboost(train, test, features=COMPACT_FEATURES),
        }

        for rule, (scored, probabilities) in scored_rules.items():
            for capacity in MAIN_CAPACITIES:
                yearly.append(evaluate_scored(scored, capacity, rule, train_start, train_end, test_year))
            if probabilities is not None and len(probabilities) == len(y_test):
                calibration[rule]["y"].extend(y_test)
                calibration[rule]["p"].extend([float(value) for value in probabilities])
                calibration[rule]["years"].add(test_year)

    aggregate = aggregate_selection(yearly)
    calibration_rows = [
        calibration_row(rule, item["y"], item["p"], item["years"])
        for rule, item in calibration.items()
        if item["y"] and len(set(item["y"])) > 1
    ]
    write_csv(out_dir / "feature_comparison_yearly.csv", yearly)
    write_csv(out_dir / "feature_comparison_aggregate.csv", aggregate)
    write_csv(out_dir / "feature_comparison_calibration.csv", sorted(calibration_rows, key=lambda row: row["brier_score"]))
    print(f"Feature comparison built in {out_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact-feature comparison rules for AMM Bayesian set model.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_feature_comparison(2006, 2006)
    else:
        run_feature_comparison(1995, 2025)


if __name__ == "__main__":
    main()


