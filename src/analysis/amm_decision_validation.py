from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayesian_set_method_evidence import (  # noqa: E402
    CAPACITIES,
    FULL_LEVEL,
    PRIOR_WEIGHT,
    TARGET,
    bayesian_set_cell_scores,
    build_test_cells,
    load_parts,
    posterior_for_row,
    score_hierarchical_mean,
    selected_metrics,
    train_hierarchical_posteriors,
    tuple_key,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "decision_validation"
MAIN_RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "full"

SMOKE_YEARS = list(range(2021, 2026))
FULL_YEARS = list(range(1995, 2026))
NET_CAPACITIES = [0.01, 0.025, 0.05, 0.10, 0.20]


def unit_key(row: dict) -> tuple[str, str]:
    return str(row["event_id"]), str(row["component"])


def support_group(support: int) -> str:
    if support < 10:
        return "0--9"
    if support < 50:
        return "10--49"
    if support < 200:
        return "50--199"
    return ">=200"


def pi_bin(pi: float) -> str:
    if pi >= 0.95:
        return "[0.95, 1.00]"
    if pi >= 0.50:
        return "[0.50, 0.95)"
    if pi >= 0.10:
        return "[0.10, 0.50)"
    if pi >= 0.01:
        return "[0.01, 0.10)"
    return "[0.00, 0.01)"


def selected_rows_from_scores(scored: list[tuple[float, dict]], capacity: float) -> list[dict]:
    k = max(1, math.ceil(len(scored) * capacity))
    return [row for _, row in scored[:k]]


def evaluate_scored_rules(
    yearly: list[dict],
    score_store: dict[tuple[str, float], dict[str, list[float]]],
) -> list[dict]:
    groups: dict[tuple[str, float], dict] = defaultdict(lambda: {
        "years": set(),
        "test_units": 0,
        "damage_units": 0,
        "selected_units": 0,
        "selected_damage_units": 0,
        "cost_num": 0.0,
        "cost_den": 0.0,
        "aos_num": 0.0,
        "aos_den": 0.0,
        "annual_lift": [],
    })
    for row in yearly:
        key = (row["rule"], row["capacity"])
        item = groups[key]
        item["years"].add(row["test_year"])
        item["test_units"] += row["test_units"]
        item["damage_units"] += row["damage_units"]
        item["selected_units"] += row["selected_units"]
        item["selected_damage_units"] += row["selected_damage_units"]
        item["cost_num"] += row["selected_cost"]
        item["cost_den"] += row["total_cost"]
        item["aos_num"] += row["selected_aos"]
        item["aos_den"] += row["total_aos"]
        item["annual_lift"].append(row["damage_lift"])

    out = []
    for (rule, capacity), item in groups.items():
        hit_rate = item["selected_damage_units"] / item["selected_units"] if item["selected_units"] else 0.0
        overall_rate = item["damage_units"] / item["test_units"] if item["test_units"] else 0.0
        annual = sorted(item["annual_lift"])
        score_key = (rule, capacity)
        scores = score_store.get(score_key, {"y": [], "score": []})
        brier = brier_score_loss(scores["y"], scores["score"]) if scores["y"] else float("nan")
        out.append({
            "rule": rule,
            "capacity": capacity,
            "test_years": len(item["years"]),
            "test_units": item["test_units"],
            "damage_units": item["damage_units"],
            "selected_units": item["selected_units"],
            "selected_damage_units": item["selected_damage_units"],
            "damage_capture": item["selected_damage_units"] / item["damage_units"] if item["damage_units"] else 0.0,
            "selected_hit_rate": hit_rate,
            "damage_lift": hit_rate / overall_rate if overall_rate else 0.0,
            "cost_capture": item["cost_num"] / item["cost_den"] if item["cost_den"] else 0.0,
            "aos_capture": item["aos_num"] / item["aos_den"] if item["aos_den"] else 0.0,
            "annual_lift_p05": float(np.quantile(annual, 0.05)) if annual else 0.0,
            "annual_lift_sd": float(np.std(annual, ddof=1)) if len(annual) > 1 else 0.0,
            "score_brier": brier,
        })
    order = {
        "Posterior mean top-k": 0,
        "Robust q10 top-k": 1,
        "Lower credible-bound top-k": 2,
        "Upper credible-bound top-k": 3,
        "Thompson top-k": 4,
        "Membership top-k": 5,
        "Boundary-product top-k": 6,
        "Boundary-excess top-k": 7,
    }
    return sorted(out, key=lambda row: (row["capacity"], order.get(row["rule"], 99)))


def event_totals(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        key = str(row["event_id"])
        if key not in out:
            out[key] = {"cost": float(row["cost"]), "aos": float(row["aos"])}
    return out


def score_from_cell_metric(test: list[dict], metric: dict[tuple, float]) -> list[tuple[float, dict]]:
    scored = [(float(metric.get(tuple_key(row, FULL_LEVEL), 0.0)), row) for row in test]
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    return scored


def cell_metrics(cells: list[dict], rng: np.random.Generator) -> dict[str, dict[tuple, float]]:
    out = {
        "Posterior mean top-k": {},
        "Robust q10 top-k": {},
        "Lower credible-bound top-k": {},
        "Upper credible-bound top-k": {},
        "Thompson top-k": {},
    }
    for cell in cells:
        key = cell["key"]
        alpha = float(cell["alpha"])
        beta = float(cell["beta"])
        out["Posterior mean top-k"][key] = float(cell["posterior_mean"])
        out["Robust q10 top-k"][key] = float(beta_dist.ppf(0.10, alpha, beta))
        out["Lower credible-bound top-k"][key] = float(beta_dist.ppf(0.05, alpha, beta))
        out["Upper credible-bound top-k"][key] = float(beta_dist.ppf(0.95, alpha, beta))
        out["Thompson top-k"][key] = float(rng.beta(alpha, beta))
    return out


def decision_rule_comparison(parts: list[dict], years: list[int], draws: int, out_dir: Path) -> None:
    yearly: list[dict] = []
    score_store: dict[tuple[str, float], dict[str, list[float]]] = defaultdict(lambda: {"y": [], "score": []})
    for test_year in years:
        train = [row for row in parts if test_year - 5 <= int(row["year"]) <= test_year - 1]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue
        cells, _ = build_test_cells(train, test)
        rng = np.random.default_rng(20260602 + test_year)
        base_metrics = cell_metrics(cells, rng)
        cert_by_capacity = {
            capacity: bayesian_set_cell_scores(cells, capacity, draws, seed=20260602 + test_year * 101 + int(capacity * 10000))
            for capacity in CAPACITIES
        }
        events = event_totals(test)
        total_cost = sum(value["cost"] for value in events.values())
        total_aos = sum(value["aos"] for value in events.values())
        total_damage = sum(int(bool(row[TARGET])) for row in test)
        y_values = [int(bool(row[TARGET])) for row in test]

        for capacity in CAPACITIES:
            metrics_for_capacity = dict(base_metrics)
            values = cert_by_capacity[capacity]
            metrics_for_capacity["Membership top-k"] = {
                key: float(item["capacity_membership_probability"]) for key, item in values.items()
            }
            metrics_for_capacity["Boundary-product top-k"] = {
                key: float(item["membership_x_boundary_excess"]) for key, item in values.items()
            }
            metrics_for_capacity["Boundary-excess top-k"] = {
                key: float(item["capacity_boundary_excess"]) for key, item in values.items()
            }
            for rule, metric in metrics_for_capacity.items():
                scored = score_from_cell_metric(test, metric)
                selected = selected_rows_from_scores(scored, capacity)
                selected_events = {str(row["event_id"]) for row in selected}
                selected_damage = sum(int(bool(row[TARGET])) for row in selected)
                hit_rate = selected_damage / len(selected) if selected else 0.0
                overall_rate = total_damage / len(test) if test else 0.0
                yearly.append({
                    "test_year": test_year,
                    "rule": rule,
                    "capacity": capacity,
                    "test_units": len(test),
                    "damage_units": total_damage,
                    "selected_units": len(selected),
                    "selected_damage_units": selected_damage,
                    "damage_capture": selected_damage / total_damage if total_damage else 0.0,
                    "selected_hit_rate": hit_rate,
                    "damage_lift": hit_rate / overall_rate if overall_rate else 0.0,
                    "selected_cost": sum(events[event_id]["cost"] for event_id in selected_events if event_id in events),
                    "total_cost": total_cost,
                    "selected_aos": sum(events[event_id]["aos"] for event_id in selected_events if event_id in events),
                    "total_aos": total_aos,
                })
                clipped_scores = [min(max(metric.get(tuple_key(row, FULL_LEVEL), 0.0), 1e-6), 1 - 1e-6) for row in test]
                score_store[(rule, capacity)]["y"].extend(y_values)
                score_store[(rule, capacity)]["score"].extend(clipped_scores)

    write_csv(out_dir / "bayesian_decision_rule_yearly.csv", yearly)
    write_csv(out_dir / "bayesian_decision_rule_aggregate.csv", evaluate_scored_rules(yearly, score_store))


def interval_coverage(parts: list[dict], years: list[int], draws: int, out_dir: Path) -> None:
    yearly: list[dict] = []
    rng = np.random.default_rng(20260603)
    for test_year in years:
        train = [row for row in parts if test_year - 5 <= int(row["year"]) <= test_year - 1]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue
        cells, _ = build_test_cells(train, test)
        for cell in cells:
            n = int(cell["count"])
            y = int(cell["future_damage"])
            if n <= 0:
                continue
            theta = rng.beta(float(cell["alpha"]), float(cell["beta"]), size=draws)
            pred = rng.binomial(n, theta) / n
            lower80, upper80 = np.quantile(pred, [0.10, 0.90])
            lower95, upper95 = np.quantile(pred, [0.025, 0.975])
            rate = y / n
            yearly.append({
                "test_year": test_year,
                "support_group": support_group(int(cell["support"])),
                "cell_count": 1,
                "unit_count": n,
                "damage_units": y,
                "observed_rate": rate,
                "posterior_mean": float(cell["posterior_mean"]),
                "cover80": int(lower80 <= rate <= upper80),
                "cover95": int(lower95 <= rate <= upper95),
                "width80": float(upper80 - lower80),
                "width95": float(upper95 - lower95),
            })

    groups: dict[str, dict] = defaultdict(lambda: {
        "years": set(),
        "cells": 0,
        "units": 0,
        "damage_units": 0,
        "cover80": 0,
        "cover95": 0,
        "width80": [],
        "width95": [],
        "observed": [],
        "predicted": [],
    })
    for row in yearly:
        item = groups[row["support_group"]]
        item["years"].add(row["test_year"])
        item["cells"] += 1
        item["units"] += row["unit_count"]
        item["damage_units"] += row["damage_units"]
        item["cover80"] += row["cover80"]
        item["cover95"] += row["cover95"]
        item["width80"].append(row["width80"])
        item["width95"].append(row["width95"])
        item["observed"].append(row["observed_rate"])
        item["predicted"].append(row["posterior_mean"])

    order = {"0--9": 0, "10--49": 1, "50--199": 2, ">=200": 3}
    aggregate = []
    for group, item in groups.items():
        aggregate.append({
            "support_group": group,
            "test_years": len(item["years"]),
            "cells": item["cells"],
            "units": item["units"],
            "damage_rate": item["damage_units"] / item["units"] if item["units"] else 0.0,
            "mean_observed_cell_rate": float(np.mean(item["observed"])) if item["observed"] else 0.0,
            "mean_posterior_cell_rate": float(np.mean(item["predicted"])) if item["predicted"] else 0.0,
            "cover80": item["cover80"] / item["cells"] if item["cells"] else 0.0,
            "cover95": item["cover95"] / item["cells"] if item["cells"] else 0.0,
            "median_width80": float(np.median(item["width80"])) if item["width80"] else 0.0,
            "median_width95": float(np.median(item["width95"])) if item["width95"] else 0.0,
        })
    write_csv(out_dir / "certificate_interval_coverage_yearly.csv", yearly)
    write_csv(out_dir / "certificate_interval_coverage_aggregate.csv", sorted(aggregate, key=lambda row: order[row["support_group"]]))


def bootstrap_stability(parts: list[dict], years: list[int], draws: int, boot_reps: int, out_dir: Path) -> None:
    rows: list[dict] = []
    rng = np.random.default_rng(20260604)
    capacity = 0.05
    for test_year in years:
        train = [row for row in parts if test_year - 5 <= int(row["year"]) <= test_year - 1]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue
        cells, _ = build_test_cells(train, test)
        cert = bayesian_set_cell_scores(cells, capacity, draws, seed=20260604 + test_year)
        original_scored = score_hierarchical_mean(train, test)
        k = max(1, math.ceil(len(test) * capacity))
        original_selected = [row for _, row in original_scored[:k]]
        original_cells = {tuple_key(row, FULL_LEVEL) for row in original_selected}
        group_cells: dict[str, set[tuple]] = defaultdict(set)
        for row in original_selected:
            pi = float(cert.get(tuple_key(row, FULL_LEVEL), {}).get("capacity_membership_probability", 0.0))
            group_cells[pi_bin(pi)].add(tuple_key(row, FULL_LEVEL))

        train_array = np.array(train, dtype=object)
        for rep in range(boot_reps):
            sample_idx = rng.integers(0, len(train_array), size=len(train_array))
            boot_train = [dict(item) for item in train_array[sample_idx]]
            tables, global_stats = train_hierarchical_posteriors(boot_train, TARGET, prior_weight=PRIOR_WEIGHT)
            boot_scored = []
            for row in test:
                _, stats = posterior_for_row(row, tables, global_stats)
                boot_scored.append((float(stats["mean"]), row))
            boot_scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
            boot_cells = {tuple_key(row, FULL_LEVEL) for _, row in boot_scored[:k]}
            union = original_cells | boot_cells
            jaccard = len(original_cells & boot_cells) / len(union) if union else 0.0
            for group in ["[0.95, 1.00]", "[0.50, 0.95)", "[0.10, 0.50)", "[0.01, 0.10)", "[0.00, 0.01)"]:
                cells_in_group = group_cells.get(group, set())
                if not cells_in_group:
                    continue
                rows.append({
                    "test_year": test_year,
                    "replicate": rep + 1,
                    "membership_bin": group,
                    "cells": len(cells_in_group),
                    "cell_reselection_rate": len(cells_in_group & boot_cells) / len(cells_in_group),
                    "selected_cell_jaccard": jaccard,
                })

    groups: dict[str, dict] = defaultdict(lambda: {"rows": 0, "cells": [], "rate": [], "jaccard": [], "years": set()})
    for row in rows:
        item = groups[row["membership_bin"]]
        item["rows"] += 1
        item["years"].add(row["test_year"])
        item["cells"].append(row["cells"])
        item["rate"].append(row["cell_reselection_rate"])
        item["jaccard"].append(row["selected_cell_jaccard"])
    order = {"[0.95, 1.00]": 0, "[0.50, 0.95)": 1, "[0.10, 0.50)": 2, "[0.01, 0.10)": 3, "[0.00, 0.01)": 4}
    aggregate = []
    for group, item in groups.items():
        aggregate.append({
            "membership_bin": group,
            "test_years": len(item["years"]),
            "replicates": item["rows"],
            "median_cells": float(np.median(item["cells"])) if item["cells"] else 0.0,
            "mean_cell_reselection_rate": float(np.mean(item["rate"])) if item["rate"] else 0.0,
            "p05_cell_reselection_rate": float(np.quantile(item["rate"], 0.05)) if item["rate"] else 0.0,
            "mean_selected_cell_jaccard": float(np.mean(item["jaccard"])) if item["jaccard"] else 0.0,
        })
    write_csv(out_dir / "certificate_bootstrap_stability_yearly.csv", rows)
    write_csv(out_dir / "certificate_bootstrap_stability_aggregate.csv", sorted(aggregate, key=lambda row: order[row["membership_bin"]]))


def expansion_validity(parts: list[dict], years: list[int], draws: int, out_dir: Path) -> None:
    rows: list[dict] = []
    base_capacity = 0.05
    expanded_capacities = [0.06, 0.07, 0.10]
    for test_year in years:
        train = [row for row in parts if test_year - 5 <= int(row["year"]) <= test_year - 1]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue
        cells, _ = build_test_cells(train, test)
        cert = bayesian_set_cell_scores(cells, base_capacity, draws, seed=20260605 + test_year)
        scored = score_hierarchical_mean(train, test)
        base_k = max(1, math.ceil(len(test) * base_capacity))
        base_units = {unit_key(row) for _, row in scored[:base_k]}
        unselected = [row for _, row in scored if unit_key(row) not in base_units]
        unit_bins = {unit_key(row): pi_bin(float(cert.get(tuple_key(row, FULL_LEVEL), {}).get("capacity_membership_probability", 0.0))) for row in unselected}
        for expanded_capacity in expanded_capacities:
            expanded_k = max(1, math.ceil(len(test) * expanded_capacity))
            added_units = {unit_key(row) for _, row in scored[base_k:expanded_k]}
            for group in ["[0.10, 0.50)", "[0.01, 0.10)", "[0.00, 0.01)"]:
                group_units = [row for row in unselected if unit_bins[unit_key(row)] == group]
                entered = [row for row in group_units if unit_key(row) in added_units]
                rows.append({
                    "test_year": test_year,
                    "expanded_capacity": expanded_capacity,
                    "membership_bin": group,
                    "candidate_units": len(group_units),
                    "entered_units": len(entered),
                    "entry_rate": len(entered) / len(group_units) if group_units else 0.0,
                    "entered_damage_rate": sum(int(bool(row[TARGET])) for row in entered) / len(entered) if entered else 0.0,
                })
    groups: dict[tuple[float, str], dict] = defaultdict(lambda: {"years": set(), "candidates": 0, "entered": 0, "entered_damage": 0})
    for row in rows:
        key = (row["expanded_capacity"], row["membership_bin"])
        item = groups[key]
        item["years"].add(row["test_year"])
        item["candidates"] += row["candidate_units"]
        item["entered"] += row["entered_units"]
        item["entered_damage"] += int(round(row["entered_damage_rate"] * row["entered_units"]))
    order = {"[0.10, 0.50)": 0, "[0.01, 0.10)": 1, "[0.00, 0.01)": 2}
    aggregate = []
    for (capacity, group), item in groups.items():
        aggregate.append({
            "expanded_capacity": capacity,
            "membership_bin": group,
            "test_years": len(item["years"]),
            "candidate_units": item["candidates"],
            "entered_units": item["entered"],
            "entry_rate": item["entered"] / item["candidates"] if item["candidates"] else 0.0,
            "entered_damage_rate": item["entered_damage"] / item["entered"] if item["entered"] else 0.0,
        })
    write_csv(out_dir / "certificate_expansion_validity_yearly.csv", rows)
    write_csv(out_dir / "certificate_expansion_validity_aggregate.csv", sorted(aggregate, key=lambda row: (row["expanded_capacity"], order[row["membership_bin"]])))


def capacity_net_utility(out_dir: Path) -> None:
    aggregate = pd.read_csv(MAIN_RESULT_DIR / "set_model_selection_aggregate.csv")
    sub = aggregate[(aggregate["rule"] == "hierarchical_eb_execution") & (aggregate["capacity"].isin(NET_CAPACITIES))].copy()
    sub["combined_reliability_utility"] = (
        sub["damage_capture"]
        + sub["hard_consequence_capture"]
        + sub["cost_capture"]
        + sub["aos_capture"]
    ) / 4.0
    utility_specs = [
        ("Damage utility", "damage_capture"),
        ("Operational burden utility", "combined_reliability_utility"),
        ("Cost utility", "cost_capture"),
        ("AOS utility", "aos_capture"),
    ]
    cost_weights = [1.0, 2.0, 3.0, 4.0, 6.0]
    rows = []
    for utility_name, column in utility_specs:
        for cost_weight in cost_weights:
            best_capacity = None
            best_net = -1e9
            for _, row in sub.iterrows():
                capacity = float(row["capacity"])
                utility = float(row[column])
                net = utility - cost_weight * capacity
                rows.append({
                    "utility": utility_name,
                    "review_cost_weight": cost_weight,
                    "capacity": capacity,
                    "gross_utility": utility,
                    "net_utility": net,
                    "is_optimal": 0,
                })
                if net > best_net:
                    best_net = net
                    best_capacity = capacity
            for item in rows:
                if item["utility"] == utility_name and item["review_cost_weight"] == cost_weight and item["capacity"] == best_capacity:
                    item["is_optimal"] = 1
    write_csv(out_dir / "capacity_net_utility.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AMM decision validation diagnostics.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--boot", type=int, default=None)
    args = parser.parse_args()

    years = SMOKE_YEARS if args.mode == "smoke" else FULL_YEARS
    draws = args.draws if args.draws is not None else (250 if args.mode == "smoke" else 500)
    boot_reps = args.boot if args.boot is not None else (20 if args.mode == "smoke" else 80)
    out_dir = RESULT_DIR / args.mode
    parts = load_parts()

    decision_rule_comparison(parts, years, draws, out_dir)
    interval_coverage(parts, years, draws, out_dir)
    bootstrap_stability(parts, years, draws, boot_reps, out_dir)
    expansion_validity(parts, years, draws, out_dir)
    capacity_net_utility(out_dir)
    print(f"Decision validation diagnostics written to {out_dir}.")


if __name__ == "__main__":
    main()
