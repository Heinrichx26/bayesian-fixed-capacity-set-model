from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "amm_set_model"


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str


SCENARIOS = [
    Scenario("parent_child_signal", "Parent-child signal"),
    Scenario("noisy_child_parent_stable", "Noisy child, stable parent"),
    Scenario("rare_high_risk", "Rare high-risk cells"),
    Scenario("annual_drift", "Annual distribution drift"),
    Scenario("burden_damage_divergence", "Burden-damage divergence"),
]


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def allocate_top_counts(scores: np.ndarray, counts: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    selected = np.zeros(len(scores), dtype=float)
    remaining = int(k)
    boundary = float(scores[order[min(max(k, 1), int(counts.sum())) - 1]]) if int(counts.sum()) == len(scores) else 0.0
    for idx in order:
        if remaining <= 0:
            break
        take = min(int(counts[idx]), remaining)
        selected[idx] = take
        remaining -= take
        boundary = float(scores[idx])
    return selected, boundary


def make_scenario_data(rng: np.random.Generator, scenario: Scenario, rep: int) -> pd.DataFrame:
    parents = 24
    children_per_parent = 12
    rows: list[dict] = []
    cell_id = 0

    parent_logits = rng.normal(-2.45, 0.75, size=parents)
    parent_shift = rng.normal(0.0, 0.30, size=parents)
    if scenario.key == "annual_drift":
        drift_parent = rng.choice(parents, size=parents // 3, replace=False)
        parent_shift[drift_parent] += rng.normal(0.55, 0.18, size=len(drift_parent))

    high_risk_cells: set[tuple[int, int]] = set()
    if scenario.key == "rare_high_risk":
        for parent in rng.choice(parents, size=8, replace=False):
            high_risk_cells.add((int(parent), int(rng.integers(0, children_per_parent))))

    for parent in range(parents):
        parent_rate = float(logistic(parent_logits[parent]))
        for child in range(children_per_parent):
            child_noise = rng.normal(0.0, 0.85)
            if scenario.key == "noisy_child_parent_stable":
                child_noise = rng.normal(0.0, 0.25)
            if scenario.key == "rare_high_risk" and (parent, child) in high_risk_cells:
                child_noise += 2.15

            train_logit = parent_logits[parent] + child_noise
            test_logit = train_logit + parent_shift[parent]
            if scenario.key == "burden_damage_divergence":
                test_logit += rng.normal(0.0, 0.20)

            train_rate = float(logistic(train_logit))
            test_rate = float(logistic(test_logit))

            n_train = int(rng.negative_binomial(2, 0.38))
            if scenario.key in {"noisy_child_parent_stable", "rare_high_risk"} and rng.random() < 0.34:
                n_train = int(rng.integers(0, 3))
            if scenario.key == "rare_high_risk" and (parent, child) in high_risk_cells:
                n_train = int(max(1, rng.poisson(3)))

            base_test = 8 + 22 * parent_rate
            if scenario.key == "burden_damage_divergence":
                burden_factor = float(logistic(parent_logits[parent] * 0.35 + rng.normal(0.0, 0.60)))
                base_test = 6 + 12 * parent_rate + 30 * burden_factor
            n_test = int(max(1, rng.poisson(base_test)))

            d_train = int(rng.binomial(n_train, train_rate)) if n_train > 0 else 0
            d_test = int(rng.binomial(n_test, test_rate))
            rows.append(
                {
                    "rep": rep,
                    "scenario": scenario.key,
                    "scenario_label": scenario.label,
                    "cell": cell_id,
                    "parent": parent,
                    "child": child,
                    "train_rate": train_rate,
                    "true_rate": test_rate,
                    "parent_rate": parent_rate,
                    "n_train": n_train,
                    "d_train": d_train,
                    "n_test": n_test,
                    "d_test": d_test,
                }
            )
            cell_id += 1

    df = pd.DataFrame(rows)
    parent = df.groupby("parent", as_index=False).agg(parent_n=("n_train", "sum"), parent_d=("d_train", "sum"))
    parent["parent_post"] = (parent["parent_d"] + 1.0) / (parent["parent_n"] + 2.0)
    df = df.merge(parent[["parent", "parent_post"]], on="parent", how="left")
    return df


def add_scores(df: pd.DataFrame, alpha: float, rng: np.random.Generator, quantile_draws: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = df.copy()
    global_rate = (df["d_train"].sum() + 1.0) / (df["n_train"].sum() + 2.0)
    df["direct_score"] = np.where(df["n_train"] > 0, df["d_train"] / df["n_train"].clip(lower=1), global_rate)
    df["laplace_score"] = (df["d_train"] + 1.0) / (df["n_train"] + 2.0)
    df["parent_score"] = df["parent_post"].fillna(global_rate)
    df["hier_score"] = (df["d_train"] + alpha * df["parent_score"]) / (df["n_train"] + alpha)

    a = df["d_train"].to_numpy(dtype=float) + alpha * df["parent_score"].to_numpy(dtype=float)
    b = (
        df["n_train"].to_numpy(dtype=float)
        - df["d_train"].to_numpy(dtype=float)
        + alpha * (1.0 - df["parent_score"].to_numpy(dtype=float))
    )
    draws = rng.beta(a, b, size=(max(50, quantile_draws), len(df)))
    df["lower_score"] = np.quantile(draws, 0.10, axis=0)
    df["upper_score"] = np.quantile(draws, 0.90, axis=0)
    df["thompson_score"] = draws[0]
    return df, a, b


def certificate(
    rng: np.random.Generator,
    a: np.ndarray,
    b: np.ndarray,
    n_test: np.ndarray,
    k: int,
    draws: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_counts = np.zeros(len(n_test), dtype=float)
    boundary_excess = np.zeros(len(n_test), dtype=float)
    for _ in range(draws):
        theta = rng.beta(a, b)
        selected, boundary = allocate_top_counts(theta, n_test, k)
        selected_counts += selected
        boundary_excess += np.maximum(theta - boundary, 0.0)
    membership = selected_counts / (draws * n_test)
    return membership, boundary_excess / draws


def evaluate_score(df: pd.DataFrame, score_col: str, label: str, k: int, oracle_capture: float) -> dict:
    n_test = df["n_test"].to_numpy(dtype=int)
    d_test = df["d_test"].to_numpy(dtype=float)
    true_damage = df["true_rate"].to_numpy(dtype=float) * n_test
    scores = df[score_col].to_numpy(dtype=float)
    selected, _ = allocate_top_counts(scores, n_test, k)

    captured_damage = float(np.sum(selected * d_test / n_test))
    total_damage = float(d_test.sum())
    selected_expected = float(np.sum(selected * true_damage / n_test))
    total_expected = float(true_damage.sum())
    hit_rate = captured_damage / k if k else 0.0
    base_rate = total_damage / float(n_test.sum()) if n_test.sum() else 0.0
    expected_capture = selected_expected / total_expected if total_expected else 0.0

    return {
        "method": label,
        "capture": captured_damage / total_damage if total_damage else 0.0,
        "hit_rate": hit_rate,
        "lift": hit_rate / base_rate if base_rate else 0.0,
        "brier": float(np.average((scores - df["true_rate"].to_numpy(dtype=float)) ** 2, weights=n_test)),
        "oracle_regret": max(0.0, oracle_capture - expected_capture),
        "selected_cells": int((selected > 0).sum()),
    }


def simulate_once(
    rng: np.random.Generator,
    rep: int,
    scenario: Scenario,
    alpha: float,
    draws: int,
    ref_draws: int,
) -> tuple[list[dict], list[dict]]:
    df = make_scenario_data(rng, scenario, rep)
    df, a, b = add_scores(df, alpha=alpha, rng=rng, quantile_draws=min(draws, 500))
    n_test = df["n_test"].to_numpy(dtype=int)
    k = int(np.ceil(0.05 * int(n_test.sum())))
    oracle_selected, _ = allocate_top_counts(df["true_rate"].to_numpy(dtype=float), n_test, k)
    total_expected = float(np.sum(df["true_rate"].to_numpy(dtype=float) * n_test))
    oracle_capture = float(np.sum(oracle_selected * df["true_rate"].to_numpy(dtype=float)) / total_expected)

    membership, boundary_excess = certificate(rng, a, b, n_test, k, draws=draws)
    ref_membership, _ = certificate(rng, a, b, n_test, k, draws=ref_draws)
    cert_mae = float(np.average(np.abs(membership - ref_membership), weights=n_test))
    df["membership_score"] = membership
    df["boundary_score"] = boundary_excess
    df["product_score"] = membership * boundary_excess

    methods = [
        ("direct_score", "Direct cell"),
        ("laplace_score", "Laplace cell"),
        ("parent_score", "Parent cell"),
        ("hier_score", "Hierarchical shrinkage"),
        ("lower_score", "Lower credible bound"),
        ("upper_score", "Upper credible bound"),
        ("thompson_score", "Thompson top-k"),
        ("membership_score", "Membership top-k"),
        ("product_score", "Boundary-product top-k"),
    ]

    method_rows: list[dict] = []
    for score_col, label in methods:
        row = evaluate_score(df, score_col, label, k, oracle_capture)
        row.update(
            {
                "rep": rep,
                "scenario": scenario.key,
                "scenario_label": scenario.label,
                "alpha": alpha,
                "draws": draws,
                "membership_error": cert_mae if label in {"Hierarchical shrinkage", "Membership top-k", "Boundary-product top-k"} else np.nan,
            }
        )
        method_rows.append(row)

    selected_mean, _ = allocate_top_counts(df["hier_score"].to_numpy(dtype=float), n_test, k)
    selected_df = df.loc[selected_mean > 0].copy()
    selected_df["selected_units"] = selected_mean[selected_mean > 0]
    selected_df["membership"] = membership[selected_mean > 0]
    selected_df["boundary_excess"] = boundary_excess[selected_mean > 0]
    parts = [
        ("High certificate", selected_df[selected_df["membership"] >= 0.95]),
        ("Boundary", selected_df[(selected_df["membership"] >= 0.50) & (selected_df["membership"] < 0.95)]),
        ("Unstable", selected_df[selected_df["membership"] < 0.50]),
    ]
    cert_rows: list[dict] = []
    for group, part in parts:
        if len(part) == 0:
            rate = support = excess = unit_share = np.nan
        else:
            rate = float(np.sum(part["selected_units"] * part["d_test"] / part["n_test"]) / part["selected_units"].sum())
            support = float(part["n_train"].median())
            excess = float(part["boundary_excess"].median())
            unit_share = float(part["selected_units"].sum() / k)
        cert_rows.append(
            {
                "rep": rep,
                "scenario": scenario.key,
                "scenario_label": scenario.label,
                "alpha": alpha,
                "draws": draws,
                "group": group,
                "cells": int(len(part)),
                "unit_share": unit_share,
                "damage_rate": rate,
                "median_support": support,
                "median_boundary_excess": excess,
                "membership_error": cert_mae,
            }
        )
    return method_rows, cert_rows


def summarize_methods(methods: pd.DataFrame) -> pd.DataFrame:
    return (
        methods.groupby("method", as_index=False)
        .agg(
            capture=("capture", "mean"),
            lift=("lift", "mean"),
            brier=("brier", "mean"),
            oracle_regret=("oracle_regret", "mean"),
            membership_error=("membership_error", "mean"),
            lift_sd=("lift", "std"),
            selected_cells=("selected_cells", "mean"),
        )
        .sort_values("lift", ascending=False)
    )


def run(repetitions: int, draws: int, ref_draws: int, mode: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260601 if mode == "full" else 20260602)
    method_rows: list[dict] = []
    cert_rows: list[dict] = []
    for scenario in SCENARIOS:
        for rep in range(repetitions):
            mr, cr = simulate_once(rng, rep, scenario, alpha=10.0, draws=draws, ref_draws=ref_draws)
            method_rows.extend(mr)
            cert_rows.extend(cr)

    prefix = "smoke" if mode == "smoke" else "full"
    methods = pd.DataFrame(method_rows)
    certs = pd.DataFrame(cert_rows)
    methods.to_csv(OUT_DIR / f"{prefix}_synthetic_methods_raw.csv", index=False)
    certs.to_csv(OUT_DIR / f"{prefix}_synthetic_certificates_raw.csv", index=False)
    summarize_methods(methods).to_csv(OUT_DIR / f"{prefix}_synthetic_method_summary.csv", index=False)
    (
        methods.groupby(["scenario", "scenario_label", "method"], as_index=False)
        .agg(
            capture=("capture", "mean"),
            lift=("lift", "mean"),
            brier=("brier", "mean"),
            oracle_regret=("oracle_regret", "mean"),
            membership_error=("membership_error", "mean"),
            lift_sd=("lift", "std"),
        )
        .sort_values(["scenario", "lift"], ascending=[True, False])
        .to_csv(OUT_DIR / f"{prefix}_synthetic_scenario_summary.csv", index=False)
    )
    (
        certs.groupby(["scenario", "scenario_label", "group"], as_index=False)
        .agg(
            cells=("cells", "mean"),
            unit_share=("unit_share", "mean"),
            damage_rate=("damage_rate", "mean"),
            median_support=("median_support", "median"),
            median_boundary_excess=("median_boundary_excess", "median"),
            membership_error=("membership_error", "mean"),
        )
        .sort_values(["scenario", "damage_rate"], ascending=[True, False])
        .to_csv(OUT_DIR / f"{prefix}_synthetic_certificate_summary.csv", index=False)
    )

    sensitivity_rows: list[dict] = []
    sensitivity_reps = 4 if mode == "smoke" else 40
    scenario = SCENARIOS[0]
    for alpha in [2.0, 5.0, 10.0, 20.0, 50.0]:
        for j in range(sensitivity_reps):
            mr, _ = simulate_once(rng, 10_000 + int(alpha) * 100 + j, scenario, alpha=alpha, draws=draws, ref_draws=ref_draws)
            sensitivity_rows.extend([row for row in mr if row["method"] == "Hierarchical shrinkage"])
    alpha_raw = pd.DataFrame(sensitivity_rows)
    alpha_raw.to_csv(OUT_DIR / f"{prefix}_alpha_sensitivity_raw.csv", index=False)
    alpha_raw.groupby("alpha", as_index=False).agg(
        lift=("lift", "mean"),
        capture=("capture", "mean"),
        brier=("brier", "mean"),
        oracle_regret=("oracle_regret", "mean"),
        membership_error=("membership_error", "mean"),
        lift_sd=("lift", "std"),
    ).to_csv(OUT_DIR / f"{prefix}_alpha_sensitivity.csv", index=False)

    draw_rows: list[dict] = []
    draw_reps = 4 if mode == "smoke" else 20
    for draw_count in [100, 200, 500, 1000]:
        local_ref = max(ref_draws, draw_count * 2)
        for j in range(draw_reps):
            mr, cr = simulate_once(rng, 20_000 + draw_count * 100 + j, scenario, alpha=10.0, draws=draw_count, ref_draws=local_ref)
            boundary = pd.DataFrame(cr)
            hier = [r for r in mr if r["method"] == "Hierarchical shrinkage"][0]
            draw_rows.append(
                {
                    "draws": draw_count,
                    "boundary_cells": float(boundary.loc[boundary["group"] == "Boundary", "cells"].iloc[0]),
                    "boundary_damage_rate": float(boundary.loc[boundary["group"] == "Boundary", "damage_rate"].iloc[0]),
                    "hierarchical_lift": float(hier["lift"]),
                    "membership_error": float(hier["membership_error"]),
                }
            )
    draw_raw = pd.DataFrame(draw_rows)
    draw_raw.to_csv(OUT_DIR / f"{prefix}_draw_sensitivity_raw.csv", index=False)
    draw_raw.groupby("draws", as_index=False).agg(
        boundary_cells=("boundary_cells", "mean"),
        boundary_damage_rate=("boundary_damage_rate", "mean"),
        hierarchical_lift=("hierarchical_lift", "mean"),
        membership_error=("membership_error", "mean"),
    ).to_csv(OUT_DIR / f"{prefix}_draw_sensitivity.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--ref-draws", type=int, default=None)
    args = parser.parse_args()
    repetitions = args.repetitions if args.repetitions is not None else (3 if args.mode == "smoke" else 120)
    draws = args.draws if args.draws is not None else (120 if args.mode == "smoke" else 500)
    ref_draws = args.ref_draws if args.ref_draws is not None else (400 if args.mode == "smoke" else 1500)
    run(repetitions=repetitions, draws=draws, ref_draws=ref_draws, mode=args.mode)


if __name__ == "__main__":
    main()
