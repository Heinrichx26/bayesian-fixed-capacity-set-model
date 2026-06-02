from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "full"
METHOD_RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "method_evidence" / "full"
PARTIAL_2026_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "method_evidence" / "partial_2026"
DECISION_RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "decision_validation" / "full"
TABLE_DIR = PROJECT_ROOT / "results" / "artifacts" / "tables"
FIG_DIR = PROJECT_ROOT / "results" / "artifacts" / "figures"

RULE_LABELS = {
    "catboost": "CatBoost",
    "tabm": "TabM",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "ridge_logistic": "Ridge logistic",
    "random_forest": "Random forest",
    "compact_catboost": "Compact CatBoost",
    "compact_lightgbm": "Compact LightGBM",
    "hierarchical_eb_execution": "Bayesian set model",
    "bayesian_species_child": "Bayesian species-child model",
    "bayesian_posterior_mean": "Bayesian posterior mean",
    "boundary_excess": "Boundary-excess score",
    "membership_x_boundary_excess": "Membership $\\times$ boundary",
    "membership_probability": "Membership-probability score",
    "direct_full_cell": "Direct cell",
    "species_size_component": "Species-size-comp.",
    "component_only": "Component-only",
    "cost_burden_rule": "Cost-weighted rule",
    "aos_burden_rule": "AOS-weighted rule",
    "frequency": "Historical frequency",
    "Posterior mean top-k": "Posterior mean",
    "Robust q10 top-k": "Robust q10",
    "Lower credible-bound top-k": "Lower credible bound",
    "Thompson top-k": "Thompson top-k",
    "Membership top-k": "Membership",
    "Boundary-product top-k": "Membership $\\times$ boundary",
    "Boundary-excess top-k": "Boundary excess",
}

COMPONENT_LABELS = {
    "lights": "Lights",
    "wing_rotor": "Wing/rotor",
    "windshield": "Windshield",
    "engine": "Engine",
    "fuselage": "Fuselage",
    "propeller": "Propeller",
    "nose": "Nose",
    "other": "Other",
    "tail": "Tail",
}


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}\\%"


def num(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def ci_fmt(point: float, lower: float, upper: float, digits: int = 2, percent: bool = False) -> str:
    if percent:
        return f"{100 * point:.1f} [{100 * lower:.1f}, {100 * upper:.1f}]"
    return f"{point:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def bold(value: str, flag: bool) -> str:
    return f"\\textbf{{{value}}}" if flag else value


def escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = pd.read_csv(RESULT_DIR / "set_model_selection_aggregate.csv")
    yearly = pd.read_csv(RESULT_DIR / "set_model_selection_yearly.csv")
    calibration = pd.read_csv(RESULT_DIR / "set_model_probability_calibration.csv")
    certificates = pd.read_csv(RESULT_DIR / "set_model_certificate_cells_2025.csv")
    return aggregate, yearly, calibration, certificates


def write_table(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def table_main_performance(aggregate: pd.DataFrame) -> None:
    ordered_rules = [
        "catboost",
        "tabm",
        "lightgbm",
        "xgboost",
        "ridge_logistic",
        "random_forest",
        "hierarchical_eb_execution",
        "direct_full_cell",
        "species_size_component",
        "component_only",
        "frequency",
    ]
    rows_by_cap: dict[float, list[str]] = {0.05: [], 0.10: []}
    for cap in (0.05, 0.10):
        sub = aggregate[aggregate["capacity"].round(4) == cap].set_index("rule")
        best_lift = sub.loc[ordered_rules, "damage_lift"].max()
        best_cost = sub.loc[ordered_rules, "cost_capture"].max()
        best_aos = sub.loc[ordered_rules, "aos_capture"].max()
        for rule in ordered_rules:
            r = sub.loc[rule]
            captured = f"{num(r['selected_damage_units'])} / {num(r['damage_units'])}"
            rows_by_cap[cap].append(
                " & ".join(
                    [
                        RULE_LABELS[rule],
                        captured,
                        pct(r["damage_capture"]),
                        pct(r["selected_hit_rate"]),
                        bold(fmt(r["damage_lift"]), math.isclose(r["damage_lift"], best_lift)),
                        bold(pct(r["cost_capture"]), math.isclose(r["cost_capture"], best_cost)),
                        bold(pct(r["aos_capture"]), math.isclose(r["aos_capture"], best_aos)),
                    ]
                )
                + r" \\"
            )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Rolling fixed-capacity reliability screening performance}
\label{tab:main_performance}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}
\toprule
\multicolumn{7}{@{}l}{Panel A. 5\% inspection capacity} \\
\midrule
Comparison rule & Captured damage & Damage & Hit rate & Lift & Cost & AOS \\
\midrule
""" + "\n".join(rows_by_cap[0.05]) + r"""
\midrule
\multicolumn{7}{@{}l}{Panel B. 10\% inspection capacity} \\
\midrule
Comparison rule & Captured damage & Damage & Hit rate & Lift & Cost & AOS \\
\midrule
""" + "\n".join(rows_by_cap[0.10]) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Damage denotes damage capture, and AOS denotes aircraft out of service. Larger damage capture, hit rate, lift, cost capture, and AOS capture indicate stronger screening. Bold entries mark the best value within each capacity among the displayed rules.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table2_main_performance.tex", body)


def table_bootstrap_intervals() -> None:
    intervals = pd.read_csv(METHOD_RESULT_DIR / "year_bootstrap_intervals.csv")
    rule_order = ["catboost", "tabm", "hierarchical_eb_execution", "direct_full_cell"]
    labels = {
        "catboost": "CatBoost",
        "tabm": "TabM",
        "hierarchical_eb_execution": "Bayesian set model",
        "direct_full_cell": "Direct cell",
    }
    rows = []
    for capacity in (0.05, 0.10):
        cap_rows = intervals[intervals["capacity"].round(4) == capacity]
        for rule in rule_order:
            sub = cap_rows[cap_rows["rule"] == rule].set_index("metric")
            rows.append(
                " & ".join(
                    [
                        "5\\%" if capacity == 0.05 else "10\\%",
                        labels[rule],
                        ci_fmt(sub.loc["damage_lift", "point"], sub.loc["damage_lift", "lower"], sub.loc["damage_lift", "upper"]),
                        ci_fmt(sub.loc["damage_capture", "point"], sub.loc["damage_capture", "lower"], sub.loc["damage_capture", "upper"], percent=True),
                        ci_fmt(sub.loc["cost_capture", "point"], sub.loc["cost_capture", "lower"], sub.loc["cost_capture", "upper"], percent=True),
                        ci_fmt(sub.loc["aos_capture", "point"], sub.loc["aos_capture", "lower"], sub.loc["aos_capture", "upper"], percent=True),
                    ]
                )
                + r" \\"
            )
    diff_rows = []
    for capacity in (0.05, 0.10):
        for rule, label in [
            ("hierarchical_eb_execution minus direct_full_cell", "Bayesian set model minus direct cell"),
            ("hierarchical_eb_execution minus catboost", "Bayesian set model minus CatBoost"),
        ]:
            r = intervals[
                (intervals["capacity"].round(4) == capacity)
                & (intervals["rule"] == rule)
                & (intervals["metric"] == "paired_annual_lift_difference")
            ].iloc[0]
            diff_rows.append(
                " & ".join(
                    [
                        "5\\%" if capacity == 0.05 else "10\\%",
                        label,
                        ci_fmt(r["point"], r["lower"], r["upper"]),
                    ]
                )
                + r" \\"
            )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Year-level bootstrap uncertainty for rolling screening metrics}
\label{tab:bootstrap_intervals}
\small
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}p{0.055\textwidth}p{0.20\textwidth}rrrr@{}}
\toprule
Cap. & Comparison rule & Lift [95\% CI] & Damage [95\% CI] & Cost [95\% CI] & AOS [95\% CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{tabular}{@{}p{0.06\textwidth}p{0.34\textwidth}r@{}}
\toprule
Cap. & Paired annual comparison & Lift difference [95\% CI] \\
\midrule
""" + "\n".join(diff_rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Intervals are computed by resampling test years with replacement. Positive paired differences favor Bayesian set model. Larger lift, damage capture, cost capture, and AOS capture are better.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table3_bootstrap_intervals.tex", body)


def table_feature_comparison(aggregate: pd.DataFrame) -> None:
    feature_path = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "feature_comparison" / "full" / "feature_comparison_aggregate.csv"
    feature = pd.read_csv(feature_path)
    combined = pd.concat([aggregate, feature], ignore_index=True)
    rules = [
        "catboost",
        "lightgbm",
        "compact_catboost",
        "compact_lightgbm",
        "hierarchical_eb_execution",
        "bayesian_species_child",
    ]
    rows_by_cap: dict[float, list[str]] = {0.05: [], 0.10: []}
    for cap in (0.05, 0.10):
        sub = combined[combined["capacity"].round(4) == cap].drop_duplicates(["rule", "capacity"], keep="last").set_index("rule")
        best_lift = sub.loc[rules, "damage_lift"].max()
        best_cost = sub.loc[rules, "cost_capture"].max()
        best_aos = sub.loc[rules, "aos_capture"].max()
        for rule in rules:
            r = sub.loc[rule]
            feature_scope = "Expanded" if rule in {"catboost", "lightgbm"} else "Compact"
            if rule == "bayesian_species_child":
                feature_scope = "Compact + species"
            rows_by_cap[cap].append(
                " & ".join(
                    [
                        RULE_LABELS[rule],
                        feature_scope,
                        bold(fmt(r["damage_lift"]), math.isclose(r["damage_lift"], best_lift)),
                        pct(r["damage_capture"]),
                        bold(pct(r["cost_capture"]), math.isclose(r["cost_capture"], best_cost)),
                        bold(pct(r["aos_capture"]), math.isclose(r["aos_capture"], best_aos)),
                    ]
                )
                + r" \\"
            )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Feature-aligned fixed-capacity comparison}
\label{tab:feature_comparison}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}
\toprule
\multicolumn{6}{@{}l}{Panel A. 5\% inspection capacity} \\
\midrule
Comparison rule & Feature scope & Lift & Damage & Cost & AOS \\
\midrule
""" + "\n".join(rows_by_cap[0.05]) + r"""
\midrule
\multicolumn{6}{@{}l}{Panel B. 10\% inspection capacity} \\
\midrule
Comparison rule & Feature scope & Lift & Damage & Cost & AOS \\
\midrule
""" + "\n".join(rows_by_cap[0.10]) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Compact feature scope uses component family, phase bucket, animal size, and aircraft mass class. Expanded feature scope adds species identifiers and derived categorical interactions. Larger values are better; bold entries mark the best value within each capacity and metric.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table2_feature_comparison.tex", body)


def table_hierarchy_ablation() -> None:
    ablation = pd.read_csv(METHOD_RESULT_DIR / "hierarchy_ablation_aggregate.csv")
    rules = [
        "Direct full cell",
        "EB component",
        "EB component-phase",
        "EB component-phase-size",
        "Full hierarchy",
        "Full hierarchy with certificates",
    ]
    sub5 = ablation[ablation["capacity"].round(4) == 0.05].set_index("rule")
    sub10 = ablation[ablation["capacity"].round(4) == 0.10].set_index("rule")
    best_5_lift = sub5.loc[rules, "damage_lift"].max()
    best_5_damage = sub5.loc[rules, "damage_capture"].max()
    best_10_lift = sub10.loc[rules, "damage_lift"].max()
    best_10_damage = sub10.loc[rules, "damage_capture"].max()
    best_brier = sub5.loc[rules, "brier_score"].min()
    best_log = sub5.loc[rules, "log_loss"].min()
    screening_rows = []
    diagnostic_rows = []
    for rule in rules:
        sub = ablation[ablation["rule"] == rule].set_index("capacity")
        screening_rows.append(
            " & ".join(
                [
                    rule,
                    bold(fmt(sub.loc[0.05, "damage_lift"]), math.isclose(sub.loc[0.05, "damage_lift"], best_5_lift)),
                    bold(pct(sub.loc[0.05, "damage_capture"]), math.isclose(sub.loc[0.05, "damage_capture"], best_5_damage)),
                    bold(fmt(sub.loc[0.10, "damage_lift"]), math.isclose(sub.loc[0.10, "damage_lift"], best_10_lift)),
                    bold(pct(sub.loc[0.10, "damage_capture"]), math.isclose(sub.loc[0.10, "damage_capture"], best_10_damage)),
                ]
            )
            + r" \\"
        )
        diagnostic_rows.append(
            " & ".join(
                [
                    rule,
                    bold(fmt(sub.loc[0.05, "brier_score"], 3), math.isclose(sub.loc[0.05, "brier_score"], best_brier)),
                    bold(fmt(sub.loc[0.05, "log_loss"], 3), math.isclose(sub.loc[0.05, "log_loss"], best_log)),
                    pct(sub.loc[0.05, "fallback_share"], 1),
                ]
            )
            + r" \\"
        )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Mechanism ablation for hierarchical screening}
\label{tab:hierarchy_ablation}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
Rule component & 5\% lift & 5\% damage & 10\% lift & 10\% damage \\
\midrule
""" + "\n".join(screening_rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
Rule component & Brier & Log loss & Fallback \\
\midrule
""" + "\n".join(diagnostic_rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Larger lift and damage capture are better. Brier score and log loss are all-unit probability scores across rolling test-year component units; lower values indicate better rate estimation. Fallback is the share of selected units scored from a parent level at the 5\% capacity. Certificates are audit outputs and do not change the execution ranking.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table4_hierarchy_ablation.tex", body)


def table_calibration_stability(aggregate: pd.DataFrame, calibration: pd.DataFrame) -> None:
    ablation = pd.read_csv(METHOD_RESULT_DIR / "hierarchy_ablation_aggregate.csv")
    method_cal = {
        "direct_full_cell": ablation[(ablation["rule"] == "Direct full cell") & (ablation["capacity"].round(4) == 0.05)].iloc[0],
        "hierarchical_eb_execution": ablation[(ablation["rule"] == "Full hierarchy") & (ablation["capacity"].round(4) == 0.05)].iloc[0],
    }
    rules = [
        "direct_full_cell",
        "hierarchical_eb_execution",
        "ridge_logistic",
        "random_forest",
        "catboost",
        "tabm",
        "lightgbm",
        "xgboost",
    ]
    cal = calibration.set_index("rule")
    agg5 = aggregate[aggregate["capacity"].round(4) == 0.05].set_index("rule")
    brier_values = []
    log_values = []
    ece_values = []
    for rule in rules:
        source = method_cal.get(rule)
        if source is not None:
            brier_values.append(float(source["brier_score"]))
            log_values.append(float(source["log_loss"]))
            ece_values.append(float(source["expected_calibration_error"]))
        else:
            brier_values.append(float(cal.loc[rule, "brier_score"]))
            log_values.append(float(cal.loc[rule, "log_loss"]))
            ece_values.append(float(cal.loc[rule, "expected_calibration_error"]))
    best_brier = min(brier_values)
    best_log = min(log_values)
    best_ece = min(ece_values)
    best_p5 = agg5.loc[rules, "annual_lift_p05"].max()
    rows = []
    for rule in rules:
        source = method_cal.get(rule)
        if source is not None:
            brier = float(source["brier_score"])
            log = float(source["log_loss"])
            ece = float(source["expected_calibration_error"])
        else:
            c = cal.loc[rule]
            brier = float(c["brier_score"])
            log = float(c["log_loss"])
            ece = float(c["expected_calibration_error"])
        a = agg5.loc[rule]
        rows.append(
            " & ".join(
                [
                    RULE_LABELS[rule],
                    bold(fmt(brier, 3), math.isclose(brier, best_brier)),
                    bold(fmt(log, 3), math.isclose(log, best_log)),
                    bold(fmt(ece, 3), math.isclose(ece, best_ece)),
                    bold(fmt(a["annual_lift_p05"]), math.isclose(a["annual_lift_p05"], best_p5)),
                    fmt(a["annual_lift_min"]),
                    fmt(a["annual_lift_sd"]),
                ]
            )
            + r" \\"
        )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Probability calibration and annual stability}
\label{tab:calibration_stability}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{0.31\textwidth}rrrrrr@{}}
\toprule
Comparison rule & Brier & Log loss & ECE & Lift P5 & Worst lift & Lift SD \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. ECE denotes expected calibration error. Brier score, log loss, and ECE are all-unit probability diagnostics across rolling test-year component units. Lift P5 is the fifth percentile of annual lift across 31 rolling years at the 5\% capacity. Lower Brier score, log loss, and ECE are better; larger Lift P5 and worst lift indicate more stable screening.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table3_calibration_stability.tex", body)


def table_certificate_ablation(aggregate: pd.DataFrame) -> None:
    decision = pd.read_csv(DECISION_RESULT_DIR / "bayesian_decision_rule_aggregate.csv")
    rules = [
        "Posterior mean top-k",
        "Robust q10 top-k",
        "Lower credible-bound top-k",
        "Thompson top-k",
        "Membership top-k",
        "Boundary-product top-k",
        "Boundary-excess top-k",
    ]
    rows_by_cap: dict[float, list[str]] = {0.05: [], 0.10: []}
    for cap in (0.05, 0.10):
        sub = decision[decision["capacity"].round(4) == cap].set_index("rule")
        best_lift = sub.loc[rules, "damage_lift"].max()
        best_cost = sub.loc[rules, "cost_capture"].max()
        best_aos = sub.loc[rules, "aos_capture"].max()
        best_p5 = sub.loc[rules, "annual_lift_p05"].max()
        for rule in rules:
            r = sub.loc[rule]
            rows_by_cap[cap].append(
                " & ".join(
                    [
                        RULE_LABELS[rule],
                        bold(fmt(r["damage_lift"]), math.isclose(r["damage_lift"], best_lift)),
                        bold(pct(r["cost_capture"]), math.isclose(r["cost_capture"], best_cost)),
                        bold(pct(r["aos_capture"]), math.isclose(r["aos_capture"], best_aos)),
                        bold(fmt(r["annual_lift_p05"]), math.isclose(r["annual_lift_p05"], best_p5)),
                    ]
                )
                + r" \\"
            )
    body = r"""\begin{table}[H]
\centering
\caption{Bayesian selected-set decision rules}
\label{tab:certificate_ablation}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
\multicolumn{5}{@{}l}{Panel A. 5\% inspection capacity} \\
\midrule
Decision rule & Lift & Cost & AOS & Lift P5 \\
\midrule
""" + "\n".join(rows_by_cap[0.05]) + r"""
\midrule
\multicolumn{5}{@{}l}{Panel B. 10\% inspection capacity} \\
\midrule
Decision rule & Lift & Cost & AOS & Lift P5 \\
\midrule
""" + "\n".join(rows_by_cap[0.10]) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Posterior mean is the Bayes action for additive utility. Robust q10 and lower credible bound use posterior quantiles. Thompson top-\(k\) uses one seeded posterior draw per test year. Membership and boundary rules use selected-set certificate quantities. Larger values are better; bold entries mark the best value within each capacity and metric.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table4_certificate_ablation.tex", body)


def table_certificate_validation() -> None:
    coverage = pd.read_csv(DECISION_RESULT_DIR / "certificate_interval_coverage_aggregate.csv")
    stability = pd.read_csv(DECISION_RESULT_DIR / "certificate_bootstrap_stability_aggregate.csv")
    expansion = pd.read_csv(DECISION_RESULT_DIR / "certificate_expansion_validity_aggregate.csv")

    coverage_rows = []
    for _, r in coverage.iterrows():
        support_label = r"\(\geq200\)" if str(r["support_group"]) == ">=200" else str(r["support_group"])
        coverage_rows.append(
            " & ".join(
                [
                    support_label,
                    num(r["cells"]),
                    num(r["units"]),
                    pct(r["cover80"]),
                    bold(pct(r["cover95"]), 0.90 <= float(r["cover95"]) <= 0.99),
                    pct(r["median_width95"]),
                ]
            )
            + r" \\"
        )

    def interval_label(value: str) -> str:
        return {
            "[0.95, 1.00]": r"\([0.95,1.00]\)",
            "[0.50, 0.95)": r"\([0.50,0.95)\)",
            "[0.10, 0.50)": r"\([0.10,0.50)\)",
            "[0.01, 0.10)": r"\([0.01,0.10)\)",
            "[0.00, 0.01)": r"\([0.00,0.01)\)",
        }.get(str(value), escape(value))

    stability_rows = []
    keep_bins = ["[0.95, 1.00]", "[0.50, 0.95)", "[0.10, 0.50)"]
    for _, r in stability[stability["membership_bin"].isin(keep_bins)].iterrows():
        stability_rows.append(
            " & ".join(
                [
                    interval_label(r["membership_bin"]),
                    num(r["replicates"]),
                    pct(r["mean_cell_reselection_rate"]),
                    pct(r["p05_cell_reselection_rate"]),
                    pct(r["mean_selected_cell_jaccard"]),
                ]
            )
            + r" \\"
        )

    expansion_rows = []
    pivot = expansion[expansion["membership_bin"].isin(["[0.10, 0.50)", "[0.01, 0.10)", "[0.00, 0.01)"])]
    for bin_name in ["[0.10, 0.50)", "[0.01, 0.10)", "[0.00, 0.01)"]:
        row6 = pivot[(pivot["membership_bin"] == bin_name) & (pivot["expanded_capacity"].round(4) == 0.06)].iloc[0]
        row10 = pivot[(pivot["membership_bin"] == bin_name) & (pivot["expanded_capacity"].round(4) == 0.10)].iloc[0]
        expansion_rows.append(
            " & ".join(
                [
                    interval_label(bin_name),
                    num(row6["candidate_units"]),
                    bold(pct(row6["entry_rate"]), bin_name == "[0.10, 0.50)"),
                    bold(pct(row10["entry_rate"]), bin_name == "[0.10, 0.50)"),
                    pct(row10["entered_damage_rate"]),
                ]
            )
            + r" \\"
        )

    body = r"""\begin{table}[!htbp]
\centering
\caption{Posterior selected-set certificate validation}
\label{tab:certificate_validation}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{0.16\textwidth}rrrrr@{}}
\toprule
\multicolumn{6}{@{}l}{Panel A. Posterior predictive interval coverage by historical support} \\
\midrule
Support & Cells & Units & 80\% cover & 95\% cover & 95\% width \\
\midrule
""" + "\n".join(coverage_rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{tabular}{@{}p{0.18\textwidth}rrrr@{}}
\toprule
\multicolumn{5}{@{}l}{Panel B. Training-window bootstrap stability at 5\% capacity} \\
\midrule
\(\pi_g(5\%)\) bin & Rep. & Reselect & Reselect P5 & Jaccard \\
\midrule
""" + "\n".join(stability_rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{tabular}{@{}p{0.18\textwidth}rrrr@{}}
\toprule
\multicolumn{5}{@{}l}{Panel C. Entry into expanded posterior-mean lists from the 5\% boundary} \\
\midrule
\(\pi_g(5\%)\) bin & Units & Entry at 6\% & Entry at 10\% & Damage at 10\% \\
\midrule
""" + "\n".join(expansion_rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Coverage is evaluated for future cell damage rates. Bootstrap stability resamples the five-year historical window and reports cell re-selection. Entry rates use units outside the 5\% list and measure whether they enter expanded 6\% or 10\% lists. Coverage near the nominal level, higher re-selection, larger Jaccard, and higher boundary-entry rates indicate stronger certificate validity.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table6_certificate_validation.tex", body)


def table_capacity_net_utility() -> None:
    data = pd.read_csv(DECISION_RESULT_DIR / "capacity_net_utility.csv")
    optimal = data[data["is_optimal"] == 1].copy()
    rows = []
    for _, r in optimal.sort_values(["utility", "review_cost_weight"]).iterrows():
        rows.append(
            " & ".join(
                [
                    escape(r["utility"]),
                    fmt(r["review_cost_weight"], 1),
                    pct(r["capacity"]),
                    fmt(r["gross_utility"], 3),
                    fmt(r["net_utility"], 3),
                ]
            )
            + r" \\"
        )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Capacity net-utility optima under review-cost weights}
\label{tab:capacity_net_utility}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}p{0.28\textwidth}rrrr@{}}
\toprule
Utility & Cost weight & Optimal capacity & Gross utility & Net utility \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.92\textwidth}
\small Note. Net utility equals normalized reliability utility minus the review-cost weight multiplied by selected-unit capacity. Larger net utility is better under the stated utility and cost weight.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table_capacity_net_utility.tex", body)


def table_certificate_examples(certificates: pd.DataFrame) -> None:
    rows = []
    for _, r in certificates.iterrows():
        cell = " / ".join(
            [
                COMPONENT_LABELS.get(str(r["component"]), str(r["component"]).title()),
                str(r["phase"]).replace("_", " ").title(),
                str(r["size"]).title(),
                f"mass {int(r['aircraft_mass_class'])}",
            ]
        )
        rows.append(
            " & ".join(
                [
                    escape(r["certificate_group"]),
                    escape(cell),
                    fmt(r["posterior_mean"], 3),
                    fmt(r["capacity_membership_probability"], 3),
                    fmt(r["capacity_boundary_excess"], 3),
                    num(r["historical_support"]),
                    num(r["future_damage"]),
                ]
            )
            + r" \\"
        )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Cell-level list certificates for selected 2025 units}
\label{tab:list_certificate_examples}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{0.13\textwidth}p{0.34\textwidth}rrrrr@{}}
\toprule
Group & Cell & Mean & \(\pi_g\) & \(B_g\) & Support & Damage \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Cells are selected by the 2025 5\% Bayesian set model list and have at least ten historical units. Mean is the posterior damage-rate estimate; \(\pi_g\) is capacity membership probability; \(B_g\) is capacity-boundary excess. Larger certificate values indicate higher posterior support for inclusion in the fixed-capacity list.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table5_certificate_examples.tex", body)


def table_certificate_audit() -> None:
    groups = pd.read_csv(METHOD_RESULT_DIR / "certificate_group_aggregate.csv")
    candidates = pd.read_csv(METHOD_RESULT_DIR / "boundary_candidate_audit_aggregate.csv")
    candidate_yearly = pd.read_csv(METHOD_RESULT_DIR / "boundary_candidate_audit_yearly.csv")
    best_group_damage_rate = groups["damage_rate"].max()
    best_group_cost = groups["cost_capture"].max()
    best_group_aos = groups["aos_capture"].max()
    group_rows_by_cap: dict[float, list[str]] = {0.05: [], 0.10: []}
    for _, r in groups.iterrows():
        cap = 0.05 if round(r["capacity"], 4) == 0.05 else 0.10
        group_rows_by_cap[cap].append(
            " & ".join(
                [
                    str(r["certificate_group"]),
                    num(r["units"]),
                    pct(r["unit_share"]),
                    bold(pct(r["damage_rate"]), math.isclose(r["damage_rate"], best_group_damage_rate)),
                    pct(r["damage_capture"]),
                    bold(pct(r["cost_capture"]), math.isclose(r["cost_capture"], best_group_cost)),
                    bold(pct(r["aos_capture"]), math.isclose(r["aos_capture"], best_group_aos)),
                    fmt(r["median_boundary_excess"], 3),
                ]
            )
            + r" \\"
        )
    candidate_rows = []
    rng = np.random.default_rng(20260529)
    candidate_intervals: dict[str, dict[str, tuple[float, float]]] = {}
    for (rule, share), frame in candidate_yearly.groupby(["candidate_rule", "candidate_share"]):
        years = sorted(frame["test_year"].unique())
        boot_rows = []
        for _ in range(1000):
            sampled_years = rng.choice(years, size=len(years), replace=True)
            sample = pd.concat([frame[frame["test_year"] == year] for year in sampled_years], ignore_index=True)
            boot_rows.append({
                "damage_rate": float(sample["damage_rate"].mean()),
                "damage_lift": float(sample["damage_lift"].mean()),
            })
        boot = pd.DataFrame(boot_rows)
        candidate_intervals[f"{rule}|{share:.4f}"] = {
            "damage_rate": tuple(boot["damage_rate"].quantile([0.025, 0.975])),
            "damage_lift": tuple(boot["damage_lift"].quantile([0.025, 0.975])),
        }
    best_candidate_lift = candidates["mean_damage_lift"].max()
    best_candidate_cost = candidates["mean_cost_capture"].max()
    best_candidate_aos = candidates["mean_aos_capture"].max()
    for _, r in candidates.iterrows():
        rule = str(r["candidate_rule"])
        share = float(r["candidate_share"]) if "candidate_share" in r.index else 0.01
        interval_key = f"{rule}|{share:.4f}"
        dr_low, dr_high = candidate_intervals[interval_key]["damage_rate"]
        lift_low, lift_high = candidate_intervals[interval_key]["damage_lift"]
        damage_text = f"{r['mean_damage_rate'] * 100:.1f} [{dr_low * 100:.1f}, {dr_high * 100:.1f}]"
        lift_text = f"{fmt(r['mean_damage_lift'])} [{fmt(lift_low)}, {fmt(lift_high)}]"
        short_rule = {
            "Boundary-certificate expansion": "Boundary cert.",
            "Boundary-excess expansion": "Boundary excess",
            "Posterior-mean expansion": "Posterior mean",
            "Random unselected expansion": "Random",
        }.get(rule, str(rule).replace(" expansion", ""))
        candidate_rows.append(
            " & ".join(
                [
                    pct(share),
                    short_rule,
                    damage_text,
                    bold(lift_text, math.isclose(r["mean_damage_lift"], best_candidate_lift)),
                    bold(pct(r["mean_cost_capture"]), math.isclose(r["mean_cost_capture"], best_candidate_cost)),
                    bold(pct(r["mean_aos_capture"]), math.isclose(r["mean_aos_capture"], best_candidate_aos)),
                ]
            )
            + r" \\"
        )
    group_body = r"""\begin{table}[!htbp]
\centering
\caption{List-certificate audit value under rolling validation}
\label{tab:certificate_audit}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}
\toprule
\multicolumn{8}{@{}l}{Panel A. 5\% inspection capacity} \\
\midrule
Certificate group & Units & Unit share & Damage rate & Damage & Cost & AOS & Median \(B_g\) \\
\midrule
""" + "\n".join(group_rows_by_cap[0.05]) + r"""
\midrule
\multicolumn{8}{@{}l}{Panel B. 10\% inspection capacity} \\
\midrule
Certificate group & Units & Unit share & Damage rate & Damage & Cost & AOS & Median \(B_g\) \\
\midrule
""" + "\n".join(group_rows_by_cap[0.10]) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Certificate groups partition the Bayesian set model list by capacity membership probability. Larger damage rate, damage capture, cost capture, and AOS capture indicate stronger selected-list value.
\end{minipage}
\end{table}
"""

    candidate_body = r"""\begin{table}[H]
\centering
\caption{Additional boundary-candidate audit under rolling validation}
\label{tab:boundary_candidate_audit}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.10\textwidth}p{0.18\textwidth}p{0.24\textwidth}p{0.22\textwidth}rr@{}}
\toprule
Extra cap. & Candidate rule & Hit rate [95\% CI] & Lift [95\% CI] & Cost & AOS \\
\midrule
""" + "\n".join(candidate_rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Extra capacity is measured relative to the 5\% Bayesian set model list. Boundary-excess candidates are unselected units ranked by \(B_g(0.05)\), matching the fixed-budget expansion utility. Boundary-certificate candidates use smaller \(|\pi_g(0.05)-0.50|\), then larger \(B_g(0.05)\), then larger \(\pi_g(0.05)\). Posterior-mean candidates are the next unselected units by the Bayesian posterior mean. Random unselected candidates report the mean of 200 equal-size without-replacement random samples from unselected units in each test year. Intervals use year-level bootstrap resampling. Larger values indicate stronger candidate review value.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table6_certificate_audit.tex", group_body)
    write_table(TABLE_DIR / "table6_boundary_candidate_audit.tex", candidate_body)


def table_partial_2026_update() -> None:
    data = pd.read_csv(PARTIAL_2026_DIR / "partial_2026_update.csv")
    rows = []
    max_lift = data["damage_lift"].max()
    max_aos = data["aos_capture"].max()
    for _, r in data.sort_values("capacity").iterrows():
        rows.append(
            " & ".join(
                [
                    pct(r["capacity"]),
                    num(r["test_component_units"]),
                    f"{num(r['selected_damage_units'])}/{num(r['damage_units'])}",
                    pct(r["damage_capture"]),
                    bold(fmt(r["damage_lift"]), math.isclose(r["damage_lift"], max_lift)),
                    bold(pct(r["cost_capture"]), math.isclose(r["cost_capture"], 1.0)),
                    bold(pct(r["aos_capture"]), math.isclose(r["aos_capture"], max_aos)),
                    pct(r["high_certificate_share"]),
                    pct(r["boundary_share"]),
                    pct(r["unstable_share"]),
                ]
            )
            + r" \\"
        )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Partial 2026 prospective update}
\label{tab:partial_2026_update}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
Cap. & Units & Damage & D cap. & D lift & Cost & AOS & High cert. & Boundary & Unstable \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. The update trains on 2021--2025 and evaluates the public 2026 records available in the data release. Larger capture and lift values indicate stronger prospective concentration. Certificate shares partition the selected list.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table7_partial_2026_update.tex", body)


def prospective_window_rows() -> pd.DataFrame:
    yearly = pd.read_csv(RESULT_DIR / "set_model_selection_yearly.csv")
    groups = pd.read_csv(METHOD_RESULT_DIR / "certificate_group_yearly.csv")
    partial = pd.read_csv(PARTIAL_2026_DIR / "partial_2026_update.csv")
    rule = "hierarchical_eb_execution"
    rows = []
    for year in range(2021, 2026):
        sub = yearly[(yearly["test_year"] == year) & (yearly["rule"] == rule)].set_index("capacity")
        if 0.05 not in sub.index or 0.10 not in sub.index:
            continue
        group_sub = groups[(groups["test_year"] == year) & (groups["capacity"].round(4) == 0.05)]
        group_units = {
            item["certificate_group"]: float(item["units"])
            for _, item in group_sub.iterrows()
        }
        selected_units = float(sub.loc[0.05, "selected_units"])
        rows.append({
            "test_window": str(year),
            "train_start": int(sub.loc[0.05, "train_start"]),
            "train_end": int(sub.loc[0.05, "train_end"]),
            "test_component_units": int(sub.loc[0.05, "test_component_units"]),
            "damage_units": int(sub.loc[0.05, "damage_units"]),
            "damage_capture_5": float(sub.loc[0.05, "damage_capture"]),
            "damage_lift_5": float(sub.loc[0.05, "damage_lift"]),
            "damage_capture_10": float(sub.loc[0.10, "damage_capture"]),
            "damage_lift_10": float(sub.loc[0.10, "damage_lift"]),
            "cost_capture_5": float(sub.loc[0.05, "cost_capture"]),
            "aos_capture_5": float(sub.loc[0.05, "aos_capture"]),
            "high_certificate_share_5": group_units.get("High certificate", 0.0) / selected_units if selected_units else 0.0,
            "boundary_share_5": group_units.get("Boundary", 0.0) / selected_units if selected_units else 0.0,
            "unstable_share_5": group_units.get("Unstable", 0.0) / selected_units if selected_units else 0.0,
            "partial_update": 0,
        })
    part5 = partial[partial["capacity"].round(4) == 0.05].iloc[0]
    part10 = partial[partial["capacity"].round(4) == 0.10].iloc[0]
    rows.append({
        "test_window": "2026 partial",
        "train_start": int(part5["train_start"]),
        "train_end": int(part5["train_end"]),
        "test_component_units": int(part5["test_component_units"]),
        "damage_units": int(part5["damage_units"]),
        "damage_capture_5": float(part5["damage_capture"]),
        "damage_lift_5": float(part5["damage_lift"]),
        "damage_capture_10": float(part10["damage_capture"]),
        "damage_lift_10": float(part10["damage_lift"]),
        "cost_capture_5": float(part5["cost_capture"]),
        "aos_capture_5": float(part5["aos_capture"]),
        "high_certificate_share_5": float(part5["high_certificate_share"]),
        "boundary_share_5": float(part5["boundary_share"]),
        "unstable_share_5": float(part5["unstable_share"]),
        "partial_update": 1,
    })
    out = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "prospective_windows"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "prospective_window_validation.csv", index=False)
    return out


def table_prospective_window_validation() -> None:
    data = prospective_window_rows()
    full_year = data[data["partial_update"] == 0]
    max_lift_5 = full_year["damage_lift_5"].max()
    max_lift_10 = full_year["damage_lift_10"].max()
    max_cost = full_year["cost_capture_5"].max()
    max_aos = full_year["aos_capture_5"].max()
    performance_rows = []
    certificate_rows = []
    for _, r in full_year.iterrows():
        performance_rows.append(
            " & ".join(
                [
                    str(r["test_window"]),
                    num(r["test_component_units"]),
                    num(r["damage_units"]),
                    pct(r["damage_capture_5"]),
                    bold(fmt(r["damage_lift_5"]), math.isclose(r["damage_lift_5"], max_lift_5)),
                    pct(r["damage_capture_10"]),
                    bold(fmt(r["damage_lift_10"]), math.isclose(r["damage_lift_10"], max_lift_10)),
                ]
            )
            + r" \\"
        )
        certificate_rows.append(
            " & ".join(
                [
                    str(r["test_window"]),
                    bold(pct(r["cost_capture_5"]), math.isclose(r["cost_capture_5"], max_cost)),
                    bold(pct(r["aos_capture_5"]), math.isclose(r["aos_capture_5"], max_aos)),
                    pct(r["high_certificate_share_5"]),
                    pct(r["boundary_share_5"]),
                ]
            )
            + r" \\"
        )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Late-period prospective-window validation}
\label{tab:prospective_window_validation}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Test window & Units & Damage & 5\% D cap. & 5\% lift & 10\% D cap. & 10\% lift \\
\midrule
""" + "\n".join(performance_rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{tabular}{@{}lrrrr@{}}
\toprule
Test window & 5\% cost & 5\% AOS & High cert. & Boundary \\
\midrule
""" + "\n".join(certificate_rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Each row uses a five-year training window ending in the year before the test window. D cap. denotes damage capture. Larger damage capture, lift, cost capture, AOS capture, and high-certificate share are better; boundary share indicates selected units near the capacity edge.
\end{minipage}
\end{table}
"""
    write_table(TABLE_DIR / "table8_prospective_window_validation.tex", body)


def figure_prospective_window_lift() -> None:
    data = prospective_window_rows()
    hist = data[data["partial_update"] == 0].copy()
    partial = data[data["partial_update"] == 1].iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=240)
    ax.plot(hist["test_window"], hist["damage_lift_5"], marker="o", lw=1.5, color="#1b6f6a", label="5% lift, 2021-2025")
    ax.plot(hist["test_window"], hist["damage_lift_10"], marker="s", lw=1.5, color="#2455c3", label="10% lift, 2021-2025")
    ax.scatter(["2026"], [partial["damage_lift_5"]], marker="o", s=52, color="#1b6f6a", edgecolor="#222", linewidth=0.5, label="5% partial 2026")
    ax.scatter(["2026"], [partial["damage_lift_10"]], marker="s", s=52, color="#2455c3", edgecolor="#222", linewidth=0.5, label="10% partial 2026")
    for x_value, y_value in zip(hist["test_window"], hist["damage_lift_5"]):
        ax.text(x_value, y_value + 0.08, f"{y_value:.2f}", ha="center", va="bottom", fontsize=6.2)
    for x_value, y_value in zip(hist["test_window"], hist["damage_lift_10"]):
        ax.text(x_value, y_value - 0.13, f"{y_value:.2f}", ha="center", va="top", fontsize=6.2)
    ax.text("2026", partial["damage_lift_5"] + 0.08, f"{partial['damage_lift_5']:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax.text("2026", partial["damage_lift_10"] - 0.13, f"{partial['damage_lift_10']:.2f}", ha="center", va="top", fontsize=6.2)
    ax.axhline(1, color="#444", lw=0.8)
    ax.set_ylabel("Damage lift", fontsize=9.2)
    ax.set_xlabel("Prospective test window", fontsize=9.2)
    ax.set_ylim(0.75, max(float(data["damage_lift_5"].max()), float(data["damage_lift_10"].max())) + 0.55)
    ax.tick_params(axis="both", labelsize=8.4)
    ax.grid(axis="y", color="#d7dde1", lw=0.6)
    fig.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.94, bottom=0.27)
    fig.savefig(FIG_DIR / "fig9_prospective_window_lift.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def figure_capacity_frontier(aggregate: pd.DataFrame) -> None:
    def label_points(ax, x_values, y_values, color, offset):
        for x_val, y_val in zip(x_values, y_values):
            ax.annotate(
                f"{float(y_val):.1f}",
                (float(x_val), float(y_val)),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=5.8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#f7f7f2", edgecolor="none", alpha=0.82),
            )

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=260)
    for rule, color, marker in [
        ("hierarchical_eb_execution", "#005a54", "o"),
        ("catboost", "#4c1d95", "s"),
        ("frequency", "#4b5563", "^"),
    ]:
        sub = aggregate[aggregate["rule"] == rule].sort_values("capacity")
        x = sub["capacity"] * 100
        y = sub["damage_capture"] * 100
        axes[0].plot(x, y, marker=marker, color=color, label=RULE_LABELS[rule], lw=2.0, ms=4.5)
        label_offset = {
            "hierarchical_eb_execution": (0, -9),
            "catboost": (0, 8),
            "frequency": (0, 8),
        }[rule]
        label_points(axes[0], x, y, color, label_offset)
    axes[0].set_xlabel("Inspection capacity (%)", fontsize=9.4)
    axes[0].set_ylabel("Damage capture (%)", fontsize=9.4)
    axes[0].set_xlim(-0.6, 20.9)
    axes[0].set_ylim(0, 92)
    axes[0].grid(axis="y", color="#d7dde1", lw=0.6)
    axes[0].set_title("(a) Component damage", fontsize=10.2)
    axes[0].tick_params(axis="both", labelsize=8.6)

    sub = aggregate[aggregate["rule"] == "hierarchical_eb_execution"].sort_values("capacity")
    x = sub["capacity"] * 100
    cost_y = sub["cost_capture"] * 100
    aos_y = sub["aos_capture"] * 100
    axes[1].plot(x, cost_y, marker="o", color="#7a3500", label="Cost", lw=2.0, ms=4.5)
    axes[1].plot(x, aos_y, marker="s", color="#17365d", label="AOS", lw=2.0, ms=4.5)
    label_points(axes[1], x, cost_y, "#7a3500", (0, 8))
    label_points(axes[1], x, aos_y, "#17365d", (0, -9))
    axes[1].set_xlabel("Inspection capacity (%)", fontsize=9.4)
    axes[1].set_ylabel("Burden capture (%)", fontsize=9.4)
    axes[1].set_xlim(-0.6, 20.9)
    axes[1].set_ylim(0, 103)
    axes[1].grid(axis="y", color="#d7dde1", lw=0.6)
    axes[1].set_title("(b) Operational burden", fontsize=10.2)
    axes[1].tick_params(axis="both", labelsize=8.6)
    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, frameon=False, fontsize=7.8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.29, wspace=0.26)
    fig.savefig(FIG_DIR / "fig3_capacity_frontier_amm.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def figure_marginal_review_value() -> None:
    data = pd.read_csv(METHOD_RESULT_DIR / "boundary_candidate_audit_aggregate.csv")
    order = [
        ("Boundary-excess expansion", "Boundary excess", "#005a54", "o"),
        ("Boundary-certificate expansion", "Boundary certificate", "#a15c00", "s"),
        ("Posterior-mean expansion", "Posterior mean", "#2455c3", "^"),
        ("Random unselected expansion", "Random", "#6f7782", "D"),
    ]
    metrics = [
        ("mean_damage_rate", "Damage rate (%)", "(a) Damage"),
        ("mean_cost_capture", "Cost capture (%)", "(b) Cost"),
        ("mean_aos_capture", "AOS capture (%)", "(c) AOS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.55, 3.15), dpi=260)
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for rule, label, color, marker in order:
            sub = data[data["candidate_rule"] == rule].sort_values("candidate_share")
            x = sub["candidate_share"] * 100
            y = sub[metric] * 100
            ax.plot(x, y, marker=marker, color=color, lw=1.75, ms=4.6, label=label)
        ax.set_xlabel("Extra capacity (%)", fontsize=9.0)
        ax.set_ylabel(ylabel, fontsize=9.0)
        ax.set_xlim(0.25, 5.18)
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ymax = max(5.0, float(data[metric].max() * 100) * 1.16)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#d7dde1", lw=0.6)
        ax.set_title(title, fontsize=10.0)
        ax.tick_params(axis="both", labelsize=8.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.0, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.31, wspace=0.31)
    fig.savefig(FIG_DIR / "fig8_marginal_review_value_amm.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def figure_ml_calibration(aggregate: pd.DataFrame, calibration: pd.DataFrame) -> None:
    rules = ["catboost", "tabm", "lightgbm", "xgboost", "ridge_logistic", "random_forest", "hierarchical_eb_execution"]
    agg5 = aggregate[aggregate["capacity"].round(4) == 0.05].set_index("rule").loc[rules]
    cal = calibration.set_index("rule").loc[rules].copy()
    ablation = pd.read_csv(METHOD_RESULT_DIR / "hierarchy_ablation_aggregate.csv")
    bayesian_cal = ablation[(ablation["rule"] == "Full hierarchy") & (ablation["capacity"].round(4) == 0.05)].iloc[0]
    cal.loc["hierarchical_eb_execution", "brier_score"] = bayesian_cal["brier_score"]
    cal.loc["hierarchical_eb_execution", "expected_calibration_error"] = bayesian_cal["expected_calibration_error"]
    short_labels = {
        "catboost": "CatBoost",
        "tabm": "TabM",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "ridge_logistic": "Ridge",
        "random_forest": "RF",
        "hierarchical_eb_execution": "Bayes set",
    }
    legend_labels = {
        "catboost": "CatBoost",
        "tabm": "TabM",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "ridge_logistic": "Ridge",
        "random_forest": "RF",
        "hierarchical_eb_execution": "Bayes set",
    }
    labels = [short_labels[r] for r in rules]
    audit = ablation[
        (ablation["rule"].isin(["Direct full cell", "Full hierarchy"]))
        & (ablation["capacity"].isin([0.05, 0.10]))
    ].copy()
    audit["coverage"] = 100 * (1 - audit["fallback_share"])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25), dpi=260)
    lift_bars = axes[0].bar(range(len(rules)), agg5["damage_lift"], color=["#8a3ffc", "#b0477d", "#2455c3", "#4c6f91", "#287a4f", "#6f7782", "#1b6f6a"])
    for bar in lift_bars:
        value = float(bar.get_height())
        label = axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.07,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#222",
            rotation=0,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="#f8fafc", edgecolor="none", alpha=0.86),
        )
        label.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white")])
    axes[0].axhline(1, color="#444", lw=0.8)
    axes[0].set_xticks(range(len(rules)), labels, rotation=22, ha="right", fontsize=8.3)
    axes[0].tick_params(axis="y", labelsize=8.6)
    axes[0].set_ylabel("5% lift", fontsize=9.4)
    axes[0].set_ylim(0, 7.75)
    axes[0].grid(axis="y", color="#d7dde1", lw=0.6)
    axes[0].set_title("(a) Screening concentration", fontsize=10.2)
    colors = {
        "catboost": "#8a3ffc",
        "tabm": "#b0477d",
        "lightgbm": "#2455c3",
        "xgboost": "#4c6f91",
        "ridge_logistic": "#287a4f",
        "random_forest": "#6f7782",
        "hierarchical_eb_execution": "#1b6f6a",
    }
    for rule in rules:
        axes[1].scatter(
            cal.loc[rule, "brier_score"],
            cal.loc[rule, "expected_calibration_error"],
            s=60,
            color=colors[rule],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    label_offsets = {
        "catboost": (-4, 6),
        "tabm": (7, 1),
        "lightgbm": (4, -6),
        "xgboost": (4, 6),
        "ridge_logistic": (-7, -6),
        "random_forest": (6, 0),
        "hierarchical_eb_execution": (6, 5),
    }
    for rule in rules:
        axes[1].annotate(
            short_labels[rule],
            (cal.loc[rule, "brier_score"], cal.loc[rule, "expected_calibration_error"]),
            xytext=label_offsets[rule],
            textcoords="offset points",
            ha="left" if label_offsets[rule][0] >= 0 else "right",
            va="center",
            fontsize=7.1,
            color=colors[rule],
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.72),
        )
    axes[1].set_xlabel("Brier score", fontsize=9.2)
    axes[1].set_ylabel("Expected calibration error", fontsize=9.2)
    axes[1].tick_params(axis="both", labelsize=8.4)
    x_values = cal.loc[rules, "brier_score"].astype(float)
    y_values = cal.loc[rules, "expected_calibration_error"].astype(float)
    x_pad = max(0.012, float(x_values.max() - x_values.min()) * 0.14)
    y_pad = max(0.035, float(y_values.max() - y_values.min()) * 0.12)
    axes[1].set_xlim(max(0.0, float(x_values.min()) - x_pad), float(x_values.max()) + x_pad)
    axes[1].set_ylim(max(0.0, float(y_values.min()) - y_pad), float(y_values.max()) + y_pad)
    axes[1].grid(color="#d7dde1", lw=0.6)
    axes[1].set_title("(b) Probability diagnostics", fontsize=10.2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.24, wspace=0.30)
    fig.savefig(FIG_DIR / "fig5_ml_calibration_amm.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.9, 2.85), dpi=260)
    x_pos = np.arange(2)
    width = 0.34
    direct = audit[audit["rule"] == "Direct full cell"].sort_values("capacity")
    full = audit[audit["rule"] == "Full hierarchy"].sort_values("capacity")
    direct_bars = ax.bar(x_pos - width / 2, direct["coverage"], width=width, color="#6f7782", label="Direct cell")
    full_bars = ax.bar(x_pos + width / 2, full["coverage"], width=width, color="#1b6f6a", label="Bayesian set model")
    for bars in (direct_bars, full_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.16,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                color="#222",
            )
    ax.set_xticks(x_pos, ["5%", "10%"], fontsize=9)
    ax.tick_params(axis="y", labelsize=8.8)
    ax.set_ylabel("Full-cell support (%)", fontsize=9.5)
    ax.set_ylim(94.8, 101.1)
    ax.grid(axis="y", color="#d7dde1", lw=0.6)
    fig.legend(frameon=False, fontsize=8.4, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.94, bottom=0.25)
    fig.savefig(FIG_DIR / "fig5_support_coverage_amm.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def figure_certificates(certificates: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=220)
    for ax, group in zip(axes, ["High-certainty", "Boundary"]):
        sub = certificates[certificates["certificate_group"] == group].copy()
        sub["label"] = sub["component"].map(COMPONENT_LABELS).fillna(sub["component"])
        y = list(range(len(sub)))
        bars = ax.barh(y, sub["posterior_mean"], color="#1b6f6a", alpha=0.85, label="Mean")
        ax.scatter(sub["capacity_membership_probability"], y, color="#a15c00", s=25, label=r"$\pi_g$")
        for bar, mean_value, membership_value in zip(bars, sub["posterior_mean"], sub["capacity_membership_probability"]):
            y_mid = bar.get_y() + bar.get_height() / 2
            mean_x = float(mean_value)
            mean_label_x = max(0.06, mean_x - 0.055) if mean_x > 0.16 else mean_x + 0.025
            ax.text(
                mean_label_x,
                y_mid,
                f"{float(mean_value):.2f}",
                ha="right" if mean_x > 0.16 else "left",
                va="center",
                fontsize=6.6,
                color="white" if mean_x > 0.16 else "#1b6f6a",
            )
            ax.text(
                min(float(membership_value) + 0.025, 1.13),
                y_mid,
                f"{float(membership_value):.2f}",
                ha="left",
                va="center",
                fontsize=6.6,
                color="#a15c00",
            )
        ax.set_yticks(y, sub["label"], fontsize=7)
        ax.set_xlim(0, 1.16)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#d7dde1", lw=0.6)
        ax.set_title("(a) High-certainty cells" if group == "High-certainty" else "(b) Boundary cells")
        ax.set_xlabel("Posterior quantity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.22, wspace=0.34)
    fig.savefig(FIG_DIR / "fig6_certificate_examples_amm.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def figure_stability(yearly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.25), dpi=260)
    endpoint_label_y = {
        "catboost": 9.82,
        "lightgbm": 9.20,
        "tabm": 8.58,
        "hierarchical_eb_execution": 7.96,
        "species_size_component": 7.32,
        "frequency": 0.45,
    }
    for rule, color in [
        ("catboost", "#8a3ffc"),
        ("tabm", "#b0477d"),
        ("lightgbm", "#2455c3"),
        ("hierarchical_eb_execution", "#1b6f6a"),
        ("species_size_component", "#994f00"),
        ("frequency", "#8a8f98"),
    ]:
        sub = yearly[(yearly["rule"] == rule) & (yearly["capacity"].round(4) == 0.05)].sort_values("test_year")
        ax.plot(sub["test_year"], sub["damage_lift"], color=color, lw=1.7, label=RULE_LABELS[rule])
        last = sub.iloc[-1]
        label_x = float(last["test_year"]) + 0.55
        label_y = endpoint_label_y[rule]
        ax.plot(
            [float(last["test_year"]), label_x - 0.10],
            [float(last["damage_lift"]), label_y],
            color=color,
            lw=0.55,
            alpha=0.75,
        )
        ax.text(
            label_x,
            label_y,
            f"{float(last['damage_lift']):.1f}",
            color=color,
            fontsize=7.4,
            va="center",
            ha="left",
        )
    ax.axhline(1, color="#444", lw=0.8)
    ax.set_xlabel("Test year")
    ax.set_ylabel("5% annual lift")
    ax.set_xlim(1994.5, 2026.65)
    ax.set_ylim(0, 10.30)
    ax.grid(axis="y", color="#d7dde1", lw=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.4, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.31)
    fig.savefig(FIG_DIR / "fig7_annual_stability_amm.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    aggregate, yearly, calibration, certificates = load()
    table_main_performance(aggregate)
    table_bootstrap_intervals()
    table_feature_comparison(aggregate)
    table_hierarchy_ablation()
    table_calibration_stability(aggregate, calibration)
    table_certificate_ablation(aggregate)
    table_certificate_validation()
    table_certificate_examples(certificates)
    table_certificate_audit()
    table_capacity_net_utility()
    table_partial_2026_update()
    table_prospective_window_validation()
    figure_capacity_frontier(aggregate)
    figure_marginal_review_value()
    figure_prospective_window_lift()
    figure_ml_calibration(aggregate, calibration)
    figure_certificates(certificates)
    figure_stability(yearly)
    print("AMM tables and data figures built.")


if __name__ == "__main__":
    main()


