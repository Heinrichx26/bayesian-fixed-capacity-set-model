from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "amm_set_model"
TABLE_DIR = PROJECT_ROOT / "article" / "tables"
FIG_DIR = PROJECT_ROOT / "article" / "figures"


def fmt(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{100 * float(x):.1f}\\%"


def bold_if(value: str, condition: bool) -> str:
    return f"\\textbf{{{value}}}" if condition else value


def write_synthetic_table() -> None:
    scenario = pd.read_csv(RESULT_DIR / "full_synthetic_scenario_summary.csv")
    alpha = pd.read_csv(RESULT_DIR / "full_alpha_sensitivity.csv")
    rows = []
    order = [
        "Parent-child signal",
        "Noisy child, stable parent",
        "Rare high-risk cells",
        "Annual distribution drift",
        "Burden-damage divergence",
    ]
    for label in order:
        part = scenario.loc[scenario["scenario_label"] == label].set_index("method")
        direct = part.loc["Direct cell"]
        hier = part.loc["Hierarchical shrinkage"]
        rows.append(
            f"{label} & {fmt(direct['lift'])} & "
            f"{bold_if(fmt(hier['lift']), float(hier['lift']) >= float(direct['lift']))} & "
            f"\\textbf{{{fmt(hier['brier'], 3)}}} & {fmt(hier['oracle_regret'], 3)} & "
            f"{fmt(hier['membership_error'], 3)} \\\\"
        )

    alpha_rows = []
    for _, r in alpha.iterrows():
        alpha_rows.append(
            f"$\\alpha={int(float(r['alpha']))}$ & {pct(r['capture'])} & {fmt(r['lift'])} & "
            f"{fmt(r['brier'], 3)} & {fmt(r['oracle_regret'], 3)} & {fmt(r['membership_error'], 3)} \\\\"
        )

    body = r"""\begin{table}[!htbp]
\centering
\caption{Synthetic validation of sparse selected-set inference}
\label{tab:synthetic_validation}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
\multicolumn{6}{@{}l}{Panel A. Controlled sparse-cell scenarios} \\
\midrule
Scenario & Direct lift & Hier. lift & Hier. Brier & Norm. regret & Cert. MAE \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{6}{@{}l}{Panel B. Prior sample-size sensitivity for hierarchical shrinkage} \\
\midrule
Setting & Capture & Lift & Brier & Norm. regret & Cert. MAE \\
\midrule
""" + "\n".join(alpha_rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. Larger capture and lift are better; lower Brier, normalized regret, and certificate mean absolute error (MAE) are better. Normalized regret is the oracle capture-share gap under known simulated rates.
\end{minipage}
\end{table}
"""
    (TABLE_DIR / "table5_synthetic_validation.tex").write_text(body, encoding="utf-8")


def write_supplemental_synthetic_table() -> None:
    methods = pd.read_csv(RESULT_DIR / "full_synthetic_method_summary.csv")
    ordered = [
        "Hierarchical shrinkage",
        "Lower credible bound",
        "Membership top-k",
        "Boundary-product top-k",
        "Upper credible bound",
        "Direct cell",
        "Laplace cell",
        "Parent cell",
        "Thompson top-k",
    ]
    rows = []
    for label in ordered:
        r = methods.loc[methods["method"] == label].iloc[0]
        rows.append(
            f"{label} & {pct(r['capture'])} & {fmt(r['lift'])} & {fmt(r['brier'], 3)} & "
            f"{fmt(r['oracle_regret'], 3)} & {fmt(r['membership_error'], 3)} \\\\"
        )
    body = r"""\begin{table}[!htbp]
\centering
\caption{Synthetic decision-rule comparison}
\label{tab:synthetic_decision_comparison}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
Rule & Capture & Lift & Brier & Norm. regret & Cert. MAE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{0.94\textwidth}
\small Note. The table aggregates the five controlled scenarios used in the reported synthetic validation. Normalized regret is the oracle capture-share gap under known simulated rates. Certificate MAE is reported for rules that use the hierarchical posterior selected-set certificate.
\end{minipage}
\end{table}
"""
    (TABLE_DIR / "table_synthetic_decision_comparison.tex").write_text(body, encoding="utf-8")


def write_synthetic_figure() -> None:
    methods = pd.read_csv(RESULT_DIR / "full_synthetic_method_summary.csv")
    alpha = pd.read_csv(RESULT_DIR / "full_alpha_sensitivity.csv")
    draws = pd.read_csv(RESULT_DIR / "full_draw_sensitivity.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.4, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(8.85, 2.75), dpi=260)

    ordered = [
        "Direct cell",
        "Parent cell",
        "Hierarchical shrinkage",
        "Lower credible bound",
        "Membership top-k",
    ]
    labels = ["Direct\ncell", "Parent\ncell", "Hier.\nEB", "LCB", "Member.\ntop-k"]
    colors = ["#355C7D", "#6C9A8B", "#C04B36", "#7A5195", "#D98E04"]
    m = methods.set_index("method").loc[ordered]
    axes[0].bar(range(len(m)), m["lift"], color=colors)
    axes[0].set_xticks(range(len(m)))
    axes[0].set_xticklabels(labels, rotation=0, fontsize=7.3)
    axes[0].set_ylabel("Lift", fontsize=9.0)
    axes[0].set_ylim(0, max(m["lift"]) * 1.20)
    axes[0].set_title("(a) Decision-rule lift", fontsize=10.0)
    for i, v in enumerate(m["lift"]):
        axes[0].text(
            i,
            v + 0.04,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="#f8fafc", edgecolor="none", alpha=0.84),
        )

    axes[1].plot(alpha["alpha"], alpha["lift"], marker="o", color="#C04B36", linewidth=1.4)
    axes[1].set_xscale("log")
    axes[1].set_xticks([2, 5, 10, 20, 50])
    axes[1].set_xticklabels(["2", "5", "10", "20", "50"])
    axes[1].set_ylim(0, max(alpha["lift"]) * 1.17)
    axes[1].set_xlabel("Prior sample size", fontsize=9.0)
    axes[1].set_title("(b) Shrinkage sensitivity", fontsize=10.0)
    for _, r in alpha.iterrows():
        axes[1].text(
            r["alpha"],
            r["lift"] + 0.04,
            f"{r['lift']:.2f}",
            ha="center",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="#f8fafc", edgecolor="none", alpha=0.82),
        )

    axes[2].plot(draws["draws"], draws["membership_error"], marker="o", color="#355C7D", linewidth=1.4)
    axes[2].set_xscale("log")
    axes[2].set_xticks([100, 200, 500, 1000])
    axes[2].set_xticklabels(["100", "200", "500", "1000"])
    axes[2].set_ylim(0, max(draws["membership_error"]) * 1.25)
    axes[2].set_xlabel("Posterior draws", fontsize=9.0)
    axes[2].set_ylabel("Cert. MAE", fontsize=9.0)
    axes[2].set_title("(c) Certificate stability", fontsize=10.0)
    for _, r in draws.iterrows():
        axes[2].text(
            r["draws"],
            r["membership_error"] + 0.00045,
            f"{r['membership_error']:.3f}",
            ha="center",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="#f8fafc", edgecolor="none", alpha=0.82),
        )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8.0)
        ax.grid(axis="y", color="#d7dde1", lw=0.55, alpha=0.75)
    fig.tight_layout(w_pad=1.35)
    fig.savefig(FIG_DIR / "fig4_amm_synthetic_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_amm_synthetic_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    write_synthetic_table()
    write_supplemental_synthetic_table()
    write_synthetic_figure()


if __name__ == "__main__":
    main()
