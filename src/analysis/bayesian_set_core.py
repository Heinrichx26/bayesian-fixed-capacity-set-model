from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from screening_comparison_rules import (  # noqa: E402
    SCORE_SPECS,
    key_for,
    score_records,
    train_scores,
)
from smoke_faa_wildlife import enrich, load_rows  # noqa: E402
from wildlife_component_data import component_rows  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "smoke"

TARGET = "part_damage"
TRAIN_START = 2000
TRAIN_END = 2005
TEST_YEAR = 2006
POSTERIOR_DRAWS = 200
PRIOR_WEIGHT = 10.0

MAIN_SCORE = "component_phase_size_mass_rate"
HIERARCHY_LEVELS = [
    ["component"],
    ["component", "phase_bucket"],
    ["component", "phase_bucket", "size"],
    ["component", "phase_bucket", "size", "aircraft_mass_class"],
]
FULL_LEVEL = HIERARCHY_LEVELS[-1]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_parts() -> list[dict]:
    events = [enrich(row) for row in load_rows()]
    return [row for row in component_rows(events) if 1990 <= int(row["year"]) <= 2025]


def tuple_key(row: dict, level: Iterable[str]) -> tuple:
    return tuple(row.get(column, "") for column in level)


def train_hierarchical_posteriors(
    train: list[dict],
    target: str,
    prior_weight: float = PRIOR_WEIGHT,
) -> tuple[dict[tuple, dict[tuple, dict]], dict]:
    global_n = len(train)
    global_y = sum(int(bool(row[target])) for row in train)
    global_alpha = global_y + 1.0
    global_beta = global_n - global_y + 1.0
    global_mean = global_alpha / (global_alpha + global_beta) if global_n else 0.0
    global_stats = {
        "alpha": global_alpha,
        "beta": global_beta,
        "mean": global_mean,
        "n": global_n,
        "y": global_y,
        "level": "global",
    }

    tables: dict[tuple, dict[tuple, dict]] = {}
    parent_means: dict[tuple, float] = {(): global_mean}
    for level in HIERARCHY_LEVELS:
        stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "y": 0})
        for row in train:
            key = tuple_key(row, level)
            stats[key]["n"] += 1
            stats[key]["y"] += int(bool(row[target]))

        level_table: dict[tuple, dict] = {}
        for key, item in stats.items():
            parent_key = key[:-1] if len(key) > 1 else ()
            parent_mean = parent_means.get(parent_key, global_mean)
            alpha = item["y"] + prior_weight * parent_mean
            beta = item["n"] - item["y"] + prior_weight * (1.0 - parent_mean)
            level_table[key] = {
                "alpha": alpha,
                "beta": beta,
                "mean": alpha / (alpha + beta),
                "n": item["n"],
                "y": item["y"],
                "level": "|".join(level),
            }
        tables[tuple(level)] = level_table
        parent_means = {key: value["mean"] for key, value in level_table.items()}

    return tables, global_stats


def posterior_for_row(row: dict, tables: dict[tuple, dict[tuple, dict]], global_stats: dict) -> tuple[tuple, dict]:
    full_key = tuple_key(row, FULL_LEVEL)
    for level in reversed(HIERARCHY_LEVELS):
        key = tuple_key(row, level)
        stats = tables.get(tuple(level), {}).get(key)
        if stats is not None:
            return full_key, stats
    return full_key, global_stats


def score_hierarchical_mean(train: list[dict], test: list[dict]) -> list[tuple[float, dict]]:
    tables, global_stats = train_hierarchical_posteriors(train, TARGET)
    scored = []
    for row in test:
        _, stats = posterior_for_row(row, tables, global_stats)
        scored.append((float(stats["mean"]), row))
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    return scored


def event_totals(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        event_id = str(row["event_id"])
        if event_id not in out:
            out[event_id] = {"cost": float(row["cost"]), "aos": float(row["aos"])}
    return out


def evaluate_scored(scored: list[tuple[float, dict]], budget: float, score_name: str) -> dict:
    k = max(1, math.ceil(len(scored) * budget))
    selected = [row for _, row in scored[:k]]
    test_rows = [row for _, row in scored]
    target_records = sum(int(bool(row[TARGET])) for row in test_rows)
    captured = sum(int(bool(row[TARGET])) for row in selected)
    overall_rate = target_records / len(test_rows) if test_rows else 0.0
    hit_rate = captured / k if k else 0.0

    all_events = event_totals(test_rows)
    selected_event_ids = {str(row["event_id"]) for row in selected}
    total_cost = sum(value["cost"] for value in all_events.values())
    total_aos = sum(value["aos"] for value in all_events.values())
    selected_cost = sum(all_events[event_id]["cost"] for event_id in selected_event_ids if event_id in all_events)
    selected_aos = sum(all_events[event_id]["aos"] for event_id in selected_event_ids if event_id in all_events)

    return {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_year": TEST_YEAR,
        "score": score_name,
        "budget_share": budget,
        "test_component_records": len(test_rows),
        "target_records": target_records,
        "selected_component_records": k,
        "captured_target_records": captured,
        "capture_rate": captured / target_records if target_records else 0.0,
        "selected_target_rate": hit_rate,
        "overall_target_rate": overall_rate,
        "lift": hit_rate / overall_rate if overall_rate else 0.0,
        "event_deduplicated_cost_capture": selected_cost / total_cost if total_cost else 0.0,
        "event_deduplicated_aos_capture": selected_aos / total_aos if total_aos else 0.0,
    }


def build_test_cells(train: list[dict], test: list[dict]) -> tuple[list[dict], dict[tuple, int]]:
    tables, global_stats = train_hierarchical_posteriors(train, TARGET)
    cells: dict[tuple, dict] = {}
    row_cell_index: dict[tuple, int] = {}

    for row in test:
        full_key, stats = posterior_for_row(row, tables, global_stats)
        item = cells.setdefault(
            full_key,
            {
                "key": full_key,
                "count": 0,
                "alpha": float(stats["alpha"]),
                "beta": float(stats["beta"]),
                "posterior_mean": float(stats["mean"]),
                "support": int(stats["n"]),
                "training_damage": int(stats["y"]),
                "fallback_level": str(stats["level"]),
                "future_damage": 0,
                "future_cost": 0.0,
                "future_aos": 0.0,
            },
        )
        item["count"] += 1
        item["future_damage"] += int(bool(row[TARGET]))
        item["future_cost"] += float(row["cost"])
        item["future_aos"] += float(row["aos"])

    ordered_cells = sorted(cells.values(), key=lambda item: item["key"])
    for idx, item in enumerate(ordered_cells):
        row_cell_index[item["key"]] = idx
    return ordered_cells, row_cell_index


def bayesian_set_cell_scores(
    cells: list[dict],
    budget: float,
    draws: int = POSTERIOR_DRAWS,
    seed: int = 20260528,
) -> dict[tuple, dict]:
    rng = np.random.default_rng(seed)
    alpha = np.array([cell["alpha"] for cell in cells], dtype=float)
    beta = np.array([cell["beta"] for cell in cells], dtype=float)
    counts = np.array([cell["count"] for cell in cells], dtype=float)
    total_records = int(counts.sum())
    selected_records = max(1, math.ceil(total_records * budget))

    membership = np.zeros(len(cells), dtype=float)
    boundary_excess = np.zeros(len(cells), dtype=float)
    lower_tail = np.zeros(len(cells), dtype=float)

    for _ in range(draws):
        theta = rng.beta(alpha, beta)
        order = np.argsort(-theta, kind="mergesort")
        ordered_counts = counts[order]
        cumulative = np.cumsum(ordered_counts)
        previous = cumulative - ordered_counts

        selected_units = np.clip(selected_records - previous, 0, ordered_counts)
        membership[order] += selected_units / ordered_counts

        if selected_records < total_records:
            boundary_position = int(np.searchsorted(cumulative, selected_records + 1, side="left"))
            boundary = theta[order[min(boundary_position, len(order) - 1)]]
        else:
            boundary = float(theta[order[-1]])
        boundary_excess += np.maximum(theta - boundary, 0.0)

        lower_tail += theta

    out: dict[tuple, dict] = {}
    for idx, cell in enumerate(cells):
        pi = float(membership[idx] / draws)
        capacity_boundary_excess = float(boundary_excess[idx] / draws)
        mean_draw = float(lower_tail[idx] / draws)
        out[cell["key"]] = {
            **cell,
            "capacity_membership_probability": pi,
            "capacity_boundary_excess": capacity_boundary_excess,
            "membership_x_boundary_excess": pi * capacity_boundary_excess,
            "posterior_draw_mean": mean_draw,
        }
    return out


def score_from_cell_values(test: list[dict], values: dict[tuple, dict], value_name: str) -> list[tuple[float, dict]]:
    scored = []
    for row in test:
        key = tuple_key(row, FULL_LEVEL)
        score = float(values.get(key, {}).get(value_name, 0.0))
        scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    return scored


def cell_audit_rows(values: dict[tuple, dict], budget: float, limit: int = 10) -> list[dict]:
    rows = []
    ordered = sorted(values.values(), key=lambda row: (-row["membership_x_boundary_excess"], row["key"]))
    for rank, item in enumerate(ordered[:limit], start=1):
        key = item["key"]
        rows.append({
            "rank": rank,
            "budget_share": budget,
            "component": key[0],
            "phase_bucket": key[1],
            "size": key[2],
            "aircraft_mass_class": key[3],
            "posterior_mean": item["posterior_mean"],
            "capacity_membership_probability": item["capacity_membership_probability"],
            "capacity_boundary_excess": item["capacity_boundary_excess"],
            "membership_x_boundary_excess": item["membership_x_boundary_excess"],
            "test_component_records": item["count"],
            "training_support": item["support"],
            "training_damage": item["training_damage"],
            "fallback_level": item["fallback_level"],
            "observed_future_damage": item["future_damage"],
            "observed_future_cost": item["future_cost"],
            "observed_future_aos": item["future_aos"],
        })
    return rows


def make_report(metrics: list[dict], audit: list[dict]) -> str:
    by_score_budget = {(row["score"], row["budget_share"]): row for row in metrics}
    hier = by_score_budget.get(("hierarchical_eb_execution", 0.05))
    bayesian_set_model = by_score_budget.get(("bayesian_set_model_full", 0.05))
    pass_lines = []
    if hier and bayesian_set_model:
        lift_gap = bayesian_set_model["lift"] - hier["lift"]
        cost_gap = bayesian_set_model["event_deduplicated_cost_capture"] - hier["event_deduplicated_cost_capture"]
        aos_gap = bayesian_set_model["event_deduplicated_aos_capture"] - hier["event_deduplicated_aos_capture"]
        if lift_gap > 0:
            pass_lines.append("Bayesian set model improves 5% damage lift over deterministic hierarchical EB.")
        if abs(lift_gap) <= 0.15 and (cost_gap > 0 or aos_gap > 0):
            pass_lines.append("Bayesian set model keeps 5% lift close to hierarchical EB and improves cost or AOS capture.")
        if not pass_lines:
            pass_lines.append("The single-window smoke test does not yet show a clear Bayesian set model gain over hierarchical EB.")

    lines = [
        "# Bayesian set model smoke test",
        "",
        f"Training window: {TRAIN_START}-{TRAIN_END}; test year: {TEST_YEAR}; posterior draws: {POSTERIOR_DRAWS}.",
        "",
        "## Screening metrics",
        "",
        "| Score | Capacity | Captured damage | Capture | Hit rate | Lift | Cost capture | AOS capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(metrics, key=lambda item: (item["budget_share"], -item["lift"], item["score"])):
        lines.append(
            f"| {row['score']} | {row['budget_share']:.1%} | "
            f"{row['captured_target_records']:,}/{row['target_records']:,} | "
            f"{row['capture_rate']:.1%} | {row['selected_target_rate']:.1%} | "
            f"{row['lift']:.2f} | {row['event_deduplicated_cost_capture']:.1%} | "
            f"{row['event_deduplicated_aos_capture']:.1%} |"
        )

    lines.extend([
        "",
        "## Smoke-test summary",
        "",
        *[f"- {line}" for line in pass_lines],
        "",
        "## Top Bayesian set model certificate cells at 5% capacity",
        "",
        "| Rank | Component | Phase | Size | Mass | Mean | Pi | boundary_excess | Support | Future damage |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in audit[:10]:
        lines.append(
            f"| {row['rank']} | {row['component']} | {row['phase_bucket']} | {row['size']} | "
            f"{row['aircraft_mass_class']} | {row['posterior_mean']:.3f} | "
            f"{row['capacity_membership_probability']:.3f} | {row['capacity_boundary_excess']:.4f} | "
            f"{row['training_support']:,} | {row['observed_future_damage']:,} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    parts = load_parts()
    train = [row for row in parts if TRAIN_START <= int(row["year"]) <= TRAIN_END]
    test = [row for row in parts if int(row["year"]) == TEST_YEAR]
    if not train or not test:
        raise RuntimeError("Training or test data are empty.")

    metrics: list[dict] = []
    audit_rows: list[dict] = []
    for budget in [0.05, 0.10]:
        direct_scored = score_records(train, test, MAIN_SCORE, TARGET)
        hier_scored = score_hierarchical_mean(train, test)
        cells, _ = build_test_cells(train, test)
        values = bayesian_set_cell_scores(cells, budget, POSTERIOR_DRAWS, seed=20260528 + int(budget * 1000))
        scored_variants = {
            "direct_smoothed_cell": direct_scored,
            "hierarchical_eb_execution": hier_scored,
            "bayesian_set_model_posterior_mean": score_from_cell_values(test, values, "posterior_mean"),
            "bayesian_set_model_membership_probability": score_from_cell_values(test, values, "capacity_membership_probability"),
            "bayesian_set_model_capacity_boundary_excess": score_from_cell_values(test, values, "capacity_boundary_excess"),
            "bayesian_set_model_full": score_from_cell_values(test, values, "membership_x_boundary_excess"),
        }
        for score_name, scored in scored_variants.items():
            metrics.append(evaluate_scored(scored, budget, score_name))
        if budget == 0.05:
            audit_rows = cell_audit_rows(values, budget, limit=25)

    write_csv(RESULT_DIR / "bayesian_set_model_smoke_metrics.csv", metrics)
    write_csv(RESULT_DIR / "bayesian_set_model_smoke_cell_audit.csv", audit_rows)
    (RESULT_DIR / "bayesian_set_model_smoke_report.md").write_text(make_report(metrics, audit_rows), encoding="utf-8")


if __name__ == "__main__":
    main()





