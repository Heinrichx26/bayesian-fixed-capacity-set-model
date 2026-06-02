from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_WILDLIFE_DIR = PROJECT_ROOT / "data" / "raw" / "faa_wildlife"
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "method_evidence"
FULL_RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model" / "full"

CAPACITIES = [0.05, 0.10]
CANDIDATE_SHARES = [0.005, 0.01, 0.02, 0.05]
TARGET = "part_damage"
PRIOR_WEIGHT = 10.0
HIERARCHY_LEVELS = [
    ["component"],
    ["component", "phase_bucket"],
    ["component", "phase_bucket", "size"],
    ["component", "phase_bucket", "size", "aircraft_mass_class"],
]
FULL_LEVEL = HIERARCHY_LEVELS[-1]

COMPONENTS: dict[str, tuple[list[str], list[str]]] = {
    "radome": (["STR_RAD"], ["DAM_RAD"]),
    "windshield": (["STR_WINDSHLD"], ["DAM_WINDSHLD"]),
    "nose": (["STR_NOSE"], ["DAM_NOSE"]),
    "engine": (
        ["STR_ENG1", "STR_ENG2", "STR_ENG3", "STR_ENG4"],
        ["DAM_ENG1", "DAM_ENG2", "DAM_ENG3", "DAM_ENG4"],
    ),
    "propeller": (["STR_PROP"], ["DAM_PROP"]),
    "wing_rotor": (["STR_WING_ROT"], ["DAM_WING_ROT"]),
    "fuselage": (["STR_FUSE"], ["DAM_FUSE"]),
    "landing_gear": (["STR_LG"], ["DAM_LG"]),
    "tail": (["STR_TAIL"], ["DAM_TAIL"]),
    "lights": (["STR_LGHTS"], ["DAM_LGHTS"]),
    "other": (["STR_OTHER"], ["DAM_OTHER"]),
}

DAMAGE_PARTS = [
    "DAM_RAD",
    "DAM_WINDSHLD",
    "DAM_NOSE",
    "DAM_ENG1",
    "DAM_ENG2",
    "DAM_ENG3",
    "DAM_ENG4",
    "DAM_PROP",
    "DAM_WING_ROT",
    "DAM_FUSE",
    "DAM_LG",
    "DAM_TAIL",
    "DAM_LGHTS",
    "DAM_OTHER",
]


def clean_key(key: str) -> str:
    return key.strip().upper().replace(" ", "_")


def text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def number(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return 0.0


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return text(value).upper() in {"TRUE", "T", "YES", "Y", "1"}


def phase_bucket(phase: str) -> str:
    p = phase.lower()
    if not p:
        return "unknown"
    if any(item in p for item in ["take-off", "takeoff", "climb", "departure"]):
        return "departure"
    if any(item in p for item in ["approach", "landing", "descent", "arrival"]):
        return "arrival"
    if any(item in p for item in ["en route", "enroute"]):
        return "enroute"
    if any(item in p for item in ["taxi", "parked", "pushback", "local"]):
        return "ground"
    return "unknown"


def load_events() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RAW_WILDLIFE_DIR.glob("faa_wildlife_export_*.json")):
        with path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        for raw in payload.get("Result", []):
            row = {clean_key(k): v for k, v in raw.items()}
            rows.append(row)
    dedup: dict[str, dict] = {}
    for row in rows:
        key = text(row.get("INDX_NR"))
        if key:
            dedup[key] = row
    return list(dedup.values())


def enrich_event(row: dict) -> dict:
    year = int(number(row.get("INCIDENT_YEAR")))
    month = int(number(row.get("INCIDENT_MONTH")))
    phase = text(row.get("PHASE_OF_FLIGHT"))
    damage_level = text(row.get("DAMAGE_LEVEL")).upper()
    indicated_damage = text(row.get("INDICATED_DAMAGE")).upper() == "TRUE"
    any_part_damage = any(truthy(row.get(part)) for part in DAMAGE_PARTS)
    cost = number(row.get("COST_REPAIRS_INFL_ADJ")) + number(row.get("COST_OTHER_INFL_ADJ"))
    aos = number(row.get("AOS"))
    injuries = number(row.get("NR_INJURIES"))
    fatalities = number(row.get("NR_FATALITIES"))
    effect = text(row.get("EFFECT")).upper()
    meaningful_effect = effect not in {"", "NONE", "NULL"}
    damaged = indicated_damage or any_part_damage or (damage_level not in {"", "N", "NO DAMAGE"})
    row["_YEAR"] = year
    row["_MONTH"] = month
    row["_PHASE_BUCKET"] = phase_bucket(phase)
    row["_SIZE"] = text(row.get("SIZE")).upper() or "UNKNOWN"
    row["_HARD_EVENT"] = damaged or cost > 0 or aos > 0 or meaningful_effect or injuries > 0 or fatalities > 0
    row["_COST"] = cost
    row["_AOS"] = aos
    return row


def load_parts_through(end_year: int = 2025) -> list[dict]:
    out: list[dict] = []
    for event in (enrich_event(row) for row in load_events()):
        if not (1990 <= int(event["_YEAR"]) <= end_year):
            continue
        event_id = text(event.get("INDX_NR"))
        if not event_id:
            continue
        for component, (struck_fields, damage_fields) in COMPONENTS.items():
            if not any(truthy(event.get(field)) for field in struck_fields):
                continue
            out.append({
                "event_id": event_id,
                "year": int(event["_YEAR"]),
                "month": int(event["_MONTH"]),
                "component": component,
                "part_damage": any(truthy(event.get(field)) for field in damage_fields),
                "event_hard": bool(event.get("_HARD_EVENT")),
                "cost": float(event.get("_COST") or 0.0),
                "aos": float(event.get("_AOS") or 0.0),
                "phase_bucket": event.get("_PHASE_BUCKET") or "unknown",
                "size": event.get("_SIZE") or "UNKNOWN",
                "aircraft_mass_class": text(event.get("AC_MASS")) or "UNKNOWN",
                "species_id": text(event.get("SPECIES_ID")) or text(event.get("SPECIES")) or "UNKNOWN",
                "airport_id": text(event.get("AIRPORT_ID")) or "UNKNOWN",
                "aircraft": text(event.get("AIRCRAFT")),
            })
    return out


def load_parts() -> list[dict]:
    return load_parts_through(2025)


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


def bayesian_set_cell_scores(cells: list[dict], capacity: float, draws: int, seed: int) -> dict[tuple, dict]:
    rng = np.random.default_rng(seed)
    alpha = np.array([cell["alpha"] for cell in cells], dtype=float)
    beta = np.array([cell["beta"] for cell in cells], dtype=float)
    counts = np.array([cell["count"] for cell in cells], dtype=float)
    total_units = int(counts.sum())
    selected_units_total = max(1, math.ceil(total_units * capacity))
    membership = np.zeros(len(cells), dtype=float)
    boundary_excess = np.zeros(len(cells), dtype=float)
    draw_mean = np.zeros(len(cells), dtype=float)
    for _ in range(draws):
        theta = rng.beta(alpha, beta)
        order = np.argsort(-theta, kind="mergesort")
        ordered_counts = counts[order]
        cumulative = np.cumsum(ordered_counts)
        previous = cumulative - ordered_counts
        selected_units = np.clip(selected_units_total - previous, 0, ordered_counts)
        membership[order] += selected_units / ordered_counts
        if selected_units_total < total_units:
            boundary_position = int(np.searchsorted(cumulative, selected_units_total + 1, side="left"))
            boundary = theta[order[min(boundary_position, len(order) - 1)]]
        else:
            boundary = float(theta[order[-1]])
        boundary_excess += np.maximum(theta - boundary, 0.0)
        draw_mean += theta
    out: dict[tuple, dict] = {}
    for idx, cell in enumerate(cells):
        pi = float(membership[idx] / draws)
        excess = float(boundary_excess[idx] / draws)
        out[cell["key"]] = {
            **cell,
            "capacity_membership_probability": pi,
            "capacity_boundary_excess": excess,
            "membership_x_boundary_excess": pi * excess,
            "posterior_draw_mean": float(draw_mean[idx] / draws),
        }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def event_totals(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        event_id = str(row["event_id"])
        if event_id not in out:
            out[event_id] = {"cost": float(row["cost"]), "aos": float(row["aos"])}
    return out


def selected_metrics(test: list[dict], selected: list[dict]) -> dict:
    total_damage = sum(int(bool(row[TARGET])) for row in test)
    selected_damage = sum(int(bool(row[TARGET])) for row in selected)
    total_rate = total_damage / len(test) if test else 0.0
    selected_rate = selected_damage / len(selected) if selected else 0.0
    all_events = event_totals(test)
    selected_event_ids = {str(row["event_id"]) for row in selected}
    total_cost = sum(value["cost"] for value in all_events.values())
    total_aos = sum(value["aos"] for value in all_events.values())
    selected_cost = sum(all_events[event_id]["cost"] for event_id in selected_event_ids if event_id in all_events)
    selected_aos = sum(all_events[event_id]["aos"] for event_id in selected_event_ids if event_id in all_events)
    return {
        "units": len(selected),
        "damage_units": selected_damage,
        "selected_hit_rate": selected_rate,
        "damage_capture": selected_damage / total_damage if total_damage else 0.0,
        "damage_lift": selected_rate / total_rate if total_rate else 0.0,
        "cost_capture": selected_cost / total_cost if total_cost else 0.0,
        "aos_capture": selected_aos / total_aos if total_aos else 0.0,
    }


def group_name(pi: float) -> str:
    if pi >= 0.95:
        return "High certificate"
    if pi >= 0.50:
        return "Boundary"
    return "Unstable"


def aggregate_group_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "years": set(),
        "units": 0,
        "damage_units": 0,
        "selected_units": 0,
        "selected_damage_units": 0,
        "total_damage_units": 0,
        "selected_cost": 0.0,
        "total_cost": 0.0,
        "selected_aos": 0.0,
        "total_aos": 0.0,
        "supports": [],
        "boundary_excess": [],
    })
    for row in rows:
        key = (row["capacity"], row["certificate_group"])
        item = groups[key]
        item["years"].add(row["test_year"])
        for field in ["units", "damage_units", "selected_units", "selected_damage_units", "total_damage_units"]:
            item[field] += row[field]
        for field in ["selected_cost", "total_cost", "selected_aos", "total_aos"]:
            item[field] += row[field]
        item["supports"].extend(row["support_values"])
        item["boundary_excess"].extend(row["boundary_excess_values"])

    out = []
    order = {"High certificate": 0, "Boundary": 1, "Unstable": 2}
    for (capacity, certificate_group), item in groups.items():
        out.append({
            "capacity": capacity,
            "certificate_group": certificate_group,
            "test_years": len(item["years"]),
            "units": item["units"],
            "unit_share": item["units"] / item["selected_units"] if item["selected_units"] else 0.0,
            "damage_units": item["damage_units"],
            "damage_rate": item["damage_units"] / item["units"] if item["units"] else 0.0,
            "damage_capture": item["damage_units"] / item["total_damage_units"] if item["total_damage_units"] else 0.0,
            "cost_capture": item["selected_cost"] / item["total_cost"] if item["total_cost"] else 0.0,
            "aos_capture": item["selected_aos"] / item["total_aos"] if item["total_aos"] else 0.0,
            "median_support": float(np.median(item["supports"])) if item["supports"] else 0.0,
            "median_boundary_excess": float(np.median(item["boundary_excess"])) if item["boundary_excess"] else 0.0,
        })
    return sorted(out, key=lambda row: (row["capacity"], order.get(row["certificate_group"], 99)))


def score_level(train: list[dict], test: list[dict], max_level: int) -> tuple[list[tuple[float, dict]], np.ndarray, list[str]]:
    tables, global_stats = train_hierarchical_posteriors(train, TARGET, prior_weight=PRIOR_WEIGHT)
    allowed = [tuple(level) for level in HIERARCHY_LEVELS[:max_level]]
    scored: list[tuple[float, dict]] = []
    probabilities: list[float] = []
    used_levels: list[str] = []
    for row in test:
        stats = None
        for level in reversed(allowed):
            key = tuple_key(row, level)
            stats = tables.get(level, {}).get(key)
            if stats is not None:
                break
        if stats is None:
            stats = global_stats
        value = float(stats["mean"])
        row_copy = dict(row)
        row_copy["_used_level"] = str(stats["level"])
        scored.append((value, row_copy))
        probabilities.append(min(max(value, 1e-6), 1 - 1e-6))
        used_levels.append(str(stats["level"]))
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    return scored, np.array(probabilities, dtype=float), used_levels


def direct_probability(train: list[dict], test: list[dict]) -> tuple[list[tuple[float, dict]], np.ndarray]:
    stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "y": 0})
    global_y = sum(int(bool(row[TARGET])) for row in train)
    global_rate = global_y / len(train) if train else 0.0
    for row in train:
        key = tuple_key(row, FULL_LEVEL)
        stats[key]["n"] += 1
        stats[key]["y"] += int(bool(row[TARGET]))
    values = {
        key: (item["y"] + PRIOR_WEIGHT * global_rate) / (item["n"] + PRIOR_WEIGHT)
        for key, item in stats.items()
    }
    scored = []
    probabilities = []
    for row in test:
        score = float(values.get(tuple_key(row, FULL_LEVEL), global_rate))
        scored.append((score, row))
        probabilities.append(score)
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    scores = np.array(probabilities, dtype=float)
    return scored, np.clip(scores, 1e-6, 1 - 1e-6)


def ablation_and_certificates(parts: list[dict], start_year: int, end_year: int, draws: int) -> None:
    group_yearly: list[dict] = []
    audit_yearly: list[dict] = []
    ablation_yearly: list[dict] = []
    probability_store: dict[str, dict] = defaultdict(lambda: {"y": [], "p": []})

    for test_year in range(start_year, end_year + 1):
        train = [row for row in parts if test_year - 5 <= int(row["year"]) <= test_year - 1]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue

        y_test = [int(bool(row[TARGET])) for row in test]
        total_damage = sum(y_test)
        all_events = event_totals(test)
        total_cost = sum(value["cost"] for value in all_events.values())
        total_aos = sum(value["aos"] for value in all_events.values())

        cells, _ = build_test_cells(train, test)
        cert_by_capacity = {
            capacity: bayesian_set_cell_scores(cells, capacity, draws, seed=20260528 + test_year * 1000 + int(capacity * 10000))
            for capacity in CAPACITIES
        }

        hierarchical_scored = score_hierarchical_mean(train, test)
        for capacity in CAPACITIES:
            k = max(1, math.ceil(len(test) * capacity))
            selected = [row for _, row in hierarchical_scored[:k]]
            values = cert_by_capacity[capacity]
            selected_total = len(selected)
            selected_damage_total = sum(int(bool(row[TARGET])) for row in selected)

            per_group: dict[str, list[dict]] = defaultdict(list)
            for row in selected:
                cert = values.get(tuple_key(row, FULL_LEVEL), {})
                per_group[group_name(float(cert.get("capacity_membership_probability", 0.0)))].append(row)

            for name in ["High certificate", "Boundary", "Unstable"]:
                rows = per_group.get(name, [])
                selected_event_ids = {str(row["event_id"]) for row in rows}
                support_values = [
                    float(values.get(tuple_key(row, FULL_LEVEL), {}).get("support", 0.0))
                    for row in rows
                ]
                boundary_values = [
                    float(values.get(tuple_key(row, FULL_LEVEL), {}).get("capacity_boundary_excess", 0.0))
                    for row in rows
                ]
                group_yearly.append({
                    "test_year": test_year,
                    "capacity": capacity,
                    "certificate_group": name,
                    "units": len(rows),
                    "damage_units": sum(int(bool(row[TARGET])) for row in rows),
                    "selected_units": selected_total,
                    "selected_damage_units": selected_damage_total,
                    "total_damage_units": total_damage,
                    "selected_cost": sum(all_events[event_id]["cost"] for event_id in selected_event_ids if event_id in all_events),
                    "total_cost": total_cost,
                    "selected_aos": sum(all_events[event_id]["aos"] for event_id in selected_event_ids if event_id in all_events),
                    "total_aos": total_aos,
                    "support_values": support_values,
                    "boundary_excess_values": boundary_values,
                })

            if abs(capacity - 0.05) < 1e-9:
                selected_ids = {id(row) for row in selected}
                unselected = [row for row in test if id(row) not in selected_ids]
                scored_candidates = []
                posterior_candidates = []
                boundary_excess_candidates = []
                for row in unselected:
                    cert = values.get(tuple_key(row, FULL_LEVEL), {})
                    pi = float(cert.get("capacity_membership_probability", 0.0))
                    boundary = float(cert.get("capacity_boundary_excess", 0.0))
                    posterior = float(cert.get("posterior_mean", 0.0))
                    scored_candidates.append(((1.0 - abs(pi - 0.50), boundary, pi), row))
                    posterior_candidates.append((posterior, row))
                    boundary_excess_candidates.append((boundary, row))
                scored_candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1]["event_id"], item[1]["component"]))
                posterior_candidates.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
                boundary_excess_candidates.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
                random_pool = unselected[:]
                rng = np.random.default_rng(20260528 + test_year)
                for candidate_share in CANDIDATE_SHARES:
                    candidate_count = max(1, math.ceil(len(test) * candidate_share))
                    certificate_candidates = [row for _, row in scored_candidates[:candidate_count]]
                    posterior_next = [row for _, row in posterior_candidates[:candidate_count]]
                    boundary_excess_next = [row for _, row in boundary_excess_candidates[:candidate_count]]
                    for label, rows in [
                        ("Boundary-certificate expansion", certificate_candidates),
                        ("Boundary-excess expansion", boundary_excess_next),
                        ("Posterior-mean expansion", posterior_next),
                    ]:
                        metrics = selected_metrics(test, rows)
                        audit_yearly.append({
                            "test_year": test_year,
                            "candidate_rule": label,
                            "candidate_share": candidate_share,
                            "candidate_units": len(rows),
                            "damage_rate": metrics["selected_hit_rate"],
                            "damage_lift": metrics["damage_lift"],
                            "damage_capture": metrics["damage_capture"],
                            "cost_capture": metrics["cost_capture"],
                            "aos_capture": metrics["aos_capture"],
                        })
                    random_damage = []
                    random_lift = []
                    random_capture = []
                    random_cost = []
                    random_aos = []
                    for _ in range(200):
                        sample = rng.choice(len(random_pool), size=min(candidate_count, len(random_pool)), replace=False)
                        chosen = [random_pool[int(idx)] for idx in sample]
                        metrics = selected_metrics(test, chosen)
                        random_damage.append(metrics["selected_hit_rate"])
                        random_lift.append(metrics["damage_lift"])
                        random_capture.append(metrics["damage_capture"])
                        random_cost.append(metrics["cost_capture"])
                        random_aos.append(metrics["aos_capture"])
                    audit_yearly.append({
                        "test_year": test_year,
                        "candidate_rule": "Random unselected expansion",
                        "candidate_share": candidate_share,
                        "candidate_units": min(candidate_count, len(random_pool)),
                        "damage_rate": float(np.mean(random_damage)),
                        "damage_lift": float(np.mean(random_lift)),
                        "damage_capture": float(np.mean(random_capture)),
                        "cost_capture": float(np.mean(random_cost)),
                        "aos_capture": float(np.mean(random_aos)),
                    })

        rules = [
            ("Direct full cell", direct_probability(train, test), "component|phase_bucket|size|aircraft_mass_class"),
            ("EB component", score_level(train, test, 1), "component"),
            ("EB component-phase", score_level(train, test, 2), "component|phase_bucket"),
            ("EB component-phase-size", score_level(train, test, 3), "component|phase_bucket|size"),
            ("Full hierarchy", score_level(train, test, 4), "component|phase_bucket|size|aircraft_mass_class"),
            ("Full hierarchy with certificates", score_level(train, test, 4), "component|phase_bucket|size|aircraft_mass_class"),
        ]
        for label, result, target_level in rules:
            if label == "Direct full cell":
                scored, probabilities = result  # type: ignore[misc]
                used_levels = [target_level] * len(test)
            else:
                scored, probabilities, used_levels = result  # type: ignore[misc]
            probability_store[label]["y"].extend(y_test)
            probability_store[label]["p"].extend(probabilities.tolist())
            for capacity in CAPACITIES:
                k = max(1, math.ceil(len(scored) * capacity))
                selected = [row for _, row in scored[:k]]
                metrics = selected_metrics(test, selected)
                if label == "Direct full cell":
                    fallback_share = 0.0
                else:
                    fallback_share = sum(1 for _, row in scored[:k] if row.get("_used_level") != target_level) / k if k else 0.0
                ablation_yearly.append({
                    "test_year": test_year,
                    "rule": label,
                    "capacity": capacity,
                    "test_component_units": len(test),
                    "damage_units": total_damage,
                    "selected_units": k,
                    "selected_damage_units": metrics["damage_units"],
                    "damage_capture": metrics["damage_capture"],
                    "selected_hit_rate": metrics["selected_hit_rate"],
                    "overall_damage_rate": total_damage / len(test) if test else 0.0,
                    "damage_lift": metrics["damage_lift"],
                    "cost_capture": metrics["cost_capture"],
                    "aos_capture": metrics["aos_capture"],
                    "fallback_share": fallback_share,
                })

    group_aggregate = aggregate_group_rows(group_yearly)
    audit_aggregate = aggregate_audit_rows(audit_yearly)
    ablation_aggregate = aggregate_ablation_rows(ablation_yearly, probability_store)
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "certificate_group_yearly.csv", flatten_group_rows(group_yearly))
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "certificate_group_aggregate.csv", group_aggregate)
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "boundary_candidate_audit_yearly.csv", audit_yearly)
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "boundary_candidate_audit_aggregate.csv", audit_aggregate)
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "hierarchy_ablation_yearly.csv", ablation_yearly)
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "hierarchy_ablation_aggregate.csv", ablation_aggregate)


def boundary_candidate_audit_only(parts: list[dict], start_year: int, end_year: int, draws: int) -> None:
    audit_yearly: list[dict] = []
    capacity = 0.05
    for test_year in range(start_year, end_year + 1):
        train = [row for row in parts if test_year - 5 <= int(row["year"]) <= test_year - 1]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue
        cells, _ = build_test_cells(train, test)
        values = bayesian_set_cell_scores(
            cells,
            capacity,
            draws,
            seed=20260528 + test_year * 1000 + int(capacity * 10000),
        )
        hierarchical_scored = score_hierarchical_mean(train, test)
        k = max(1, math.ceil(len(test) * capacity))
        selected = [row for _, row in hierarchical_scored[:k]]
        selected_ids = {id(row) for row in selected}
        unselected = [row for row in test if id(row) not in selected_ids]
        scored_candidates = []
        boundary_excess_candidates = []
        posterior_candidates = []
        for row in unselected:
            cert = values.get(tuple_key(row, FULL_LEVEL), {})
            pi = float(cert.get("capacity_membership_probability", 0.0))
            boundary = float(cert.get("capacity_boundary_excess", 0.0))
            posterior = float(cert.get("posterior_mean", 0.0))
            scored_candidates.append(((1.0 - abs(pi - 0.50), boundary, pi), row))
            boundary_excess_candidates.append((boundary, row))
            posterior_candidates.append((posterior, row))
        scored_candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1]["event_id"], item[1]["component"]))
        boundary_excess_candidates.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
        posterior_candidates.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
        random_pool = unselected[:]
        rng = np.random.default_rng(20260528 + test_year)
        for candidate_share in CANDIDATE_SHARES:
            candidate_count = max(1, math.ceil(len(test) * candidate_share))
            rule_sets = [
                ("Boundary-certificate expansion", [row for _, row in scored_candidates[:candidate_count]]),
                ("Boundary-excess expansion", [row for _, row in boundary_excess_candidates[:candidate_count]]),
                ("Posterior-mean expansion", [row for _, row in posterior_candidates[:candidate_count]]),
            ]
            for label, rows in rule_sets:
                metrics = selected_metrics(test, rows)
                audit_yearly.append({
                    "test_year": test_year,
                    "candidate_rule": label,
                    "candidate_share": candidate_share,
                    "candidate_units": len(rows),
                    "damage_rate": metrics["selected_hit_rate"],
                    "damage_lift": metrics["damage_lift"],
                    "damage_capture": metrics["damage_capture"],
                    "cost_capture": metrics["cost_capture"],
                    "aos_capture": metrics["aos_capture"],
                })
            random_damage = []
            random_lift = []
            random_capture = []
            random_cost = []
            random_aos = []
            for _ in range(200):
                sample = rng.choice(len(random_pool), size=min(candidate_count, len(random_pool)), replace=False)
                chosen = [random_pool[int(idx)] for idx in sample]
                metrics = selected_metrics(test, chosen)
                random_damage.append(metrics["selected_hit_rate"])
                random_lift.append(metrics["damage_lift"])
                random_capture.append(metrics["damage_capture"])
                random_cost.append(metrics["cost_capture"])
                random_aos.append(metrics["aos_capture"])
            audit_yearly.append({
                "test_year": test_year,
                "candidate_rule": "Random unselected expansion",
                "candidate_share": candidate_share,
                "candidate_units": min(candidate_count, len(random_pool)),
                "damage_rate": float(np.mean(random_damage)),
                "damage_lift": float(np.mean(random_lift)),
                "damage_capture": float(np.mean(random_capture)),
                "cost_capture": float(np.mean(random_cost)),
                "aos_capture": float(np.mean(random_aos)),
            })
    audit_aggregate = aggregate_audit_rows(audit_yearly)
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "boundary_candidate_audit_yearly.csv", audit_yearly)
    write_csv(RESULT_DIR / ("smoke" if start_year == end_year else "full") / "boundary_candidate_audit_aggregate.csv", audit_aggregate)


def flatten_group_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in {"support_values", "boundary_excess_values"}}
        item["median_support"] = float(np.median(row["support_values"])) if row["support_values"] else 0.0
        item["median_boundary_excess"] = float(np.median(row["boundary_excess_values"])) if row["boundary_excess_values"] else 0.0
        out.append(item)
    return out


def aggregate_audit_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "years": set(),
        "candidate_units": 0,
        "damage_rates": [],
        "damage_lifts": [],
        "damage_captures": [],
        "cost_captures": [],
        "aos_captures": [],
    })
    for row in rows:
        item = groups[(row["candidate_rule"], row["candidate_share"])]
        item["years"].add(row["test_year"])
        item["candidate_units"] += row["candidate_units"]
        for source, target in [
            ("damage_rate", "damage_rates"),
            ("damage_lift", "damage_lifts"),
            ("damage_capture", "damage_captures"),
            ("cost_capture", "cost_captures"),
            ("aos_capture", "aos_captures"),
        ]:
            item[target].append(float(row[source]))
    out = []
    order = {
        "Boundary-certificate expansion": 0,
        "Boundary-excess expansion": 1,
        "Posterior-mean expansion": 2,
        "Random unselected expansion": 3,
    }
    for (rule, share), item in groups.items():
        out.append({
            "candidate_rule": rule,
            "candidate_share": share,
            "test_years": len(item["years"]),
            "candidate_units": item["candidate_units"],
            "mean_damage_rate": float(np.mean(item["damage_rates"])) if item["damage_rates"] else 0.0,
            "mean_damage_lift": float(np.mean(item["damage_lifts"])) if item["damage_lifts"] else 0.0,
            "mean_damage_capture": float(np.mean(item["damage_captures"])) if item["damage_captures"] else 0.0,
            "mean_cost_capture": float(np.mean(item["cost_captures"])) if item["cost_captures"] else 0.0,
            "mean_aos_capture": float(np.mean(item["aos_captures"])) if item["aos_captures"] else 0.0,
        })
    return sorted(out, key=lambda row: (row["candidate_share"], order.get(row["candidate_rule"], 99)))


def partial_update_2026(parts: list[dict], draws: int) -> None:
    train = [row for row in parts if 2021 <= int(row["year"]) <= 2025]
    test = [row for row in parts if int(row["year"]) == 2026]
    if not train or not test:
        raise RuntimeError("Partial 2026 update requires 2021--2026 public data files.")
    cells, _ = build_test_cells(train, test)
    scored = score_hierarchical_mean(train, test)
    rows: list[dict] = []
    group_rows: list[dict] = []
    for capacity in CAPACITIES:
        values = bayesian_set_cell_scores(
            cells,
            capacity,
            draws,
            seed=20260601 + int(capacity * 10000),
        )
        k = max(1, math.ceil(len(test) * capacity))
        selected = [row for _, row in scored[:k]]
        metrics = selected_metrics(test, selected)
        per_group: dict[str, list[dict]] = defaultdict(list)
        for row in selected:
            cert = values.get(tuple_key(row, FULL_LEVEL), {})
            per_group[group_name(float(cert.get("capacity_membership_probability", 0.0)))].append(row)
        group_counts = {name: len(per_group.get(name, [])) for name in ["High certificate", "Boundary", "Unstable"]}
        rows.append({
            "train_start": 2021,
            "train_end": 2025,
            "test_year": 2026,
            "capacity": capacity,
            "test_component_units": len(test),
            "damage_units": sum(int(bool(row[TARGET])) for row in test),
            "selected_units": metrics["units"],
            "selected_damage_units": metrics["damage_units"],
            "damage_capture": metrics["damage_capture"],
            "selected_hit_rate": metrics["selected_hit_rate"],
            "damage_lift": metrics["damage_lift"],
            "cost_capture": metrics["cost_capture"],
            "aos_capture": metrics["aos_capture"],
            "high_certificate_share": group_counts["High certificate"] / metrics["units"] if metrics["units"] else 0.0,
            "boundary_share": group_counts["Boundary"] / metrics["units"] if metrics["units"] else 0.0,
            "unstable_share": group_counts["Unstable"] / metrics["units"] if metrics["units"] else 0.0,
        })
        for name in ["High certificate", "Boundary", "Unstable"]:
            group_rows.append({
                "capacity": capacity,
                "certificate_group": name,
                "units": group_counts[name],
                "unit_share": group_counts[name] / metrics["units"] if metrics["units"] else 0.0,
            })
    out_dir = RESULT_DIR / "partial_2026"
    write_csv(out_dir / "partial_2026_update.csv", rows)
    write_csv(out_dir / "partial_2026_certificate_groups.csv", group_rows)


def aggregate_ablation_rows(rows: list[dict], probability_store: dict[str, dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "years": set(),
        "test_component_units": 0,
        "damage_units": 0,
        "selected_units": 0,
        "selected_damage_units": 0,
        "fallback_units": 0.0,
    })
    for row in rows:
        key = (row["rule"], row["capacity"])
        item = groups[key]
        item["years"].add(row["test_year"])
        for field in ["test_component_units", "damage_units", "selected_units", "selected_damage_units"]:
            item[field] += row[field]
        item["fallback_units"] += row["fallback_share"] * row["selected_units"]
    order = {
        "Direct full cell": 0,
        "EB component": 1,
        "EB component-phase": 2,
        "EB component-phase-size": 3,
        "Full hierarchy": 4,
        "Full hierarchy with certificates": 5,
    }
    out = []
    for (rule, capacity), item in groups.items():
        overall_rate = item["damage_units"] / item["test_component_units"] if item["test_component_units"] else 0.0
        hit_rate = item["selected_damage_units"] / item["selected_units"] if item["selected_units"] else 0.0
        y = np.array(probability_store[rule]["y"], dtype=int)
        p = np.clip(np.array(probability_store[rule]["p"], dtype=float), 1e-6, 1 - 1e-6)
        probability_order = np.argsort(p)
        bins = np.array_split(probability_order, 10)
        ece = 0.0
        for bin_index in bins:
            if len(bin_index) == 0:
                continue
            ece += len(bin_index) / len(p) * abs(float(y[bin_index].mean()) - float(p[bin_index].mean()))
        out.append({
            "rule": rule,
            "capacity": capacity,
            "test_years": len(item["years"]),
            "selected_damage_units": item["selected_damage_units"],
            "damage_units": item["damage_units"],
            "damage_capture": item["selected_damage_units"] / item["damage_units"] if item["damage_units"] else 0.0,
            "damage_lift": hit_rate / overall_rate if overall_rate else 0.0,
            "brier_score": brier_score_loss(y, p) if len(y) else 0.0,
            "log_loss": log_loss(y, p, labels=[0, 1]) if len(y) else 0.0,
            "expected_calibration_error": ece if len(y) else 0.0,
            "fallback_share": item["fallback_units"] / item["selected_units"] if item["selected_units"] else 0.0,
        })
    return sorted(out, key=lambda row: (row["capacity"], order.get(row["rule"], 99)))


def bootstrap_intervals(iterations: int = 1000, seed: int = 20260528) -> None:
    import pandas as pd

    yearly = pd.read_csv(FULL_RESULT_DIR / "set_model_selection_yearly.csv")
    years = sorted(yearly["test_year"].unique())
    rng = np.random.default_rng(seed)
    rules = ["catboost", "hierarchical_eb_execution", "direct_full_cell", "tabm"]
    capacities = [0.05, 0.10]
    metric_rows = []

    def pooled_metric(frame: pd.DataFrame, metric: str) -> float:
        if metric == "damage_lift":
            selected_rate = frame["selected_damage_units"].sum() / frame["selected_units"].sum()
            overall_rate = frame["damage_units"].sum() / frame["test_component_units"].sum()
            return selected_rate / overall_rate if overall_rate else 0.0
        if metric == "damage_capture":
            return frame["selected_damage_units"].sum() / frame["damage_units"].sum()
        if metric == "cost_capture":
            return frame["selected_event_cost"].sum() / frame["total_event_cost"].sum()
        if metric == "aos_capture":
            return frame["selected_event_aos"].sum() / frame["total_event_aos"].sum()
        raise ValueError(metric)

    for capacity in capacities:
        for rule in rules:
            base = yearly[(yearly["rule"] == rule) & (yearly["capacity"].round(4) == capacity)]
            for metric in ["damage_lift", "damage_capture", "cost_capture", "aos_capture"]:
                values = []
                for _ in range(iterations):
                    sample_years = rng.choice(years, size=len(years), replace=True)
                    sample = pd.concat([base[base["test_year"] == year] for year in sample_years], ignore_index=True)
                    values.append(pooled_metric(sample, metric))
                point = pooled_metric(base, metric)
                metric_rows.append({
                    "rule": rule,
                    "capacity": capacity,
                    "metric": metric,
                    "point": point,
                    "lower": float(np.quantile(values, 0.025)),
                    "upper": float(np.quantile(values, 0.975)),
                })

        for comparison in ["direct_full_cell", "catboost"]:
            diff_values = []
            for _ in range(iterations):
                sample_years = rng.choice(years, size=len(years), replace=True)
                diffs = []
                for year in sample_years:
                    bayesian = yearly[
                        (yearly["test_year"] == year)
                        & (yearly["rule"] == "hierarchical_eb_execution")
                        & (yearly["capacity"].round(4) == capacity)
                    ]["damage_lift"].iloc[0]
                    other = yearly[
                        (yearly["test_year"] == year)
                        & (yearly["rule"] == comparison)
                        & (yearly["capacity"].round(4) == capacity)
                    ]["damage_lift"].iloc[0]
                    diffs.append(bayesian - other)
                diff_values.append(float(np.mean(diffs)))
            actual = []
            for year in years:
                bayesian = yearly[
                    (yearly["test_year"] == year)
                    & (yearly["rule"] == "hierarchical_eb_execution")
                    & (yearly["capacity"].round(4) == capacity)
                ]["damage_lift"].iloc[0]
                other = yearly[
                    (yearly["test_year"] == year)
                    & (yearly["rule"] == comparison)
                    & (yearly["capacity"].round(4) == capacity)
                ]["damage_lift"].iloc[0]
                actual.append(bayesian - other)
            metric_rows.append({
                "rule": f"hierarchical_eb_execution minus {comparison}",
                "capacity": capacity,
                "metric": "paired_annual_lift_difference",
                "point": float(np.mean(actual)),
                "lower": float(np.quantile(diff_values, 0.025)),
                "upper": float(np.quantile(diff_values, 0.975)),
            })
    out_dir = RESULT_DIR / "full"
    write_csv(out_dir / "year_bootstrap_intervals.csv", metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bayesian set model method evidence tables.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--partial-2026", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    parts = load_parts_through(2026) if args.partial_2026 else load_parts()
    if args.partial_2026:
        partial_update_2026(parts, args.draws or 500)
        print(f"Partial 2026 update built in {RESULT_DIR / 'partial_2026'}.")
        return
    if args.mode == "smoke":
        if args.candidate_only:
            boundary_candidate_audit_only(parts, 2006, 2006, args.draws or 120)
        else:
            ablation_and_certificates(parts, 2006, 2006, args.draws or 120)
    else:
        if args.candidate_only:
            boundary_candidate_audit_only(parts, 1995, 2025, args.draws or 500)
        else:
            ablation_and_certificates(parts, 1995, 2025, args.draws or 500)
            bootstrap_intervals(args.bootstrap)
    print(f"Method evidence built in {RESULT_DIR / args.mode}.")


if __name__ == "__main__":
    main()


