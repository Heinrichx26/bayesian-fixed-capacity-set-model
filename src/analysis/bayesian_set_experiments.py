from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

try:
    import tabm
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    tabm = None
    torch = None
    DataLoader = None
    TensorDataset = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayesian_set_core import (  # noqa: E402
    FULL_LEVEL,
    TARGET,
    build_test_cells,
    bayesian_set_cell_scores,
    load_parts,
    score_from_cell_values,
    score_hierarchical_mean,
    tuple_key,
)
from screening_comparison_rules import score_records, train_scores  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "experiments" / "bayesian_set_model"

CAPACITIES = [0.01, 0.025, 0.05, 0.10, 0.20]
MAIN_CAPACITIES = [0.05, 0.10]
FIELD_FEATURES = ["component", "phase_bucket", "size", "aircraft_mass_class", "species_id"]
COMPACT_FEATURES = ["component", "phase_bucket", "size", "aircraft_mass_class"]
DOMAIN_RULES = {
    "frequency": "component_phase_size_mass_frequency",
    "component_only": "component_only_rate",
    "species_size_component": "species_size_component_hazard_rate",
    "direct_full_cell": "component_phase_size_mass_rate",
}


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


def evaluate_scored(
    scored: list[tuple[float, dict]],
    capacity: float,
    rule: str,
    train_start: int,
    train_end: int,
    test_year: int,
) -> dict:
    selected_count = max(1, math.ceil(len(scored) * capacity))
    selected = [row for _, row in scored[:selected_count]]
    test_rows = [row for _, row in scored]
    target_records = sum(int(bool(row[TARGET])) for row in test_rows)
    captured = sum(int(bool(row[TARGET])) for row in selected)
    overall_rate = target_records / len(test_rows) if test_rows else 0.0
    selected_rate = captured / selected_count if selected_count else 0.0

    all_events = event_totals(test_rows)
    selected_event_ids = {str(row["event_id"]) for row in selected}
    total_cost = sum(value["cost"] for value in all_events.values())
    total_aos = sum(value["aos"] for value in all_events.values())
    selected_cost = sum(all_events[event_id]["cost"] for event_id in selected_event_ids if event_id in all_events)
    selected_aos = sum(all_events[event_id]["aos"] for event_id in selected_event_ids if event_id in all_events)
    selected_hard = sum(int(bool(row["event_hard"])) for row in selected)
    hard_records = sum(int(bool(row["event_hard"])) for row in test_rows)

    return {
        "train_start": train_start,
        "train_end": train_end,
        "test_year": test_year,
        "rule": rule,
        "capacity": capacity,
        "test_component_units": len(test_rows),
        "damage_units": target_records,
        "selected_units": selected_count,
        "selected_damage_units": captured,
        "damage_capture": captured / target_records if target_records else 0.0,
        "selected_hit_rate": selected_rate,
        "overall_damage_rate": overall_rate,
        "damage_lift": selected_rate / overall_rate if overall_rate else 0.0,
        "hard_consequence_units": hard_records,
        "selected_hard_consequence_units": selected_hard,
        "hard_consequence_capture": selected_hard / hard_records if hard_records else 0.0,
        "total_event_cost": total_cost,
        "selected_event_cost": selected_cost,
        "cost_capture": selected_cost / total_cost if total_cost else 0.0,
        "total_event_aos": total_aos,
        "selected_event_aos": selected_aos,
        "aos_capture": selected_aos / total_aos if total_aos else 0.0,
    }


def field_dict(row: dict, include_species: bool = True) -> dict[str, int]:
    component = str(row.get("component", "UNKNOWN") or "UNKNOWN")
    phase = str(row.get("phase_bucket", "UNKNOWN") or "UNKNOWN")
    size = str(row.get("size", "UNKNOWN") or "UNKNOWN")
    mass = str(row.get("aircraft_mass_class", "UNKNOWN") or "UNKNOWN")
    species = str(row.get("species_id", "UNKNOWN") or "UNKNOWN")
    item = {
        f"component={component}": 1,
        f"phase={phase}": 1,
        f"size={size}": 1,
        f"mass={mass}": 1,
        f"component_phase={component}|{phase}": 1,
        f"component_size={component}|{size}": 1,
        f"phase_size={phase}|{size}": 1,
        f"component_phase_size={component}|{phase}|{size}": 1,
        f"component_phase_size_mass={component}|{phase}|{size}|{mass}": 1,
    }
    if include_species:
        item.update({
            f"species={species}": 1,
            f"species_phase_size={species}|{phase}|{size}": 1,
            f"species_size_component={species}|{size}|{component}": 1,
        })
    return item


def cat_frame(rows: list[dict], features: list[str] | None = None) -> pd.DataFrame:
    columns = features or FIELD_FEATURES
    return pd.DataFrame(
        [{feature: str(row.get(feature, "UNKNOWN") or "UNKNOWN") for feature in columns} for row in rows],
        columns=columns,
    )


def train_custom_posteriors(train: list[dict], levels: list[list[str]], prior_weight: float = 10.0) -> tuple[dict[tuple, dict[tuple, dict]], dict]:
    global_n = len(train)
    global_y = sum(int(bool(row[TARGET])) for row in train)
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
    for level in levels:
        stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "y": 0})
        for row in train:
            key = tuple(row.get(column, "") for column in level)
            stats[key]["n"] += 1
            stats[key]["y"] += int(bool(row[TARGET]))
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


def score_bayesian_species_child(train: list[dict], test: list[dict]) -> tuple[list[tuple[float, dict]], np.ndarray]:
    levels = [
        ["component"],
        ["component", "phase_bucket"],
        ["component", "phase_bucket", "size"],
        ["component", "phase_bucket", "size", "aircraft_mass_class"],
        ["component", "phase_bucket", "size", "aircraft_mass_class", "species_id"],
    ]
    tables, global_stats = train_custom_posteriors(train, levels)
    scored: list[tuple[float, dict]] = []
    probabilities: list[float] = []
    for row in test:
        stats = None
        for level in reversed(levels):
            key = tuple(row.get(column, "") for column in level)
            stats = tables.get(tuple(level), {}).get(key)
            if stats is not None:
                break
        if stats is None:
            stats = global_stats
        value = float(stats["mean"])
        scored.append((value, row))
        probabilities.append(value)
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    return scored, np.clip(np.array(probabilities, dtype=float), 1e-6, 1 - 1e-6)


def tabm_arrays(train: list[dict], test: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    maps: list[dict[str, int]] = []
    for feature in FIELD_FEATURES:
        values = sorted({str(row.get(feature, "UNKNOWN") or "UNKNOWN") for row in train})
        maps.append({value: idx + 1 for idx, value in enumerate(values)})

    def encode(rows: list[dict]) -> np.ndarray:
        encoded = []
        for row in rows:
            encoded.append([
                maps[idx].get(str(row.get(feature, "UNKNOWN") or "UNKNOWN"), 0)
                for idx, feature in enumerate(FIELD_FEATURES)
            ])
        return np.array(encoded, dtype=np.int64)

    cardinalities = [len(item) + 1 for item in maps]
    return encode(train), encode(test), cardinalities


def as_scored(probabilities: np.ndarray, test: list[dict]) -> list[tuple[float, dict]]:
    scored = [(float(score), row) for score, row in zip(probabilities, test)]
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    return scored


def score_logistic(train: list[dict], test: list[dict], include_species: bool = True) -> tuple[list[tuple[float, dict]], np.ndarray]:
    y = np.array([int(bool(row[TARGET])) for row in train], dtype=int)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(field_dict(row, include_species=include_species) for row in train)
    x_test = vectorizer.transform(field_dict(row, include_species=include_species) for row in test)
    model = LogisticRegression(
        C=0.5,
        penalty="l2",
        solver="liblinear",
        max_iter=220,
        class_weight="balanced",
        random_state=20260528,
    )
    model.fit(x_train, y)
    probabilities = model.predict_proba(x_test)[:, 1]
    return as_scored(probabilities, test), probabilities


def score_random_forest(train: list[dict], test: list[dict], include_species: bool = True) -> tuple[list[tuple[float, dict]], np.ndarray]:
    y = np.array([int(bool(row[TARGET])) for row in train], dtype=int)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(field_dict(row, include_species=include_species) for row in train)
    x_test = vectorizer.transform(field_dict(row, include_species=include_species) for row in test)
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=20260528,
    )
    model.fit(x_train, y)
    probabilities = model.predict_proba(x_test)[:, 1]
    return as_scored(probabilities, test), probabilities


def score_xgboost(train: list[dict], test: list[dict], include_species: bool = True) -> tuple[list[tuple[float, dict]], np.ndarray]:
    y = np.array([int(bool(row[TARGET])) for row in train], dtype=int)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(field_dict(row, include_species=include_species) for row in train)
    x_test = vectorizer.transform(field_dict(row, include_species=include_species) for row in test)
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        reg_lambda=6.0,
        scale_pos_weight=negatives / positives,
        tree_method="hist",
        n_jobs=-1,
        random_state=20260528,
    )
    model.fit(x_train, y)
    probabilities = model.predict_proba(x_test)[:, 1]
    return as_scored(probabilities, test), probabilities


def score_lightgbm(train: list[dict], test: list[dict], include_species: bool = True) -> tuple[list[tuple[float, dict]], np.ndarray]:
    y = np.array([int(bool(row[TARGET])) for row in train], dtype=int)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(field_dict(row, include_species=include_species) for row in train)
    x_test = vectorizer.transform(field_dict(row, include_species=include_species) for row in test)
    model = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        class_weight="balanced",
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=6.0,
        n_jobs=-1,
        random_state=20260528,
        verbose=-1,
    )
    model.fit(x_train, y)
    probabilities = model.predict_proba(x_test)[:, 1]
    return as_scored(probabilities, test), probabilities


def score_catboost(train: list[dict], test: list[dict], features: list[str] | None = None) -> tuple[list[tuple[float, dict]], np.ndarray]:
    columns = features or FIELD_FEATURES
    y = np.array([int(bool(row[TARGET])) for row in train], dtype=int)
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    model = CatBoostClassifier(
        iterations=220,
        depth=6,
        learning_rate=0.045,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=8.0,
        random_seed=20260528,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
        scale_pos_weight=negatives / positives,
    )
    train_pool = Pool(cat_frame(train, columns), label=y, cat_features=columns)
    test_pool = Pool(cat_frame(test, columns), cat_features=columns)
    model.fit(train_pool)
    probabilities = model.predict_proba(test_pool)[:, 1]
    return as_scored(probabilities, test), probabilities


def score_tabm(train: list[dict], test: list[dict]) -> tuple[list[tuple[float, dict]], np.ndarray]:
    if tabm is None or torch is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("TabM comparison requires tabm and torch to be installed.")
    y = np.array([int(bool(row[TARGET])) for row in train], dtype=np.float32)
    x_train, x_test, cardinalities = tabm_arrays(train, test)

    rng = np.random.default_rng(20260528)
    indices = np.arange(len(train))
    rng.shuffle(indices)
    val_size = max(500, int(0.12 * len(indices)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = TensorDataset(torch.tensor(x_train[train_indices]), torch.tensor(y[train_indices]))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_x = torch.tensor(x_train[val_indices])
    val_y = torch.tensor(y[val_indices])

    model = tabm.TabM(
        n_num_features=0,
        cat_cardinalities=cardinalities,
        d_out=1,
        n_blocks=2,
        d_block=64,
        dropout=0.08,
        k=8,
        arch_type="tabm-mini",
        start_scaling_init="random-signs",
    )
    positives = max(1.0, float(y[train_indices].sum()))
    negatives = max(1.0, float(len(train_indices) - positives))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    patience = 3
    stale = 0
    for _ in range(14):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(None, batch_x).squeeze(-1).mean(1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(None, val_x).squeeze(-1).mean(1)
            val_loss = float(loss_fn(val_logits, val_y).item())
        if val_loss + 1e-5 < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probabilities: list[float] = []
    test_tensor = torch.tensor(x_test)
    with torch.no_grad():
        for start in range(0, len(test), 4096):
            logits = model(None, test_tensor[start:start + 4096]).squeeze(-1).mean(1)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    probabilities_array = np.array(probabilities, dtype=float)
    return as_scored(probabilities_array, test), probabilities_array


def probability_array(scored: list[tuple[float, dict]], test_order: list[dict]) -> np.ndarray:
    score_by_row = {id(row): float(score) for score, row in scored}
    probabilities = [score_by_row[id(row)] for row in test_order]
    return np.clip(np.array(probabilities, dtype=float), 1e-6, 1 - 1e-6)


def direct_smoothed_probability(train: list[dict], test: list[dict], alpha: float = 10.0) -> np.ndarray:
    keys = ["component", "phase_bucket", "size", "aircraft_mass_class"]
    stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "y": 0})
    total_y = sum(int(bool(row[TARGET])) for row in train)
    global_rate = total_y / len(train) if train else 0.0
    for row in train:
        key = tuple(row.get(field, "") for field in keys)
        stats[key]["n"] += 1
        stats[key]["y"] += int(bool(row[TARGET]))
    values = {
        key: (item["y"] + alpha * global_rate) / (item["n"] + alpha)
        for key, item in stats.items()
    }
    probabilities = [
        float(values.get(tuple(row.get(field, "") for field in keys), global_rate))
        for row in test
    ]
    return np.clip(np.array(probabilities, dtype=float), 1e-6, 1 - 1e-6)


def normalize_scores(scored: list[tuple[float, dict]]) -> tuple[list[tuple[float, dict]], np.ndarray]:
    scores = np.array([score for score, _ in scored], dtype=float)
    if len(scores) == 0:
        return scored, np.array([])
    lo = float(scores.min())
    hi = float(scores.max())
    if hi <= lo:
        probabilities = np.full(len(scores), float(np.clip(hi, 0.0, 1.0)))
    else:
        probabilities = (scores - lo) / (hi - lo)
    return scored, probabilities


def score_burden_rule(train: list[dict], test: list[dict], burden: str) -> tuple[list[tuple[float, dict]], np.ndarray]:
    keys = ["component", "phase_bucket", "size", "aircraft_mass_class"]
    stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "value": 0.0})
    global_value = 0.0
    for row in train:
        key = tuple(row.get(column, "") for column in keys)
        value = float(row[burden])
        stats[key]["n"] += 1
        stats[key]["value"] += value
        global_value += value
    global_mean = global_value / len(train) if train else 0.0
    values = {
        key: (item["value"] + 10.0 * global_mean) / (item["n"] + 10.0)
        for key, item in stats.items()
    }
    scored = []
    for row in test:
        key = tuple(row.get(column, "") for column in keys)
        scored.append((float(values.get(key, global_mean)), row))
    scored.sort(key=lambda item: (-item[0], item[1]["event_id"], item[1]["component"]))
    return normalize_scores(scored)


def calibration_row(rule: str, y_true: list[int], probability: list[float], test_years: set[int]) -> dict:
    y = np.array(y_true, dtype=int)
    p = np.clip(np.array(probability, dtype=float), 1e-6, 1 - 1e-6)
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    ece = 0.0
    for item in bins:
        if len(item) == 0:
            continue
        weight = len(item) / len(p)
        ece += weight * abs(float(y[item].mean()) - float(p[item].mean()))
    return {
        "rule": rule,
        "test_years": len(test_years),
        "records": len(y_true),
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "expected_calibration_error": ece,
        "mean_predicted_rate": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def aggregate_selection(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "years": set(),
        "test_component_units": 0,
        "damage_units": 0,
        "selected_units": 0,
        "selected_damage_units": 0,
        "hard_consequence_units": 0,
        "selected_hard_consequence_units": 0,
        "total_event_cost": 0.0,
        "selected_event_cost": 0.0,
        "total_event_aos": 0.0,
        "selected_event_aos": 0.0,
        "annual_lifts": [],
    })
    for row in rows:
        key = (row["rule"], row["capacity"])
        item = groups[key]
        item["years"].add(row["test_year"])
        for field in [
            "test_component_units",
            "damage_units",
            "selected_units",
            "selected_damage_units",
            "hard_consequence_units",
            "selected_hard_consequence_units",
        ]:
            item[field] += row[field]
        for field in ["total_event_cost", "selected_event_cost", "total_event_aos", "selected_event_aos"]:
            item[field] += row[field]
        item["annual_lifts"].append(row["damage_lift"])
    out = []
    for (rule, capacity), item in groups.items():
        overall_rate = item["damage_units"] / item["test_component_units"] if item["test_component_units"] else 0.0
        hit_rate = item["selected_damage_units"] / item["selected_units"] if item["selected_units"] else 0.0
        annual = np.array(item["annual_lifts"], dtype=float)
        out.append({
            "rule": rule,
            "capacity": capacity,
            "test_years": len(item["years"]),
            "test_component_units": item["test_component_units"],
            "damage_units": item["damage_units"],
            "selected_units": item["selected_units"],
            "selected_damage_units": item["selected_damage_units"],
            "damage_capture": item["selected_damage_units"] / item["damage_units"] if item["damage_units"] else 0.0,
            "selected_hit_rate": hit_rate,
            "overall_damage_rate": overall_rate,
            "damage_lift": hit_rate / overall_rate if overall_rate else 0.0,
            "hard_consequence_capture": item["selected_hard_consequence_units"] / item["hard_consequence_units"] if item["hard_consequence_units"] else 0.0,
            "cost_capture": item["selected_event_cost"] / item["total_event_cost"] if item["total_event_cost"] else 0.0,
            "aos_capture": item["selected_event_aos"] / item["total_event_aos"] if item["total_event_aos"] else 0.0,
            "annual_lift_mean": float(annual.mean()) if len(annual) else 0.0,
            "annual_lift_p05": float(np.quantile(annual, 0.05)) if len(annual) else 0.0,
            "annual_lift_min": float(annual.min()) if len(annual) else 0.0,
            "annual_lift_sd": float(annual.std(ddof=1)) if len(annual) > 1 else 0.0,
        })
    return sorted(out, key=lambda row: (row["capacity"], -row["damage_lift"], row["rule"]))


def cell_certificate_rows(values: dict[tuple, dict], limit_each: int = 6) -> list[dict]:
    candidates = [
        row for row in values.values()
        if int(row["support"]) >= 10 and float(row["capacity_membership_probability"]) > 0.0
    ]
    high = sorted(candidates, key=lambda row: (-row["capacity_membership_probability"], -row["posterior_mean"], row["key"]))[:limit_each]
    boundary = sorted(
        [row for row in candidates if 0.45 <= float(row["capacity_membership_probability"]) <= 0.97],
        key=lambda row: (abs(float(row["capacity_membership_probability"]) - 0.75), -row["posterior_mean"], row["key"]),
    )[:limit_each]
    out = []
    for group, rows in [("High-certainty", high), ("Boundary", boundary)]:
        for rank, row in enumerate(rows, 1):
            key = row["key"]
            out.append({
                "certificate_group": group,
                "rank_within_group": rank,
                "component": key[0],
                "phase": key[1],
                "size": key[2],
                "aircraft_mass_class": key[3],
                "posterior_mean": row["posterior_mean"],
                "capacity_membership_probability": row["capacity_membership_probability"],
                "capacity_boundary_excess": row["capacity_boundary_excess"],
                "historical_support": row["support"],
                "historical_damage": row["training_damage"],
                "fallback_level": row["fallback_level"],
                "future_units": row["count"],
                "future_damage": row["future_damage"],
                "future_cost": row["future_cost"],
                "future_aos": row["future_aos"],
            })
    return out


def run_experiment(start_year: int, end_year: int, draws: int, capacities: list[float], include_ml: bool) -> None:
    parts = load_parts()
    out_dir = RESULT_ROOT / ("smoke" if start_year == end_year else "full")
    yearly: list[dict] = []
    calibration: dict[str, dict] = defaultdict(lambda: {"y": [], "p": [], "years": set()})
    certificate_rows: list[dict] = []

    for test_year in range(start_year, end_year + 1):
        train_start = test_year - 5
        train_end = test_year - 1
        train = [row for row in parts if train_start <= int(row["year"]) <= train_end]
        test = [row for row in parts if int(row["year"]) == test_year]
        if not train or not test:
            continue
        y_test = [int(bool(row[TARGET])) for row in test]
        cells, _ = build_test_cells(train, test)
        certificate_values = {
            capacity: bayesian_set_cell_scores(cells, capacity, draws, seed=20260528 + test_year * 1000 + int(capacity * 10000))
            for capacity in capacities
        }

        direct_scored = score_records(train, test, DOMAIN_RULES["direct_full_cell"], TARGET)
        hierarchical_scored = score_hierarchical_mean(train, test)
        component_scored = score_records(train, test, DOMAIN_RULES["component_only"], TARGET)
        species_component_scored = score_records(train, test, DOMAIN_RULES["species_size_component"], TARGET)
        frequency_scored = score_records(train, test, DOMAIN_RULES["frequency"], TARGET)
        cost_scored, _ = score_burden_rule(train, test, "cost")
        aos_scored, _ = score_burden_rule(train, test, "aos")
        scored_common: dict[str, tuple[list[tuple[float, dict]], np.ndarray | None]] = {
            "direct_full_cell": (direct_scored, direct_smoothed_probability(train, test)),
            "hierarchical_eb_execution": (hierarchical_scored, probability_array(hierarchical_scored, test)),
            "bayesian_species_child": score_bayesian_species_child(train, test),
            "frequency": (frequency_scored, None),
            "component_only": (component_scored, probability_array(component_scored, test)),
            "species_size_component": (species_component_scored, probability_array(species_component_scored, test)),
            "cost_burden_rule": (cost_scored, None),
            "aos_burden_rule": (aos_scored, None),
        }
        if include_ml:
            scored_common["ridge_logistic"] = score_logistic(train, test)
            scored_common["random_forest"] = score_random_forest(train, test)
            scored_common["xgboost"] = score_xgboost(train, test)
            scored_common["lightgbm"] = score_lightgbm(train, test)
            scored_common["catboost"] = score_catboost(train, test)
            scored_common["compact_lightgbm"] = score_lightgbm(train, test, include_species=False)
            scored_common["compact_catboost"] = score_catboost(train, test, features=COMPACT_FEATURES)
            scored_common["tabm"] = score_tabm(train, test)

        for capacity in capacities:
            values = certificate_values[capacity]
            scored_capacity = {
                "bayesian_posterior_mean": score_from_cell_values(test, values, "posterior_mean"),
                "membership_probability": score_from_cell_values(test, values, "capacity_membership_probability"),
                "boundary_excess": score_from_cell_values(test, values, "capacity_boundary_excess"),
                "membership_x_boundary_excess": score_from_cell_values(test, values, "membership_x_boundary_excess"),
            }
            for rule, scored in scored_capacity.items():
                yearly.append(evaluate_scored(scored, capacity, rule, train_start, train_end, test_year))
            if test_year == end_year and abs(capacity - 0.05) < 1e-9:
                certificate_rows = cell_certificate_rows(values)

        for rule, (scored, probabilities) in scored_common.items():
            for capacity in capacities:
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
    write_csv(out_dir / "set_model_selection_yearly.csv", yearly)
    write_csv(out_dir / "set_model_selection_aggregate.csv", aggregate)
    write_csv(out_dir / "set_model_probability_calibration.csv", sorted(calibration_rows, key=lambda row: row["brier_score"]))
    write_csv(out_dir / f"set_model_certificate_cells_{end_year}.csv", certificate_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMM Bayesian set model experiments.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--skip-ml", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_experiment(2006, 2006, args.draws or 200, MAIN_CAPACITIES, include_ml=not args.skip_ml)
    else:
        run_experiment(1995, 2025, args.draws or 500, CAPACITIES, include_ml=not args.skip_ml)


if __name__ == "__main__":
    main()




