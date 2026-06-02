from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "faa_wildlife"
RESULT_DIR = PROJECT_ROOT / "results" / "smoke_tests" / "faa_wildlife"


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

INGESTION_FLAGS = ["ING_ENG1", "ING_ENG2", "ING_ENG3", "ING_ENG4", "INGESTED_OTHER"]

SIZE_WEIGHT = {
    "LARGE": 3.0,
    "MEDIUM": 1.7,
    "SMALL": 1.0,
}

PHASE_WEIGHT = {
    "departure": 1.6,
    "arrival": 1.4,
    "enroute": 1.8,
    "ground": 0.7,
    "unknown": 1.0,
}


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
    if any(x in p for x in ["take-off", "takeoff", "climb", "departure"]):
        return "departure"
    if any(x in p for x in ["approach", "landing", "descent", "arrival"]):
        return "arrival"
    if any(x in p for x in ["en route", "enroute"]):
        return "enroute"
    if any(x in p for x in ["taxi", "parked", "pushback", "local"]):
        return "ground"
    return "unknown"


def mass_weight(ac_mass: str) -> float:
    value = number(ac_mass)
    if value >= 5:
        return 1.5
    if value >= 4:
        return 1.3
    if value >= 3:
        return 1.15
    return 1.0


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RAW_DIR.glob("faa_wildlife_export_*.json")):
        with path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        for raw in payload.get("Result", []):
            rows.append({clean_key(k): v for k, v in raw.items()})

    dedup: dict[str, dict] = {}
    for row in rows:
        key = text(row.get("INDX_NR"))
        if key:
            dedup[key] = row
    return list(dedup.values())


def enrich(row: dict) -> dict:
    year = int(number(row.get("INCIDENT_YEAR")))
    month = int(number(row.get("INCIDENT_MONTH")))
    phase = text(row.get("PHASE_OF_FLIGHT"))
    bucket = phase_bucket(phase)
    size = text(row.get("SIZE")).upper() or "UNKNOWN"
    effect = text(row.get("EFFECT")).upper()
    damage_level = text(row.get("DAMAGE_LEVEL")).upper()
    indicated_damage = text(row.get("INDICATED_DAMAGE")).upper() == "TRUE"
    any_part_damage = any(truthy(row.get(part)) for part in DAMAGE_PARTS)
    cost = number(row.get("COST_REPAIRS_INFL_ADJ")) + number(row.get("COST_OTHER_INFL_ADJ"))
    aos = number(row.get("AOS"))
    injuries = number(row.get("NR_INJURIES"))
    fatalities = number(row.get("NR_FATALITIES"))
    ingested = any(truthy(row.get(flag)) for flag in INGESTION_FLAGS)
    meaningful_effect = effect not in {"", "NONE", "NULL"}
    damaged = indicated_damage or any_part_damage or (damage_level not in {"", "N", "NO DAMAGE"})
    hard_event = damaged or cost > 0 or aos > 0 or meaningful_effect or injuries > 0 or fatalities > 0
    size_w = SIZE_WEIGHT.get(size, 1.0)
    phase_w = PHASE_WEIGHT.get(bucket, 1.0)
    mass_w = mass_weight(text(row.get("AC_MASS")))
    mechanism_weight = size_w * phase_w * mass_w * (1.25 if ingested else 1.0)

    row["_YEAR"] = year
    row["_MONTH"] = month
    row["_PHASE_BUCKET"] = bucket
    row["_SIZE"] = size
    row["_DAMAGED"] = damaged
    row["_HARD_EVENT"] = hard_event
    row["_COST"] = cost
    row["_AOS"] = aos
    row["_INGESTED"] = ingested
    row["_MECH_WEIGHT"] = mechanism_weight
    return row


def group_key(row: dict, mode: str) -> tuple:
    airport = text(row.get("AIRPORT_ID")) or "UNKNOWN"
    month = row["_MONTH"]
    phase = row["_PHASE_BUCKET"]
    size = row["_SIZE"]
    if mode == "airport_month_size_phase":
        return airport, month, size, phase
    species = text(row.get("SPECIES_ID")) or text(row.get("SPECIES")) or "UNKNOWN"
    return airport, month, species, phase


def group_stats(rows: list[dict], mode: str) -> dict[tuple, dict]:
    stats: dict[tuple, dict] = defaultdict(
        lambda: {
            "n": 0,
            "hard": 0,
            "damage": 0,
            "cost": 0.0,
            "aos": 0.0,
            "mech_sum": 0.0,
        }
    )
    for row in rows:
        key = group_key(row, mode)
        item = stats[key]
        item["n"] += 1
        item["hard"] += int(row["_HARD_EVENT"])
        item["damage"] += int(row["_DAMAGED"])
        item["cost"] += row["_COST"]
        item["aos"] += row["_AOS"]
        item["mech_sum"] += row["_MECH_WEIGHT"]
    return stats


def score_groups(stats: dict[tuple, dict], score_name: str) -> dict[tuple, float]:
    total_n = sum(v["n"] for v in stats.values())
    total_hard = sum(v["hard"] for v in stats.values())
    global_rate = total_hard / total_n if total_n else 0.0
    scores: dict[tuple, float] = {}
    for key, value in stats.items():
        n = value["n"]
        hard = value["hard"]
        avg_mech = value["mech_sum"] / n if n else 1.0
        if score_name == "volume":
            score = n
        elif score_name == "past_hard":
            score = hard
        elif score_name == "smoothed_hard_rate":
            score = (hard + 3.0 * global_rate) / (n + 3.0)
        elif score_name == "mechanism_potential":
            score = math.sqrt(n) * avg_mech
        else:
            raise ValueError(score_name)
        scores[key] = score
    return scores


def evaluate_split(rows: list[dict], train_years: set[int], test_years: set[int], mode: str) -> list[dict]:
    train = [r for r in rows if r["_YEAR"] in train_years]
    test = [r for r in rows if r["_YEAR"] in test_years]
    stats = group_stats(train, mode)
    test_stats = group_stats(test, mode)
    total_test_hard = sum(v["hard"] for v in test_stats.values())
    total_test_rows = sum(v["n"] for v in test_stats.values())
    if not stats or total_test_hard == 0:
        return []

    out = []
    score_names = ["volume", "past_hard", "smoothed_hard_rate", "mechanism_potential"]
    for budget_share in [0.05, 0.10, 0.20]:
        k = max(1, int(math.ceil(len(stats) * budget_share)))
        for score_name in score_names:
            scores = score_groups(stats, score_name)
            selected = {key for key, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]}
            selected_test = [v for key, v in test_stats.items() if key in selected]
            captured_hard = sum(v["hard"] for v in selected_test)
            captured_rows = sum(v["n"] for v in selected_test)
            selected_share = captured_rows / total_test_rows if total_test_rows else 0.0
            out.append(
                {
                    "train_years": "-".join(str(y) for y in sorted(train_years)),
                    "test_years": "-".join(str(y) for y in sorted(test_years)),
                    "group_mode": mode,
                    "score": score_name,
                    "budget_share": budget_share,
                    "selected_groups": k,
                    "train_groups": len(stats),
                    "test_rows": total_test_rows,
                    "test_hard_events": total_test_hard,
                    "captured_rows": captured_rows,
                    "captured_hard_events": captured_hard,
                    "hard_capture_rate": captured_hard / total_test_hard,
                    "selected_event_share": selected_share,
                    "hard_rate_in_selected": captured_hard / captured_rows if captured_rows else 0.0,
                    "hard_lift_over_selected_share": (captured_hard / total_test_hard) / selected_share
                    if selected_share
                    else 0.0,
                }
            )
    return out


def escape_md(value) -> str:
    return str(value).replace("|", "\\|")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_counterfactual_examples(rows: list[dict], mode: str) -> list[dict]:
    train = [r for r in rows if r["_YEAR"] in {2023, 2024}]
    test = [r for r in rows if r["_YEAR"] == 2025]
    stats = group_stats(train, mode)
    if not stats:
        return []
    k = max(1, int(math.ceil(len(stats) * 0.10)))
    volume_scores = score_groups(stats, "volume")
    mechanism_scores = score_groups(stats, "mechanism_potential")
    volume_keys = {key for key, _ in sorted(volume_scores.items(), key=lambda x: (-x[1], x[0]))[:k]}
    mechanism_keys = {key for key, _ in sorted(mechanism_scores.items(), key=lambda x: (-x[1], x[0]))[:k]}
    added = mechanism_keys - volume_keys
    test_stats = group_stats(test, mode)
    examples = []
    for key in added:
        value = test_stats.get(key)
        if not value or value["hard"] == 0:
            continue
        examples.append(
            {
                "group_mode": mode,
                "group_key": " | ".join(map(str, key)),
                "train_strikes": stats[key]["n"],
                "train_hard_events": stats[key]["hard"],
                "test_2025_strikes": value["n"],
                "test_2025_hard_events": value["hard"],
                "test_2025_cost": round(value["cost"], 2),
                "test_2025_aos": round(value["aos"], 2),
                "mechanism_score": round(mechanism_scores[key], 4),
                "volume_score": round(volume_scores[key], 4),
            }
        )
    return sorted(examples, key=lambda x: (-x["test_2025_hard_events"], -x["test_2025_cost"]))[:20]


def year_summary(rows: list[dict]) -> list[dict]:
    out = []
    for year in sorted({r["_YEAR"] for r in rows if r["_YEAR"]}):
        part = [r for r in rows if r["_YEAR"] == year]
        out.append(
            {
                "year": year,
                "rows": len(part),
                "damage_records": sum(r["_DAMAGED"] for r in part),
                "hard_events": sum(r["_HARD_EVENT"] for r in part),
                "engine_ingestion_records": sum(r["_INGESTED"] for r in part),
                "inflation_adjusted_cost": round(sum(r["_COST"] for r in part), 2),
                "aircraft_out_of_service_hours": round(sum(r["_AOS"] for r in part), 2),
            }
        )
    return out


def top_cost_events(rows: list[dict], limit: int = 30) -> list[dict]:
    selected = sorted(rows, key=lambda r: (-r["_COST"], -r["_AOS"], -int(r["_HARD_EVENT"])))[:limit]
    out = []
    for row in selected:
        out.append(
            {
                "incident_date": text(row.get("INCIDENT_DATE")),
                "incident_year": row["_YEAR"],
                "airport_id": text(row.get("AIRPORT_ID")),
                "airport": text(row.get("AIRPORT")),
                "aircraft": text(row.get("AIRCRAFT")),
                "aircraft_mass_class": text(row.get("AC_MASS")),
                "phase_of_flight": text(row.get("PHASE_OF_FLIGHT")),
                "phase_bucket": row["_PHASE_BUCKET"],
                "species": text(row.get("SPECIES")),
                "size": row["_SIZE"],
                "damage_level": text(row.get("DAMAGE_LEVEL")),
                "effect": text(row.get("EFFECT")),
                "cost": round(row["_COST"], 2),
                "aos_hours": round(row["_AOS"], 2),
                "engine_ingested": row["_INGESTED"],
            }
        )
    return out


def build_report(rows: list[dict], metrics: list[dict], examples: list[dict]) -> str:
    if not rows:
        return "# FAA Wildlife Strike Database smoke test\n\nNo raw rows were found.\n"

    years = sorted({r["_YEAR"] for r in rows if r["_YEAR"]})
    by_year = year_summary(rows)
    metric_best = sorted(
        metrics,
        key=lambda x: (x["test_years"], x["group_mode"], x["budget_share"], -x["hard_capture_rate"]),
    )
    hard_events = [r["_HARD_EVENT"] for r in rows]
    cost_values = [r["_COST"] for r in rows if r["_COST"] > 0]
    aos_values = [r["_AOS"] for r in rows if r["_AOS"] > 0]

    lines = [
        "# FAA Wildlife Strike Database smoke test",
        "",
        "## Data window",
        "",
        f"- Public export records loaded: {len(rows):,}.",
        f"- Year range: {min(years)}-{max(years)}.",
        "- Hard outcomes combine reported component damage, repair cost, other cost, aircraft out of service, flight effect, injury, or fatality.",
        "- Screening scores use fields available before detailed evidence inspection: airport, month, animal size, species, phase of flight, aircraft mass class, and engine-ingestion flags.",
        "",
        "## Year summary",
        "",
        "| Year | Records | Damage records | Hard outcomes | Engine ingestion | Cost | AOS hours |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in by_year:
        lines.append(
            f"| {item['year']} | {item['rows']:,} | {item['damage_records']:,} | {item['hard_events']:,} | "
            f"{item['engine_ingestion_records']:,} | {item['inflation_adjusted_cost']:,.0f} | "
            f"{item['aircraft_out_of_service_hours']:,.1f} |"
        )

    lines.extend(
        [
            "",
            "## Forward split check",
            "",
            "| Train years | Test years | Grouping | Capacity | Rule | Captured hard outcomes | Capture | Selected event share | Lift |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in metric_best:
        if item["budget_share"] not in {0.05, 0.10}:
            continue
        lines.append(
            f"| {item['train_years']} | {item['test_years']} | {item['group_mode']} | "
            f"{item['budget_share']:.0%} | {item['score']} | "
            f"{item['captured_hard_events']:,}/{item['test_hard_events']:,} | "
            f"{item['hard_capture_rate']:.1%} | {item['selected_event_share']:.1%} | "
            f"{item['hard_lift_over_selected_share']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Frequency-missed examples",
            "",
            "The rows below enter the top 10% under the mechanism score and fall outside the top 10% under the frequency rule.",
            "",
            "| Grouping | Group | Train strikes | Train hard outcomes | 2025 strikes | 2025 hard outcomes | 2025 cost | 2025 AOS hours |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in examples[:12]:
        lines.append(
            f"| {item['group_mode']} | {escape_md(item['group_key'])} | "
            f"{item['train_strikes']} | {item['train_hard_events']} | "
            f"{item['test_2025_strikes']} | {item['test_2025_hard_events']} | "
            f"{item['test_2025_cost']:,.0f} | {item['test_2025_aos']:,.1f} |"
        )

    lines.extend(
        [
            "",
            "## Smoke-test interpretation",
            "",
            f"- Full-window hard-outcome rate: {sum(hard_events) / len(hard_events):.1%}.",
            f"- Cost-bearing records: {len(cost_values):,}; median cost: {median(cost_values) if cost_values else 0:,.0f}.",
            f"- AOS-bearing records: {len(aos_values):,}; median AOS hours: {median(aos_values) if aos_values else 0:,.1f}.",
            "- A full run is useful when damage-potential rules concentrate future hard outcomes above frequency-only screening in the smoke window.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [enrich(row) for row in load_rows()]
    rows = [r for r in rows if 2023 <= r["_YEAR"] <= 2026]

    metrics: list[dict] = []
    modes = ["airport_month_size_phase", "airport_month_species_phase"]
    splits = [
        ({2023, 2024}, {2025}),
        ({2024, 2025}, {2026}),
    ]
    for mode in modes:
        for train_years, test_years in splits:
            metrics.extend(evaluate_split(rows, train_years, test_years, mode))

    examples: list[dict] = []
    for mode in modes:
        examples.extend(top_counterfactual_examples(rows, mode))
    examples = sorted(examples, key=lambda x: (-x["test_2025_hard_events"], -x["test_2025_cost"]))[:30]

    write_csv(RESULT_DIR / "wildlife_smoke_metrics.csv", metrics)
    write_csv(RESULT_DIR / "wildlife_counterfactual_examples.csv", examples)
    write_csv(RESULT_DIR / "wildlife_year_summary.csv", year_summary(rows))
    write_csv(RESULT_DIR / "wildlife_top_cost_events.csv", top_cost_events(rows))
    report = build_report(rows, metrics, examples)
    (RESULT_DIR / "smoke_test_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()


