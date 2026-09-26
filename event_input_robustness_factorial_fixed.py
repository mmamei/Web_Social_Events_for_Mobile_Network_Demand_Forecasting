#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness analysis for unreliable event information.

The script perturbs only the exogenous event-information layer. The observed
mobile-traffic series is never edited.

Three perturbations are supported:
  1. Cancellations: remove an absolute number of real event records from the event feed.
  2. False positives: inject an absolute number of event records in periods with no real event.
  3. Temporal shifts: move a fixed subset of real events by +/- N hours.

After perturbing the canonical event calendar, all derived event features are
regenerated. Non-event exogenous variables already present in the baseline
feature CSV (for example Google Trends) are preserved unchanged.

Optionally, the script can evaluate previously saved Keras models. In that
mode, the models are NOT retrained: saved clean models are evaluated on the
same held-out traffic targets while only the event inputs are changed.

Typical workflow
----------------
1) Train the clean event-aware model and save models/scalers::

    python ml_exps_lstm_optuna_custom_loss_v3_eventaware.py \
        --event-features-file outputs/events/event_features_hourly.csv \
        --require-event-features --save-models ...

2) Generate the 27 event-input perturbation scenarios::

    python event_input_robustness_factorial_fixed.py \
        --canonical-events outputs/events/events_canonical.csv \
        --baseline-features outputs/events/event_features_hourly.csv \
        --output-dir outputs/event_robustness

3) Generate scenarios and evaluate the saved clean models::

    python event_input_robustness_factorial_fixed.py \
        --canonical-events outputs/events/events_canonical.csv \
        --baseline-features outputs/events/event_features_hourly.csv \
        --experiment-dir outputs/custom_event_1000 \
        --evaluate-models \
        --output-dir outputs/event_robustness

The default grid is 3 x 3 x 3 = 27 scenarios:
  cancelled event records: 1, 2, 3
  false-positive event records: 1, 2, 3
  temporal shift magnitude: 0 h, 1 h, 2 h

Within each replicate, perturbations are nested across severity levels. For
example, the event cancelled in C=1 is also among those cancelled in C=2 and
C=3; likewise, FP=1 is a subset of FP=2 and FP=3. The events shifted by 1 h
are the same events shifted by 2 h. This avoids the discretization artifacts
that arise when percentage levels map to the same integer count in a small
evaluation window and reduces sampling noise when inspecting severity trends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("event_input_robustness")
PANEL_META_COLS = ("date", "hour", "far_edge", "value")
EXOG_PREFIX = "exog__"
CAP_LEVELS = (0.70, 0.80, 0.90)


# ---------------------------------------------------------------------------
# Canonical event representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    event_id: str
    date: pd.Timestamp
    event_type: str
    title: str = ""
    start_hour: int | None = None
    duration_hours: int = 3
    venue: str = ""
    latitude: float | None = None
    longitude: float | None = None
    intensity: float = 1.0
    base_relevance: float = 1.0
    source: str = "manual"
    synthetic: bool = False

    def validate(self) -> None:
        if pd.isna(self.date):
            raise ValueError(f"{self.event_id}: invalid event date")
        if self.start_hour is not None and not 0 <= int(self.start_hour) <= 23:
            raise ValueError(f"{self.event_id}: start_hour must be in 0..23")
        if int(self.duration_hours) < 1:
            raise ValueError(f"{self.event_id}: duration_hours must be >= 1")
        if not 0.0 <= float(self.intensity) <= 1.0:
            raise ValueError(f"{self.event_id}: intensity must be in [0,1]")
        if not 0.0 <= float(self.base_relevance) <= 1.0:
            raise ValueError(f"{self.event_id}: base_relevance must be in [0,1]")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError(
                f"{self.event_id}: latitude and longitude must both be present or absent"
            )


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return value or "other"


def _none_if_na(value: Any) -> Any:
    return None if pd.isna(value) else value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def load_canonical_events(path: Path) -> list[Event]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"date", "event_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    events: list[Event] = []
    for index, row in frame.iterrows():
        event_id = str(row.get("event_id", "")).strip()
        if not event_id or event_id.lower() == "nan":
            event_id = f"E{index:05d}"
        start_hour_raw = _none_if_na(row.get("start_hour", None))
        start_hour = None if start_hour_raw is None else int(float(start_hour_raw))
        lat_raw = _none_if_na(row.get("latitude", None))
        lon_raw = _none_if_na(row.get("longitude", None))
        event = Event(
            event_id=event_id,
            date=pd.Timestamp(row["date"]).normalize(),
            event_type=str(row["event_type"]),
            title="" if pd.isna(row.get("title", "")) else str(row.get("title", "")),
            start_hour=start_hour,
            duration_hours=int(float(row.get("duration_hours", 3))),
            venue="" if pd.isna(row.get("venue", "")) else str(row.get("venue", "")),
            latitude=None if lat_raw is None else float(lat_raw),
            longitude=None if lon_raw is None else float(lon_raw),
            intensity=float(row.get("intensity", 1.0)),
            base_relevance=float(row.get("base_relevance", 1.0)),
            source="manual" if pd.isna(row.get("source", "manual")) else str(row.get("source", "manual")),
            synthetic=_to_bool(row.get("synthetic", False)),
        )
        event.validate()
        events.append(event)

    if not events:
        raise ValueError(f"{path}: no canonical events found")
    ids = [event.event_id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: event_id values must be unique")
    return events


def event_start_end(
    event: Event,
    *,
    date_only_mode: str,
    default_start_hour: int,
    default_duration_hours: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if event.start_hour is None:
        if date_only_mode == "full_day":
            start = event.date.normalize()
            end = start + pd.Timedelta(days=1)
            return start, end
        if date_only_mode != "default_time":
            raise ValueError("date_only_mode must be full_day or default_time")
        start_hour = int(default_start_hour)
        duration = int(default_duration_hours)
    else:
        start_hour = int(event.start_hour)
        duration = int(event.duration_hours)
    start = event.date.normalize() + pd.Timedelta(hours=start_hour)
    end = start + pd.Timedelta(hours=max(1, duration))
    return start, end


def make_explicit_events(
    events: Sequence[Event],
    *,
    date_only_mode: str,
    default_start_hour: int,
    default_duration_hours: int,
) -> list[Event]:
    """Convert every event to an explicit start hour and duration.

    This makes +/- hour perturbations well-defined even when the original
    calendar contains date-only events.
    """
    explicit: list[Event] = []
    for event in events:
        start, end = event_start_end(
            event,
            date_only_mode=date_only_mode,
            default_start_hour=default_start_hour,
            default_duration_hours=default_duration_hours,
        )
        duration = max(1, int(round((end - start).total_seconds() / 3600.0)))
        explicit.append(
            replace(
                event,
                date=start.normalize(),
                start_hour=int(start.hour),
                duration_hours=duration,
            )
        )
    return explicit


def explicit_start(event: Event) -> pd.Timestamp:
    if event.start_hour is None:
        raise ValueError("explicit_start requires explicit events")
    return event.date.normalize() + pd.Timedelta(hours=int(event.start_hour))


def explicit_end(event: Event) -> pd.Timestamp:
    return explicit_start(event) + pd.Timedelta(hours=int(event.duration_hours))


def shift_event(event: Event, hours: int) -> Event:
    start = explicit_start(event) + pd.Timedelta(hours=int(hours))
    return replace(event, date=start.normalize(), start_hour=int(start.hour))


def events_to_frame(events: Sequence[Event]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        row = asdict(event)
        row["date"] = event.date.strftime("%Y-%m-%d")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event-feature generation
# ---------------------------------------------------------------------------


def load_node_locations(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"far_edge", "lat", "lon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing node-location columns {sorted(missing)}")
    out = frame.copy()
    out["far_edge"] = out["far_edge"].astype(str)
    out["lat"] = pd.to_numeric(out["lat"], errors="raise")
    out["lon"] = pd.to_numeric(out["lon"], errors="raise")
    if "cluster" in out.columns:
        out["cluster"] = pd.to_numeric(out["cluster"], errors="raise").astype(int)
    key_cols = (["cluster"] if "cluster" in out.columns else []) + ["far_edge"]
    return out.drop_duplicates(subset=key_cols).reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def spatial_relevance(
    event: Event,
    node_lat: float | None,
    node_lon: float | None,
    sigma_km: float,
) -> float:
    base = float(event.base_relevance)
    if event.latitude is None or event.longitude is None or node_lat is None or node_lon is None:
        return base
    if sigma_km <= 0:
        raise ValueError("spatial_sigma_km must be > 0")
    distance = haversine_km(
        float(event.latitude),
        float(event.longitude),
        float(node_lat),
        float(node_lon),
    )
    return float(np.clip(base * math.exp(-0.5 * (distance / sigma_km) ** 2), 0.0, 1.0))


def context_values(
    ts: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lead_hours: int,
    lag_hours: int,
) -> tuple[float, float, float, int]:
    if start <= ts < end:
        return 0.0, 0.0, 1.0, 1
    if ts < start and lead_hours > 0:
        gap = (start - ts).total_seconds() / 3600.0
        if 0.0 < gap <= lead_hours:
            before = max(0.0, 1.0 - (gap - 1.0) / max(1.0, float(lead_hours)))
            return before, 0.0, before, 0
    if ts >= end and lag_hours > 0:
        gap = (ts - end).total_seconds() / 3600.0 + 1.0
        if 0.0 < gap <= lag_hours:
            after = max(0.0, 1.0 - (gap - 1.0) / max(1.0, float(lag_hours)))
            return 0.0, after, after, 0
    return 0.0, 0.0, 0.0, 0


def build_hourly_event_features(
    events: Sequence[Event],
    *,
    node_locations: pd.DataFrame | None,
    lead_hours: int,
    lag_hours: int,
    spatial_sigma_km: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    type_names = sorted({slug(event.event_type) for event in events})
    source_names = sorted({slug(event.source) for event in events})
    node_records = [None] if node_locations is None else node_locations.to_dict(orient="records")
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}

    for event in events:
        event.validate()
        start = explicit_start(event)
        end = explicit_end(event)
        first = start - pd.Timedelta(hours=int(lead_hours))
        last = end + pd.Timedelta(hours=int(lag_hours)) - pd.Timedelta(hours=1)

        for node in node_records:
            if node is None:
                node_lat = node_lon = None
                key_prefix: tuple[Any, ...] = ()
            else:
                node_lat = float(node["lat"])
                node_lon = float(node["lon"])
                if "cluster" in node:
                    key_prefix = (int(node["cluster"]), str(node["far_edge"]))
                else:
                    key_prefix = (str(node["far_edge"]),)

            relevance = spatial_relevance(event, node_lat, node_lon, spatial_sigma_km)
            if relevance <= 1e-12:
                continue

            for ts in pd.date_range(first, last, freq="h"):
                if ts.normalize() < start_date.normalize() or ts.normalize() > end_date.normalize():
                    continue
                before, after, proximity, active = context_values(
                    ts, start, end, int(lead_hours), int(lag_hours)
                )
                if proximity <= 0.0 and not active:
                    continue

                if node is None:
                    key = (ts.normalize(), int(ts.hour))
                else:
                    key = (*key_prefix, ts.normalize(), int(ts.hour))

                if key not in rows:
                    base: dict[str, Any] = {
                        "date": ts.normalize(),
                        "hour": int(ts.hour),
                        "event_active": 0.0,
                        "event_count": 0.0,
                        "event_context_count": 0.0,
                        "event_intensity": 0.0,
                        "event_relevance": 0.0,
                        "event_before": 0.0,
                        "event_after": 0.0,
                        "event_temporal_proximity": 0.0,
                        "event_synthetic": 0.0,
                    }
                    if node is not None:
                        base["far_edge"] = str(node["far_edge"])
                        if "cluster" in node:
                            base["cluster"] = int(node["cluster"])
                    for name in type_names:
                        base[f"event_type_{name}"] = 0.0
                    for name in source_names:
                        base[f"event_source_{name}"] = 0.0
                    rows[key] = base

                row = rows[key]
                row["event_active"] = max(float(row["event_active"]), float(active))
                row["event_context_count"] = float(row["event_context_count"]) + 1.0
                if active:
                    row["event_count"] = float(row["event_count"]) + 1.0
                    row["event_intensity"] = max(
                        float(row["event_intensity"]),
                        float(event.intensity) * relevance,
                    )
                row["event_relevance"] = max(float(row["event_relevance"]), relevance)
                row["event_before"] = max(float(row["event_before"]), before * relevance)
                row["event_after"] = max(float(row["event_after"]), after * relevance)
                row["event_temporal_proximity"] = max(
                    float(row["event_temporal_proximity"]), proximity * relevance
                )
                row["event_synthetic"] = max(float(row["event_synthetic"]), float(event.synthetic))
                row[f"event_type_{slug(event.event_type)}"] = 1.0
                row[f"event_source_{slug(event.source)}"] = 1.0

    key_cols = ["date", "hour"]
    if node_locations is not None:
        key_cols = (["cluster"] if "cluster" in node_locations.columns else []) + ["far_edge", "date", "hour"]

    if not rows:
        return pd.DataFrame(columns=key_cols)

    out = pd.DataFrame(rows.values())
    feature_cols = [column for column in out.columns if column not in key_cols]
    out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="raise").astype(np.float32)
    out = out.sort_values(key_cols).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[key_cols + feature_cols]


def normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timestamp" in out.columns:
        timestamp = pd.to_datetime(out["timestamp"], errors="coerce")
        if "date" not in out.columns:
            out["date"] = timestamp.dt.normalize()
        if "hour" not in out.columns:
            out["hour"] = timestamp.dt.hour
    if "date" not in out.columns or "hour" not in out.columns:
        raise ValueError("Feature CSV must contain date+hour or timestamp")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out = out.dropna(subset=["date", "hour"]).copy()
    out["hour"] = out["hour"].astype(int)
    if "cluster" in out.columns:
        out["cluster"] = pd.to_numeric(out["cluster"], errors="raise").astype(int)
    if "far_edge" in out.columns:
        out["far_edge"] = out["far_edge"].astype(str)
    return out


def feature_keys(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("cluster", "far_edge", "date", "hour") if column in frame.columns]


def expand_global_features_to_entities(
    scenario_events: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    baseline_keys = feature_keys(baseline)
    entity_keys = [column for column in ("cluster", "far_edge") if column in baseline_keys]
    scenario_keys = feature_keys(scenario_events)
    if not entity_keys or all(column in scenario_keys for column in entity_keys):
        return scenario_events

    entities = baseline[entity_keys].drop_duplicates().copy()
    global_frame = scenario_events.copy()
    entities["__join"] = 1
    global_frame["__join"] = 1
    out = entities.merge(global_frame, on="__join", how="outer").drop(columns=["__join"])
    return out


def align_scenario_to_baseline(
    baseline: pd.DataFrame,
    scenario_events: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the baseline external-feature schema while replacing event_* values.

    Columns not starting with ``event_`` are treated as independent exogenous
    signals and are carried over unchanged. This is how Google Trends remains
    fixed while scheduled-event information is corrupted.
    """
    base = normalize_feature_frame(baseline)
    scenario = normalize_feature_frame(scenario_events)
    scenario = expand_global_features_to_entities(scenario, base)

    keys = feature_keys(base)
    if not keys:
        raise ValueError("No feature join keys found")
    if any(key not in scenario.columns for key in keys):
        raise ValueError(
            f"Scenario feature keys do not match baseline keys {keys}; "
            "provide --node-locations when node-specific event features are required"
        )

    baseline_event_cols = [column for column in base.columns if column.startswith("event_")]
    passthrough_cols = [
        column
        for column in base.columns
        if column not in keys and column not in baseline_event_cols and column != "timestamp"
    ]

    scenario_event_cols = [column for column in scenario.columns if column.startswith("event_")]
    event_part = scenario[keys + scenario_event_cols].copy()
    event_part = event_part.drop_duplicates(subset=keys)
    for column in baseline_event_cols:
        if column not in event_part.columns:
            event_part[column] = 0.0
    event_part = event_part[keys + baseline_event_cols]

    base_passthrough = base[keys + passthrough_cols].drop_duplicates(subset=keys)
    universe = pd.concat([base[keys], event_part[keys]], ignore_index=True).drop_duplicates()
    out = universe.merge(base_passthrough, on=keys, how="left", validate="one_to_one")
    out = out.merge(event_part, on=keys, how="left", validate="one_to_one")

    feature_cols = passthrough_cols + baseline_event_cols
    for column in feature_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(np.float32)
    out = out.sort_values(keys).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[keys + feature_cols]


def baseline_event_difference(baseline: pd.DataFrame, regenerated: pd.DataFrame) -> float:
    base = normalize_feature_frame(baseline)
    regen = normalize_feature_frame(regenerated)
    keys = feature_keys(base)
    event_cols = [column for column in base.columns if column.startswith("event_")]
    if not event_cols:
        return 0.0
    for column in event_cols:
        if column not in regen.columns:
            regen[column] = 0.0
    left = base[keys + event_cols].copy()
    right = regen[keys + event_cols].copy()
    merged = left.merge(right, on=keys, how="outer", suffixes=("_base", "_regen"))
    max_diff = 0.0
    for column in event_cols:
        a = pd.to_numeric(merged[f"{column}_base"], errors="coerce").fillna(0.0).to_numpy(float)
        b = pd.to_numeric(merged[f"{column}_regen"], errors="coerce").fillna(0.0).to_numpy(float)
        if len(a):
            max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    return max_diff


# ---------------------------------------------------------------------------
# Nested perturbation plan
# ---------------------------------------------------------------------------


def intervals_overlap(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> bool:
    return start_a < end_b and start_b < end_a


def event_intersects_window(event: Event, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    window_end = end.normalize() + pd.Timedelta(days=1)
    return intervals_overlap(explicit_start(event), explicit_end(event), start.normalize(), window_end)


def fraction_count(rate: float, n: int) -> int:
    if rate <= 0.0 or n <= 0:
        return 0
    return min(n, max(1, int(math.ceil(float(rate) * n))))


@dataclass
class PerturbationPlan:
    cancel_order: list[str]
    shift_order: list[str]
    shift_sign: dict[str, int]
    false_positive_events: list[Event]


def active_hours(events: Sequence[Event], start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    occupied: set[pd.Timestamp] = set()
    first_allowed = start.normalize()
    last_allowed = end.normalize() + pd.Timedelta(hours=23)
    for event in events:
        first = max(explicit_start(event), first_allowed)
        last = min(explicit_end(event) - pd.Timedelta(hours=1), last_allowed)
        if last < first:
            continue
        occupied.update(pd.date_range(first, last, freq="h").tolist())
    return occupied


def build_perturbation_plan(
    events: Sequence[Event],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_fp_count: int,
    shift_fraction: float,
    fixed_event_types: set[str],
    seed: int,
) -> PerturbationPlan:
    rng = np.random.default_rng(seed)
    eligible = [
        event
        for event in events
        if event_intersects_window(event, start, end)
        and slug(event.event_type) not in fixed_event_types
    ]
    if not eligible:
        raise ValueError("No events intersect the perturbation window")

    ids = np.asarray([event.event_id for event in eligible], dtype=object)
    cancel_order = rng.permutation(ids).tolist()
    shift_order = rng.permutation(ids).tolist()
    shift_sign = {str(event_id): (1 if rng.random() < 0.5 else -1) for event_id in ids}

    occupied = active_hours(eligible, start, end)
    all_hours = pd.date_range(start.normalize(), end.normalize() + pd.Timedelta(hours=23), freq="h")
    non_event_hours = [timestamp for timestamp in all_hours if timestamp not in occupied]
    rng.shuffle(non_event_hours)

    max_fp_count = max(0, int(max_fp_count))

    templates = list(eligible)
    template_order = rng.integers(0, len(templates), size=max(1, max_fp_count * 4))
    true_intervals = [(explicit_start(event), explicit_end(event)) for event in eligible]
    chosen_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    false_positive_events: list[Event] = []

    template_cursor = 0
    for candidate in non_event_hours:
        if len(false_positive_events) >= max_fp_count:
            break
        template = templates[int(template_order[template_cursor % len(template_order)])]
        template_cursor += 1
        duration = int(template.duration_hours)
        candidate_end = candidate + pd.Timedelta(hours=duration)
        if candidate.normalize() < start.normalize() or candidate > end.normalize() + pd.Timedelta(hours=23):
            continue
        if any(intervals_overlap(candidate, candidate_end, a, b) for a, b in true_intervals):
            continue
        if any(intervals_overlap(candidate, candidate_end, a, b) for a, b in chosen_intervals):
            continue

        fp_index = len(false_positive_events) + 1
        false_positive_events.append(
            replace(
                template,
                event_id=f"FP{fp_index:05d}",
                date=candidate.normalize(),
                start_hour=int(candidate.hour),
                title=f"False-positive proxy #{fp_index}: {template.title}",
                # Keep synthetic=False intentionally: the forecasting model must
                # not be told that the injected event is a false positive.
                synthetic=False,
            )
        )
        chosen_intervals.append((candidate, candidate_end))

    if len(false_positive_events) < max_fp_count:
        LOGGER.warning(
            "Only %d/%d non-overlapping false-positive events could be generated",
            len(false_positive_events),
            max_fp_count,
        )

    # Keep a deterministic priority order for temporal-shift candidates.
    # The actual shifted IDs are selected per scenario *after* cancellations
    # so that a cancelled record cannot silently mask the SHIFT factor.
    return PerturbationPlan(
        cancel_order=[str(value) for value in cancel_order],
        shift_order=[str(value) for value in shift_order],
        shift_sign=shift_sign,
        false_positive_events=false_positive_events,
    )


def apply_scenario(
    events: Sequence[Event],
    plan: PerturbationPlan,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cancel_count: int,
    fp_count: int,
    fixed_event_types: set[str],
    shift_hours: int,
    shift_fraction: float,
) -> tuple[list[Event], list[dict[str, Any]], dict[str, int]]:
    eligible = [
        event
        for event in events
        if event_intersects_window(event, start, end)
        and slug(event.event_type) not in fixed_event_types
    ]
    eligible_ids = {event.event_id for event in eligible}
    cancel_count = min(max(0, int(cancel_count)), len(eligible))
    cancel_ids = set(plan.cancel_order[:cancel_count])

    # Select temporal-shift candidates only among records that survive CANC.
    # This prevents an unlucky overlap between cancel_order and shift_order from
    # making SH00/SH01/SH02 numerically identical. The priority order remains
    # deterministic for a given replicate/seed.
    surviving_ids = eligible_ids.difference(cancel_ids)
    if shift_hours > 0 and surviving_ids:
        requested_shift_count = fraction_count(shift_fraction, len(surviving_ids))
        ordered_survivors = [
            event_id for event_id in plan.shift_order if event_id in surviving_ids
        ]
        shift_ids = set(ordered_survivors[:requested_shift_count])
    else:
        shift_ids = set()

    fp_count = min(max(0, int(fp_count)), len(plan.false_positive_events))

    perturbed: list[Event] = []
    audit: list[dict[str, Any]] = []
    n_shifted = 0

    for event in events:
        if event.event_id in eligible_ids and event.event_id in cancel_ids:
            audit.append(
                {
                    "action": "cancelled",
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "original_start": explicit_start(event),
                    "perturbed_start": pd.NaT,
                }
            )
            continue

        if event.event_id in eligible_ids and event.event_id in shift_ids:
            signed_shift = int(plan.shift_sign[event.event_id]) * int(shift_hours)
            shifted = shift_event(event, signed_shift)
            perturbed.append(shifted)
            n_shifted += 1
            audit.append(
                {
                    "action": "shifted",
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "original_start": explicit_start(event),
                    "perturbed_start": explicit_start(shifted),
                    "shift_hours": signed_shift,
                }
            )
        else:
            perturbed.append(event)

    for fp_event in plan.false_positive_events[:fp_count]:
        perturbed.append(fp_event)
        audit.append(
            {
                "action": "false_positive",
                "event_id": fp_event.event_id,
                "event_type": fp_event.event_type,
                "original_start": pd.NaT,
                "perturbed_start": explicit_start(fp_event),
            }
        )

    counts = {
        "cancelled_events": len(cancel_ids),
        "shifted_events": n_shifted,
        "false_positive_events": fp_count,
    }
    perturbed.sort(key=lambda event: (explicit_start(event), event.event_type, event.event_id))
    return perturbed, audit, counts


# ---------------------------------------------------------------------------
# Evaluation using saved clean models
# ---------------------------------------------------------------------------


def load_external_features(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = normalize_feature_frame(pd.read_csv(path))
    join_keys = ["date", "hour"]
    if "cluster" in frame.columns:
        join_keys.append("cluster")
    if "far_edge" in frame.columns:
        join_keys.append("far_edge")
    ignored = set(join_keys) | {"timestamp"}
    source_features = [column for column in frame.columns if column not in ignored]
    if not source_features:
        raise ValueError(f"{path}: no exogenous feature columns")
    for column in source_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(np.float32)
    rename = {column: f"{EXOG_PREFIX}{column}" for column in source_features}
    frame = frame[[*join_keys, *source_features]].rename(columns=rename)
    return frame, join_keys, list(rename.values())


def load_hourly_panel(data_dir: Path, cluster: int) -> pd.DataFrame:
    path = data_dir / "hourly_panels" / f"hourly_cluster_{cluster}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(PANEL_META_COLS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame[list(PANEL_META_COLS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["far_edge"] = frame["far_edge"].astype(str)
    frame = frame.dropna(subset=["date", "hour", "far_edge", "value"])
    frame["hour"] = frame["hour"].astype(int)
    frame = frame.loc[frame["hour"].between(0, 23)].copy()
    return frame.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)


def merge_external_features(
    panel: pd.DataFrame,
    cluster: int,
    external: pd.DataFrame,
    join_keys: Sequence[str],
    model_feature_names: Sequence[str],
) -> pd.DataFrame:
    left = panel.copy()
    right = external
    if "cluster" in join_keys:
        right = right.loc[right["cluster"] == int(cluster)].copy()
        left["cluster"] = int(cluster)
    if "far_edge" in join_keys:
        left["far_edge"] = left["far_edge"].astype(str)
    merged = left.merge(right, on=list(join_keys), how="left", validate="many_to_one")
    for column in model_feature_names:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    if "cluster" in merged.columns and "cluster" not in PANEL_META_COLS:
        merged = merged.drop(columns=["cluster"])
    return merged.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)


def make_supervised(
    node_frame: pd.DataFrame,
    *,
    n_lags: int,
    include_weekend_onehot: bool,
) -> pd.DataFrame:
    frame = node_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["dow"] = frame["date"].dt.weekday.astype(int)
    frame["weekend"] = (frame["dow"] >= 5).astype(int)
    frame = frame.sort_values(["date", "hour"]).reset_index(drop=True)

    lag_columns: list[str] = []
    for lag in range(1, int(n_lags) + 1):
        column = f"lag{lag}"
        lag_columns.append(column)
        frame[column] = frame["value"].shift(lag)

    hour_dummies = pd.get_dummies(
        pd.Categorical(frame["hour"], categories=range(24)), prefix="h", dtype=float
    )
    dow_dummies = pd.get_dummies(
        pd.Categorical(frame["dow"], categories=range(7)), prefix="d", dtype=float
    )
    hour_dummies.index = frame.index
    dow_dummies.index = frame.index

    parts: list[pd.DataFrame] = [frame[list(PANEL_META_COLS)], hour_dummies, dow_dummies]
    if include_weekend_onehot:
        weekend_dummies = pd.get_dummies(
            pd.Categorical(frame["weekend"], categories=(0, 1)),
            prefix="weekend",
            dtype=float,
        )
        weekend_dummies.index = frame.index
        parts.append(weekend_dummies)

    exogenous = sorted(column for column in frame.columns if column.startswith(EXOG_PREFIX))
    if exogenous:
        parts.append(frame[exogenous].astype(float))
    parts.append(frame[lag_columns])

    supervised = pd.concat(parts, axis=1)
    features = [column for column in supervised.columns if column not in PANEL_META_COLS]
    supervised = supervised.replace([np.inf, -np.inf], np.nan)
    supervised = supervised.dropna(subset=["value", *features])
    return supervised.reset_index(drop=True)


def test_block(
    supervised: pd.DataFrame,
    *,
    test_fraction: float,
    validation_fraction: float,
    min_train_rows: int,
) -> pd.DataFrame:
    n_rows = len(supervised)
    n_test = max(1, int(round(n_rows * float(test_fraction))))
    n_pretest = n_rows - n_test
    n_validation = max(1, int(round(n_pretest * float(validation_fraction))))
    n_train = n_pretest - n_validation
    if n_train < int(min_train_rows) or n_validation < 1 or n_test < 1:
        raise ValueError(
            f"series too short: train={n_train}, validation={n_validation}, test={n_test}"
        )
    return supervised.iloc[n_pretest:].copy().reset_index(drop=True)


def safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return text or "node"


def model_index(models_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for metadata_path in sorted(models_dir.glob("*_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        key = (int(metadata["cluster"]), str(metadata["far_edge"]))
        model_path = models_dir / metadata.get("model_path", "")
        scaler_path = models_dir / metadata.get("scaler_path", "")
        if not model_path.exists():
            model_path = models_dir / f"cluster_{key[0]}_far_edge_{safe_component(key[1])}.keras"
        if not scaler_path.exists():
            scaler_path = models_dir / f"cluster_{key[0]}_far_edge_{safe_component(key[1])}_scaler.npz"
        if model_path.exists() and scaler_path.exists():
            index[key] = {
                "model_path": model_path,
                "scaler_path": scaler_path,
                "metadata_path": metadata_path,
            }
    if not index:
        raise FileNotFoundError(f"No saved model/scaler pairs found in {models_dir}")
    return index


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / denom)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def traffic_reduction(y_true: np.ndarray, y_pred: np.ndarray, capacity_level: float) -> float:
    y = np.asarray(y_true, float)
    yhat = np.asarray(y_pred, float)
    capacity = float(np.nanpercentile(y, float(capacity_level) * 100.0))
    overflow_static = np.nansum(np.maximum(0.0, y - capacity))
    if overflow_static <= 1e-12:
        return float("nan")
    dynamic_capacity = np.maximum(capacity, yhat)
    overflow_dynamic = np.nansum(np.maximum(0.0, y - dynamic_capacity))
    return float(100.0 * (overflow_static - overflow_dynamic) / overflow_static)


def directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, float) - np.asarray(y_true, float)
    under = np.maximum(0.0, -error)
    over = np.maximum(0.0, error)
    return {
        "Underprediction_rate_pct": float(np.mean(error < 0.0) * 100.0),
        "Overprediction_rate_pct": float(np.mean(error > 0.0) * 100.0),
        "Mean_underprediction": float(np.mean(under)),
        "Mean_overprediction": float(np.mean(over)),
        "P95_underprediction": float(np.percentile(under, 95)),
        "P95_overprediction": float(np.percentile(over, 95)),
    }


def load_run_config(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Evaluation expects an experiment produced with --save-models."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_scenarios(
    *,
    scenario_files: Sequence[tuple[str, int, int, int, int, Path]],
    baseline_features_path: Path,
    experiment_dir: Path,
    output_dir: Path,
) -> None:
    # TensorFlow is imported only when model evaluation is explicitly requested.
    import tensorflow as tf  # type: ignore

    config = load_run_config(experiment_dir)
    data_dir = Path(config.get("data_dir", "data"))
    clusters_raw = config.get("clusters", None)
    if clusters_raw:
        clusters = [int(value) for value in clusters_raw]
    else:
        models = model_index(experiment_dir / "models")
        clusters = sorted({cluster for cluster, _ in models})

    n_lags = int(config.get("n_lags", 24))
    test_fraction = float(config.get("test_fraction", 0.20))
    validation_fraction = float(config.get("validation_fraction", 0.20))
    min_train_rows = int(config.get("min_train_rows", 48))
    include_weekend_onehot = bool(config.get("include_weekend_onehot", True))
    clip_negative = bool(config.get("clip_negative_predictions", False))

    models_dir = experiment_dir / "models"
    model_meta = model_index(models_dir)
    panel_cache = {cluster: load_hourly_panel(data_dir, cluster) for cluster in clusters}

    all_scenarios = [("CLEAN", 0, 0, 0, 0, baseline_features_path)] + list(scenario_files)
    rows: list[dict[str, Any]] = []
    prediction_chunks: list[pd.DataFrame] = []

    # Scenario-major evaluation keeps memory bounded. Models are loaded per node
    # for each scenario; this is slower than caching all models but is robust on
    # machines with limited GPU/CPU memory.
    for scenario_id, replicate, cancel_count, fp_count, shift_hours, feature_path in all_scenarios:
        LOGGER.info("Evaluating %s using %s", scenario_id, feature_path)
        external, join_keys, external_model_features = load_external_features(feature_path)

        for cluster in clusters:
            panel = merge_external_features(
                panel_cache[cluster],
                cluster,
                external,
                join_keys,
                external_model_features,
            )
            for far_edge, node_frame in panel.groupby("far_edge", sort=False):
                key = (int(cluster), str(far_edge))
                if key not in model_meta:
                    continue
                supervised = make_supervised(
                    node_frame,
                    n_lags=n_lags,
                    include_weekend_onehot=include_weekend_onehot,
                )
                try:
                    test = test_block(
                        supervised,
                        test_fraction=test_fraction,
                        validation_fraction=validation_fraction,
                        min_train_rows=min_train_rows,
                    )
                except ValueError:
                    continue

                scaler = np.load(model_meta[key]["scaler_path"], allow_pickle=False)
                saved_features = [str(value) for value in scaler["feature_names"].tolist()]
                missing_features = [column for column in saved_features if column not in test.columns]
                if missing_features:
                    raise RuntimeError(
                        f"Scenario {scenario_id}, node {key}: missing saved model features "
                        f"{missing_features[:10]}"
                    )
                X_raw = test[saved_features].to_numpy(dtype=np.float32, copy=True)
                x_min = scaler["x_min"].astype(np.float32)
                x_range = scaler["x_range"].astype(np.float32)
                y_min = float(np.asarray(scaler["y_min"]).reshape(-1)[0])
                y_range = float(np.asarray(scaler["y_range"]).reshape(-1)[0])
                X_scaled = ((X_raw - x_min) / x_range).reshape((len(X_raw), 1, len(saved_features)))

                model = tf.keras.models.load_model(model_meta[key]["model_path"], compile=False)
                pred_scaled = model.predict(X_scaled, verbose=0).reshape(-1)
                prediction = pred_scaled.astype(float) * y_range + y_min
                if clip_negative:
                    prediction = np.maximum(0.0, prediction)
                y_true = test["value"].to_numpy(dtype=float)

                row: dict[str, Any] = {
                    "scenario": scenario_id,
                    "replicate": int(replicate),
                    "cancel_count": int(cancel_count),
                    "fp_count": int(fp_count),
                    "shift_hours": int(shift_hours),
                    "cluster": int(cluster),
                    "far_edge": str(far_edge),
                    "MAE": mae(y_true, prediction),
                    "RMSE": rmse(y_true, prediction),
                    "MAPE": mape(y_true, prediction),
                    "R2": r2(y_true, prediction),
                }
                row.update(directional_metrics(y_true, prediction))
                for capacity_level in CAP_LEVELS:
                    row[f"Dpp@{int(capacity_level * 100)}"] = traffic_reduction(
                        y_true, prediction, capacity_level
                    )
                rows.append(row)

                pred_frame = test[["date", "hour", "far_edge"]].copy()
                pred_frame.insert(0, "cluster", int(cluster))
                pred_frame.insert(0, "scenario", scenario_id)
                pred_frame["y_true"] = y_true
                pred_frame["y_pred"] = prediction
                prediction_chunks.append(pred_frame)
                del model
                tf.keras.backend.clear_session()

    if not rows:
        raise RuntimeError("No saved models were evaluated")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "robustness_metrics_by_node.csv", index=False)
    if prediction_chunks:
        pd.concat(prediction_chunks, ignore_index=True).to_csv(
            output_dir / "robustness_predictions.csv", index=False
        )

    numeric_metrics = [
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "Underprediction_rate_pct",
        "Overprediction_rate_pct",
        "Mean_underprediction",
        "Mean_overprediction",
        "P95_underprediction",
        "P95_overprediction",
        "Dpp@70",
        "Dpp@80",
        "Dpp@90",
    ]
    summary = (
        metrics.groupby(["scenario", "replicate", "cancel_count", "fp_count", "shift_hours"], as_index=False)[numeric_metrics]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)

    clean = metrics.loc[metrics["scenario"] == "CLEAN"].copy()
    clean = clean.set_index(["cluster", "far_edge"])
    delta_rows: list[dict[str, Any]] = []
    for _, row in metrics.loc[metrics["scenario"] != "CLEAN"].iterrows():
        key = (int(row["cluster"]), str(row["far_edge"]))
        if key not in clean.index:
            continue
        base = clean.loc[key]
        out: dict[str, Any] = {
            "scenario": row["scenario"],
            "replicate": int(row["replicate"]),
            "cancel_count": int(row["cancel_count"]),
            "fp_count": int(row["fp_count"]),
            "shift_hours": int(row["shift_hours"]),
            "cluster": int(row["cluster"]),
            "far_edge": str(row["far_edge"]),
        }
        for metric in numeric_metrics:
            out[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
        out["delta_RMSE_pct"] = (
            100.0 * (float(row["RMSE"]) - float(base["RMSE"])) / float(base["RMSE"])
            if float(base["RMSE"]) > 1e-12
            else float("nan")
        )
        delta_rows.append(out)

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(output_dir / "robustness_deltas_by_node.csv", index=False)
    if not deltas.empty:
        delta_numeric = [column for column in deltas.columns if column.startswith("delta_")]
        delta_summary = (
            deltas.groupby(["scenario", "replicate", "cancel_count", "fp_count", "shift_hours"], as_index=False)[delta_numeric]
            .agg(["mean", "std"])
        )
        delta_summary.columns = [
            "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
            for col in delta_summary.columns.to_flat_index()
        ]
        delta_summary.to_csv(output_dir / "robustness_deltas_summary.csv", index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally evaluate event-input robustness scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--canonical-events", type=Path, required=True)
    parser.add_argument(
        "--baseline-features",
        type=Path,
        required=True,
        help="Clean event-aware feature CSV used by the forecasting pipeline",
    )
    parser.add_argument(
        "--node-locations",
        type=Path,
        default=None,
        help="Optional CSV with far_edge,lat,lon[,cluster] for spatial relevance",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--perturb-start-date",
        default=None,
        help=(
            "First date on which event-feed perturbations may be applied. "
            "When omitted with --evaluate-models, it is inferred from the saved-model held-out test block."
        ),
    )
    parser.add_argument(
        "--perturb-end-date",
        default=None,
        help=(
            "Last date on which event-feed perturbations may be applied. "
            "When omitted with --evaluate-models, it is inferred from the saved-model held-out test block."
        ),
    )
    parser.add_argument("--lead-hours", type=int, default=6)
    parser.add_argument("--lag-hours", type=int, default=4)
    parser.add_argument("--spatial-sigma-km", type=float, default=3.0)
    parser.add_argument(
        "--date-only-mode",
        choices=("full_day", "default_time"),
        default="full_day",
    )
    parser.add_argument("--default-start-hour", type=int, default=20)
    parser.add_argument("--default-duration-hours", type=int, default=3)

    parser.add_argument(
        "--cancel-count-grid",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Absolute numbers of real event records removed from the event feed",
    )
    parser.add_argument(
        "--fp-count-grid",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Absolute numbers of false-positive event records injected into non-event periods",
    )
    parser.add_argument("--shift-grid", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--shift-fraction",
        type=float,
        default=0.20,
        help="Fraction of real events selected for temporal shifting",
    )
    parser.add_argument(
        "--fixed-event-types",
        nargs="*",
        default=("public_holiday",),
        help=(
            "Event types kept deterministic and excluded from cancellation, false-positive "
            "templates, and temporal shifting. Public holidays are fixed by default."
        ),
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-baseline-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require regenerated clean event features to match the supplied baseline",
    )
    parser.add_argument("--baseline-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--save-scenario-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--evaluate-models",
        action="store_true",
        help="Evaluate previously saved clean models on the corrupted event inputs",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment directory containing run_config.json and models/",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/event_robustness"))
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.lead_hours < 0 or args.lag_hours < 0:
        raise ValueError("lead-hours and lag-hours must be >= 0")
    if args.replicates < 1:
        raise ValueError("replicates must be >= 1")
    if not 0.0 <= args.shift_fraction <= 1.0:
        raise ValueError("shift-fraction must be in [0,1]")
    for name, values in (("cancel-count-grid", args.cancel_count_grid), ("fp-count-grid", args.fp_count_grid)):
        if any(int(value) < 0 for value in values):
            raise ValueError(f"{name} values must be >= 0")
    if any(value < 0 for value in args.shift_grid):
        raise ValueError("shift-grid values must be >= 0")
    if args.evaluate_models and args.experiment_dir is None:
        raise ValueError("--evaluate-models requires --experiment-dir")


def infer_heldout_test_window(
    experiment_dir: Path,
    baseline_features_path: Path,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer the dates covered by the held-out test block used for evaluation."""
    config = load_run_config(experiment_dir)
    data_dir = Path(config.get("data_dir", "data"))
    clusters_raw = config.get("clusters", None)
    if clusters_raw:
        clusters = [int(value) for value in clusters_raw]
    else:
        models = model_index(experiment_dir / "models")
        clusters = sorted({cluster for cluster, _ in models})

    n_lags = int(config.get("n_lags", 24))
    test_fraction = float(config.get("test_fraction", 0.20))
    validation_fraction = float(config.get("validation_fraction", 0.20))
    min_train_rows = int(config.get("min_train_rows", 48))
    include_weekend_onehot = bool(config.get("include_weekend_onehot", True))

    external, join_keys, external_model_features = load_external_features(baseline_features_path)
    model_meta = model_index(experiment_dir / "models")
    first_ts: list[pd.Timestamp] = []
    last_ts: list[pd.Timestamp] = []

    for cluster in clusters:
        panel = merge_external_features(
            load_hourly_panel(data_dir, cluster),
            cluster,
            external,
            join_keys,
            external_model_features,
        )
        for far_edge, node_frame in panel.groupby("far_edge", sort=False):
            key = (int(cluster), str(far_edge))
            if key not in model_meta:
                continue
            supervised = make_supervised(
                node_frame,
                n_lags=n_lags,
                include_weekend_onehot=include_weekend_onehot,
            )
            try:
                test = test_block(
                    supervised,
                    test_fraction=test_fraction,
                    validation_fraction=validation_fraction,
                    min_train_rows=min_train_rows,
                )
            except ValueError:
                continue
            if test.empty:
                continue
            timestamps = (
                pd.to_datetime(test["date"], errors="coerce").dt.normalize()
                + pd.to_timedelta(pd.to_numeric(test["hour"], errors="coerce"), unit="h")
            ).dropna()
            if timestamps.empty:
                continue
            first_ts.append(pd.Timestamp(timestamps.min()))
            last_ts.append(pd.Timestamp(timestamps.max()))

    if not first_ts:
        raise RuntimeError("Could not infer a held-out test window from the saved experiment")
    return min(first_ts).normalize(), max(last_ts).normalize()


def scenario_name(replicate: int, cancel_count: int, fp_count: int, shift_hours: int) -> str:
    return (
        f"R{replicate:02d}_C{int(cancel_count):02d}_"
        f"FP{int(fp_count):02d}_SH{int(shift_hours):02d}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir = args.output_dir / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    baseline = normalize_feature_frame(pd.read_csv(args.baseline_features))
    start = (
        pd.Timestamp(args.start_date).normalize()
        if args.start_date
        else pd.Timestamp(baseline["date"].min()).normalize()
    )
    end = (
        pd.Timestamp(args.end_date).normalize()
        if args.end_date
        else pd.Timestamp(baseline["date"].max()).normalize()
    )
    if end < start:
        raise ValueError("end-date is before start-date")

    if args.perturb_start_date or args.perturb_end_date:
        perturb_start = (
            pd.Timestamp(args.perturb_start_date).normalize()
            if args.perturb_start_date
            else start
        )
        perturb_end = (
            pd.Timestamp(args.perturb_end_date).normalize()
            if args.perturb_end_date
            else end
        )
    elif args.evaluate_models and args.experiment_dir is not None:
        perturb_start, perturb_end = infer_heldout_test_window(
            args.experiment_dir, args.baseline_features
        )
        LOGGER.info(
            "Automatically inferred held-out perturbation window: %s to %s",
            perturb_start.date(), perturb_end.date(),
        )
    else:
        perturb_start, perturb_end = start, end

    perturb_start = max(perturb_start, start)
    perturb_end = min(perturb_end, end)
    if perturb_end < perturb_start:
        raise ValueError("Perturbation window does not overlap the feature-data window")

    canonical = load_canonical_events(args.canonical_events)
    explicit_events = make_explicit_events(
        canonical,
        date_only_mode=args.date_only_mode,
        default_start_hour=args.default_start_hour,
        default_duration_hours=args.default_duration_hours,
    )
    nodes = load_node_locations(args.node_locations)

    clean_event_features = build_hourly_event_features(
        explicit_events,
        node_locations=nodes,
        lead_hours=args.lead_hours,
        lag_hours=args.lag_hours,
        spatial_sigma_km=args.spatial_sigma_km,
        start_date=start,
        end_date=end,
    )
    clean_aligned = align_scenario_to_baseline(baseline, clean_event_features)
    clean_diff = baseline_event_difference(baseline, clean_aligned)
    LOGGER.info("Clean regenerated event-feature max abs difference: %.8g", clean_diff)
    if args.strict_baseline_check and clean_diff > float(args.baseline_tolerance):
        raise RuntimeError(
            "Regenerated clean event features do not match the supplied baseline. "
            f"max_abs_diff={clean_diff:.6g}. Check --date-only-mode, lead/lag hours, "
            "node locations, and spatial sigma; or use --no-strict-baseline-check only "
            "after verifying the intended feature configuration."
        )
    clean_aligned.to_csv(args.output_dir / "clean_features_regenerated.csv", index=False)
    events_to_frame(explicit_events).to_csv(args.output_dir / "clean_events_explicit.csv", index=False)

    fixed_event_types = {slug(value) for value in args.fixed_event_types}
    eligible_for_perturbation = [
        event for event in explicit_events
        if event_intersects_window(event, perturb_start, perturb_end)
        and slug(event.event_type) not in fixed_event_types
    ]
    max_cancel_count = max(int(value) for value in args.cancel_count_grid)
    max_fp_count = max(int(value) for value in args.fp_count_grid)
    if max_cancel_count > len(eligible_for_perturbation):
        raise ValueError(
            f"cancel-count-grid requests up to {max_cancel_count} cancellations, but only "
            f"{len(eligible_for_perturbation)} perturbable event records intersect the evaluation window"
        )
    if any(int(value) > 0 for value in args.shift_grid) and max_cancel_count >= len(eligible_for_perturbation):
        raise ValueError(
            "The largest cancellation scenario removes every perturbable true event, "
            "so the SHIFT factor cannot be evaluated independently. Reduce the maximum "
            "--cancel-count-grid value (for example, with 3 eligible events use 0 1 2)."
        )
    LOGGER.info(
        "Held-out perturbation window %s..%s contains %d eligible event records; absolute-count grids: C=%s, FP=%s, SHIFT=%s",
        perturb_start.date(), perturb_end.date(), len(eligible_for_perturbation),
        list(args.cancel_count_grid),
        list(args.fp_count_grid),
        list(args.shift_grid),
    )
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    scenario_files: list[tuple[str, int, int, int, int, Path]] = []

    for replicate in range(1, int(args.replicates) + 1):
        plan_seed = int(args.seed) + replicate * 100_003
        plan = build_perturbation_plan(
            explicit_events,
            start=perturb_start,
            end=perturb_end,
            max_fp_count=max_fp_count,
            shift_fraction=float(args.shift_fraction),
            fixed_event_types=fixed_event_types,
            seed=plan_seed,
        )

        if len(plan.false_positive_events) < max_fp_count:
            raise RuntimeError(
                f"Requested up to {max_fp_count} false-positive events, but only "
                f"{len(plan.false_positive_events)} non-overlapping candidates could be generated"
            )

        for cancel_count in [int(value) for value in args.cancel_count_grid]:
            for fp_count in [int(value) for value in args.fp_count_grid]:
                for shift_hours in [int(value) for value in args.shift_grid]:
                    scenario_id = scenario_name(replicate, cancel_count, fp_count, shift_hours)
                    perturbed, audit, counts = apply_scenario(
                        explicit_events,
                        plan,
                        start=perturb_start,
                        end=perturb_end,
                        cancel_count=cancel_count,
                        fp_count=fp_count,
                        fixed_event_types=fixed_event_types,
                        shift_hours=shift_hours,
                        shift_fraction=float(args.shift_fraction),
                    )
                    event_features = build_hourly_event_features(
                        perturbed,
                        node_locations=nodes,
                        lead_hours=args.lead_hours,
                        lag_hours=args.lag_hours,
                        spatial_sigma_km=args.spatial_sigma_km,
                        start_date=start,
                        end_date=end,
                    )
                    aligned = align_scenario_to_baseline(baseline, event_features)

                    feature_path = scenario_dir / f"{scenario_id}_features.csv"
                    event_path = scenario_dir / f"{scenario_id}_events.csv"
                    if args.save_scenario_features or args.evaluate_models:
                        aligned.to_csv(feature_path, index=False)
                        events_to_frame(perturbed).to_csv(event_path, index=False)
                    scenario_files.append(
                        (scenario_id, replicate, cancel_count, fp_count, shift_hours, feature_path)
                    )

                    manifest_rows.append(
                        {
                            "scenario": scenario_id,
                            "replicate": replicate,
                            "seed": plan_seed,
                            "cancel_count_requested": cancel_count,
                            "fp_count_requested": fp_count,
                            "shift_fraction": float(args.shift_fraction),
                            "shift_hours": shift_hours,
                            "eligible_event_records": len(eligible_for_perturbation),
                            **counts,
                            "event_rows_after_perturbation": len(perturbed),
                            "feature_rows": len(aligned),
                            "events_file": str(event_path),
                            "features_file": str(feature_path),
                        }
                    )
                    for record in audit:
                        record = dict(record)
                        record.update(
                            {
                                "scenario": scenario_id,
                                "replicate": replicate,
                                "cancel_count_requested": cancel_count,
                                "fp_count_requested": fp_count,
                                "shift_hours_axis": shift_hours,
                            }
                        )
                        audit_rows.append(record)

    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "scenario_manifest.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "scenario_event_audit.csv", index=False)

    config_payload = {
        "canonical_events": str(args.canonical_events),
        "baseline_features": str(args.baseline_features),
        "node_locations": str(args.node_locations) if args.node_locations else None,
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "perturb_start_date": str(perturb_start.date()),
        "perturb_end_date": str(perturb_end.date()),
        "perturbation_scope": "held_out_test_window" if args.evaluate_models else "explicit_or_feature_window",
        "lead_hours": int(args.lead_hours),
        "lag_hours": int(args.lag_hours),
        "spatial_sigma_km": float(args.spatial_sigma_km),
        "date_only_mode": args.date_only_mode,
        "default_start_hour": int(args.default_start_hour),
        "default_duration_hours": int(args.default_duration_hours),
        "cancel_count_grid": [int(value) for value in args.cancel_count_grid],
        "fp_count_grid": [int(value) for value in args.fp_count_grid],
        "shift_grid": [int(value) for value in args.shift_grid],
        "shift_fraction": float(args.shift_fraction),
        "fixed_event_types": sorted(fixed_event_types),
        "eligible_event_records": len(eligible_for_perturbation),
        "replicates": int(args.replicates),
        "seed": int(args.seed),
        "clean_regenerated_max_abs_difference": float(clean_diff),
        "ground_truth_traffic_modified": False,
        "model_retrained_per_scenario": False,
        "scenario_count": len(manifest_rows),
    }
    serialized = json.dumps(config_payload, sort_keys=True).encode("utf-8")
    config_payload["configuration_sha256"] = hashlib.sha256(serialized).hexdigest()
    with (args.output_dir / "robustness_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)
        handle.write("\n")

    LOGGER.info("Generated %d perturbation scenarios", len(manifest_rows))
    LOGGER.info("Ground-truth traffic was not modified")

    if args.evaluate_models:
        evaluate_scenarios(
            scenario_files=scenario_files,
            baseline_features_path=args.baseline_features,
            experiment_dir=args.experiment_dir,
            output_dir=args.output_dir,
        )
        LOGGER.info("Saved robustness metrics under %s", args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
