#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utilities for reproducible event-aware forecasting features.

The output produced by :func:`build_hourly_features` is intentionally compatible
with ``ml_exps_lstm_optuna_custom_loss_v3.py``: the only non-numeric columns are
join keys (date/hour and, optionally, cluster/far_edge).
"""
from __future__ import annotations

import calendar
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
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
            raise ValueError("event date is NaT")
        if self.start_hour is not None and not 0 <= int(self.start_hour) <= 23:
            raise ValueError(f"start_hour must be in 0..23, got {self.start_hour}")
        if int(self.duration_hours) < 1:
            raise ValueError("duration_hours must be >= 1")
        if not 0.0 <= float(self.intensity) <= 1.0:
            raise ValueError("intensity must be in [0,1]")
        if not 0.0 <= float(self.base_relevance) <= 1.0:
            raise ValueError("base_relevance must be in [0,1]")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must both be present or both absent")


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return value or "other"


def parse_date_flexible(value: str, dayfirst: bool | None = None) -> pd.Timestamp:
    value = str(value).strip()
    if not value:
        return pd.NaT
    attempts: list[bool] = []
    if dayfirst is not None:
        attempts.append(dayfirst)
    attempts.extend([True, False])
    seen: set[bool] = set()
    for df in attempts:
        if df in seen:
            continue
        seen.add(df)
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=df)
        if not pd.isna(parsed):
            return pd.Timestamp(parsed).normalize()
    return pd.NaT


def read_one_column_dates(path: Path, dayfirst: bool | None = None) -> list[pd.Timestamp]:
    """Read a legacy one-column file robustly, including unquoted commas in dates.

    ``partite.csv`` contains lines such as ``Aug 11, 2019`` without CSV quoting;
    reading raw lines avoids pandas interpreting the comma as a delimiter.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        return []
    values = [line.strip() for line in lines[1:] if line.strip()]
    dates = [parse_date_flexible(value, dayfirst=dayfirst) for value in values]
    bad = [value for value, date in zip(values, dates) if pd.isna(date)]
    if bad:
        raise ValueError(f"Could not parse dates in {path}: {bad[:5]}")
    return dates


def easter_sunday(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def france_public_holidays(year: int, intensity: float = 0.6) -> list[Event]:
    easter = easter_sunday(year)
    named = [
        (pd.Timestamp(year, 1, 1), "New Year's Day"),
        (easter + pd.Timedelta(days=1), "Easter Monday"),
        (pd.Timestamp(year, 5, 1), "Labour Day"),
        (pd.Timestamp(year, 5, 8), "Victory in Europe Day"),
        (easter + pd.Timedelta(days=39), "Ascension Day"),
        (easter + pd.Timedelta(days=50), "Whit Monday"),
        (pd.Timestamp(year, 7, 14), "Bastille Day"),
        (pd.Timestamp(year, 8, 15), "Assumption of Mary"),
        (pd.Timestamp(year, 11, 1), "All Saints' Day"),
        (pd.Timestamp(year, 11, 11), "Armistice Day"),
        (pd.Timestamp(year, 12, 25), "Christmas Day"),
    ]
    return [
        Event(
            date=date.normalize(),
            event_type="public_holiday",
            title=title,
            start_hour=None,
            duration_hours=24,
            intensity=float(intensity),
            base_relevance=1.0,
            source="calendar_fr",
        )
        for date, title in named
    ]


def load_node_locations(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"far_edge", "lat", "lon"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing node-location columns {sorted(missing)}")
    out = df.copy()
    out["far_edge"] = out["far_edge"].astype(str)
    out["lat"] = pd.to_numeric(out["lat"], errors="raise")
    out["lon"] = pd.to_numeric(out["lon"], errors="raise")
    if "cluster" in out.columns:
        out["cluster"] = pd.to_numeric(out["cluster"], errors="raise").astype(int)
    cols = ["cluster"] if "cluster" in out.columns else []
    cols += ["far_edge", "lat", "lon"]
    out = out[cols].drop_duplicates(subset=[c for c in cols if c not in {"lat", "lon"}])
    return out.reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def spatial_relevance(
    event: Event,
    node_lat: float | None,
    node_lon: float | None,
    sigma_km: float,
) -> float:
    base = float(event.base_relevance)
    if event.latitude is None or event.longitude is None or node_lat is None or node_lon is None:
        return base
    distance = haversine_km(float(event.latitude), float(event.longitude), float(node_lat), float(node_lon))
    if sigma_km <= 0:
        raise ValueError("spatial sigma must be > 0")
    weight = math.exp(-0.5 * (distance / sigma_km) ** 2)
    return float(np.clip(base * weight, 0.0, 1.0))


def event_bounds(
    event: Event,
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


def _context_values(
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


def events_to_frame(events: Sequence[Event]) -> pd.DataFrame:
    rows = []
    for event in events:
        event.validate()
        row = asdict(event)
        row["date"] = event.date.strftime("%Y-%m-%d")
        rows.append(row)
    columns = [
        "date", "event_type", "title", "start_hour", "duration_hours", "venue",
        "latitude", "longitude", "intensity", "base_relevance", "source", "synthetic",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_hourly_features(
    events: Sequence[Event],
    *,
    node_locations: pd.DataFrame | None = None,
    lead_hours: int = 6,
    lag_hours: int = 4,
    spatial_sigma_km: float = 3.0,
    date_only_mode: str = "full_day",
    default_start_hour: int = 20,
    default_duration_hours: int = 3,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    keep_zero_rows: bool = False,
) -> pd.DataFrame:
    """Expand canonical events into numeric hourly model features.

    Features include an active-event indicator,
    intensity-like signal, before/after temporal proximity, event type indicators,
    and an optional geographical relevance in [0,1].
    """
    if lead_hours < 0 or lag_hours < 0:
        raise ValueError("lead_hours and lag_hours must be >= 0")
    if not 0 <= default_start_hour <= 23:
        raise ValueError("default_start_hour must be in 0..23")

    clean_events = []
    for event in events:
        event.validate()
        if start_date is not None and event.date < pd.Timestamp(start_date).normalize() - pd.Timedelta(days=2):
            continue
        if end_date is not None and event.date > pd.Timestamp(end_date).normalize() + pd.Timedelta(days=2):
            continue
        clean_events.append(event)

    type_names = sorted({slug(e.event_type) for e in clean_events})
    source_names = sorted({slug(e.source) for e in clean_events})

    if node_locations is None:
        node_records = [None]
    else:
        node_records = node_locations.to_dict(orient="records")

    rows: dict[tuple, dict] = {}

    for event in clean_events:
        start, end = event_bounds(event, date_only_mode, default_start_hour, default_duration_hours)
        first = start - pd.Timedelta(hours=lead_hours)
        last = end + pd.Timedelta(hours=lag_hours) - pd.Timedelta(hours=1)
        timestamps = pd.date_range(first, last, freq="h")

        for node in node_records:
            if node is None:
                node_lat = node_lon = None
                key_prefix: tuple = ()
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

            for ts in timestamps:
                if start_date is not None and ts.normalize() < pd.Timestamp(start_date).normalize():
                    continue
                if end_date is not None and ts.normalize() > pd.Timestamp(end_date).normalize():
                    continue
                before, after, proximity, active = _context_values(ts, start, end, lead_hours, lag_hours)
                if proximity <= 0 and not active:
                    continue

                if node is None:
                    key = (ts.normalize(), int(ts.hour))
                else:
                    key = (*key_prefix, ts.normalize(), int(ts.hour))

                if key not in rows:
                    base = {
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
                    for type_name in type_names:
                        base[f"event_type_{type_name}"] = 0.0
                    for source_name in source_names:
                        base[f"event_source_{source_name}"] = 0.0
                    rows[key] = base

                row = rows[key]
                weighted_relevance = relevance
                row["event_active"] = max(row["event_active"], float(active))
                row["event_context_count"] += 1.0
                if active:
                    row["event_count"] += 1.0
                    row["event_intensity"] = max(
                        row["event_intensity"],
                        float(event.intensity) * weighted_relevance,
                    )
                row["event_relevance"] = max(row["event_relevance"], weighted_relevance)
                row["event_before"] = max(row["event_before"], before * weighted_relevance)
                row["event_after"] = max(row["event_after"], after * weighted_relevance)
                row["event_temporal_proximity"] = max(
                    row["event_temporal_proximity"],
                    proximity * weighted_relevance,
                )
                row["event_synthetic"] = max(row["event_synthetic"], float(event.synthetic))
                row[f"event_type_{slug(event.event_type)}"] = 1.0
                row[f"event_source_{slug(event.source)}"] = 1.0

    if not rows:
        key_cols = ["date", "hour"]
        if node_locations is not None:
            key_cols = (["cluster"] if "cluster" in node_locations.columns else []) + ["far_edge"] + key_cols
        feature_cols = [
            "event_active", "event_count", "event_context_count", "event_intensity",
            "event_relevance", "event_before", "event_after", "event_temporal_proximity",
            "event_synthetic",
        ]
        return pd.DataFrame(columns=key_cols + feature_cols)

    out = pd.DataFrame(rows.values())
    key_cols = ["date", "hour"]
    if node_locations is not None:
        key_cols = (["cluster"] if "cluster" in node_locations.columns else []) + ["far_edge"] + key_cols

    numeric_cols = [c for c in out.columns if c not in key_cols]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="raise").astype(np.float32)
    if not keep_zero_rows:
        active_cols = [
            c for c in numeric_cols
            if c not in {"event_source_manual", "event_source_online", "event_source_ai"}
        ]
        if active_cols:
            out = out.loc[(out[active_cols].abs().sum(axis=1) > 0)].copy()
    out = out.sort_values(key_cols).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[key_cols + [c for c in out.columns if c not in key_cols]]


def infer_panel_date_range(data_dir: Path | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if data_dir is None:
        return None, None
    panel_dir = data_dir / "hourly_panels"
    files = sorted(panel_dir.glob("hourly_cluster_*.csv"))
    if not files:
        return None, None
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for path in files:
        try:
            values = pd.read_csv(path, usecols=["date"])["date"]
        except Exception:
            continue
        dates = pd.to_datetime(values, errors="coerce").dropna()
        if not dates.empty:
            mins.append(pd.Timestamp(dates.min()).normalize())
            maxs.append(pd.Timestamp(dates.max()).normalize())
    if not mins:
        return None, None
    return min(mins), max(maxs)


def write_outputs(
    events: Sequence[Event],
    features: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / f"{prefix}_events_canonical.csv"
    feature_path = output_dir / f"{prefix}_event_features_hourly.csv"
    events_to_frame(events).to_csv(event_path, index=False)
    features.to_csv(feature_path, index=False)
    return event_path, feature_path
