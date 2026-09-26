#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Version 1: build event-aware features from user-supplied CSV files.

Supports the three legacy files used in the experiment:
  - partite.csv      -> FC Nantes / Nantes football fixture-day proxy
  - champions.csv    -> UEFA Champions League match-day proxy
  - eu_league.csv    -> UEFA Europa League match-day proxy

The legacy files contain dates only. By default a date-only event is represented
as a full-day scheduled-event proxy. Use --date-only-mode default_time if you
want an assumed start hour and duration instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from event_features_common import (
    Event,
    build_hourly_features,
    france_public_holidays,
    infer_panel_date_range,
    load_node_locations,
    read_one_column_dates,
    write_outputs,
)

from google_trends_features import (
    combine_with_base,
    read_google_trends_export,
    to_hourly as trends_to_hourly,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--partite", type=Path, default=Path("partite.csv"))
    p.add_argument("--champions", type=Path, default=Path("champions.csv"))
    p.add_argument("--eu-league", type=Path, default=Path("eu_league.csv"))
    p.add_argument("--node-locations", type=Path, default=None,
                   help="Optional CSV: far_edge,lat,lon[,cluster]. Enables geographic relevance.")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="If hourly panels exist here, their date range is used automatically.")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--lead-hours", type=int, default=6)
    p.add_argument("--lag-hours", type=int, default=4)
    p.add_argument("--spatial-sigma-km", type=float, default=3.0)
    p.add_argument("--date-only-mode", choices=("full_day", "default_time"), default="full_day")
    p.add_argument("--default-start-hour", type=int, default=20)
    p.add_argument("--default-duration-hours", type=int, default=3)
    p.add_argument("--partite-intensity", type=float, default=1.0)
    p.add_argument("--champions-intensity", type=float, default=1.0)
    p.add_argument("--europa-intensity", type=float, default=1.0)
    p.add_argument("--partite-relevance", type=float, default=1.0)
    p.add_argument("--champions-relevance", type=float, default=1.0)
    p.add_argument("--europa-relevance", type=float, default=1.0)
    p.add_argument("--include-fr-holidays", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--holiday-intensity", type=float, default=0.6)
    p.add_argument("--google-trends-csv", type=Path, default=None,
                   help="Optional archived Google Trends CSV export. When supplied, trend signals are merged with scheduled-event features.")
    p.add_argument("--trends-fill", choices=("ffill", "linear"), default="ffill")
    p.add_argument("--trends-lag-hours", type=int, default=0,
                   help="Optional lag applied to Google Trends features to avoid contemporaneous/future signal use.")
    p.add_argument("--trends-lt-one-value", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/events_v1_csv"))
    return p.parse_args()


def add_legacy(events: list[Event], path: Path, event_type: str, intensity: float,
               relevance: float, dayfirst: bool | None, title_prefix: str) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found; skipping")
        return
    for date in read_one_column_dates(path, dayfirst=dayfirst):
        events.append(Event(
            date=date,
            event_type=event_type,
            title=f"{title_prefix} {date.date()}",
            start_hour=None,
            duration_hours=24,
            intensity=float(intensity),
            base_relevance=float(relevance),
            source="manual_csv",
        ))


def main() -> int:
    args = parse_args()
    events: list[Event] = []
    add_legacy(events, args.partite, "nantes_fixture", args.partite_intensity,
               args.partite_relevance, None, "Nantes football fixture-day")
    add_legacy(events, args.champions, "champions_matchday", args.champions_intensity,
               args.champions_relevance, True, "Champions League match-day")
    add_legacy(events, args.eu_league, "europa_matchday", args.europa_intensity,
               args.europa_relevance, True, "Europa League match-day")

    panel_start, panel_end = infer_panel_date_range(args.data_dir)
    start = pd.Timestamp(args.start_date).normalize() if args.start_date else panel_start
    end = pd.Timestamp(args.end_date).normalize() if args.end_date else panel_end

    if start is None:
        start = min(e.date for e in events) if events else pd.Timestamp("2019-03-01")
    if end is None:
        end = max(e.date for e in events) if events else pd.Timestamp("2019-05-31")

    if args.include_fr_holidays:
        for year in range(start.year, end.year + 1):
            events.extend(france_public_holidays(year, intensity=args.holiday_intensity))

    # Keep canonical event output relevant to the study window (+ context margin).
    events = sorted(
        [e for e in events if start - pd.Timedelta(days=2) <= e.date <= end + pd.Timedelta(days=2)],
        key=lambda e: (e.date, e.event_type, e.title),
    )
    nodes = load_node_locations(args.node_locations)
    features = build_hourly_features(
        events,
        node_locations=nodes,
        lead_hours=args.lead_hours,
        lag_hours=args.lag_hours,
        spatial_sigma_km=args.spatial_sigma_km,
        date_only_mode=args.date_only_mode,
        default_start_hour=args.default_start_hour,
        default_duration_hours=args.default_duration_hours,
        start_date=start,
        end_date=end,
    )
    if args.google_trends_csv is not None:
        trends_raw = read_google_trends_export(
            args.google_trends_csv, lt_one_value=args.trends_lt_one_value
        )
        trends_hourly = trends_to_hourly(
            trends_raw, start, end, fill=args.trends_fill, lag_hours=args.trends_lag_hours
        )
        features = combine_with_base(features, trends_hourly, nodes)
    event_path, feature_path = write_outputs(events, features, args.output_dir, "v1")
    print(f"Study window: {start.date()} .. {end.date()}")
    print(f"Canonical events: {len(events)} -> {event_path}")
    print(f"Hourly feature rows: {len(features)} -> {feature_path}")
    if args.google_trends_csv is not None:
        print(f"Google Trends source: {args.google_trends_csv}")
    print("Use with the forecasting model as:")
    print(f"  --event-features-file {feature_path} --require-event-features")
    if args.node_locations is None:
        print("NOTE: no --node-locations supplied: event relevance is global, not cell-distance weighted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
