#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Version 2: retrieve historical Nantes events from online structured sources.

Current implementation uses football-data.co.uk's season CSV for French Ligue 1.
For local stadium events, home matches of Nantes are mapped to Stade de la
Beaujoire. Away fixtures may optionally be included as weaker citywide signals.

No API key is required. The downloaded raw source is cached for reproducibility.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

from event_features_common import (
    Event,
    build_hourly_features,
    france_public_holidays,
    infer_panel_date_range,
    load_node_locations,
    write_outputs,
)

from google_trends_features import (
    combine_with_base,
    fetch_google_trends,
    read_google_trends_export,
    to_hourly as trends_to_hourly,
)

STADIUM_LAT = 47.255631
STADIUM_LON = -1.525375
STADIUM_NAME = "Stade de la Beaujoire"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--season-code", default="1819", help="football-data season code; 1819 = 2018/19")
    p.add_argument("--team", default="Nantes")
    p.add_argument("--source-url", default=None,
                   help="Override the football-data CSV URL.")
    p.add_argument("--include-away", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--home-intensity", type=float, default=1.0)
    p.add_argument("--away-intensity", type=float, default=0.35)
    p.add_argument("--away-relevance", type=float, default=0.35)
    p.add_argument("--date-only-mode", choices=("full_day", "default_time"), default="default_time")
    p.add_argument("--default-start-hour", type=int, default=20,
                   help="Used when the historical source has no kickoff time.")
    p.add_argument("--default-duration-hours", type=int, default=3)
    p.add_argument("--lead-hours", type=int, default=6)
    p.add_argument("--lag-hours", type=int, default=4)
    p.add_argument("--node-locations", type=Path, default=None)
    p.add_argument("--spatial-sigma-km", type=float, default=3.0)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--include-fr-holidays", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--holiday-intensity", type=float, default=0.6)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/events_v2_online"))
    trends = p.add_mutually_exclusive_group()
    trends.add_argument("--google-trends-csv", type=Path, default=None,
                        help="Archived Google Trends export to merge with online scheduled events.")
    trends.add_argument("--fetch-google-trends", action="store_true",
                        help="Fetch Google Trends through pytrends and cache the raw result.")
    p.add_argument("--trends-keywords", nargs="+", default=["FC Nantes"])
    p.add_argument("--trends-geo", default="FR")
    p.add_argument("--trends-fill", choices=("ffill", "linear"), default="ffill")
    p.add_argument("--trends-lag-hours", type=int, default=0)
    p.add_argument("--trends-lt-one-value", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=30.0)
    return p.parse_args()


def fetch_source(args: argparse.Namespace) -> tuple[pd.DataFrame, Path, str]:
    url = args.source_url or f"https://www.football-data.co.uk/mmz4281/{args.season_code}/F1.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"football_data_F1_{args.season_code}.csv"
    if raw_path.exists():
        content = raw_path.read_bytes()
    else:
        response = requests.get(url, timeout=args.timeout, headers={"User-Agent": "event-aware-research/1.0"})
        response.raise_for_status()
        content = response.content
        raw_path.write_bytes(content)
    df = pd.read_csv(io.BytesIO(content))
    return df, raw_path, url


def parse_match_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    for dayfirst in (True, False):
        date = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        if not pd.isna(date):
            return pd.Timestamp(date).normalize()
    return pd.NaT


def matches_to_events(df: pd.DataFrame, args: argparse.Namespace) -> list[Event]:
    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"online source missing columns {sorted(missing)}")
    events: list[Event] = []
    team_cf = args.team.casefold()
    for _, row in df.iterrows():
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        is_home = home.casefold() == team_cf
        is_away = away.casefold() == team_cf
        if not is_home and not (args.include_away and is_away):
            continue
        date = parse_match_date(row["Date"])
        if pd.isna(date):
            continue
        start_hour = None
        if "Time" in df.columns and pd.notna(row.get("Time")):
            try:
                start_hour = int(str(row["Time"]).split(":", 1)[0])
            except Exception:
                start_hour = None
        if is_home:
            events.append(Event(
                date=date,
                event_type="nantes_home_match",
                title=f"{home} vs {away}",
                start_hour=start_hour,
                duration_hours=args.default_duration_hours,
                venue=STADIUM_NAME,
                latitude=STADIUM_LAT,
                longitude=STADIUM_LON,
                intensity=args.home_intensity,
                base_relevance=1.0,
                source="football_data_online",
            ))
        else:
            events.append(Event(
                date=date,
                event_type="nantes_away_match",
                title=f"{home} vs {away}",
                start_hour=start_hour,
                duration_hours=args.default_duration_hours,
                venue=str(row.get("HomeTeam", "")),
                intensity=args.away_intensity,
                base_relevance=args.away_relevance,
                source="football_data_online",
            ))
    return events


def main() -> int:
    args = parse_args()
    df, raw_path, url = fetch_source(args)
    events = matches_to_events(df, args)

    panel_start, panel_end = infer_panel_date_range(args.data_dir)
    start = pd.Timestamp(args.start_date).normalize() if args.start_date else panel_start
    end = pd.Timestamp(args.end_date).normalize() if args.end_date else panel_end
    if start is None:
        start = min(e.date for e in events)
    if end is None:
        end = max(e.date for e in events)

    if args.include_fr_holidays:
        for year in range(start.year, end.year + 1):
            events.extend(france_public_holidays(year, intensity=args.holiday_intensity))
    events = sorted([e for e in events if start - pd.Timedelta(days=2) <= e.date <= end + pd.Timedelta(days=2)],
                    key=lambda e: (e.date, e.event_type, e.title))

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
    trends_source = None
    if args.google_trends_csv is not None:
        trends_raw = read_google_trends_export(
            args.google_trends_csv, lt_one_value=args.trends_lt_one_value
        )
        trends_source = str(args.google_trends_csv)
    elif args.fetch_google_trends:
        trends_raw = fetch_google_trends(
            args.trends_keywords, start, end, args.trends_geo
        )
        trends_cache = args.output_dir / "google_trends_raw.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trends_raw.to_csv(trends_cache, index=False)
        trends_source = str(trends_cache)
    else:
        trends_raw = None

    if trends_raw is not None:
        trends_hourly = trends_to_hourly(
            trends_raw, start, end, fill=args.trends_fill, lag_hours=args.trends_lag_hours
        )
        features = combine_with_base(features, trends_hourly, nodes)
    event_path, feature_path = write_outputs(events, features, args.output_dir, "v2")
    print(f"Source: {url}")
    print(f"Cached raw source: {raw_path}")
    print(f"Study window: {start.date()} .. {end.date()}")
    print(f"Canonical events: {len(events)} -> {event_path}")
    print(f"Hourly feature rows: {len(features)} -> {feature_path}")
    if trends_source is not None:
        print(f"Google Trends source: {trends_source}")
    print(f"  --event-features-file {feature_path} --require-event-features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
