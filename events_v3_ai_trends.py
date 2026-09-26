#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Version 3: ask an LLM to generate *synthetic* event scenarios.

This is deliberately NOT a factual event-retrieval mechanism. Generated events
are tagged synthetic and are suitable for what-if / robustness experiments, not
for claiming that an event actually occurred in Nantes.

Default backend: local Ollama HTTP API (no cloud dependency).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests

from event_features_common import (
    Event,
    build_hourly_features,
    france_public_holidays,
    load_node_locations,
    write_outputs,
)

from google_trends_features import (
    combine_with_base,
    fetch_google_trends,
    read_google_trends_export,
    to_hourly as trends_to_hourly,
)

NANTES_LAT = 47.2184
NANTES_LON = -1.5536


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--start-date", default="2019-03-01")
    p.add_argument("--end-date", default="2019-05-31")
    p.add_argument("--city", default="Nantes, France")
    p.add_argument("--n-events", type=int, default=12)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=42,
                   help="Included in the prompt for reproducible scenario intent; LLM output may still vary.")
    p.add_argument("--event-types", nargs="+",
                   default=["sport", "concert", "festival", "fair", "exhibition"])
    p.add_argument("--node-locations", type=Path, default=None)
    p.add_argument("--lead-hours", type=int, default=6)
    p.add_argument("--lag-hours", type=int, default=4)
    p.add_argument("--spatial-sigma-km", type=float, default=3.0)
    p.add_argument("--include-fr-holidays", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--holiday-intensity", type=float, default=0.6)
    trends = p.add_mutually_exclusive_group()
    trends.add_argument("--google-trends-csv", type=Path, default=None,
                        help="Optional archived Google Trends export merged with AI-generated events.")
    trends.add_argument("--fetch-google-trends", action="store_true",
                        help="Fetch Google Trends for --trends-keywords or AI-proposed keywords.")
    p.add_argument("--trends-keywords", nargs="*", default=None,
                   help="Explicit keywords. If omitted with --fetch-google-trends, use keywords proposed by the AI scenario.")
    p.add_argument("--trends-geo", default="FR")
    p.add_argument("--trends-fill", choices=("ffill", "linear"), default="ffill")
    p.add_argument("--trends-lag-hours", type=int, default=0)
    p.add_argument("--trends-lt-one-value", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/events_v3_ai"))
    return p.parse_args()


def prompt_for(args: argparse.Namespace) -> str:
    return f"""Generate a SYNTHETIC event scenario for research, not factual history.
City: {args.city}
Date range: {args.start_date} to {args.end_date}
Number of synthetic events: {args.n_events}
Allowed event types: {', '.join(args.event_types)}
Scenario seed label: {args.seed}

Return ONLY valid JSON with this schema:
{{
  "trend_keywords": ["FC Nantes", "Nantes concert"],
  "events": [
    {{
      "date": "YYYY-MM-DD",
      "start_hour": 0,
      "duration_hours": 3,
      "event_type": "sport",
      "title": "synthetic descriptive title",
      "venue": "plausible generic venue or district",
      "latitude": 47.22,
      "longitude": -1.55,
      "intensity": 0.0,
      "base_relevance": 0.0
    }}
  ]
}}
Constraints:
- Every event is fictional/synthetic; do not claim historical truth.
- intensity and base_relevance must be numbers in [0,1].
- Use plausible Nantes-area coordinates near latitude {NANTES_LAT}, longitude {NANTES_LON}.
- Dates must stay inside the requested range.
- Use a mix of event types and hours.
- trend_keywords must contain 1 to 5 plausible search queries related to the synthetic scenario.
"""


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def query_ollama(args: argparse.Namespace) -> tuple[dict, dict]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You generate synthetic research scenarios and return strict JSON."},
            {"role": "user", "content": prompt_for(args)},
        ],
        "stream": False,
        "format": "json",
        "options": {"seed": args.seed, "temperature": 0.8},
    }
    response = requests.post(args.ollama_url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    raw = response.json()
    content = raw.get("message", {}).get("content", "")
    return extract_json(content), raw


def to_events(data: dict, args: argparse.Namespace) -> list[Event]:
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    allowed = {x.casefold() for x in args.event_types}
    events: list[Event] = []
    for item in data.get("events", []):
        date = pd.to_datetime(item.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        date = pd.Timestamp(date).normalize()
        if not start <= date <= end:
            continue
        event_type = str(item.get("event_type", "other")).strip().lower()
        if event_type.casefold() not in allowed:
            event_type = "other"
        try:
            start_hour = int(item.get("start_hour", 18))
            duration = max(1, int(item.get("duration_hours", 3)))
            intensity = float(item.get("intensity", 0.7))
            relevance = float(item.get("base_relevance", 0.7))
            lat = float(item.get("latitude", NANTES_LAT))
            lon = float(item.get("longitude", NANTES_LON))
        except (TypeError, ValueError):
            continue
        events.append(Event(
            date=date,
            event_type=event_type,
            title=str(item.get("title", "Synthetic event")),
            start_hour=max(0, min(23, start_hour)),
            duration_hours=duration,
            venue=str(item.get("venue", "Synthetic Nantes venue")),
            latitude=lat,
            longitude=lon,
            intensity=max(0.0, min(1.0, intensity)),
            base_relevance=max(0.0, min(1.0, relevance)),
            source="ai_synthetic",
            synthetic=True,
        ))
    return events


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    if end < start:
        raise ValueError("end-date is before start-date")

    data, raw = query_ollama(args)
    events = to_events(data, args)
    if args.include_fr_holidays:
        for year in range(start.year, end.year + 1):
            events.extend(france_public_holidays(year, intensity=args.holiday_intensity))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ai_raw_response.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "ai_parsed_scenario.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    nodes = load_node_locations(args.node_locations)
    features = build_hourly_features(
        events,
        node_locations=nodes,
        lead_hours=args.lead_hours,
        lag_hours=args.lag_hours,
        spatial_sigma_km=args.spatial_sigma_km,
        date_only_mode="default_time",
        default_start_hour=18,
        default_duration_hours=3,
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
        keywords = list(args.trends_keywords or data.get("trend_keywords", []) or ["FC Nantes"])
        keywords = [str(k) for k in keywords[:5]]
        trends_raw = fetch_google_trends(keywords, start, end, args.trends_geo)
        trends_cache = args.output_dir / "google_trends_raw.csv"
        trends_raw.to_csv(trends_cache, index=False)
        trends_source = str(trends_cache)
    else:
        trends_raw = None

    if trends_raw is not None:
        trends_hourly = trends_to_hourly(
            trends_raw, start, end, fill=args.trends_fill, lag_hours=args.trends_lag_hours
        )
        features = combine_with_base(features, trends_hourly, nodes)
    event_path, feature_path = write_outputs(events, features, args.output_dir, "v3")
    print("WARNING: these events are synthetic AI-generated scenarios, not historical facts.")
    print(f"Canonical events: {len(events)} -> {event_path}")
    print(f"Hourly feature rows: {len(features)} -> {feature_path}")
    if trends_source is not None:
        print(f"Google Trends source: {trends_source}")
    print(f"  --event-features-file {feature_path} --require-event-features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
