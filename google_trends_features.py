#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build hourly Google Trends exogenous features.

Two input modes are supported:
  1. ``--trends-csv`` consumes a local Google Trends export. This mode is
     deterministic once the raw export is archived with the experiment.
  2. ``--fetch`` queries Google Trends through the unofficial ``pytrends``
     client, caches the returned table, and processes it with the same pipeline.

The resulting CSV is compatible with the event-aware ``ml_exps*`` scripts in
this package and can be merged with scheduled-event features.

Google Trends values are treated as optional web/search context. Experiments can
therefore be run with scheduled events only, Trends only, both sources, or no
exogenous context.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def slug(text: str) -> str:
    text = re.sub(r"\s*:\s*\([^)]*\)\s*$", "", str(text).strip())
    value = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return value or "signal"


def _coerce_interest(series: pd.Series, lt_one_value: float) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"<1": str(lt_one_value), "< 1": str(lt_one_value)})
    text = text.str.replace("%", "", regex=False)
    values = pd.to_numeric(text, errors="coerce")
    return values


def read_google_trends_export(path: Path, lt_one_value: float = 0.5) -> pd.DataFrame:
    """Read common Google Trends CSV export layouts, including metadata lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    candidate: pd.DataFrame | None = None
    for skip in range(0, 15):
        try:
            df = pd.read_csv(path, skiprows=skip, sep=None, engine="python")
        except Exception:
            continue
        if df.shape[1] < 2 or df.empty:
            continue
        first = df.columns[0]
        dates = pd.to_datetime(df[first], errors="coerce")
        good = float(dates.notna().mean())
        if good >= 0.5:
            candidate = df.copy()
            candidate.insert(0, "__parsed_date", dates)
            break
    if candidate is None:
        raise ValueError(f"Could not identify a dated Google Trends table in {path}")

    candidate = candidate.dropna(subset=["__parsed_date"]).copy()
    date_col = candidate.columns[1]
    signal_cols = [
        c for c in candidate.columns[2:]
        if str(c).strip().casefold() not in {"ispartial", "is_partial"}
    ]
    if not signal_cols:
        # If insertion shifted differently because of odd input, use all original non-date cols.
        signal_cols = [
            c for c in candidate.columns
            if c not in {"__parsed_date", date_col}
            and str(c).strip().casefold() not in {"ispartial", "is_partial"}
        ]
    if not signal_cols:
        raise ValueError(f"No interest columns found in {path}")

    out = pd.DataFrame({"date": pd.to_datetime(candidate["__parsed_date"]).dt.normalize()})
    for column in signal_cols:
        values = _coerce_interest(candidate[column], lt_one_value=lt_one_value)
        if values.notna().sum() == 0:
            continue
        out[f"trend_google_{slug(column)}"] = values.astype(float)
    if out.shape[1] == 1:
        raise ValueError(f"All Google Trends value columns were non-numeric in {path}")
    return out


def fetch_google_trends(
    keywords: Sequence[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    geo: str,
) -> pd.DataFrame:
    if not keywords:
        raise ValueError("At least one Google Trends keyword is required")
    if len(keywords) > 5:
        raise ValueError("pytrends accepts at most five keywords per comparable request")
    try:
        from pytrends.request import TrendReq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pytrends is not installed. Install it with: pip install pytrends"
        ) from exc

    pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.5)
    timeframe = f"{pd.Timestamp(start_date).date()} {pd.Timestamp(end_date).date()}"
    pytrends.build_payload(list(keywords), timeframe=timeframe, geo=geo)
    raw = pytrends.interest_over_time()
    if raw is None or raw.empty:
        raise RuntimeError("Google Trends returned no data")
    raw = raw.reset_index()
    if "isPartial" in raw.columns:
        raw = raw.drop(columns=["isPartial"])
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    out = raw[["date", *[k for k in keywords if k in raw.columns]]].copy()
    rename = {k: f"trend_google_{slug(k)}" for k in keywords if k in out.columns}
    return out.rename(columns=rename)


def normalize_interest(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    signal_cols = [c for c in out.columns if c != "date"]
    for column in signal_cols:
        values = pd.to_numeric(out[column], errors="coerce").astype(float)
        # Native Trends exports are 0..100. Already-normalized input is kept.
        if values.notna().any() and float(values.max()) > 1.5:
            values = values / 100.0
        out[column] = values.clip(lower=0.0, upper=1.0)
    return out


def to_hourly(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    fill: str = "ffill",
    lag_hours: int = 0,
) -> pd.DataFrame:
    if lag_hours < 0:
        raise ValueError("lag_hours must be >= 0")
    frame = normalize_interest(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).groupby("date", as_index=False).mean(numeric_only=True)
    start = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(hours=23)
    idx = pd.date_range(start, end_ts, freq="h")
    values = frame.set_index("date").sort_index().reindex(idx)
    if fill == "ffill":
        values = values.ffill()
    elif fill == "linear":
        values = values.interpolate(method="time").ffill()
    else:
        raise ValueError("fill must be ffill or linear")
    values = values.fillna(0.0)
    if lag_hours:
        values = values.shift(int(lag_hours)).fillna(0.0)

    result = values.reset_index().rename(columns={"index": "timestamp"})
    result.insert(0, "date", result["timestamp"].dt.normalize())
    result.insert(1, "hour", result["timestamp"].dt.hour.astype(int))
    result = result.drop(columns=["timestamp"])
    signal_cols = [c for c in result.columns if c not in {"date", "hour"}]
    if signal_cols:
        result["trend_google_mean"] = result[signal_cols].mean(axis=1)
        result["trend_google_max"] = result[signal_cols].max(axis=1)
    result["trend_source_google"] = 1.0
    result[signal_cols + ["trend_google_mean", "trend_google_max", "trend_source_google"]] = (
        result[signal_cols + ["trend_google_mean", "trend_google_max", "trend_source_google"]]
        .astype(np.float32)
    )
    return result


def _normalize_base(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        if "date" not in out.columns:
            out["date"] = ts.dt.normalize()
        if "hour" not in out.columns:
            out["hour"] = ts.dt.hour
    if "date" not in out.columns or "hour" not in out.columns:
        raise ValueError("base feature CSV must contain date+hour or timestamp")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out = out.dropna(subset=["date", "hour"]).copy()
    out["hour"] = out["hour"].astype(int)
    if "far_edge" in out.columns:
        out["far_edge"] = out["far_edge"].astype(str)
    if "cluster" in out.columns:
        out["cluster"] = pd.to_numeric(out["cluster"], errors="raise").astype(int)
    return out


def combine_with_base(
    base: pd.DataFrame | None,
    trends: pd.DataFrame,
    node_locations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    trends = _normalize_base(trends)
    if base is None:
        out = trends
    else:
        base = _normalize_base(base)
        entity_keys = [k for k in ("cluster", "far_edge") if k in base.columns]
        if entity_keys:
            if node_locations is not None:
                entities = node_locations[entity_keys].drop_duplicates().copy()
            else:
                entities = base[entity_keys].drop_duplicates().copy()
            entities["__key"] = 1
            trend_expanded = trends.copy()
            trend_expanded["__key"] = 1
            trend_expanded = entities.merge(trend_expanded, on="__key", how="outer").drop(columns=["__key"])
            join_keys = [*entity_keys, "date", "hour"]
            out = trend_expanded.merge(base, on=join_keys, how="outer", validate="one_to_one")
        else:
            out = trends.merge(base, on=["date", "hour"], how="outer", validate="one_to_one")
    key_cols = [k for k in ("cluster", "far_edge", "date", "hour") if k in out.columns]
    feature_cols = [c for c in out.columns if c not in key_cols and c != "timestamp"]
    for column in feature_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(np.float32)
    out = out.sort_values(key_cols).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[key_cols + feature_cols]


def load_node_locations(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"far_edge"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: expected far_edge column")
    df = df.copy()
    df["far_edge"] = df["far_edge"].astype(str)
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="raise").astype(int)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--trends-csv", type=Path, help="Archived Google Trends CSV export")
    source.add_argument("--fetch", action="store_true", help="Fetch through pytrends")
    p.add_argument("--keywords", nargs="+", default=["FC Nantes"], help="Used with --fetch")
    p.add_argument("--geo", default="FR", help="Google Trends geo code used with --fetch")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--fill", choices=("ffill", "linear"), default="ffill")
    p.add_argument("--lag-hours", type=int, default=0,
                   help="Shift the web signal into the past-facing feature vector; 24 means only yesterday's value is used.")
    p.add_argument("--lt-one-value", type=float, default=0.5,
                   help="Numeric replacement for '<1' in exported Trends CSVs before dividing by 100.")
    p.add_argument("--base-event-features", type=Path, default=None,
                   help="Optional scheduled-event feature CSV to merge with Trends.")
    p.add_argument("--node-locations", type=Path, default=None,
                   help="Optional far_edge[,cluster] table when base features are node-specific.")
    p.add_argument("--output", type=Path, default=Path("outputs/event_features_with_google_trends.csv"))
    p.add_argument("--raw-cache", type=Path, default=Path("outputs/google_trends_raw.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    if end < start:
        raise ValueError("end-date is before start-date")

    if args.fetch:
        raw = fetch_google_trends(args.keywords, start, end, args.geo)
        args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(args.raw_cache, index=False)
        print(f"Cached Google Trends source: {args.raw_cache}")
    else:
        raw = read_google_trends_export(args.trends_csv, lt_one_value=args.lt_one_value)

    hourly = to_hourly(raw, start, end, fill=args.fill, lag_hours=args.lag_hours)
    base = pd.read_csv(args.base_event_features) if args.base_event_features else None
    nodes = load_node_locations(args.node_locations)
    combined = combine_with_base(base, hourly, nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Hourly rows: {len(combined)}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
