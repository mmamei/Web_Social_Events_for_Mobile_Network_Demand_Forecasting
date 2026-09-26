#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared loader for numeric exogenous/event-aware features.

The loader accepts a CSV keyed by ``date,hour`` and optionally by ``cluster``
and/or ``far_edge``. Every remaining column must be numeric. Feature names are
prefixed with ``exog__`` before they are merged into the traffic panels, so the
forecasting scripts can include them without colliding with endogenous columns.

This deliberately treats scheduled-event indicators and Google-Trends-like
signals identically at model level: both are exogenous contextual features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

EXOG_PREFIX = "exog__"
PANEL_META_COLS = ("date", "hour", "far_edge", "value")


@dataclass(frozen=True)
class EventFeatureTable:
    frame: pd.DataFrame
    join_keys: tuple[str, ...]
    source_feature_names: tuple[str, ...]
    model_feature_names: tuple[str, ...]
    source_path: Path


def _normalize_time_columns(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    frame = frame.copy()
    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        if "date" not in frame.columns:
            frame["date"] = timestamp.dt.normalize()
        if "hour" not in frame.columns:
            frame["hour"] = timestamp.dt.hour
    if "date" not in frame.columns or "hour" not in frame.columns:
        raise ValueError(f"{source}: expected date+hour or timestamp columns")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    frame = frame.dropna(subset=["date", "hour"]).copy()
    frame["hour"] = frame["hour"].astype(int)
    if not frame["hour"].between(0, 23).all():
        raise ValueError(f"{source}: hour must be in 0..23")
    return frame


def load_event_features(path: Path | None) -> EventFeatureTable | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Event feature file not found: {path}")

    frame = _normalize_time_columns(pd.read_csv(path), path)
    if "cluster" in frame.columns:
        frame["cluster"] = pd.to_numeric(frame["cluster"], errors="coerce")
        if frame["cluster"].isna().any():
            raise ValueError(f"{path}: cluster contains non-numeric values")
        frame["cluster"] = frame["cluster"].astype(int)
    if "far_edge" in frame.columns:
        frame["far_edge"] = frame["far_edge"].astype(str)

    join_keys = ["date", "hour"]
    if "cluster" in frame.columns:
        join_keys.append("cluster")
    if "far_edge" in frame.columns:
        join_keys.append("far_edge")

    ignored = set(join_keys) | {"timestamp"}
    source_features = [c for c in frame.columns if c not in ignored]
    if not source_features:
        raise ValueError(f"{path}: no exogenous feature columns found")

    reserved = set(PANEL_META_COLS)
    conflicts = [c for c in source_features if c in reserved]
    if conflicts:
        raise ValueError(f"{path}: reserved feature names: {conflicts}")

    for column in source_features:
        original_non_null = frame[column].notna()
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = original_non_null & converted.isna()
        if bad.any():
            examples = frame.loc[bad, column].astype(str).head(3).tolist()
            raise ValueError(
                f"{path}: exogenous feature {column!r} is not numeric; examples={examples}"
            )
        frame[column] = converted.fillna(0.0).astype(np.float32)
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: feature {column!r} contains inf values")

    duplicate_mask = frame.duplicated(subset=join_keys, keep=False)
    if duplicate_mask.any():
        example = frame.loc[duplicate_mask, join_keys].head(5).to_dict(orient="records")
        raise ValueError(
            f"{path}: duplicate rows for join keys {join_keys}; pre-aggregate first. "
            f"Examples: {example}"
        )

    rename_map = {c: f"{EXOG_PREFIX}{c}" for c in source_features}
    frame = frame[[*join_keys, *source_features]].rename(columns=rename_map)
    return EventFeatureTable(
        frame=frame,
        join_keys=tuple(join_keys),
        source_feature_names=tuple(source_features),
        model_feature_names=tuple(rename_map[c] for c in source_features),
        source_path=path,
    )


def merge_event_features(
    panel: pd.DataFrame,
    cluster: int,
    events: EventFeatureTable | None,
) -> tuple[pd.DataFrame, float]:
    """Merge event features into one cluster panel; missing exogenous values -> 0."""
    panel = panel.copy()
    if events is None:
        return panel, 0.0

    right = events.frame
    if "cluster" in events.join_keys:
        right = right.loc[right["cluster"] == int(cluster)].copy()
        panel["cluster"] = int(cluster)
    if "far_edge" in events.join_keys:
        panel["far_edge"] = panel["far_edge"].astype(str)

    merged = panel.merge(
        right,
        on=list(events.join_keys),
        how="left",
        validate="many_to_one",
        indicator="_event_merge",
    )
    matched = float((merged["_event_merge"] == "both").mean() * 100.0)
    merged = merged.drop(columns=["_event_merge"])
    for column in events.model_feature_names:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).astype(np.float32)
    if "cluster" in merged.columns and "cluster" not in PANEL_META_COLS:
        merged = merged.drop(columns=["cluster"])
    return merged, matched


def exogenous_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if c.startswith(EXOG_PREFIX))


def add_exogenous_to_supervised_parts(frame: pd.DataFrame, parts: list[pd.DataFrame]) -> None:
    cols = exogenous_columns(frame)
    if cols:
        parts.append(frame[cols].astype(float))
