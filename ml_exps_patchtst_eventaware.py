#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone PatchTST-style forecasting experiment with optional exogenous features.

The script evaluates a single PatchTST-style model independently for every
far-edge node and service cluster. It can run in three modes:

1. no-event mode: traffic history + calendar context only;
2. event-aware mode: traffic history + calendar context + aligned numeric
   exogenous features;
3. paired comparison: run both modes with the same split and model settings.

The exogenous CSV can contain scheduled-event indicators, Google Trends
signals, or any other numeric contextual variables. It must be keyed by
``date,hour`` and may optionally also contain ``cluster`` and/or ``far_edge``.
Missing contextual values are interpreted as zero.

Default PatchTST-style configuration:
    context length : 24 hours
    patch length   : 6
    patch stride   : 3
    d_model        : 64
    attention heads: 4
    encoder layers : 2
    dropout        : 0.10
    optimizer      : AdamW
    learning rate  : 1e-3
    weight decay   : 1e-4
    batch size     : 32
    max epochs     : 50

The final test block is chronological and never used for model selection.
An internal chronological validation block is taken from the pre-test data for
early stopping. Scaling statistics are estimated from the training block only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required. Install dependencies with: "
        "pip install torch pandas numpy scikit-learn"
    ) from exc

LOGGER = logging.getLogger("patchtst_eventaware")
META_COLS = ("date", "hour", "far_edge", "value")
CAP_LEVELS = (0.70, 0.80, 0.90)
EXOG_PREFIX = "exog__"


@dataclass(frozen=True)
class EventFeatureTable:
    frame: pd.DataFrame
    join_keys: tuple[str, ...]
    source_feature_names: tuple[str, ...]
    model_feature_names: tuple[str, ...]
    source_path: Path


@dataclass(frozen=True)
class PatchConfig:
    context_length: int
    patch_length: int
    patch_stride: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    validation_fraction: float


@dataclass(frozen=True)
class ScaleState:
    traffic_mean: float
    traffic_std: float
    side_mean: np.ndarray
    side_std: np.ndarray


@dataclass
class NodeSamples:
    cluster: int
    far_edge: str
    history: np.ndarray
    side: np.ndarray
    target: np.ndarray
    keys: pd.DataFrame
    side_feature_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone PatchTST-style event-aware forecasting experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clusters", type=int, nargs="*", default=None)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-train-samples", type=int, default=48)
    parser.add_argument("--event-features-file", type=Path, default=None)
    parser.add_argument("--require-event-features", action="store_true")
    parser.add_argument(
        "--compare-event-awareness",
        action="store_true",
        help="Run paired no-event and event-aware experiments with identical settings.",
    )
    parser.add_argument(
        "--include-weekend-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-nodes-per-cluster", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--context-length", type=int, default=24)
    parser.add_argument("--patch-length", type=int, default=6)
    parser.add_argument("--patch-stride", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--loss",
        choices=("mae", "mse", "huber"),
        default="mae",
        help="Training and early-stopping loss.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/patchtst_eventaware"),
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    if not 0.0 < args.test_fraction < 0.5:
        parser.error("--test-fraction must be in (0, 0.5)")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    if args.context_length < 2:
        parser.error("--context-length must be >= 2")
    if args.patch_length < 1 or args.patch_length > args.context_length:
        parser.error("--patch-length must be in [1, context-length]")
    if args.patch_stride < 1:
        parser.error("--patch-stride must be >= 1")
    if args.d_model < 1 or args.heads < 1 or args.layers < 1:
        parser.error("--d-model, --heads, and --layers must be >= 1")
    if args.d_model % args.heads != 0:
        parser.error("--d-model must be divisible by --heads")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0,1)")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        parser.error("--epochs and --batch-size must be >= 1; --patience must be >= 0")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate must be > 0 and --weight-decay must be >= 0")
    if args.max_nodes_per_cluster < 0:
        parser.error("--max-nodes-per-cluster cannot be negative")
    if args.require_event_features and args.event_features_file is None:
        parser.error("--require-event-features requires --event-features-file")
    if args.compare_event_awareness and args.event_features_file is None:
        parser.error("--compare-event-awareness requires --event-features-file")
    return args


def setup(seed: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / den)) * 100.0)


def traffic_reduction(
    y_true: np.ndarray,
    prediction: np.ndarray,
    capacity_percentile: float,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    capacity = np.nanpercentile(y_true, capacity_percentile * 100.0)
    overflow_fixed = np.nansum(np.maximum(0.0, y_true - capacity))
    dynamic_capacity = np.maximum(capacity, prediction)
    overflow_dynamic = np.nansum(np.maximum(0.0, y_true - dynamic_capacity))
    if overflow_fixed <= 1e-12:
        return float("nan")
    return float(100.0 * (overflow_fixed - overflow_dynamic) / overflow_fixed)


# -----------------------------------------------------------------------------
# Exogenous feature loading
# -----------------------------------------------------------------------------
def load_event_features(path: Path | None) -> EventFeatureTable | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exogenous feature file not found: {path}")

    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        if "date" not in frame.columns:
            frame["date"] = timestamp.dt.normalize()
        if "hour" not in frame.columns:
            frame["hour"] = timestamp.dt.hour
    if "date" not in frame.columns or "hour" not in frame.columns:
        raise ValueError(f"{path}: expected date+hour or timestamp columns")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    frame = frame.dropna(subset=["date", "hour"]).copy()
    frame["hour"] = frame["hour"].astype(int)
    if not frame["hour"].between(0, 23).all():
        raise ValueError(f"{path}: hour must be in 0..23")

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
        raise ValueError(f"{path}: no numeric exogenous columns were found")

    for column in source_features:
        original_non_null = frame[column].notna()
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = original_non_null & converted.isna()
        if bad.any():
            examples = frame.loc[bad, column].astype(str).head(3).tolist()
            raise ValueError(
                f"{path}: feature {column!r} is not numeric; examples={examples}"
            )
        frame[column] = converted.fillna(0.0).astype(np.float32)
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: feature {column!r} contains non-finite values")

    duplicate_mask = frame.duplicated(subset=join_keys, keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, join_keys].head(5).to_dict(orient="records")
        raise ValueError(
            f"{path}: duplicate rows for join keys {join_keys}; examples={examples}"
        )

    rename_map = {column: f"{EXOG_PREFIX}{column}" for column in source_features}
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
        indicator="_context_merge",
    )
    matched = float((merged["_context_merge"] == "both").mean() * 100.0)
    merged = merged.drop(columns=["_context_merge"])
    for column in events.model_feature_names:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).astype(np.float32)
    if "cluster" in merged.columns and "cluster" not in META_COLS:
        merged = merged.drop(columns=["cluster"])
    return merged, matched


# -----------------------------------------------------------------------------
# Traffic panel loading
# -----------------------------------------------------------------------------
def ensure_services_clusters(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "services_clusters.csv"
    if path.exists():
        return pd.read_csv(path)

    raw_files = sorted(data_dir.glob("nantes_antenna_serv_*.csv"))
    service_cluster_path = data_dir / "service_clustering.csv"
    if not raw_files or not service_cluster_path.exists():
        raise FileNotFoundError(
            f"Missing {path}; rebuilding also requires nantes_antenna_serv_*.csv "
            f"and {service_cluster_path}"
        )

    raw = pd.concat((pd.read_csv(p) for p in raw_files), ignore_index=True).drop_duplicates()
    service_clusters = pd.read_csv(service_cluster_path)
    merged = raw.merge(service_clusters, on="service", how="left")
    value_columns = [str(i) for i in range(96)]
    missing = [c for c in value_columns if c not in merged.columns]
    if missing:
        raise ValueError(f"Raw data are missing expected time-slot columns: {missing[:5]}")

    aggregated = merged.groupby(["date", "labels", "lon", "lat"], as_index=False)[value_columns].sum()
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(path, index=False)
    LOGGER.info("Created %s", path)
    return aggregated


def build_hourly_for_cluster(
    cluster: int,
    services: pd.DataFrame,
    antennas: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    value_columns = [str(i) for i in range(96)]
    subset = services.loc[
        services["labels"] == cluster,
        ["date", "lon", "lat", *value_columns],
    ].copy()
    if subset.empty:
        raise ValueError(f"No traffic data found for cluster {cluster}")

    subset = subset.groupby(["date", "lon", "lat"], as_index=False)[value_columns].sum()
    subset = antennas[["lat", "lon", "far_edge"]].merge(
        subset,
        on=["lat", "lon"],
        how="inner",
    )
    subset = subset.groupby(["lat", "lon", "far_edge", "date"], as_index=False)[value_columns].sum()
    long_frame = subset.melt(
        id_vars=["date", "far_edge"],
        value_vars=value_columns,
        var_name="slot",
        value_name="value",
    )
    slot_text = long_frame["slot"].astype(str)
    if slot_text.str.fullmatch(r"\d+").all():
        long_frame["hour"] = slot_text.astype(int) // 4
    else:
        long_frame["hour"] = pd.to_numeric(
            slot_text.str.extract(r"^(\d+)")[0],
            errors="coerce",
        )
    long_frame["date"] = pd.to_datetime(long_frame["date"], errors="coerce").dt.normalize()
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="coerce")
    hourly = (
        long_frame.dropna(subset=["date", "hour", "far_edge", "value"])
        .groupby(["date", "far_edge", "hour"], as_index=False)["value"]
        .sum()
    )
    hourly["hour"] = hourly["hour"].astype(int)
    hourly = hourly.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, index=False)
    return hourly


def discover_clusters(data_dir: Path, requested: Sequence[int] | None) -> list[int]:
    if requested:
        return sorted(set(int(v) for v in requested))
    found: list[int] = []
    for path in (data_dir / "hourly_panels").glob("hourly_cluster_*.csv"):
        match = re.search(r"hourly_cluster_(-?\d+)\.csv$", path.name)
        if match:
            found.append(int(match.group(1)))
    if found:
        return sorted(set(found))
    services = ensure_services_clusters(data_dir)
    return sorted(services["labels"].dropna().astype(int).unique().tolist())


def load_panels(
    data_dir: Path,
    clusters: Sequence[int],
    events: EventFeatureTable | None,
    force_rebuild: bool,
) -> dict[int, pd.DataFrame]:
    panels: dict[int, pd.DataFrame] = {}
    services: pd.DataFrame | None = None
    antennas: pd.DataFrame | None = None

    for cluster in clusters:
        path = data_dir / "hourly_panels" / f"hourly_cluster_{cluster}.csv"
        if path.exists() and not force_rebuild:
            frame = pd.read_csv(path)
        else:
            if services is None:
                services = ensure_services_clusters(data_dir)
            if antennas is None:
                antennas_path = data_dir / "nantes_antenna_clustering.csv"
                if not antennas_path.exists():
                    raise FileNotFoundError(antennas_path)
                antennas = pd.read_csv(antennas_path)
            frame = build_hourly_for_cluster(cluster, services, antennas, path)

        missing = set(META_COLS).difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        frame = frame[list(META_COLS)].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["far_edge"] = frame["far_edge"].astype(str)
        frame = frame.dropna(subset=["date", "hour", "far_edge", "value"])
        frame["hour"] = frame["hour"].astype(int)
        frame = frame.loc[frame["hour"].between(0, 23)].copy()

        frame, matched = merge_event_features(frame, cluster, events)
        frame = frame.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)
        panels[int(cluster)] = frame
        if events is not None:
            LOGGER.info(
                "Cluster %s: %d nodes, contextual rows matched %.2f%%",
                cluster,
                frame["far_edge"].nunique(),
                matched,
            )
        else:
            LOGGER.info("Cluster %s: %d nodes", cluster, frame["far_edge"].nunique())
    return panels


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------
def prepare_node_frame(
    node_frame: pd.DataFrame,
    include_weekend_onehot: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = node_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce").astype(int)
    frame = frame.sort_values(["date", "hour"]).reset_index(drop=True)
    frame["dow"] = frame["date"].dt.weekday.astype(int)
    frame["weekend"] = (frame["dow"] >= 5).astype(int)

    hour_dummies = pd.get_dummies(
        pd.Categorical(frame["hour"], categories=range(24)),
        prefix="h",
        dtype=float,
    )
    dow_dummies = pd.get_dummies(
        pd.Categorical(frame["dow"], categories=range(7)),
        prefix="d",
        dtype=float,
    )
    hour_dummies.index = frame.index
    dow_dummies.index = frame.index

    side_parts = [hour_dummies, dow_dummies]
    if include_weekend_onehot:
        weekend_dummies = pd.get_dummies(
            pd.Categorical(frame["weekend"], categories=(0, 1)),
            prefix="weekend",
            dtype=float,
        )
        weekend_dummies.index = frame.index
        side_parts.append(weekend_dummies)

    exogenous = sorted(c for c in frame.columns if c.startswith(EXOG_PREFIX))
    if exogenous:
        side_parts.append(frame[exogenous].astype(float))

    side = pd.concat(side_parts, axis=1)
    side_names = tuple(side.columns.astype(str).tolist())
    out = pd.concat(
        [frame[["date", "hour", "far_edge", "value"]], side],
        axis=1,
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out, side_names


def build_node_samples(
    node_frame: pd.DataFrame,
    cluster: int,
    far_edge: str,
    context_length: int,
    include_weekend_onehot: bool,
) -> NodeSamples:
    frame, side_names = prepare_node_frame(node_frame, include_weekend_onehot)
    values = frame["value"].to_numpy(dtype=np.float32)
    side_matrix = frame.loc[:, side_names].to_numpy(dtype=np.float32)

    if len(frame) <= context_length:
        raise ValueError("Series is shorter than the context window")

    history: list[np.ndarray] = []
    side: list[np.ndarray] = []
    target: list[float] = []
    key_rows: list[dict[str, Any]] = []

    for index in range(context_length, len(frame)):
        history.append(values[index - context_length : index])
        side.append(side_matrix[index])
        target.append(float(values[index]))
        key_rows.append(
            {
                "date": frame.at[index, "date"],
                "hour": int(frame.at[index, "hour"]),
                "far_edge": str(frame.at[index, "far_edge"]),
            }
        )

    return NodeSamples(
        cluster=int(cluster),
        far_edge=str(far_edge),
        history=np.asarray(history, dtype=np.float32),
        side=np.asarray(side, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        keys=pd.DataFrame(key_rows),
        side_feature_names=side_names,
    )


def split_indices(
    n_samples: int,
    test_fraction: float,
    validation_fraction: float,
    min_train_samples: int,
) -> tuple[slice, slice, slice]:
    n_test = max(1, int(round(n_samples * test_fraction)))
    n_pretest = n_samples - n_test
    n_val = max(1, int(round(n_pretest * validation_fraction)))
    n_train = n_pretest - n_val
    if n_train < min_train_samples or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Insufficient samples: train={n_train}, val={n_val}, test={n_test}"
        )
    return slice(0, n_train), slice(n_train, n_pretest), slice(n_pretest, n_samples)


def fit_scale(history_train: np.ndarray, side_train: np.ndarray) -> ScaleState:
    traffic_values = history_train.reshape(-1).astype(float)
    traffic_mean = float(np.mean(traffic_values))
    traffic_std = float(np.std(traffic_values))
    if traffic_std <= 1e-9:
        traffic_std = 1.0

    if side_train.shape[1] == 0:
        side_mean = np.zeros((0,), dtype=np.float32)
        side_std = np.ones((0,), dtype=np.float32)
    else:
        side_mean = np.mean(side_train, axis=0).astype(np.float32)
        side_std = np.std(side_train, axis=0).astype(np.float32)
        side_std = np.where(side_std <= 1e-9, 1.0, side_std).astype(np.float32)

    return ScaleState(
        traffic_mean=traffic_mean,
        traffic_std=traffic_std,
        side_mean=side_mean,
        side_std=side_std,
    )


def scale_history(history: np.ndarray, state: ScaleState) -> np.ndarray:
    return ((history - state.traffic_mean) / state.traffic_std).astype(np.float32)


def scale_side(side: np.ndarray, state: ScaleState) -> np.ndarray:
    if side.shape[1] == 0:
        return side.astype(np.float32, copy=False)
    return ((side - state.side_mean) / state.side_std).astype(np.float32)


def scale_target(target: np.ndarray, state: ScaleState) -> np.ndarray:
    return ((target - state.traffic_mean) / state.traffic_std).astype(np.float32)


def inverse_target(target: np.ndarray, state: ScaleState) -> np.ndarray:
    return np.asarray(target, dtype=np.float32) * state.traffic_std + state.traffic_mean


# -----------------------------------------------------------------------------
# PatchTST-style model
# -----------------------------------------------------------------------------
class PatchTSTRegressor(nn.Module):
    """Patch-based Transformer with optional target-time contextual features."""

    def __init__(
        self,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        n_side_features: int,
    ) -> None:
        super().__init__()
        if context_length < patch_length:
            raise ValueError("context_length must be >= patch_length")
        self.context_length = int(context_length)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.n_patches = 1 + (context_length - patch_length) // patch_stride
        if self.n_patches < 1:
            raise ValueError("Patch configuration produces no patches")

        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.n_side_features = int(n_side_features)
        if self.n_side_features > 0:
            self.side_encoder = nn.Sequential(
                nn.Linear(self.n_side_features, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input = self.n_patches * d_model + d_model
        else:
            self.side_encoder = None
            head_input = self.n_patches * d_model

        self.head = nn.Sequential(
            nn.Linear(head_input, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, history: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        patches = history.unfold(
            dimension=1,
            size=self.patch_length,
            step=self.patch_stride,
        )
        encoded = self.patch_projection(patches)
        encoded = encoded + self.position_embedding[:, : encoded.shape[1], :]
        encoded = self.encoder(encoded)
        representation = encoded.reshape(encoded.shape[0], -1)

        if self.side_encoder is not None:
            side_representation = self.side_encoder(side)
            representation = torch.cat([representation, side_representation], dim=1)

        return self.head(representation).squeeze(-1)


def make_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    return nn.L1Loss()


def make_loader(
    history: np.ndarray,
    side: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(history, dtype=torch.float32),
        torch.tensor(side, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def train_node_model(
    samples: NodeSamples,
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
    config: PatchConfig,
    loss_name: str,
    device: torch.device,
    seed: int,
    num_workers: int,
) -> tuple[np.ndarray, int, float, ScaleState, PatchTSTRegressor]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    state = fit_scale(samples.history[train_slice], samples.side[train_slice])
    history_train = scale_history(samples.history[train_slice], state)
    side_train = scale_side(samples.side[train_slice], state)
    target_train = scale_target(samples.target[train_slice], state)

    history_val = scale_history(samples.history[val_slice], state)
    side_val = scale_side(samples.side[val_slice], state)
    target_val = scale_target(samples.target[val_slice], state)

    history_test = scale_history(samples.history[test_slice], state)
    side_test = scale_side(samples.side[test_slice], state)

    train_loader = make_loader(
        history_train,
        side_train,
        target_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )

    model = PatchTSTRegressor(
        context_length=config.context_length,
        patch_length=config.patch_length,
        patch_stride=config.patch_stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        n_side_features=samples.side.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = make_loss(loss_name)

    val_history_tensor = torch.tensor(history_val, dtype=torch.float32, device=device)
    val_side_tensor = torch.tensor(side_val, dtype=torch.float32, device=device)
    val_target_tensor = torch.tensor(target_val, dtype=torch.float32, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for history_batch, side_batch, target_batch in train_loader:
            history_batch = history_batch.to(device)
            side_batch = side_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(history_batch, side_batch)
            loss = loss_fn(prediction, target_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_history_tensor, val_side_tensor)
            val_loss = float(loss_fn(val_prediction, val_target_tensor).item())

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    test_history_tensor = torch.tensor(history_test, dtype=torch.float32, device=device)
    test_side_tensor = torch.tensor(side_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction_scaled = model(test_history_tensor, test_side_tensor).detach().cpu().numpy()
    prediction = inverse_target(prediction_scaled, state).astype(float)
    return prediction, best_epoch, best_val, state, model


def evaluate_node(
    samples: NodeSamples,
    args: argparse.Namespace,
    config: PatchConfig,
    device: torch.device,
    model_dir: Path,
    node_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_slice, val_slice, test_slice = split_indices(
        n_samples=len(samples.target),
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        min_train_samples=args.min_train_samples,
    )

    prediction, best_epoch, best_val, state, model = train_node_model(
        samples=samples,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        config=config,
        loss_name=args.loss,
        device=device,
        seed=node_seed,
        num_workers=args.num_workers,
    )

    y_true = samples.target[test_slice].astype(float)
    history_test = samples.history[test_slice].astype(float)
    naive = history_test[:, -1]

    row: dict[str, Any] = {
        "cluster": samples.cluster,
        "far_edge": samples.far_edge,
        "model": "PatchTST-style",
        "event_aware": bool(any(name.startswith(EXOG_PREFIX) for name in samples.side_feature_names)),
        "train_samples": train_slice.stop - train_slice.start,
        "validation_samples": val_slice.stop - val_slice.start,
        "test_samples": test_slice.stop - test_slice.start,
        "best_epoch": best_epoch,
        "validation_loss_scaled": best_val,
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": rmse(y_true, prediction),
        "MAPE": mape(y_true, prediction),
        "R2": float(r2_score(y_true, prediction)),
        "Naive_MAE": float(mean_absolute_error(y_true, naive)),
        "Naive_RMSE": rmse(y_true, naive),
        "Naive_MAPE": mape(y_true, naive),
        "Naive_R2": float(r2_score(y_true, naive)),
    }
    row["RMSE_improvement_vs_naive_pct"] = (
        100.0 * (row["Naive_RMSE"] - row["RMSE"]) / row["Naive_RMSE"]
        if row["Naive_RMSE"] > 1e-12
        else float("nan")
    )
    for cap_level in CAP_LEVELS:
        suffix = int(cap_level * 100)
        row[f"Dpp@{suffix}"] = traffic_reduction(y_true, prediction, cap_level)
        row[f"Naive_Dpp@{suffix}"] = traffic_reduction(y_true, naive, cap_level)

    prediction_frame = samples.keys.iloc[test_slice].reset_index(drop=True).copy()
    prediction_frame.insert(0, "cluster", samples.cluster)
    prediction_frame["y_true"] = y_true
    prediction_frame["yhat_patchtst"] = prediction
    prediction_frame["yhat_naive"] = naive

    if args.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_edge = re.sub(r"[^A-Za-z0-9_.-]+", "_", samples.far_edge).strip("_.") or "node"
        stem = f"cluster_{samples.cluster}_far_edge_{safe_edge}"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(config),
                "side_feature_names": list(samples.side_feature_names),
                "scale_state": {
                    "traffic_mean": state.traffic_mean,
                    "traffic_std": state.traffic_std,
                    "side_mean": state.side_mean,
                    "side_std": state.side_std,
                },
                "best_epoch": best_epoch,
            },
            model_dir / f"{stem}.pt",
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, prediction_frame


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "Dpp@70",
        "Dpp@80",
        "Dpp@90",
        "Naive_MAE",
        "Naive_RMSE",
        "Naive_MAPE",
        "Naive_R2",
        "Naive_Dpp@70",
        "Naive_Dpp@80",
        "Naive_Dpp@90",
        "RMSE_improvement_vs_naive_pct",
        "best_epoch",
        "validation_loss_scaled",
    ]
    row: dict[str, Any] = {
        "model": "PatchTST-style",
        "nodes": int(len(metrics)),
        "event_aware": bool(metrics["event_aware"].all()),
    }
    for column in numeric:
        values = pd.to_numeric(metrics[column], errors="coerce")
        row[f"{column}_mean"] = float(values.mean())
        row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return pd.DataFrame([row])


def run_mode(
    mode_name: str,
    args: argparse.Namespace,
    clusters: Sequence[int],
    events: EventFeatureTable | None,
    config: PatchConfig,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    LOGGER.info("Starting mode=%s | device=%s", mode_name, device)
    panels = load_panels(
        data_dir=args.data_dir,
        clusters=clusters,
        events=events,
        force_rebuild=args.force_rebuild,
    )

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_dir = output_dir / "models"
    reference_side_names: tuple[str, ...] | None = None

    global_node_index = 0
    for cluster in clusters:
        panel = panels[cluster]
        nodes = sorted(panel["far_edge"].dropna().astype(str).unique().tolist())
        if args.max_nodes_per_cluster > 0:
            nodes = nodes[: args.max_nodes_per_cluster]
        LOGGER.info("Cluster %s: evaluating %d nodes", cluster, len(nodes))

        for node_index, far_edge in enumerate(nodes):
            node_frame = panel.loc[panel["far_edge"].astype(str) == str(far_edge)].copy()
            try:
                samples = build_node_samples(
                    node_frame=node_frame,
                    cluster=cluster,
                    far_edge=far_edge,
                    context_length=config.context_length,
                    include_weekend_onehot=args.include_weekend_onehot,
                )
                if reference_side_names is None:
                    reference_side_names = samples.side_feature_names
                elif samples.side_feature_names != reference_side_names:
                    raise RuntimeError("Side-feature columns differ across nodes")

                row, predictions = evaluate_node(
                    samples=samples,
                    args=args,
                    config=config,
                    device=device,
                    model_dir=model_dir,
                    node_seed=args.seed + global_node_index,
                )
                metrics_rows.append(row)
                if args.save_predictions:
                    prediction_frames.append(predictions)
                global_node_index += 1
            except ValueError as exc:
                LOGGER.warning(
                    "Skipping cluster=%s far_edge=%s: %s",
                    cluster,
                    far_edge,
                    exc,
                )

    if not metrics_rows:
        raise RuntimeError(f"No eligible nodes were evaluated in mode {mode_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)
    summary = summarize_metrics(metrics)
    metrics.to_csv(output_dir / "final_metrics_by_node.csv", index=False)
    summary.to_csv(output_dir / "final_summary.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "final_predictions.csv",
            index=False,
        )

    run_config = {
        "mode": mode_name,
        "clusters": list(clusters),
        "event_features_file": str(events.source_path) if events is not None else None,
        "event_feature_names": list(events.source_feature_names) if events is not None else [],
        "side_feature_names": list(reference_side_names or ()),
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "loss": args.loss,
        "seed": args.seed,
        "device": str(device),
        "patch_config": asdict(config),
    }
    write_json(output_dir / "run_config.json", run_config)

    LOGGER.info("Final evaluation completed on %d nodes", len(metrics))
    LOGGER.info("Mean RMSE PatchTST-style: %.6f", float(summary.at[0, "RMSE_mean"]))
    LOGGER.info("Mean RMSE Naive: %.6f", float(summary.at[0, "Naive_RMSE_mean"]))
    LOGGER.info("Mean Dpp@70: %.4f", float(summary.at[0, "Dpp@70_mean"]))
    LOGGER.info("Mean Dpp@80: %.4f", float(summary.at[0, "Dpp@80_mean"]))
    LOGGER.info("Mean Dpp@90: %.4f", float(summary.at[0, "Dpp@90_mean"]))
    return summary


def main() -> int:
    args = parse_args()
    setup(args.seed)
    device = select_device(args.device)
    clusters = discover_clusters(args.data_dir, args.clusters)
    if not clusters:
        raise RuntimeError("No service clusters were found")

    config = PatchConfig(
        context_length=args.context_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
    )

    events = load_event_features(args.event_features_file)
    if args.require_event_features and events is None:
        raise RuntimeError("Event-aware run requested without an exogenous feature file")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_event_awareness:
        no_event_summary = run_mode(
            mode_name="no_event",
            args=args,
            clusters=clusters,
            events=None,
            config=config,
            device=device,
            output_dir=args.output_dir / "no_event",
        )
        event_summary = run_mode(
            mode_name="event_aware",
            args=args,
            clusters=clusters,
            events=events,
            config=config,
            device=device,
            output_dir=args.output_dir / "event_aware",
        )

        comparison = pd.DataFrame(
            [
                {
                    "mode": "no_event",
                    **no_event_summary.iloc[0].to_dict(),
                },
                {
                    "mode": "event_aware",
                    **event_summary.iloc[0].to_dict(),
                },
            ]
        )
        comparison.to_csv(args.output_dir / "event_awareness_comparison.csv", index=False)

        no_rmse = float(no_event_summary.at[0, "RMSE_mean"])
        event_rmse = float(event_summary.at[0, "RMSE_mean"])
        improvement = (
            100.0 * (no_rmse - event_rmse) / no_rmse
            if no_rmse > 1e-12
            else float("nan")
        )
        LOGGER.info("Event-aware RMSE change vs no-event: %.2f%%", improvement)
    else:
        mode_name = "event_aware" if events is not None else "no_event"
        run_mode(
            mode_name=mode_name,
            args=args,
            clusters=clusters,
            events=events,
            config=config,
            device=device,
            output_dir=args.output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
