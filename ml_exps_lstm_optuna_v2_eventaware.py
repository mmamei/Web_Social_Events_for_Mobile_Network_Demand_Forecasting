#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Optuna search for the S3 LSTM forecasting model.

The script loads cached hourly panels from ``data/hourly_panels`` or rebuilds
them from the same raw inputs used by the benchmark pipeline. Endogenous
features include traffic lags, hour-of-day and day-of-week encodings. Optional
aligned exogenous features can be added through ``--event-features-file``.

Evaluation protocol
-------------------
1. The final ``test_fraction`` of each series is reserved for testing.
2. The preceding block is split chronologically into training and validation.
3. Optuna minimizes mean validation RMSE across the selected tuning nodes.
4. The final model is retrained with the selected hyperparameters and evaluated
   on the held-out test block.

Examples
--------
Quick search using one tuning node per cluster::

    python ml_exps_lstm_optuna_v2_eventaware.py --clusters 0 1 2 3 4 --n-trials 30

Longer search with model export::

    python ml_exps_lstm_optuna_v2_eventaware.py \
        --clusters 0 1 2 3 4 \
        --nodes-per-cluster 3 \
        --n-trials 100 \
        --save-models

Main dependencies::

    pip install numpy pandas scikit-learn tensorflow optuna

TensorFlow runs sequentially within this process to avoid GPU/CPU contention.
Existing Optuna studies are resumed unless ``--reset-study`` is supplied.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Must be set before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from sklearn.metrics import mean_absolute_error, mean_squared_error

from event_feature_adapter import (
    EventFeatureTable,
    exogenous_columns,
    load_event_features,
    merge_event_features,
)

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None  # type: ignore[assignment]


LOGGER = logging.getLogger("ml_exps_lstm_optuna")
CAP_LEVELS = (0.70, 0.80, 0.90)
META_COLUMNS = ("date", "hour", "far_edge", "value")


@dataclass(frozen=True)
class MinMaxState:
    """Parametri di scaling appresi esclusivamente sul train."""

    x_min: np.ndarray
    x_range: np.ndarray
    y_min: float
    y_range: float


@dataclass
class NodeSplit:
    """Raw chronological split for one far-edge node."""

    cluster: int
    far_edge: str
    feature_names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_keys: pd.DataFrame
    test_lag1: np.ndarray


@dataclass(frozen=True)
class TuningNode:
    """Scaled data reused across all Optuna trials."""

    cluster: int
    far_edge: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    n_features: int


if tf is not None:

    class OptunaPruningCallback(tf.keras.callbacks.Callback):
        """Minimal callback without an ``optuna-integration`` dependency."""

        def __init__(
            self,
            trial: optuna.Trial,
            monitor: str = "val_rmse",
            step_offset: int = 0,
        ) -> None:
            super().__init__()
            self.trial = trial
            self.monitor = monitor
            self.step_offset = step_offset

        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            logs = logs or {}
            value = logs.get(self.monitor)
            if value is None or not np.isfinite(value):
                return
            step = self.step_offset + int(epoch)
            self.trial.report(float(value), step=step)
            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial potato a epoch={epoch + 1}, {self.monitor}={value:.6f}"
                )

else:

    class OptunaPruningCallback:  # pragma: no cover - used only with TensorFlow
        pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ottimizzazione Optuna del modello LSTM di ml_exps.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="*",
        default=None,
        help="Clusters to process; omit to use all available clusters.",
    )
    parser.add_argument(
        "--nodes-per-cluster",
        type=int,
        default=1,
        help="Nodes per cluster used by the objective; 0 uses all eligible nodes.",
    )
    parser.add_argument("--n-lags", type=int, default=24)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of the pre-test block reserved for validation.",
    )
    parser.add_argument("--min-train-rows", type=int, default=48)
    parser.add_argument("--event-features-file", type=Path, default=None)
    parser.add_argument("--require-event-features", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout Optuna in secondi.")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", default="netmob_lstm_global")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL; omit to use SQLite inside output-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lstm_optuna_V2"))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--reset-study", action="store_true")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument(
        "--max-final-nodes",
        type=int,
        default=0,
        help="Global limit on nodes in final evaluation; 0 uses all nodes.",
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Richiede operazioni TensorFlow deterministiche quando disponibili.",
    )
    parser.add_argument("--verbose-fit", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)

    if args.n_lags < 1:
        parser.error("--n-lags must be >= 1")
    if not 0.0 < args.test_fraction < 0.5:
        parser.error("--test-fraction must be in (0, 0.5)")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    if args.min_train_rows < 2:
        parser.error("--min-train-rows must be >= 2")
    if args.n_trials < 1:
        parser.error("--n-trials must be >= 1")
    if args.max_epochs < 1:
        parser.error("--max-epochs must be >= 1")
    if args.nodes_per_cluster < 0:
        parser.error("--nodes-per-cluster cannot be negative")
    if args.max_final_nodes < 0:
        parser.error("--max-final-nodes cannot be negative")
    if args.require_event_features and args.event_features_file is None:
        parser.error("--require-event-features requires --event-features-file")
    return args


def require_tensorflow() -> None:
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies, for example: "
            "pip install tensorflow optuna pandas numpy scikit-learn"
        )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is None:
        return
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            LOGGER.warning("Operazioni deterministiche TensorFlow non disponibili.")

    # Prevent TensorFlow from reserving all GPU memory up front.
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def traffic_reduction(
    y: np.ndarray,
    yhat: np.ndarray,
    cap_percentile: float,
) -> float:
    """Riduzione percentuale dell'overflow rispetto a capacita' fissa."""

    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    capacity = np.nanpercentile(y, cap_percentile * 100.0)
    overflow_fixed = np.nansum(np.maximum(0.0, y - capacity))
    capacity_dynamic = np.maximum(capacity, yhat)
    overflow_dynamic = np.nansum(np.maximum(0.0, y - capacity_dynamic))
    if overflow_fixed <= 1e-12:
        return float("nan")
    return float(100.0 * (overflow_fixed - overflow_dynamic) / overflow_fixed)


def ensure_services_clusters(data_dir: Path) -> pd.DataFrame:
    """Carica o ricostruisce ``services_clusters.csv`` come in ml_exps.py."""

    services_path = data_dir / "services_clusters.csv"
    if services_path.exists():
        LOGGER.info("Loading %s", services_path)
        return pd.read_csv(services_path)

    raw_files = sorted(data_dir.glob("nantes_antenna_serv_*.csv"))
    service_cluster_path = data_dir / "service_clustering.csv"
    if not raw_files:
        raise FileNotFoundError(
            f"Missing {services_path} and no files match {data_dir / 'nantes_antenna_serv_*.csv'}"
        )
    if not service_cluster_path.exists():
        raise FileNotFoundError(f"Missing {service_cluster_path}")

    LOGGER.info("Ricostruisco services_clusters da %d file raw", len(raw_files))
    raw = pd.concat(
        (pd.read_csv(path) for path in raw_files),
        axis=0,
        ignore_index=True,
    ).drop_duplicates()
    service_clusters = pd.read_csv(service_cluster_path)
    merged = raw.merge(service_clusters, on="service", how="left")
    value_columns = [str(i) for i in range(96)]
    missing = [column for column in value_columns if column not in merged.columns]
    if missing:
        raise ValueError(
            "Raw data are missing expected time-slot columns, for example: "
            + ", ".join(missing[:5])
        )
    aggregated = (
        merged.groupby(["date", "labels", "lon", "lat"], as_index=False)[value_columns]
        .sum()
        .reset_index(drop=True)
    )
    services_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(services_path, index=False)
    LOGGER.info("Created %s", services_path)
    return aggregated


def build_hourly_for_cluster(
    cluster_label: int,
    services_clusters: pd.DataFrame,
    antennas: pd.DataFrame,
    hourly_dir: Path,
) -> pd.DataFrame:
    """Build the hourly panel by mapping 15-minute slots 0..95 to hours 0..23."""

    value_columns = [str(i) for i in range(96)]
    subset = services_clusters.loc[
        services_clusters["labels"] == cluster_label,
        ["date", "lon", "lat", *value_columns],
    ].copy()
    if subset.empty:
        raise ValueError(f"No data available for cluster {cluster_label}")

    subset = subset.groupby(["date", "lon", "lat"], as_index=False)[value_columns].sum()
    subset = antennas[["lat", "lon", "far_edge"]].merge(
        subset,
        on=["lat", "lon"],
        how="inner",
    )
    subset = subset.groupby(
        ["lat", "lon", "far_edge", "date"],
        as_index=False,
    )[value_columns].sum()
    long_df = subset.melt(
        id_vars=["date", "far_edge"],
        value_vars=value_columns,
        var_name="slot",
        value_name="value",
    )

    slot_text = long_df["slot"].astype(str)
    if slot_text.str.fullmatch(r"\d+").all():
        long_df["hour"] = slot_text.astype(int) // 4
    else:
        extracted = slot_text.str.extract(r"^(\d+)", expand=False)
        long_df["hour"] = pd.to_numeric(extracted, errors="coerce")

    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "far_edge", "hour", "value"])
    long_df["hour"] = long_df["hour"].astype(int)
    hourly = (
        long_df.groupby(["date", "far_edge", "hour"], as_index=False)["value"]
        .sum()
        .sort_values(["far_edge", "date", "hour"])
        .reset_index(drop=True)
    )

    hourly_dir.mkdir(parents=True, exist_ok=True)
    output_path = hourly_dir / f"hourly_cluster_{cluster_label}.csv"
    hourly.to_csv(output_path, index=False)
    LOGGER.info("Created %s", output_path)
    return hourly


def normalize_hourly_panel(panel: pd.DataFrame, source: Path | None = None) -> pd.DataFrame:
    required = {"date", "far_edge", "hour", "value"}
    missing = required.difference(panel.columns)
    if missing:
        where = f" in {source}" if source is not None else ""
        raise ValueError(f"Missing columns{where}: {sorted(missing)}")

    panel = panel.loc[:, ["date", "hour", "far_edge", "value"]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["hour"] = pd.to_numeric(panel["hour"], errors="coerce")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel = panel.dropna(subset=["date", "hour", "far_edge", "value"])
    panel["hour"] = panel["hour"].astype(int)
    panel = panel.loc[panel["hour"].between(0, 23)].copy()
    panel = panel.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)
    return panel


def discover_clusters(data_dir: Path, requested: Sequence[int] | None) -> list[int]:
    if requested:
        return sorted(set(int(value) for value in requested))

    services_path = data_dir / "services_clusters.csv"
    if services_path.exists():
        labels = pd.read_csv(services_path, usecols=["labels"])["labels"]
        return sorted(labels.dropna().astype(int).unique().tolist())

    cached = []
    for path in (data_dir / "hourly_panels").glob("hourly_cluster_*.csv"):
        match = re.search(r"hourly_cluster_(-?\d+)\.csv$", path.name)
        if match:
            cached.append(int(match.group(1)))
    if cached:
        return sorted(set(cached))

    services = ensure_services_clusters(data_dir)
    return sorted(services["labels"].dropna().astype(int).unique().tolist())


def load_hourly_panels(
    data_dir: Path,
    clusters: Sequence[int],
    force_rebuild: bool = False,
    events: EventFeatureTable | None = None,
) -> dict[int, pd.DataFrame]:
    hourly_dir = data_dir / "hourly_panels"
    panels: dict[int, pd.DataFrame] = {}
    services_clusters: pd.DataFrame | None = None
    antennas: pd.DataFrame | None = None

    for cluster in clusters:
        cache_path = hourly_dir / f"hourly_cluster_{cluster}.csv"
        if cache_path.exists() and not force_rebuild:
            LOGGER.info("Loading cache %s", cache_path)
            panel = pd.read_csv(cache_path)
        else:
            if services_clusters is None:
                services_clusters = ensure_services_clusters(data_dir)
            if antennas is None:
                antennas_path = data_dir / "nantes_antenna_clustering.csv"
                if not antennas_path.exists():
                    raise FileNotFoundError(f"Missing {antennas_path}")
                antennas = pd.read_csv(antennas_path)
            panel = build_hourly_for_cluster(
                cluster,
                services_clusters,
                antennas,
                hourly_dir,
            )
        normalized = normalize_hourly_panel(panel, cache_path)
        normalized, matched = merge_event_features(normalized, int(cluster), events)
        panels[int(cluster)] = normalized
        LOGGER.info(
            "Cluster %s: event-aware=%s | feature-row match=%.2f%%",
            cluster, events is not None, matched
        )
    return panels


def make_supervised(df_node: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
    """Feature construction with complete one-hot categories across all nodes."""

    base_cols = ["date", "hour", "far_edge", "value"]
    exog_cols = exogenous_columns(df_node)
    g = df_node.loc[:, [*base_cols, *exog_cols]].copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g["hour"] = pd.to_numeric(g["hour"], errors="coerce")
    g["value"] = pd.to_numeric(g["value"], errors="coerce")
    g = g.dropna(subset=["date", "hour", "far_edge", "value"])
    g["hour"] = g["hour"].astype(int)
    g["dow"] = g["date"].dt.weekday.astype(int)
    g = g.sort_values(["date", "hour"]).reset_index(drop=True)

    lag_columns = [f"lag{k}" for k in range(1, n_lags + 1)]
    for k, column in enumerate(lag_columns, start=1):
        g[column] = g["value"].shift(k)

    hour_cat = pd.Categorical(g["hour"], categories=range(24))
    dow_cat = pd.Categorical(g["dow"], categories=range(7))
    hour_dummies = pd.get_dummies(hour_cat, prefix="h", dtype=float)
    dow_dummies = pd.get_dummies(dow_cat, prefix="d", dtype=float)
    hour_dummies.index = g.index
    dow_dummies.index = g.index

    parts = [
        g[["date", "hour", "far_edge", "value"]],
        hour_dummies,
        dow_dummies,
    ]
    if exog_cols:
        parts.append(g[exog_cols].astype(float))
    parts.append(g[lag_columns])
    supervised = pd.concat(parts, axis=1)
    feature_columns = [
        column for column in supervised.columns if column not in META_COLUMNS
    ]
    supervised = supervised.replace([np.inf, -np.inf], np.nan)
    supervised = supervised.dropna(subset=["value", *feature_columns])
    return supervised.reset_index(drop=True)


def split_supervised_node(
    supervised: pd.DataFrame,
    cluster: int,
    far_edge: Any,
    test_fraction: float,
    validation_fraction: float,
    min_train_rows: int,
) -> NodeSplit:
    n_rows = len(supervised)
    n_test = max(1, int(round(n_rows * test_fraction)))
    n_pretest = n_rows - n_test
    n_val = max(1, int(round(n_pretest * validation_fraction)))
    n_train = n_pretest - n_val

    if n_train < min_train_rows or n_val < 1 or n_test < 1:
        raise ValueError(
            f"series too short after split: train={n_train}, val={n_val}, test={n_test}"
        )

    train = supervised.iloc[:n_train]
    val = supervised.iloc[n_train:n_pretest]
    test = supervised.iloc[n_pretest:]
    feature_names = tuple(
        column for column in supervised.columns if column not in META_COLUMNS
    )

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)

    def target(frame: pd.DataFrame) -> np.ndarray:
        return frame["value"].to_numpy(dtype=np.float32, copy=True)

    return NodeSplit(
        cluster=int(cluster),
        far_edge=str(far_edge),
        feature_names=feature_names,
        X_train=matrix(train),
        y_train=target(train),
        X_val=matrix(val),
        y_val=target(val),
        X_test=matrix(test),
        y_test=target(test),
        test_keys=test[["date", "hour", "far_edge"]].reset_index(drop=True),
        test_lag1=test["lag1"].to_numpy(dtype=np.float32, copy=True),
    )


def prepare_node_split(
    df_node: pd.DataFrame,
    cluster: int,
    far_edge: Any,
    n_lags: int,
    test_fraction: float,
    validation_fraction: float,
    min_train_rows: int,
) -> NodeSplit:
    supervised = make_supervised(df_node, n_lags=n_lags)
    return split_supervised_node(
        supervised=supervised,
        cluster=cluster,
        far_edge=far_edge,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        min_train_rows=min_train_rows,
    )


def fit_minmax(X: np.ndarray, y: np.ndarray) -> MinMaxState:
    x_min = np.nanmin(X, axis=0).astype(np.float32)
    x_max = np.nanmax(X, axis=0).astype(np.float32)
    x_range = x_max - x_min
    x_range = np.where(np.abs(x_range) <= 1e-12, 1.0, x_range).astype(np.float32)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = y_max - y_min
    if abs(y_range) <= 1e-12:
        y_range = 1.0
    return MinMaxState(x_min=x_min, x_range=x_range, y_min=y_min, y_range=y_range)


def scale_X(X: np.ndarray, state: MinMaxState) -> np.ndarray:
    return ((X - state.x_min) / state.x_range).astype(np.float32)


def scale_y(y: np.ndarray, state: MinMaxState) -> np.ndarray:
    return ((y - state.y_min) / state.y_range).astype(np.float32)


def inverse_y(y_scaled: np.ndarray, state: MinMaxState) -> np.ndarray:
    return (np.asarray(y_scaled, dtype=np.float32) * state.y_range + state.y_min).astype(
        np.float32
    )


def to_lstm_input(X: np.ndarray) -> np.ndarray:
    """Use one timestep containing all engineered features."""

    return X.reshape((X.shape[0], 1, X.shape[1])).astype(np.float32, copy=False)


def select_tuning_nodes(
    panels: dict[int, pd.DataFrame],
    nodes_per_cluster: int,
    n_lags: int,
    test_fraction: float,
    validation_fraction: float,
    min_train_rows: int,
) -> tuple[list[NodeSplit], pd.DataFrame]:
    selected: list[NodeSplit] = []
    rows: list[dict[str, Any]] = []

    for cluster, panel in panels.items():
        counts = panel.groupby("far_edge", sort=False).size()
        candidates = sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        accepted = 0
        for far_edge, raw_rows in candidates:
            node_df = panel.loc[panel["far_edge"] == far_edge].copy()
            try:
                split = prepare_node_split(
                    node_df,
                    cluster=cluster,
                    far_edge=far_edge,
                    n_lags=n_lags,
                    test_fraction=test_fraction,
                    validation_fraction=validation_fraction,
                    min_train_rows=min_train_rows,
                )
            except ValueError as exc:
                LOGGER.debug(
                    "Scarto cluster=%s far_edge=%s: %s",
                    cluster,
                    far_edge,
                    exc,
                )
                continue

            selected.append(split)
            rows.append(
                {
                    "cluster": cluster,
                    "far_edge": str(far_edge),
                    "raw_rows": int(raw_rows),
                    "train_rows": len(split.y_train),
                    "validation_rows": len(split.y_val),
                    "test_rows": len(split.y_test),
                }
            )
            accepted += 1
            if nodes_per_cluster > 0 and accepted >= nodes_per_cluster:
                break

        if accepted == 0:
            LOGGER.warning("No eligible nodes in cluster %s", cluster)
        else:
            LOGGER.info("Cluster %s: %d nodes selected for Optuna", cluster, accepted)

    if not selected:
        raise RuntimeError("No node has a sufficiently long series for tuning.")
    return selected, pd.DataFrame(rows)


def scale_tuning_nodes(node_splits: Iterable[NodeSplit]) -> list[TuningNode]:
    scaled_nodes: list[TuningNode] = []
    for split in node_splits:
        state = fit_minmax(split.X_train, split.y_train)
        scaled_nodes.append(
            TuningNode(
                cluster=split.cluster,
                far_edge=split.far_edge,
                X_train=to_lstm_input(scale_X(split.X_train, state)),
                y_train=scale_y(split.y_train, state),
                X_val=to_lstm_input(scale_X(split.X_val, state)),
                y_val=scale_y(split.y_val, state),
                n_features=split.X_train.shape[1],
            )
        )
    return scaled_nodes


def sample_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "units_1": trial.suggest_int("units_1", 32, 256, step=32),
        "dropout": trial.suggest_float("dropout", 0.0, 0.40, step=0.05),
        "dense_units": trial.suggest_categorical("dense_units", [0, 16, 32, 64]),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 5e-3, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "l2": trial.suggest_float("l2", 1e-8, 1e-3, log=True),
        "clipnorm": trial.suggest_categorical("clipnorm", [0.5, 1.0, 5.0]),
        "loss": trial.suggest_categorical("loss", ["mae", "huber", "mse"]),
        "patience": trial.suggest_categorical("patience", [5, 10, 15]),
    }
    if params["n_layers"] == 2:
        params["units_2"] = trial.suggest_int("units_2", 16, 128, step=16)
    return params


def keras_loss(name: str) -> Any:
    require_tensorflow()
    if name == "huber":
        return tf.keras.losses.Huber()
    if name in {"mae", "mse"}:
        return name
    raise ValueError(f"Loss non supportata: {name}")


def build_lstm_model(n_features: int, params: dict[str, Any]) -> Any:
    require_tensorflow()
    regularizer = tf.keras.regularizers.l2(float(params["l2"]))
    inputs = tf.keras.Input(shape=(1, n_features), name="features")
    x = inputs

    n_layers = int(params["n_layers"])
    units = [int(params["units_1"])]
    if n_layers == 2:
        units.append(int(params["units_2"]))

    for layer_index, layer_units in enumerate(units):
        return_sequences = layer_index < (len(units) - 1)
        x = tf.keras.layers.LSTM(
            layer_units,
            activation="tanh",
            return_sequences=return_sequences,
            kernel_regularizer=regularizer,
            name=f"lstm_{layer_index + 1}",
        )(x)
        dropout = float(params["dropout"])
        if dropout > 0.0:
            x = tf.keras.layers.Dropout(
                dropout,
                name=f"dropout_{layer_index + 1}",
            )(x)

    dense_units = int(params["dense_units"])
    if dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="forecast")(x)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(params["learning_rate"]),
        clipnorm=float(params["clipnorm"]),
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="S3_LSTM_Optuna")
    model.compile(
        optimizer=optimizer,
        loss=keras_loss(str(params["loss"])),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def fit_callbacks(
    patience: int,
    trial: optuna.Trial | None = None,
    step_offset: int = 0,
) -> list[Any]:
    require_tensorflow()
    callbacks: list[Any] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=int(patience),
            min_delta=1e-5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if trial is not None:
        callbacks.append(
            OptunaPruningCallback(
                trial=trial,
                monitor="val_rmse",
                step_offset=step_offset,
            )
        )
    return callbacks


def clear_tensorflow() -> None:
    if tf is not None:
        tf.keras.backend.clear_session()
    gc.collect()


def make_objective(
    nodes: Sequence[TuningNode],
    max_epochs: int,
    seed: int,
    verbose_fit: int,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        validation_scores: list[float] = []

        for node_index, node in enumerate(nodes):
            model = None
            try:
                clear_tensorflow()
                node_seed = seed + trial.number * 10_000 + node_index
                set_global_seed(node_seed, deterministic=False)
                model = build_lstm_model(node.n_features, params)
                step_offset = node_index * (max_epochs + 1)
                model.fit(
                    node.X_train,
                    node.y_train,
                    validation_data=(node.X_val, node.y_val),
                    epochs=max_epochs,
                    batch_size=int(params["batch_size"]),
                    callbacks=fit_callbacks(
                        patience=int(params["patience"]),
                        trial=trial,
                        step_offset=step_offset,
                    ),
                    shuffle=False,
                    verbose=verbose_fit,
                )
                prediction = model.predict(node.X_val, verbose=0).reshape(-1)
                score = rmse(node.y_val, prediction)
                if not np.isfinite(score):
                    raise optuna.TrialPruned("Validation RMSE non finito")
                validation_scores.append(score)
                trial.set_user_attr(
                    f"node_{node_index}_c{node.cluster}_fe{node.far_edge}_rmse",
                    float(score),
                )

                partial_mean = float(np.mean(validation_scores))
                aggregate_step = (node_index + 1) * (max_epochs + 1) - 1
                trial.report(partial_mean, step=aggregate_step)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Media parziale validation RMSE={partial_mean:.6f}"
                    )
            except optuna.TrialPruned:
                raise
            except (MemoryError, FloatingPointError) as exc:
                raise optuna.TrialPruned(f"Trial interrotto: {exc}") from exc
            except Exception as exc:
                if tf is not None and isinstance(exc, tf.errors.ResourceExhaustedError):
                    raise optuna.TrialPruned("Memoria TensorFlow esaurita") from exc
                raise
            finally:
                if model is not None:
                    del model
                clear_tensorflow()

        return float(np.mean(validation_scores))

    return objective


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def data_signature(
    args: argparse.Namespace,
    tuning_nodes: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "clusters": sorted(args.clusters) if args.clusters else None,
        "n_lags": args.n_lags,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "min_train_rows": args.min_train_rows,
        "input_layout": "samples_1_nfeatures",
        "objective": "mean_normalized_validation_rmse",
        "event_features_file": str(args.event_features_file.resolve()) if args.event_features_file else None,
        "tuning_nodes": tuning_nodes[
            ["cluster", "far_edge", "train_rows", "validation_rows"]
        ].to_dict(orient="records"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), payload


def resolve_storage(args: argparse.Namespace) -> tuple[str, Path | None]:
    if args.storage:
        return str(args.storage), None
    db_path = (args.output_dir / "optuna_study.sqlite3").resolve()
    return f"sqlite:///{db_path.as_posix()}", db_path


def create_or_resume_study(
    args: argparse.Namespace,
    signature: str,
    signature_payload: dict[str, Any],
) -> optuna.Study:
    storage_url, _ = resolve_storage(args)
    if args.reset_study:
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage_url)
            LOGGER.info("Deleted existing study: %s", args.study_name)
        except KeyError:
            pass

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(10, max(5, args.n_trials // 5)),
        n_warmup_steps=5,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    old_signature = study.user_attrs.get("data_signature")
    if old_signature and old_signature != signature and study.trials:
        raise RuntimeError(
            "The existing study uses a different data/configuration signature. "
            "Use another --study-name or pass --reset-study."
        )
    study.set_user_attr("data_signature", signature)
    study.set_user_attr("data_signature_payload", signature_payload)
    return study


def export_study(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe()
    trials.to_csv(output_dir / "study_trials.csv", index=False)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No Optuna trial completed successfully.")

    best_payload = {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_value_mean_normalized_validation_rmse": study.best_value,
        "best_params": study.best_params,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(completed),
    }
    write_json(output_dir / "best_params.json", best_payload)

    try:
        importance = get_param_importances(study)
        pd.DataFrame(
            [{"parameter": name, "importance": value} for name, value in importance.items()]
        ).to_csv(output_dir / "parameter_importance.csv", index=False)
    except Exception as exc:
        LOGGER.warning("Impossibile calcolare l'importanza dei parametri: %s", exc)


def safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return text or "node"


def train_best_on_node(
    split: NodeSplit,
    best_params: dict[str, Any],
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    require_tensorflow()

    # Phase 1: select the best epoch using train/validation only.
    train_state = fit_minmax(split.X_train, split.y_train)
    X_train = to_lstm_input(scale_X(split.X_train, train_state))
    y_train = scale_y(split.y_train, train_state)
    X_val = to_lstm_input(scale_X(split.X_val, train_state))
    y_val = scale_y(split.y_val, train_state)

    clear_tensorflow()
    set_global_seed(args.seed, deterministic=args.deterministic)
    selector_model = build_lstm_model(split.X_train.shape[1], best_params)
    history = selector_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.max_epochs,
        batch_size=int(best_params["batch_size"]),
        callbacks=fit_callbacks(patience=int(best_params["patience"])),
        shuffle=False,
        verbose=args.verbose_fit,
    )
    val_history = np.asarray(history.history.get("val_rmse", []), dtype=float)
    if len(val_history) and np.isfinite(val_history).any():
        best_epoch = int(np.nanargmin(val_history) + 1)
        best_validation_rmse = float(np.nanmin(val_history))
    else:
        best_epoch = max(1, len(history.history.get("loss", [])))
        best_validation_rmse = float("nan")
    del selector_model
    clear_tensorflow()

    # Phase 2: retrain from scratch on train+validation for the selected epoch count.
    X_fit_raw = np.concatenate([split.X_train, split.X_val], axis=0)
    y_fit_raw = np.concatenate([split.y_train, split.y_val], axis=0)
    final_state = fit_minmax(X_fit_raw, y_fit_raw)
    X_fit = to_lstm_input(scale_X(X_fit_raw, final_state))
    y_fit = scale_y(y_fit_raw, final_state)
    X_test = to_lstm_input(scale_X(split.X_test, final_state))

    set_global_seed(args.seed + 1, deterministic=args.deterministic)
    final_model = build_lstm_model(split.X_train.shape[1], best_params)
    final_model.fit(
        X_fit,
        y_fit,
        epochs=best_epoch,
        batch_size=int(best_params["batch_size"]),
        shuffle=False,
        verbose=args.verbose_fit,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )
    prediction_scaled = final_model.predict(X_test, verbose=0).reshape(-1)
    prediction = inverse_y(prediction_scaled, final_state).astype(float)
    y_true = split.y_test.astype(float)
    naive = split.test_lag1.astype(float)

    row: dict[str, Any] = {
        "cluster": split.cluster,
        "far_edge": split.far_edge,
        "model": "S3_LSTM_Optuna",
        "train_rows": len(split.y_train),
        "validation_rows": len(split.y_val),
        "test_rows": len(split.y_test),
        "best_epoch": best_epoch,
        "validation_RMSE_normalized": best_validation_rmse,
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": rmse(y_true, prediction),
        "MAPE": mape(y_true, prediction),
        "Naive_MAE": float(mean_absolute_error(y_true, naive)),
        "Naive_RMSE": rmse(y_true, naive),
    }
    row["RMSE_improvement_vs_naive_pct"] = (
        100.0 * (row["Naive_RMSE"] - row["RMSE"]) / row["Naive_RMSE"]
        if row["Naive_RMSE"] > 1e-12
        else float("nan")
    )
    for cap in CAP_LEVELS:
        row[f"Dpp@{int(cap * 100)}"] = traffic_reduction(y_true, prediction, cap)

    predictions = split.test_keys.copy()
    predictions.insert(0, "cluster", split.cluster)
    predictions["y_true"] = y_true
    predictions["yhat_S3_LSTM_Optuna"] = prediction
    predictions["yhat_S0_Naive"] = naive

    if args.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
        stem = f"cluster_{split.cluster}_far_edge_{safe_component(split.far_edge)}"
        model_path = model_dir / f"{stem}.keras"
        scaler_path = model_dir / f"{stem}_scaler.npz"
        metadata_path = model_dir / f"{stem}_metadata.json"
        final_model.save(model_path)
        np.savez_compressed(
            scaler_path,
            x_min=final_state.x_min,
            x_range=final_state.x_range,
            y_min=np.asarray([final_state.y_min], dtype=np.float64),
            y_range=np.asarray([final_state.y_range], dtype=np.float64),
            feature_names=np.asarray(split.feature_names, dtype=str),
        )
        write_json(
            metadata_path,
            {
                "cluster": split.cluster,
                "far_edge": split.far_edge,
                "model_path": model_path.name,
                "scaler_path": scaler_path.name,
                "n_lags": args.n_lags,
                "input_shape": [1, len(split.feature_names)],
                "best_epoch": best_epoch,
                "best_params": best_params,
                "metrics": row,
            },
        )

    del final_model
    clear_tensorflow()
    return row, predictions


def iter_node_groups(
    panels: dict[int, pd.DataFrame],
) -> Iterable[tuple[int, Any, pd.DataFrame]]:
    for cluster in sorted(panels):
        panel = panels[cluster]
        keys = sorted(panel["far_edge"].dropna().unique().tolist(), key=str)
        for far_edge in keys:
            yield cluster, far_edge, panel.loc[panel["far_edge"] == far_edge].copy()


def final_evaluation(
    panels: dict[int, pd.DataFrame],
    best_params: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    metrics_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    prediction_path = args.output_dir / "final_predictions.csv"
    model_dir = args.output_dir / "models"
    first_prediction_chunk = True
    processed = 0

    if args.save_predictions and prediction_path.exists():
        prediction_path.unlink()

    for cluster, far_edge, node_df in iter_node_groups(panels):
        if args.max_final_nodes and processed >= args.max_final_nodes:
            break
        try:
            split = prepare_node_split(
                node_df,
                cluster=cluster,
                far_edge=far_edge,
                n_lags=args.n_lags,
                test_fraction=args.test_fraction,
                validation_fraction=args.validation_fraction,
                min_train_rows=args.min_train_rows,
            )
        except ValueError as exc:
            skipped_rows.append(
                {
                    "cluster": cluster,
                    "far_edge": str(far_edge),
                    "reason": str(exc),
                }
            )
            continue

        LOGGER.info(
            "Final evaluation cluster=%s far_edge=%s (%d)",
            cluster,
            far_edge,
            processed + 1,
        )
        row, predictions = train_best_on_node(
            split=split,
            best_params=best_params,
            args=args,
            model_dir=model_dir,
        )
        metrics_rows.append(row)
        processed += 1

        if args.save_predictions:
            predictions.to_csv(
                prediction_path,
                mode="w" if first_prediction_chunk else "a",
                header=first_prediction_chunk,
                index=False,
            )
            first_prediction_chunk = False

    if not metrics_rows:
        raise RuntimeError("No node was evaluated in the final phase.")

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "final_metrics_by_node.csv", index=False)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            args.output_dir / "skipped_nodes.csv",
            index=False,
        )

    mean_columns = [
        "MAE",
        "RMSE",
        "MAPE",
        "Naive_MAE",
        "Naive_RMSE",
        "RMSE_improvement_vs_naive_pct",
        *[f"Dpp@{int(cap * 100)}" for cap in CAP_LEVELS],
    ]
    summary: dict[str, Any] = {
        "model": "S3_LSTM_Optuna",
        "nodes": len(metrics),
    }
    for column in mean_columns:
        summary[column] = float(pd.to_numeric(metrics[column], errors="coerce").mean())
    pd.DataFrame([summary]).to_csv(args.output_dir / "final_summary.csv", index=False)

    LOGGER.info("Final evaluation completed on %d nodes", len(metrics))
    LOGGER.info("RMSE medio LSTM: %.6f", summary["RMSE"])
    LOGGER.info("RMSE medio Naive: %.6f", summary["Naive_RMSE"])


def run(args: argparse.Namespace) -> None:
    require_tensorflow()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed, deterministic=args.deterministic)

    clusters = discover_clusters(args.data_dir, args.clusters)
    if not clusters:
        raise RuntimeError("No cluster is available.")
    args.clusters = clusters
    LOGGER.info("Cluster selezionati: %s", clusters)
    events = load_event_features(args.event_features_file)
    LOGGER.info("Event-aware features: %s", events.source_path if events is not None else "DISABLED")

    panels = load_hourly_panels(
        data_dir=args.data_dir,
        clusters=clusters,
        force_rebuild=args.force_rebuild,
        events=events,
    )
    tuning_splits, tuning_index = select_tuning_nodes(
        panels=panels,
        nodes_per_cluster=args.nodes_per_cluster,
        n_lags=args.n_lags,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        min_train_rows=args.min_train_rows,
    )
    tuning_index.to_csv(args.output_dir / "tuning_nodes.csv", index=False)
    tuning_nodes = scale_tuning_nodes(tuning_splits)

    signature, signature_payload = data_signature(args, tuning_index)
    study = create_or_resume_study(
        args=args,
        signature=signature,
        signature_payload=signature_payload,
    )

    LOGGER.info(
        "Starting Optuna: %d new trials, %d nodes in the objective",
        args.n_trials,
        len(tuning_nodes),
    )
    study.optimize(
        make_objective(
            nodes=tuning_nodes,
            max_epochs=args.max_epochs,
            seed=args.seed,
            verbose_fit=args.verbose_fit,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True,
    )
    export_study(study, args.output_dir)
    LOGGER.info("Best validation RMSE normalizzato: %.6f", study.best_value)
    LOGGER.info("Best params: %s", study.best_params)

    run_config = vars(args).copy()
    run_config.update(
        {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "optuna_version": optuna.__version__,
            "tensorflow_version": tf.__version__ if tf is not None else None,
            "data_signature": signature,
        }
    )
    write_json(args.output_dir / "run_config.json", run_config)

    if not args.tune_only:
        final_evaluation(
            panels=panels,
            best_params=dict(study.best_params),
            args=args,
        )


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        LOGGER.error("Esecuzione interrotta dall'utente.")
        return 130
    except Exception as exc:
        LOGGER.error("Errore: %s", exc)
        if os.environ.get("DEBUG") == "1":
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
