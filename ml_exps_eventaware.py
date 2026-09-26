#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified event-aware / no-event benchmark for NetMob Nantes.

Models:
  S0 Naive persistence y(t-1)
  S1 Ridge
  S2 Random Forest
  S3 LSTM (128 units, dropout 0.1, Adam+MAE)

Endogenous features are identical in both modes: lag1..lag24, hour-of-day and
day-of-week one-hot encodings. When ``--event-features-file`` is supplied,
every numeric exogenous column in that aligned CSV is appended to S1/S2/S3.
S0 remains unchanged by construction.

The same pipeline is used for event-aware and no-event runs so that the
comparison differs only in the presence of exogenous contextual features. Use ``--compare-event-awareness``
to execute both modes in one invocation and export both summaries/figures.

The event feature CSV may contain scheduled-event indicators, Google Trends
signals, or both. It must be keyed by date/hour and may additionally contain
cluster and/or far_edge.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from event_feature_adapter import (
    EXOG_PREFIX,
    EventFeatureTable,
    exogenous_columns,
    load_event_features,
    merge_event_features,
)

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

LOGGER = logging.getLogger("ml_exps_eventaware")
META_COLS = ("date", "hour", "far_edge", "value")
CAP_LEVELS = (0.70, 0.80, 0.90)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--clusters", type=int, nargs="*", default=None)
    p.add_argument("--n-lags", type=int, default=24)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--min-train-rows", type=int, default=48)
    p.add_argument("--event-features-file", type=Path, default=None)
    p.add_argument("--require-event-features", action="store_true")
    p.add_argument(
        "--compare-event-awareness",
        action="store_true",
        help="Run both no-event and event-aware pipelines and export comparable Figure-4/no-event plots.",
    )
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--rf-estimators", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--lstm-units", type=int, default=128)
    p.add_argument("--lstm-dropout", type=float, default=0.10)
    p.add_argument("--lstm-epochs", type=int, default=100)
    p.add_argument("--lstm-batch-size", type=int, default=16)
    p.add_argument("--lstm-patience", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ml_exps_eventaware"))
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.require_event_features and args.event_features_file is None:
        p.error("--require-event-features requires --event-features-file")
    if args.compare_event_awareness and args.event_features_file is None:
        p.error("--compare-event-awareness requires --event-features-file")
    if not 0.0 < args.test_fraction < 0.5:
        p.error("--test-fraction must be in (0,0.5)")
    if args.n_lags < 1:
        p.error("--n-lags must be >= 1")
    return args


def setup(seed: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def clear_tf() -> None:
    if tf is not None:
        tf.keras.backend.clear_session()
    gc.collect()


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, yhat)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), 1e-9))) * 100.0)


def traffic_reduction(y: np.ndarray, yhat: np.ndarray, cap_percentile: float) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    capacity = np.nanpercentile(y, cap_percentile * 100.0)
    overflow_fixed = np.nansum(np.maximum(0.0, y - capacity))
    dynamic_capacity = np.maximum(capacity, yhat)
    overflow_dynamic = np.nansum(np.maximum(0.0, y - dynamic_capacity))
    if overflow_fixed <= 1e-12:
        return float("nan")
    return float(100.0 * (overflow_fixed - overflow_dynamic) / overflow_fixed)


def ensure_services_clusters(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "services_clusters.csv"
    if path.exists():
        return pd.read_csv(path)
    raw_files = sorted(data_dir.glob("nantes_antenna_serv_*.csv"))
    cluster_path = data_dir / "service_clustering.csv"
    if not raw_files or not cluster_path.exists():
        raise FileNotFoundError(
            f"Missing {path}; also need raw nantes_antenna_serv_*.csv and {cluster_path} to rebuild it"
        )
    raw = pd.concat((pd.read_csv(f) for f in raw_files), ignore_index=True).drop_duplicates()
    service_clusters = pd.read_csv(cluster_path)
    merged = raw.merge(service_clusters, on="service", how="left")
    value_cols = [str(i) for i in range(96)]
    agg = merged.groupby(["date", "labels", "lon", "lat"], as_index=False)[value_cols].sum()
    path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(path, index=False)
    return agg


def build_hourly_for_cluster(cluster: int, services: pd.DataFrame, antennas: pd.DataFrame, output: Path) -> pd.DataFrame:
    value_cols = [str(i) for i in range(96)]
    subset = services.loc[services["labels"] == cluster, ["date", "lon", "lat", *value_cols]].copy()
    subset = subset.groupby(["date", "lon", "lat"], as_index=False)[value_cols].sum()
    subset = antennas[["lat", "lon", "far_edge"]].merge(subset, on=["lat", "lon"], how="inner")
    subset = subset.groupby(["lat", "lon", "far_edge", "date"], as_index=False)[value_cols].sum()
    long = subset.melt(id_vars=["date", "far_edge"], value_vars=value_cols, var_name="slot", value_name="value")
    slot = long["slot"].astype(str)
    if slot.str.fullmatch(r"\d+").all():
        long["hour"] = slot.astype(int) // 4
    else:
        long["hour"] = pd.to_numeric(slot.str.extract(r"^(\d+)")[0], errors="coerce")
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    hourly = long.dropna(subset=["date", "hour", "value"]).groupby(["date", "far_edge", "hour"], as_index=False)["value"].sum()
    hourly["hour"] = hourly["hour"].astype(int)
    hourly = hourly.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output, index=False)
    return hourly


def discover_clusters(data_dir: Path, requested: Sequence[int] | None) -> list[int]:
    if requested:
        return sorted(set(int(v) for v in requested))
    cached = []
    for path in (data_dir / "hourly_panels").glob("hourly_cluster_*.csv"):
        m = re.search(r"hourly_cluster_(-?\d+)\.csv$", path.name)
        if m:
            cached.append(int(m.group(1)))
    if cached:
        return sorted(set(cached))
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
            df = pd.read_csv(path)
        else:
            if services is None:
                services = ensure_services_clusters(data_dir)
            if antennas is None:
                antennas_path = data_dir / "nantes_antenna_clustering.csv"
                antennas = pd.read_csv(antennas_path)
            df = build_hourly_for_cluster(cluster, services, antennas, path)
        missing = set(META_COLS).difference(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        df = df[list(META_COLS)].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["far_edge"] = df["far_edge"].astype(str)
        df = df.dropna(subset=list(META_COLS)).copy()
        df["hour"] = df["hour"].astype(int)
        df = df.loc[df["hour"].between(0, 23)].copy()
        df, matched = merge_event_features(df, int(cluster), events)
        df = df.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)
        panels[int(cluster)] = df
        LOGGER.info(
            "Cluster %s: %d nodes | event-aware=%s | feature-row match=%.2f%%",
            cluster,
            df["far_edge"].nunique(),
            events is not None,
            matched,
        )
    return panels


def make_supervised(df_node: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    g = df_node.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce").dt.normalize()
    g["dow"] = g["date"].dt.weekday.astype(int)
    g = g.sort_values(["date", "hour"]).reset_index(drop=True)
    lag_cols = [f"lag{k}" for k in range(1, n_lags + 1)]
    for k, col in enumerate(lag_cols, start=1):
        g[col] = g["value"].shift(k)
    hour = pd.get_dummies(pd.Categorical(g["hour"], categories=range(24)), prefix="h", dtype=float)
    dow = pd.get_dummies(pd.Categorical(g["dow"], categories=range(7)), prefix="d", dtype=float)
    hour.index = g.index
    dow.index = g.index
    parts = [g[list(META_COLS)], hour, dow]
    exog = exogenous_columns(g)
    if exog:
        parts.append(g[exog].astype(float))
    parts.append(g[lag_cols])
    out = pd.concat(parts, axis=1)
    feature_cols = [c for c in out.columns if c not in META_COLS]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value", *feature_cols]).reset_index(drop=True)


def metric_row(model: str, y: np.ndarray, yhat: np.ndarray, event_aware: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "event_aware": bool(event_aware),
        "MAE": float(mean_absolute_error(y, yhat)),
        "RMSE": rmse(y, yhat),
        "MAPE": mape(y, yhat),
        "R2": float(r2_score(y, yhat)),
    }
    for cap in CAP_LEVELS:
        row[f"Dpp@{int(cap*100)}"] = traffic_reduction(y, yhat, cap)
    return row


def evaluate_node(df_node: pd.DataFrame, args: argparse.Namespace, event_aware: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = make_supervised(df_node, args.n_lags)
    n_test = max(1, int(round(len(s) * args.test_fraction)))
    train = s.iloc[:-n_test].copy()
    test = s.iloc[-n_test:].copy()
    if len(train) < args.min_train_rows:
        raise ValueError("series too short")
    features = [c for c in s.columns if c not in META_COLS]
    Xtr = train[features].to_numpy(dtype=np.float32)
    ytr = train["value"].to_numpy(dtype=np.float32)
    Xte = test[features].to_numpy(dtype=np.float32)
    yte = test["value"].to_numpy(dtype=np.float32)

    rows: list[dict[str, Any]] = []
    predictions = test[["date", "hour", "far_edge"]].reset_index(drop=True)
    predictions["y_true"] = yte

    yhat0 = test["lag1"].to_numpy(dtype=float)
    rows.append(metric_row("S0_Naive", yte, yhat0, event_aware))
    predictions["yhat_S0_Naive"] = yhat0

    ridge = Ridge(alpha=1.0, fit_intercept=True).fit(Xtr, ytr)
    yhat1 = ridge.predict(Xte)
    rows.append(metric_row("S1_Ridge", yte, yhat1, event_aware))
    predictions["yhat_S1_Ridge"] = yhat1

    rf = RandomForestRegressor(n_estimators=args.rf_estimators, random_state=args.seed, n_jobs=-1).fit(Xtr, ytr)
    yhat2 = rf.predict(Xte)
    rows.append(metric_row("S2_RandomForest", yte, yhat2, event_aware))
    predictions["yhat_S2_RandomForest"] = yhat2

    if not args.skip_lstm:
        if tf is None:
            raise RuntimeError("TensorFlow is not installed; use --skip-lstm or install tensorflow")
        clear_tf()
        tf.keras.utils.set_random_seed(args.seed)
        xmin = np.nanmin(Xtr, axis=0)
        xmax = np.nanmax(Xtr, axis=0)
        xrng = np.where(np.abs(xmax - xmin) <= 1e-12, 1.0, xmax - xmin)
        Xtr_mm = ((Xtr - xmin) / xrng).astype(np.float32)
        Xte_mm = ((Xte - xmin) / xrng).astype(np.float32)
        ymin = float(np.nanmin(ytr)); ymax = float(np.nanmax(ytr)); yrng = ymax - ymin
        if abs(yrng) <= 1e-12:
            yrng = 1.0
        ytr_mm = ((ytr - ymin) / yrng).astype(np.float32)
        Xtr3 = Xtr_mm.reshape((len(Xtr_mm), 1, Xtr_mm.shape[1]))
        Xte3 = Xte_mm.reshape((len(Xte_mm), 1, Xte_mm.shape[1]))
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(1, Xtr_mm.shape[1])),
            tf.keras.layers.LSTM(args.lstm_units, activation="tanh"),
            tf.keras.layers.Dropout(args.lstm_dropout),
            tf.keras.layers.Dense(1, activation="linear"),
        ])
        model.compile(optimizer="adam", loss="mae")
        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=args.lstm_patience, restore_best_weights=True
        )
        model.fit(
            Xtr3,
            ytr_mm,
            epochs=args.lstm_epochs,
            batch_size=args.lstm_batch_size,
            shuffle=False,
            verbose=0,
            callbacks=[es],
        )
        yhat3 = model.predict(Xte3, verbose=0).reshape(-1) * yrng + ymin
        rows.append(metric_row("S3_LSTM", yte, yhat3, event_aware))
        predictions["yhat_S3_LSTM"] = yhat3
        del model
        clear_tf()

    return pd.DataFrame(rows), predictions


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["MAE", "RMSE", "MAPE", "R2", "Dpp@70", "Dpp@80", "Dpp@90"]
    agg = results.groupby("model")[metrics].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    return agg.sort_values("RMSE_mean").reset_index(drop=True)


def make_comparison_plot(summary: pd.DataFrame, path: Path, title_suffix: str) -> None:
    import matplotlib.pyplot as plt
    order = summary["model"].tolist()
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    specs = [
        ("R2", "Mean R²", 1.0),
        ("MAPE", "Mean MAPE [%]", 1.0),
        ("RMSE", "Mean RMSE (×10⁶)", 1e6),
    ]
    for ax, (metric, title, scale) in zip(axes, specs):
        means = summary[f"{metric}_mean"].to_numpy(float) / scale
        stds = summary[f"{metric}_std"].fillna(0.0).to_numpy(float) / scale
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_title(title)
    fig.suptitle(f"Model comparison ({title_suffix})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_mode(args: argparse.Namespace, events: EventFeatureTable | None, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_aware = events is not None
    mode_dir = args.output_dir / label
    mode_dir.mkdir(parents=True, exist_ok=True)
    clusters = discover_clusters(args.data_dir, args.clusters)
    panels = load_panels(args.data_dir, clusters, events, args.force_rebuild)
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for cluster in clusters:
        panel = panels[cluster]
        for far_edge in sorted(panel["far_edge"].unique(), key=str):
            node = panel.loc[panel["far_edge"] == far_edge].copy()
            try:
                metrics, preds = evaluate_node(node, args, event_aware)
            except ValueError:
                continue
            metrics.insert(0, "far_edge", str(far_edge))
            metrics.insert(0, "cluster", int(cluster))
            all_metrics.append(metrics)
            preds.insert(0, "cluster", int(cluster))
            preds["event_aware"] = bool(event_aware)
            all_predictions.append(preds)
    if not all_metrics:
        raise RuntimeError("No eligible nodes")
    results = pd.concat(all_metrics, ignore_index=True)
    summary = summarize(results)
    results.to_csv(mode_dir / "forecast_results_by_node.csv", index=False)
    summary.to_csv(mode_dir / "forecast_summary.csv", index=False)
    if args.save_predictions and all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(mode_dir / "forecast_pernode_predictions.csv", index=False)
    if args.make_plots:
        make_comparison_plot(summary, mode_dir / "figure_model_comparison.pdf", label.replace("_", " "))
    LOGGER.info("%s: %d node-model rows", label, len(results))
    LOGGER.info("%s summary:\n%s", label, summary.to_string(index=False))
    return results, summary


def main() -> int:
    args = parse_args()
    setup(args.seed)
    event_table = load_event_features(args.event_features_file)
    if args.compare_event_awareness:
        no_results, no_summary = run_mode(args, None, "no_event")
        ev_results, ev_summary = run_mode(args, event_table, "event_aware")
        combined_results = pd.concat([no_results, ev_results], ignore_index=True)
        combined_results.to_csv(args.output_dir / "forecast_results_by_node_both.csv", index=False)
        no_s = no_summary.copy(); no_s.insert(1, "mode", "no_event")
        ev_s = ev_summary.copy(); ev_s.insert(1, "mode", "event_aware")
        pd.concat([no_s, ev_s], ignore_index=True).to_csv(args.output_dir / "forecast_summary_both.csv", index=False)
    else:
        label = "event_aware" if event_table is not None else "no_event"
        run_mode(args, event_table, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
