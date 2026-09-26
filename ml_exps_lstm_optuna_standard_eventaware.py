#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard LSTM hyperparameter optimization with Optuna.

Configuration
-------------
- one LSTM layer
- MAE training loss
- Adam optimizer
- n_units:       50 .. 500
- n_epochs:      50 .. 500
- batch_size:     2 .. 128
- learning_rate: 1e-5 .. 1e-1
- n_trials:      1000 by default

Protocol
--------
- chronological 80/20 split per node, without shuffling
- the final 20% is reserved for testing and never enters the Optuna objective
- a chronological validation block is extracted from the pre-test data
- the selected hyperparameters are shared across nodes
- final evaluation reports RMSE, MAE, MAPE, R2, and the naive y(t-1) baseline

Features
--------
Endogenous inputs are lag1..lag24, hour-of-day and day-of-week encodings. If
``--event-features-file`` is supplied, aligned numeric exogenous features are
added, including scheduled-event or Google Trends signals. Omitting the file
runs the traffic-only/no-event configuration.

Expected hourly panel columns: date, hour, far_edge, value.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from event_feature_adapter import (
    EventFeatureTable,
    exogenous_columns,
    load_event_features,
    merge_event_features,
)

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Run: pip install tensorflow optuna pandas numpy scikit-learn"
    ) from exc


LOGGER = logging.getLogger("lstm_optuna")
META_COLS = ("date", "hour", "far_edge", "value")


@dataclass
class NodeData:
    cluster: int
    far_edge: str
    feature_names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    lag1_test: np.ndarray


@dataclass(frozen=True)
class ScaleState:
    x_min: np.ndarray
    x_range: np.ndarray
    y_min: float
    y_range: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LSTM opt_std / Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--clusters", type=int, nargs="*", default=None)
    p.add_argument("--n-lags", type=int, default=24)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of the pre-test block reserved for chronological validation.",
    )
    p.add_argument(
        "--tuning-nodes-per-cluster",
        type=int,
        default=1,
        help=("Nodes per cluster used by Optuna. "
              "1 keeps the search manageable; use 0 for all nodes."),
    )
    p.add_argument("--min-train-rows", type=int, default=48)
    p.add_argument("--event-features-file", type=Path, default=None)
    p.add_argument("--require-event-features", action="store_true")

    # 10^3 trials.
    p.add_argument("--n-trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lstm_optuna"),
    )
    p.add_argument("--study-name", default="netmob_lstm_opt_std")
    p.add_argument("--reset-study", action="store_true")
    p.add_argument("--tune-only", action="store_true")
    p.add_argument("--verbose-fit", type=int, choices=(0, 1, 2), default=0)
    args = p.parse_args()
    if args.require_event_features and args.event_features_file is None:
        p.error("--require-event-features requires --event-features-file")
    return args


def setup(seed: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(seed)
    np.random.seed(seed)
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
    tf.keras.backend.clear_session()
    gc.collect()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / den)) * 100.0)


def discover_clusters(data_dir: Path, requested: Sequence[int] | None) -> list[int]:
    if requested:
        return sorted(set(int(x) for x in requested))

    hourly_dir = data_dir / "hourly_panels"
    found = []
    for path in hourly_dir.glob("hourly_cluster_*.csv"):
        try:
            found.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(set(found))


def load_panels(data_dir: Path, clusters: Sequence[int], events: EventFeatureTable | None = None) -> dict[int, pd.DataFrame]:
    panels: dict[int, pd.DataFrame] = {}
    for cluster in clusters:
        path = data_dir / "hourly_panels" / f"hourly_cluster_{cluster}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Build the hourly cache before running this script."
            )
        df = pd.read_csv(path)
        required = {"date", "hour", "far_edge", "value"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        df = df[["date", "hour", "far_edge", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "hour", "far_edge", "value"])
        df["hour"] = df["hour"].astype(int)
        df = df[df["hour"].between(0, 23)]
        df, matched = merge_event_features(df, int(cluster), events)
        df = df.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)
        panels[int(cluster)] = df
        LOGGER.info(
            "Cluster %s: %d nodes | event-aware=%s | feature-row match=%.2f%%",
            cluster, df["far_edge"].nunique(), events is not None, matched
        )
    return panels


def make_supervised(df_node: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    g = df_node.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g["dow"] = g["date"].dt.weekday.astype(int)
    g = g.sort_values(["date", "hour"]).reset_index(drop=True)

    lag_cols = []
    for k in range(1, n_lags + 1):
        col = f"lag{k}"
        lag_cols.append(col)
        g[col] = g["value"].shift(k)

    # Use complete categories to keep feature dimensionality identical across nodes.
    h = pd.get_dummies(
        pd.Categorical(g["hour"], categories=range(24)),
        prefix="h",
        dtype=float,
    )
    d = pd.get_dummies(
        pd.Categorical(g["dow"], categories=range(7)),
        prefix="d",
        dtype=float,
    )
    h.index = g.index
    d.index = g.index

    parts = [g[["date", "hour", "far_edge", "value"]], h, d]
    exog_cols = exogenous_columns(g)
    if exog_cols:
        parts.append(g[exog_cols].astype(float))
    parts.append(g[lag_cols])
    out = pd.concat(parts, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def split_node(
    df_node: pd.DataFrame,
    cluster: int,
    far_edge: Any,
    n_lags: int,
    test_fraction: float,
    validation_fraction: float,
    min_train_rows: int,
) -> NodeData:
    s = make_supervised(df_node, n_lags)
    n = len(s)
    n_test = max(1, int(round(n * test_fraction)))
    n_pretest = n - n_test
    n_val = max(1, int(round(n_pretest * validation_fraction)))
    n_train = n_pretest - n_val

    if n_train < min_train_rows:
        raise ValueError(f"series too short: train={n_train}, val={n_val}, test={n_test}")

    train = s.iloc[:n_train]
    val = s.iloc[n_train:n_pretest]
    test = s.iloc[n_pretest:]
    features = tuple(c for c in s.columns if c not in META_COLS)

    def X(frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, features].to_numpy(dtype=np.float32)

    def y(frame: pd.DataFrame) -> np.ndarray:
        return frame["value"].to_numpy(dtype=np.float32)

    return NodeData(
        cluster=int(cluster),
        far_edge=str(far_edge),
        feature_names=features,
        X_train=X(train),
        y_train=y(train),
        X_val=X(val),
        y_val=y(val),
        X_test=X(test),
        y_test=y(test),
        lag1_test=test["lag1"].to_numpy(dtype=np.float32),
    )


def fit_scale(X: np.ndarray, y: np.ndarray) -> ScaleState:
    xmin = np.nanmin(X, axis=0).astype(np.float32)
    xmax = np.nanmax(X, axis=0).astype(np.float32)
    xrng = xmax - xmin
    xrng = np.where(np.abs(xrng) < 1e-12, 1.0, xrng).astype(np.float32)
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    yrng = ymax - ymin
    if abs(yrng) < 1e-12:
        yrng = 1.0
    return ScaleState(xmin, xrng, ymin, yrng)


def sx(X: np.ndarray, st: ScaleState) -> np.ndarray:
    return ((X - st.x_min) / st.x_range).astype(np.float32)


def sy(y: np.ndarray, st: ScaleState) -> np.ndarray:
    return ((y - st.y_min) / st.y_range).astype(np.float32)


def iy(y: np.ndarray, st: ScaleState) -> np.ndarray:
    return np.asarray(y, dtype=np.float32) * st.y_range + st.y_min


def lstm_X(X: np.ndarray) -> np.ndarray:
    # Keep one timestep with all engineered features.
    return X.reshape((X.shape[0], 1, X.shape[1])).astype(np.float32)


def build_model(n_features: int, n_units: int, learning_rate: float) -> tf.keras.Model:
    # opt_std: LSTM standard + Adam + MAE.
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1, n_features)),
            tf.keras.layers.LSTM(int(n_units), activation="tanh"),
            tf.keras.layers.Dense(1, activation="linear"),
        ],
        name="LSTM_opt_std",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="mae",
    )
    return model


def params(trial: optuna.Trial) -> dict[str, Any]:
    # Range
    return {
        "n_units": trial.suggest_int("n_units", 50, 500),
        "n_epochs": trial.suggest_int("n_epochs", 50, 500),
        "batch_size": trial.suggest_int("batch_size", 2, 128),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-1, log=True
        ),
    }


def early_stop(patience: int) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(patience),
        restore_best_weights=True,
    )


def collect_nodes(
    panels: dict[int, pd.DataFrame],
    args: argparse.Namespace,
) -> list[NodeData]:
    nodes: list[NodeData] = []
    for cluster in sorted(panels):
        panel = panels[cluster]
        accepted = 0
        for far_edge in sorted(panel["far_edge"].unique(), key=str):
            df_node = panel[panel["far_edge"] == far_edge].copy()
            try:
                node = split_node(
                    df_node,
                    cluster=cluster,
                    far_edge=far_edge,
                    n_lags=args.n_lags,
                    test_fraction=args.test_fraction,
                    validation_fraction=args.validation_fraction,
                    min_train_rows=args.min_train_rows,
                )
            except ValueError:
                continue
            nodes.append(node)
            accepted += 1
        LOGGER.info("Cluster %s: %d eligible nodes", cluster, accepted)
    if not nodes:
        raise RuntimeError("No eligible node was found.")
    return nodes


def choose_tuning_nodes(nodes: list[NodeData], per_cluster: int) -> list[NodeData]:
    if per_cluster <= 0:
        return nodes
    chosen: list[NodeData] = []
    by_cluster: dict[int, list[NodeData]] = {}
    for node in nodes:
        by_cluster.setdefault(node.cluster, []).append(node)
    for cluster in sorted(by_cluster):
        # Select nodes deterministically to keep the tuning subset reproducible.
        chosen.extend(by_cluster[cluster][:per_cluster])
    return chosen


def objective_factory(
    tuning_nodes: Sequence[NodeData],
    patience: int,
    seed: int,
    verbose_fit: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = params(trial)
        scores: list[float] = []

        for i, node in enumerate(tuning_nodes):
            clear_tf()
            tf.keras.utils.set_random_seed(seed + trial.number * 10000 + i)

            st = fit_scale(node.X_train, node.y_train)
            Xtr = lstm_X(sx(node.X_train, st))
            ytr = sy(node.y_train, st)
            Xva = lstm_X(sx(node.X_val, st))
            yva = sy(node.y_val, st)

            model = build_model(
                n_features=Xtr.shape[2],
                n_units=params["n_units"],
                learning_rate=params["learning_rate"],
            )
            model.fit(
                Xtr,
                ytr,
                validation_data=(Xva, yva),
                epochs=int(params["n_epochs"]),
                batch_size=int(params["batch_size"]),
                shuffle=False,
                verbose=verbose_fit,
                callbacks=[early_stop(patience), tf.keras.callbacks.TerminateOnNaN()],
            )

            pred_scaled = model.predict(Xva, verbose=0).reshape(-1)
            pred = iy(pred_scaled, st)
            score = rmse(node.y_val, pred)
            if not np.isfinite(score):
                raise optuna.TrialPruned("RMSE validation non finito")
            scores.append(score)
            del model

        return float(np.mean(scores))

    return objective


def train_final_node(
    node: NodeData,
    best: dict[str, Any],
    patience: int,
    seed: int,
    verbose_fit: int,
) -> dict[str, Any]:
    # 1) Determine the best epoch without using the test block.
    clear_tf()
    tf.keras.utils.set_random_seed(seed)
    st_sel = fit_scale(node.X_train, node.y_train)
    Xtr = lstm_X(sx(node.X_train, st_sel))
    ytr = sy(node.y_train, st_sel)
    Xva = lstm_X(sx(node.X_val, st_sel))
    yva = sy(node.y_val, st_sel)

    selector = build_model(Xtr.shape[2], best["n_units"], best["learning_rate"])
    hist = selector.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=int(best["n_epochs"]),
        batch_size=int(best["batch_size"]),
        shuffle=False,
        verbose=verbose_fit,
        callbacks=[early_stop(patience), tf.keras.callbacks.TerminateOnNaN()],
    )
    val_losses = np.asarray(hist.history.get("val_loss", []), dtype=float)
    if len(val_losses) and np.isfinite(val_losses).any():
        best_epoch = int(np.nanargmin(val_losses) + 1)
    else:
        best_epoch = max(1, len(hist.history.get("loss", [])))
    del selector

    # 2) Retrain on the full pre-test block (train+validation) for best_epoch.
    clear_tf()
    tf.keras.utils.set_random_seed(seed + 1)
    Xfit_raw = np.concatenate([node.X_train, node.X_val], axis=0)
    yfit_raw = np.concatenate([node.y_train, node.y_val], axis=0)
    st = fit_scale(Xfit_raw, yfit_raw)
    Xfit = lstm_X(sx(Xfit_raw, st))
    yfit = sy(yfit_raw, st)
    Xte = lstm_X(sx(node.X_test, st))

    model = build_model(Xfit.shape[2], best["n_units"], best["learning_rate"])
    model.fit(
        Xfit,
        yfit,
        epochs=best_epoch,
        batch_size=int(best["batch_size"]),
        shuffle=False,
        verbose=verbose_fit,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )
    pred = iy(model.predict(Xte, verbose=0).reshape(-1), st).astype(float)
    ytrue = node.y_test.astype(float)
    naive = node.lag1_test.astype(float)

    row = {
        "cluster": node.cluster,
        "far_edge": node.far_edge,
        "best_epoch": best_epoch,
        "MAE": float(mean_absolute_error(ytrue, pred)),
        "RMSE": rmse(ytrue, pred),
        "MAPE": mape(ytrue, pred),
        "R2": float(r2_score(ytrue, pred)),
        "Naive_MAE": float(mean_absolute_error(ytrue, naive)),
        "Naive_RMSE": rmse(ytrue, naive),
    }
    row["RMSE_improvement_vs_naive_pct"] = (
        100.0 * (row["Naive_RMSE"] - row["RMSE"]) / row["Naive_RMSE"]
        if row["Naive_RMSE"] > 0
        else np.nan
    )
    del model
    return row


def main() -> int:
    args = parse_args()
    setup(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clusters = discover_clusters(args.data_dir, args.clusters)
    if not clusters:
        raise RuntimeError("No hourly_cluster_*.csv cache was found.")
    LOGGER.info("Cluster: %s", clusters)
    events = load_event_features(args.event_features_file)
    LOGGER.info("Event-aware features: %s", events.source_path if events is not None else "DISABLED")

    panels = load_panels(args.data_dir, clusters, events)
    nodes = collect_nodes(panels, args)
    tuning_nodes = choose_tuning_nodes(nodes, args.tuning_nodes_per_cluster)

    LOGGER.info(
        "Total nodes=%d | Optuna tuning nodes=%d | trials=%d",
        len(nodes),
        len(tuning_nodes),
        args.n_trials,
    )
    if len(tuning_nodes) == len(nodes) and args.n_trials >= 1000:
        LOGGER.warning(
            "Large search configuration: %d trials x %d nodes. "
            "For a quick run use --tuning-nodes-per-cluster 1 --n-trials 30.",
            args.n_trials,
            len(tuning_nodes),
        )

    storage_path = (args.output_dir / "optuna_study.sqlite3").resolve()
    storage = f"sqlite:///{storage_path.as_posix()}"
    if args.reset_study:
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage)
        except KeyError:
            pass

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        objective_factory(
            tuning_nodes=tuning_nodes,
            patience=args.patience,
            seed=args.seed,
            verbose_fit=args.verbose_fit,
        ),
        n_trials=args.n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best = dict(study.best_params)
    LOGGER.info("Best validation RMSE medio: %.6f", study.best_value)
    LOGGER.info("Best params: %s", best)

    with (args.output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "study_name": study.study_name,
                "best_trial": study.best_trial.number,
                "best_validation_rmse": study.best_value,
                "best_params": best,
                "search_space": {
                    "n_units": [50, 500],
                    "n_epochs": [50, 500],
                    "batch_size": [2, 128],
                    "learning_rate": [1e-5, 1e-1],
                    "loss": "mae",
                    "optimizer": "Adam",
                    "n_trials": 1000,
                },
                "event_aware": bool(events is not None),
                "event_features_file": str(events.source_path) if events is not None else None,
                "event_feature_names": list(events.source_feature_names) if events is not None else [],
                "note": "Without --event-features-file this is the traffic-only/no-event ablation; with it, aligned numeric exogenous features are appended.",
            },
            f,
            indent=2,
        )
    study.trials_dataframe().to_csv(args.output_dir / "study_trials.csv", index=False)

    if args.tune_only:
        return 0

    rows = []
    for i, node in enumerate(nodes, start=1):
        LOGGER.info(
            "Finale %d/%d: cluster=%s far_edge=%s",
            i,
            len(nodes),
            node.cluster,
            node.far_edge,
        )
        rows.append(
            train_final_node(
                node=node,
                best=best,
                patience=args.patience,
                seed=args.seed + i * 100,
                verbose_fit=args.verbose_fit,
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "final_metrics_by_node.csv", index=False)

    summary = {
        "nodes": int(len(metrics)),
        "RMSE": float(metrics["RMSE"].mean()),
        "MAE": float(metrics["MAE"].mean()),
        "MAPE": float(metrics["MAPE"].mean()),
        "R2": float(metrics["R2"].mean()),
        "event_aware": bool(events is not None),
        "Naive_RMSE": float(metrics["Naive_RMSE"].mean()),
        "Naive_MAE": float(metrics["Naive_MAE"].mean()),
        "RMSE_improvement_vs_naive_pct_mean": float(
            metrics["RMSE_improvement_vs_naive_pct"].mean()
        ),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "final_summary.csv", index=False)

    LOGGER.info("Final evaluation completed on %d nodes", summary["nodes"])
    LOGGER.info("RMSE medio LSTM opt_std: %.6f", summary["RMSE"])
    LOGGER.info("RMSE medio Naive: %.6f", summary["Naive_RMSE"])
    LOGGER.info(
        "Mean per-node improvement vs Naive: %.2f%%",
        summary["RMSE_improvement_vs_naive_pct_mean"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
