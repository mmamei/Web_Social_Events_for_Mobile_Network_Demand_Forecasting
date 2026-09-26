#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM + Optuna for the opt_cst experiment.

The network search space matches the opt_std script:
    - one LSTM layer
    - n_units:       50 .. 500
    - n_epochs:      50 .. 500
    - batch_size:     2 .. 128
    - learning_rate: 1e-5 .. 1e-1
    - Adam optimizer
    - 1000 Optuna trials by default

The training loss is the direction-aware, bounded, piecewise provisioning loss
defined below. With signed error e = prediction - demand:

    e <= 0:
        min(K, alpha - epsilon * e)

    0 < e <= e_end:
        max(eta, alpha - m_sh * e)

    e > e_end:
        eta + (K - eta) * sigma((e - e_end) / Delta)

where:
    e_end = epsilon * alpha + delta
    m_sh  = (alpha - eta) / e_end
    sigma(t) = 0.5 * (1 - cos(pi * clip(t, 0, 1)))

A quintic smootherstep tail is also available through --tail-shape.

Important scale convention
--------------------------
The target is MinMax-scaled per node using training data only. Therefore
alpha, epsilon, K, delta, eta, and Delta are interpreted in normalized target
space. The script records their exact values in the output metadata.

Event-aware features
--------------------
The hourly panel cache must contain:
    date, hour, far_edge, value

An optional aligned event-feature CSV can be supplied with
--event-features-file. It must contain either:
    date, hour
or:
    timestamp

It may also contain cluster and/or far_edge as join keys. Every remaining
column must be numeric and already aligned/encoded, for example:
    public_holiday, event_active, event_relevance,
    event_type_sport, event_type_festival, event_type_concert

Relevance columns are checked to be in [0, 1]. Missing event rows are filled
with zero. Hour, day of week, and weekend are
encoded internally with complete one-hot categories.

Without --event-features-file the script runs the traffic-only/no-event
ablation. Use --require-event-features to prevent accidental no-event runs.

Examples
--------
Quick functional run:

    python ml_exps_lstm_optuna_custom_loss_v3_eventaware.py \
        --clusters 0 1 2 3 4 \
        --n-trials 10 \
        --tuning-nodes-per-cluster 1 \
        --reset-study

Full-scale search with aligned event features:

    python ml_exps_lstm_optuna_custom_loss_v3_eventaware.py \
        --clusters 0 1 2 3 4 \
        --event-features-file data/event_features_hourly.csv \
        --require-event-features \
        --n-trials 1000 \
        --tuning-nodes-per-cluster 1 \
        --alpha 0.70 \
        --epsilon-band 0.05 \
        --loss-cap 1.0 \
        --shoulder-delta 0.01 \
        --eta 0.02 \
        --tail-width 0.25 \
        --reset-study

Notes
-----
1. The loss parameters above are explicit defaults, not best values.
2. By default the loss parameters are fixed. Use --tune-loss-params to let
   Optuna tune alpha, epsilon, delta, eta (through eta/alpha), and Delta.
   K stays fixed unless --tune-loss-cap is also supplied.
3. When loss parameters are tuned, the Optuna selection objective is forced to
   validation RMSE; minimizing a custom loss while changing its own definition
   would not be a meaningful comparison across trials.
4. Early stopping and best-epoch selection are aligned with the Optuna objective:
   val_rmse for --optuna-objective rmse, val_loss for custom_loss. The network is
   always TRAINED with DirectionAwareProvisioningLoss.
5. Optuna pruning, when enabled, is applied after each tuning node using the same
   objective returned to Optuna; no mixed-scale Keras pruning callback is used.
6. The final 20 percent test block is never used by Optuna.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Run: "
        "pip install tensorflow optuna pandas numpy scikit-learn"
    ) from exc

LOGGER = logging.getLogger("lstm_optuna_custom_loss")
PANEL_META_COLS = ("date", "hour", "far_edge", "value")
CAP_LEVELS = (0.70, 0.80, 0.90)
EXOG_PREFIX = "exog__"


@dataclass(frozen=True)
class LossConfig:
    """Parameters of the direction-aware provisioning loss."""

    alpha: float
    epsilon: float
    cap: float
    delta: float
    eta: float
    tail_width: float
    tail_shape: str = "raised_cosine"

    @property
    def e_end(self) -> float:
        return self.epsilon * self.alpha + self.delta

    @property
    def shoulder_slope(self) -> float:
        return (self.alpha - self.eta) / self.e_end

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.cap <= 0.0:
            raise ValueError("K/loss-cap must be > 0")
        if self.delta <= 0.0:
            raise ValueError("delta must be > 0")
        if self.tail_width <= 0.0:
            raise ValueError("Delta/tail-width must be > 0")
        if not 0.0 < self.eta < self.alpha:
            raise ValueError(
                "eta must satisfy 0 < eta < alpha. This is required for the "
                "descending shoulder and continuity at e=0."
            )
        if self.cap < self.alpha:
            raise ValueError(
                "K/loss-cap must be >= alpha so that the loss is bounded in [0, K]."
            )
        if self.tail_shape not in {"raised_cosine", "smootherstep"}:
            raise ValueError("tail_shape must be raised_cosine or smootherstep")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "K": payload.pop("cap"),
                "Delta": payload.pop("tail_width"),
                "e_end": self.e_end,
                "m_sh": self.shoulder_slope,
                "target_scale": "per-node MinMax [0,1] fitted on training data only",
            }
        )
        return payload


@dataclass(frozen=True)
class EventFeatureTable:
    frame: pd.DataFrame
    join_keys: tuple[str, ...]
    source_feature_names: tuple[str, ...]
    model_feature_names: tuple[str, ...]
    source_path: Path


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
    test_keys: pd.DataFrame


@dataclass(frozen=True)
class ScaleState:
    x_min: np.ndarray
    x_range: np.ndarray
    y_min: float
    y_range: float


@tf.keras.utils.register_keras_serializable(package="NetMob")
class DirectionAwareProvisioningLoss(tf.keras.losses.Loss):
    """TensorFlow implementation of the supplied piecewise loss."""

    def __init__(
        self,
        alpha: float,
        epsilon: float,
        cap: float,
        delta: float,
        eta: float,
        tail_width: float,
        tail_shape: str = "raised_cosine",
        reduction: str = "sum_over_batch_size",
        name: str = "direction_aware_provisioning_loss",
    ) -> None:
        config = LossConfig(
            alpha=float(alpha),
            epsilon=float(epsilon),
            cap=float(cap),
            delta=float(delta),
            eta=float(eta),
            tail_width=float(tail_width),
            tail_shape=str(tail_shape),
        )
        config.validate()
        super().__init__(reduction=reduction, name=name)
        self.alpha = config.alpha
        self.epsilon = config.epsilon
        self.cap = config.cap
        self.delta = config.delta
        self.eta = config.eta
        self.tail_width = config.tail_width
        self.tail_shape = config.tail_shape
        self.e_end = config.e_end
        self.m_sh = config.shoulder_slope

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        error = y_pred - y_true

        dtype = y_pred.dtype
        alpha = tf.cast(self.alpha, dtype)
        epsilon = tf.cast(self.epsilon, dtype)
        cap = tf.cast(self.cap, dtype)
        eta = tf.cast(self.eta, dtype)
        e_end = tf.cast(self.e_end, dtype)
        m_sh = tf.cast(self.m_sh, dtype)
        tail_width = tf.cast(self.tail_width, dtype)

        under = tf.minimum(cap, alpha - epsilon * error)
        shoulder = tf.maximum(eta, alpha - m_sh * error)

        t = tf.clip_by_value((error - e_end) / tail_width, 0.0, 1.0)
        if self.tail_shape == "raised_cosine":
            sigma = 0.5 * (1.0 - tf.cos(tf.cast(np.pi, dtype) * t))
        else:
            # Quintic smootherstep: 6t^5 - 15t^4 + 10t^3.
            sigma = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        tail = eta + (cap - eta) * sigma

        loss = tf.where(
            error <= 0.0,
            under,
            tf.where(error <= e_end, shoulder, tail),
        )
        loss = tf.clip_by_value(loss, 0.0, cap)
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "cap": self.cap,
                "delta": self.delta,
                "eta": self.eta,
                "tail_width": self.tail_width,
                "tail_shape": self.tail_shape,
            }
        )
        return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LSTM Optuna search with the direction-aware custom loss"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clusters", type=int, nargs="*", default=None)
    parser.add_argument("--n-lags", type=int, default=24)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of the pre-test block reserved for temporal validation.",
    )
    parser.add_argument(
        "--tuning-nodes-per-cluster",
        type=int,
        default=1,
        help="0 uses all eligible nodes in every Optuna trial.",
    )
    parser.add_argument("--min-train-rows", type=int, default=48)

    parser.add_argument("--event-features-file", type=Path, default=None)
    parser.add_argument("--require-event-features", action="store_true")
    parser.add_argument(
        "--include-weekend-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Early-stopping patience. The monitored quantity follows --optuna-objective: "
            "val_rmse for rmse, val_loss for custom_loss."
        ),
    )
    parser.add_argument(
        "--optuna-objective",
        choices=("rmse", "custom_loss"),
        default="rmse",
        help=(
            "rmse keeps the same model-selection metric as the opt_std control; "
            "custom_loss selects directly for provisioning cost."
        ),
    )
    parser.add_argument(
        "--output-activation",
        choices=("linear", "sigmoid"),
        default="linear",
        help="Use linear for architecture parity with opt_std; sigmoid bounds scaled output.",
    )
    parser.add_argument(
        "--clip-negative-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip final predictions at zero in original demand units.",
    )

    # Direction-aware loss parameters. Fixed by default; optionally tuned by Optuna.
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--epsilon-band", type=float, default=0.05)
    parser.add_argument("--loss-cap", type=float, default=1.0, dest="loss_cap")
    parser.add_argument("--shoulder-delta", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--tail-width", type=float, default=0.25)
    parser.add_argument(
        "--tail-shape",
        choices=("raised_cosine", "smootherstep"),
        default="raised_cosine",
    )
    parser.add_argument(
        "--tune-loss-params",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also tune alpha, epsilon, delta, eta/alpha and Delta. "
            "The selection objective must be RMSE when this is enabled."
        ),
    )
    parser.add_argument(
        "--tune-loss-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also tune K; requires --tune-loss-params.",
    )
    parser.add_argument(
        "--loss-alpha-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.20, 0.95),
    )
    parser.add_argument(
        "--loss-epsilon-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1e-3, 2e-1),
    )
    parser.add_argument(
        "--loss-delta-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1e-3, 1e-1),
    )
    parser.add_argument(
        "--loss-eta-ratio-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.01, 0.50),
        help="Range for eta/alpha, which guarantees 0 < eta < alpha.",
    )
    parser.add_argument(
        "--loss-tail-width-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(2e-2, 1.0),
    )
    parser.add_argument(
        "--loss-cap-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1.0, 1.5),
        help="Range for K when --tune-loss-cap is enabled.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lstm_optuna_custom_loss_v3"),
    )
    parser.add_argument("--study-name", default="netmob_lstm_opt_cst_v3")
    parser.add_argument("--reset-study", action="store_true")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-pruning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disabled by default. When enabled, pruning is evaluated after each "
            "tuning node using the actual Optuna objective."
        ),
    )
    parser.add_argument("--verbose-fit", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Validate and sample the custom loss without loading traffic data.",
    )

    args = parser.parse_args(argv)

    if args.n_lags < 1:
        parser.error("--n-lags must be >= 1")
    if not 0.0 < args.test_fraction < 0.5:
        parser.error("--test-fraction must be in (0, 0.5)")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    if args.tuning_nodes_per_cluster < 0:
        parser.error("--tuning-nodes-per-cluster cannot be negative")
    if args.min_train_rows < 2:
        parser.error("--min-train-rows must be >= 2")
    if args.n_trials < 1:
        parser.error("--n-trials must be >= 1")
    if args.patience < 0:
        parser.error("--patience cannot be negative")
    if args.require_event_features and args.event_features_file is None:
        parser.error("--require-event-features requires --event-features-file")
    if args.tune_loss_cap and not args.tune_loss_params:
        parser.error("--tune-loss-cap requires --tune-loss-params")
    if args.tune_loss_params and args.optuna_objective != "rmse":
        parser.error(
            "--tune-loss-params requires --optuna-objective rmse. "
            "Do not optimize the numerical value of a loss while simultaneously "
            "changing that loss definition across trials."
        )

    def validate_range(name: str, values: Sequence[float], *, positive: bool = False) -> None:
        low, high = map(float, values)
        if not low < high:
            parser.error(f"{name}: expected MIN < MAX")
        if positive and low <= 0.0:
            parser.error(f"{name}: MIN must be > 0")

    validate_range("--loss-alpha-range", args.loss_alpha_range)
    validate_range("--loss-epsilon-range", args.loss_epsilon_range, positive=True)
    validate_range("--loss-delta-range", args.loss_delta_range, positive=True)
    validate_range("--loss-eta-ratio-range", args.loss_eta_ratio_range, positive=True)
    validate_range("--loss-tail-width-range", args.loss_tail_width_range, positive=True)
    validate_range("--loss-cap-range", args.loss_cap_range, positive=True)

    alpha_low, alpha_high = map(float, args.loss_alpha_range)
    eta_ratio_low, eta_ratio_high = map(float, args.loss_eta_ratio_range)
    if not (0.0 < alpha_low < alpha_high < 1.0):
        parser.error("--loss-alpha-range must lie strictly inside (0, 1)")
    if not (0.0 < eta_ratio_low < eta_ratio_high < 1.0):
        parser.error("--loss-eta-ratio-range must lie strictly inside (0, 1)")
    if args.tune_loss_params and not args.tune_loss_cap and args.loss_cap < alpha_high:
        parser.error(
            "Fixed --loss-cap must be >= the upper bound of --loss-alpha-range "
            "when --tune-loss-params is enabled"
        )
    if args.tune_loss_cap and float(args.loss_cap_range[0]) < alpha_high:
        parser.error(
            "The lower bound of --loss-cap-range must be >= the upper bound of "
            "--loss-alpha-range so every sampled configuration satisfies K >= alpha"
        )

    return args


def make_loss_config(args: argparse.Namespace) -> LossConfig:
    config = LossConfig(
        alpha=float(args.alpha),
        epsilon=float(args.epsilon_band),
        cap=float(args.loss_cap),
        delta=float(args.shoulder_delta),
        eta=float(args.eta),
        tail_width=float(args.tail_width),
        tail_shape=str(args.tail_shape),
    )
    config.validate()
    return config


def loss_search_space(args: argparse.Namespace) -> dict[str, Any]:
    if not args.tune_loss_params:
        return {"enabled": False}
    payload: dict[str, Any] = {
        "enabled": True,
        "alpha": [float(args.loss_alpha_range[0]), float(args.loss_alpha_range[1])],
        "epsilon": [
            float(args.loss_epsilon_range[0]),
            float(args.loss_epsilon_range[1]),
            "log",
        ],
        "delta": [
            float(args.loss_delta_range[0]),
            float(args.loss_delta_range[1]),
            "log",
        ],
        "eta_over_alpha": [
            float(args.loss_eta_ratio_range[0]),
            float(args.loss_eta_ratio_range[1]),
        ],
        "Delta": [
            float(args.loss_tail_width_range[0]),
            float(args.loss_tail_width_range[1]),
            "log",
        ],
        "K": (
            [float(args.loss_cap_range[0]), float(args.loss_cap_range[1])]
            if args.tune_loss_cap
            else float(args.loss_cap)
        ),
        "tail_shape": str(args.tail_shape),
    }
    return payload


def trial_loss_config(
    trial: optuna.Trial,
    args: argparse.Namespace,
    fixed_config: LossConfig,
) -> LossConfig:
    if not args.tune_loss_params:
        return fixed_config

    alpha = trial.suggest_float(
        "loss_alpha",
        float(args.loss_alpha_range[0]),
        float(args.loss_alpha_range[1]),
    )
    epsilon = trial.suggest_float(
        "loss_epsilon",
        float(args.loss_epsilon_range[0]),
        float(args.loss_epsilon_range[1]),
        log=True,
    )
    delta = trial.suggest_float(
        "loss_delta",
        float(args.loss_delta_range[0]),
        float(args.loss_delta_range[1]),
        log=True,
    )
    eta_ratio = trial.suggest_float(
        "loss_eta_ratio",
        float(args.loss_eta_ratio_range[0]),
        float(args.loss_eta_ratio_range[1]),
    )
    tail_width = trial.suggest_float(
        "loss_tail_width",
        float(args.loss_tail_width_range[0]),
        float(args.loss_tail_width_range[1]),
        log=True,
    )
    cap = (
        trial.suggest_float(
            "loss_cap",
            float(args.loss_cap_range[0]),
            float(args.loss_cap_range[1]),
        )
        if args.tune_loss_cap
        else fixed_config.cap
    )
    eta = alpha * eta_ratio
    config = LossConfig(
        alpha=float(alpha),
        epsilon=float(epsilon),
        cap=float(cap),
        delta=float(delta),
        eta=float(eta),
        tail_width=float(tail_width),
        tail_shape=fixed_config.tail_shape,
    )
    config.validate()
    trial.set_user_attr("loss_eta", float(config.eta))
    trial.set_user_attr("loss_e_end", float(config.e_end))
    trial.set_user_attr("loss_m_sh", float(config.shoulder_slope))
    return config


def best_loss_config_from_study(
    study: optuna.Study,
    args: argparse.Namespace,
    fixed_config: LossConfig,
) -> LossConfig:
    if not args.tune_loss_params:
        return fixed_config
    params = study.best_params
    alpha = float(params["loss_alpha"])
    eta = alpha * float(params["loss_eta_ratio"])
    config = LossConfig(
        alpha=alpha,
        epsilon=float(params["loss_epsilon"]),
        cap=float(params["loss_cap"]) if args.tune_loss_cap else fixed_config.cap,
        delta=float(params["loss_delta"]),
        eta=eta,
        tail_width=float(params["loss_tail_width"]),
        tail_shape=fixed_config.tail_shape,
    )
    config.validate()
    return config


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
    denominator = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


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


def direction_aware_loss_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: LossConfig,
) -> np.ndarray:
    """Vectorized NumPy implementation used for diagnostics and final metrics."""

    config.validate()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    under = np.minimum(config.cap, config.alpha - config.epsilon * error)
    shoulder = np.maximum(
        config.eta,
        config.alpha - config.shoulder_slope * error,
    )
    t = np.clip((error - config.e_end) / config.tail_width, 0.0, 1.0)
    if config.tail_shape == "raised_cosine":
        sigma = 0.5 * (1.0 - np.cos(np.pi * t))
    else:
        sigma = t**3 * (t * (t * 6.0 - 15.0) + 10.0)
    tail = config.eta + (config.cap - config.eta) * sigma

    loss = np.where(error <= 0.0, under, np.where(error <= config.e_end, shoulder, tail))
    return np.clip(loss, 0.0, config.cap)


def self_test_loss(config: LossConfig) -> pd.DataFrame:
    """Check boundaries, continuity, cap, and TensorFlow/NumPy agreement."""

    config.validate()
    tiny = 1e-7

    errors = np.asarray(
        [
            -2.0,
            -tiny,
            0.0,
            tiny,
            config.e_end - tiny,
            config.e_end,
            config.e_end + tiny,
            config.e_end + 0.5 * config.tail_width,
            config.e_end + config.tail_width,
            config.e_end + 2.0 * config.tail_width,
        ],
        dtype=np.float64,
    )

    y_true = np.zeros_like(errors)

    # NumPy uses float64.
    np_loss = direction_aware_loss_numpy(y_true, errors, config)

    tf_loss_object = DirectionAwareProvisioningLoss(
        alpha=config.alpha,
        epsilon=config.epsilon,
        cap=config.cap,
        delta=config.delta,
        eta=config.eta,
        tail_width=config.tail_width,
        tail_shape=config.tail_shape,
        reduction="none",
    )

    # Keras typically uses float32 during normal training.
    tf_loss = tf_loss_object(
        tf.constant(y_true.reshape(-1, 1), dtype=tf.float32),
        tf.constant(errors.reshape(-1, 1), dtype=tf.float32),
    ).numpy().reshape(-1)

    # Compare using TensorFlow's effective numerical precision.
    np_loss_tf32 = np_loss.astype(np.float32)

    abs_diff = np.abs(np_loss_tf32 - tf_loss)
    max_diff = float(np.max(abs_diff))

    if not np.allclose(
        np_loss_tf32,
        tf_loss,
        atol=2e-6,
        rtol=2e-6,
    ):
        idx = int(np.argmax(abs_diff))
        raise AssertionError(
            "TensorFlow and NumPy loss implementations disagree: "
            f"max_diff={max_diff:.9g}, "
            f"e={errors[idx]:.9g}, "
            f"numpy={np_loss_tf32[idx]:.9g}, "
            f"tensorflow={tf_loss[idx]:.9g}"
        )

    if np.nanmin(np_loss) < -1e-10 or np.nanmax(np_loss) > config.cap + 1e-10:
        raise AssertionError("Loss is outside [0, K]")

    at_zero = direction_aware_loss_numpy(np.asarray([0.0]), np.asarray([0.0]), config)[0]
    at_end = direction_aware_loss_numpy(
        np.asarray([0.0]), np.asarray([config.e_end]), config
    )[0]
    if not np.isclose(at_zero, config.alpha, atol=1e-10):
        raise AssertionError("Loss at e=0 is not alpha")
    if not np.isclose(at_end, config.eta, atol=1e-10):
        raise AssertionError("Loss at e=e_end is not eta")

    frame = pd.DataFrame(
        {
            "error_e": errors,
            "loss_numpy": np_loss,
            "loss_numpy_float32": np_loss_tf32,
            "loss_tensorflow": tf_loss,
            "abs_diff": abs_diff,
        }
    )
    LOGGER.info(
        "Loss self-test passed: e_end=%.8f, m_sh=%.8f, "
        "range=[%.8f, %.8f], max|TF-NP|=%.3e",
        config.e_end,
        config.shoulder_slope,
        float(np.min(np_loss)),
        float(np.max(np_loss)),
        max_diff,
    )
    return frame


def write_loss_curve(
    output_dir: Path,
    config: LossConfig,
    filename: str = "loss_curve.csv",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    left = -1.0
    right = max(1.0, config.e_end + 1.25 * config.tail_width)
    errors = np.linspace(left, right, 2001)
    values = direction_aware_loss_numpy(
        np.zeros_like(errors),
        errors,
        config,
    )
    branch = np.where(
        errors <= 0.0,
        "under_provisioning",
        np.where(errors <= config.e_end, "shoulder", "saturating_tail"),
    )
    pd.DataFrame(
        {
            "error_e": errors,
            "loss": values,
            "branch": branch,
        }
    ).to_csv(output_dir / filename, index=False)


def discover_clusters(data_dir: Path, requested: Sequence[int] | None) -> list[int]:
    if requested:
        return sorted(set(int(value) for value in requested))
    found: list[int] = []
    for path in (data_dir / "hourly_panels").glob("hourly_cluster_*.csv"):
        match = re.search(r"hourly_cluster_(-?\d+)\.csv$", path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(set(found))


def _normalize_time_columns(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    frame = frame.copy()
    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        if "date" not in frame.columns:
            frame["date"] = timestamp.dt.normalize()
        if "hour" not in frame.columns:
            frame["hour"] = timestamp.dt.hour
    if "date" not in frame.columns or "hour" not in frame.columns:
        raise ValueError(
            f"{source}: expected date+hour or timestamp columns for event alignment"
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    return frame


def load_event_features(path: Path | None) -> EventFeatureTable | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Event feature file not found: {path}")

    frame = pd.read_csv(path)
    frame = _normalize_time_columns(frame, path)
    frame = frame.dropna(subset=["date", "hour"]).copy()
    frame["hour"] = frame["hour"].astype(int)
    if not frame["hour"].between(0, 23).all():
        raise ValueError(f"{path}: event feature hour must be in 0..23")

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
    source_features = [column for column in frame.columns if column not in ignored]
    if not source_features:
        raise ValueError(f"{path}: no event feature columns were found")

    reserved = set(PANEL_META_COLS)
    conflicts = [column for column in source_features if column in reserved]
    if conflicts:
        raise ValueError(f"{path}: reserved event feature names: {conflicts}")

    for column in source_features:
        original_non_null = frame[column].notna()
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = original_non_null & converted.isna()
        if bad.any():
            examples = frame.loc[bad, column].astype(str).head(3).tolist()
            raise ValueError(
                f"{path}: event feature {column!r} is not numeric; examples={examples}"
            )
        frame[column] = converted.fillna(0.0).astype(np.float32)
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: event feature {column!r} contains inf values")

        lower_name = column.lower()
        if "relevance" in lower_name or "proximity" in lower_name:
            values = frame[column].to_numpy(dtype=float)
            if np.nanmin(values) < -1e-9 or np.nanmax(values) > 1.0 + 1e-9:
                raise ValueError(
                    f"{path}: relevance/proximity column {column!r} must be in [0,1]"
                )

    duplicate_mask = frame.duplicated(subset=join_keys, keep=False)
    if duplicate_mask.any():
        example = frame.loc[duplicate_mask, join_keys].head(5).to_dict(orient="records")
        raise ValueError(
            f"{path}: duplicate event rows for join keys {join_keys}; "
            f"pre-aggregate them first. Examples: {example}"
        )

    rename_map = {column: f"{EXOG_PREFIX}{column}" for column in source_features}
    frame = frame[[*join_keys, *source_features]].rename(columns=rename_map)
    model_features = tuple(rename_map[column] for column in source_features)

    LOGGER.info(
        "Loaded event features: rows=%d, keys=%s, features=%s",
        len(frame),
        join_keys,
        list(source_features),
    )
    return EventFeatureTable(
        frame=frame,
        join_keys=tuple(join_keys),
        source_feature_names=tuple(source_features),
        model_feature_names=model_features,
        source_path=path,
    )


def merge_event_features(
    panel: pd.DataFrame,
    cluster: int,
    events: EventFeatureTable | None,
) -> pd.DataFrame:
    panel = panel.copy()
    if events is None:
        return panel

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
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    if "cluster" in merged.columns and "cluster" not in PANEL_META_COLS:
        merged = merged.drop(columns=["cluster"])

    LOGGER.info("Cluster %s: event rows matched %.2f%%", cluster, matched)
    return merged


def load_panels(
    data_dir: Path,
    clusters: Sequence[int],
    events: EventFeatureTable | None,
) -> dict[int, pd.DataFrame]:
    panels: dict[int, pd.DataFrame] = {}
    for cluster in clusters:
        path = data_dir / "hourly_panels" / f"hourly_cluster_{cluster}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Build hourly caches with ml_exps.py first."
            )
        frame = pd.read_csv(path)
        missing = set(PANEL_META_COLS).difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        frame = frame[list(PANEL_META_COLS)].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["far_edge"] = frame["far_edge"].astype(str)
        frame = frame.dropna(subset=["date", "hour", "far_edge", "value"])
        frame["hour"] = frame["hour"].astype(int)
        frame = frame.loc[frame["hour"].between(0, 23)].copy()
        frame = merge_event_features(frame, cluster, events)
        frame = frame.sort_values(["far_edge", "date", "hour"]).reset_index(drop=True)
        panels[int(cluster)] = frame
        LOGGER.info("Cluster %s: %d eligible raw nodes", cluster, frame["far_edge"].nunique())
    return panels


def make_supervised(
    node_frame: pd.DataFrame,
    n_lags: int,
    include_weekend_onehot: bool,
) -> pd.DataFrame:
    frame = node_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["dow"] = frame["date"].dt.weekday.astype(int)
    frame["weekend"] = (frame["dow"] >= 5).astype(int)
    frame = frame.sort_values(["date", "hour"]).reset_index(drop=True)

    lag_columns: list[str] = []
    for lag in range(1, n_lags + 1):
        column = f"lag{lag}"
        lag_columns.append(column)
        frame[column] = frame["value"].shift(lag)

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

    parts: list[pd.DataFrame] = [
        frame[list(PANEL_META_COLS)],
        hour_dummies,
        dow_dummies,
    ]
    if include_weekend_onehot:
        weekend_dummies = pd.get_dummies(
            pd.Categorical(frame["weekend"], categories=(0, 1)),
            prefix="weekend",
            dtype=float,
        )
        weekend_dummies.index = frame.index
        parts.append(weekend_dummies)

    exogenous_columns = sorted(
        column for column in frame.columns if column.startswith(EXOG_PREFIX)
    )
    if exogenous_columns:
        parts.append(frame[exogenous_columns].astype(float))
    parts.append(frame[lag_columns])

    supervised = pd.concat(parts, axis=1)
    feature_columns = [
        column for column in supervised.columns if column not in PANEL_META_COLS
    ]
    supervised = supervised.replace([np.inf, -np.inf], np.nan)
    supervised = supervised.dropna(subset=["value", *feature_columns])
    return supervised.reset_index(drop=True)


def split_node(
    node_frame: pd.DataFrame,
    cluster: int,
    far_edge: Any,
    n_lags: int,
    test_fraction: float,
    validation_fraction: float,
    min_train_rows: int,
    include_weekend_onehot: bool,
) -> NodeData:
    supervised = make_supervised(
        node_frame=node_frame,
        n_lags=n_lags,
        include_weekend_onehot=include_weekend_onehot,
    )
    n_rows = len(supervised)
    n_test = max(1, int(round(n_rows * test_fraction)))
    n_pretest = n_rows - n_test
    n_validation = max(1, int(round(n_pretest * validation_fraction)))
    n_train = n_pretest - n_validation

    if n_train < min_train_rows or n_validation < 1 or n_test < 1:
        raise ValueError(
            f"series too short: train={n_train}, val={n_validation}, test={n_test}"
        )

    train = supervised.iloc[:n_train]
    validation = supervised.iloc[n_train:n_pretest]
    test = supervised.iloc[n_pretest:]
    feature_names = tuple(
        column for column in supervised.columns if column not in PANEL_META_COLS
    )

    def X(frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)

    def y(frame: pd.DataFrame) -> np.ndarray:
        return frame["value"].to_numpy(dtype=np.float32, copy=True)

    return NodeData(
        cluster=int(cluster),
        far_edge=str(far_edge),
        feature_names=feature_names,
        X_train=X(train),
        y_train=y(train),
        X_val=X(validation),
        y_val=y(validation),
        X_test=X(test),
        y_test=y(test),
        lag1_test=test["lag1"].to_numpy(dtype=np.float32, copy=True),
        test_keys=test[["date", "hour", "far_edge"]].reset_index(drop=True),
    )


def collect_nodes(
    panels: dict[int, pd.DataFrame],
    args: argparse.Namespace,
) -> list[NodeData]:
    nodes: list[NodeData] = []
    reference_features: tuple[str, ...] | None = None

    for cluster in sorted(panels):
        panel = panels[cluster]
        accepted = 0
        for far_edge in sorted(panel["far_edge"].dropna().unique().tolist(), key=str):
            node_frame = panel.loc[panel["far_edge"] == far_edge].copy()
            try:
                node = split_node(
                    node_frame=node_frame,
                    cluster=cluster,
                    far_edge=far_edge,
                    n_lags=args.n_lags,
                    test_fraction=args.test_fraction,
                    validation_fraction=args.validation_fraction,
                    min_train_rows=args.min_train_rows,
                    include_weekend_onehot=args.include_weekend_onehot,
                )
            except ValueError as exc:
                LOGGER.debug(
                    "Skipped cluster=%s far_edge=%s: %s",
                    cluster,
                    far_edge,
                    exc,
                )
                continue

            if reference_features is None:
                reference_features = node.feature_names
            elif node.feature_names != reference_features:
                raise RuntimeError(
                    "Feature columns differ across nodes. Check event-feature schema and caches."
                )
            nodes.append(node)
            accepted += 1
        LOGGER.info("Cluster %s: %d eligible nodes", cluster, accepted)

    if not nodes:
        raise RuntimeError("No eligible node was found")
    return nodes


def choose_tuning_nodes(nodes: Sequence[NodeData], per_cluster: int) -> list[NodeData]:
    if per_cluster <= 0:
        return list(nodes)
    by_cluster: dict[int, list[NodeData]] = {}
    for node in nodes:
        by_cluster.setdefault(node.cluster, []).append(node)
    selected: list[NodeData] = []
    for cluster in sorted(by_cluster):
        selected.extend(by_cluster[cluster][:per_cluster])
    return selected


def fit_scale(X: np.ndarray, y: np.ndarray) -> ScaleState:
    x_min = np.nanmin(X, axis=0).astype(np.float32)
    x_max = np.nanmax(X, axis=0).astype(np.float32)
    x_range = x_max - x_min
    x_range = np.where(np.abs(x_range) <= 1e-12, 1.0, x_range).astype(np.float32)

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = y_max - y_min
    if abs(y_range) <= 1e-12:
        y_range = 1.0
    return ScaleState(x_min=x_min, x_range=x_range, y_min=y_min, y_range=y_range)


def scale_X(X: np.ndarray, state: ScaleState) -> np.ndarray:
    return ((X - state.x_min) / state.x_range).astype(np.float32)


def scale_y(y: np.ndarray, state: ScaleState) -> np.ndarray:
    return ((y - state.y_min) / state.y_range).astype(np.float32)


def inverse_y(y: np.ndarray, state: ScaleState) -> np.ndarray:
    return np.asarray(y, dtype=np.float32) * state.y_range + state.y_min


def lstm_X(X: np.ndarray) -> np.ndarray:
    # Keep the benchmark layout: one timestep, all features.
    return X.reshape((X.shape[0], 1, X.shape[1])).astype(np.float32, copy=False)


def make_keras_loss(config: LossConfig) -> DirectionAwareProvisioningLoss:
    return DirectionAwareProvisioningLoss(
        alpha=config.alpha,
        epsilon=config.epsilon,
        cap=config.cap,
        delta=config.delta,
        eta=config.eta,
        tail_width=config.tail_width,
        tail_shape=config.tail_shape,
    )


def build_model(
    n_features: int,
    n_units: int,
    learning_rate: float,
    loss_config: LossConfig,
    output_activation: str,
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1, n_features), name="features"),
            tf.keras.layers.LSTM(int(n_units), activation="tanh", name="lstm"),
            tf.keras.layers.Dense(
                1,
                activation=output_activation,
                name="provisioned_capacity",
            ),
        ],
        name="LSTM_opt_cst_direction_aware",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=make_keras_loss(loss_config),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def tparams(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_units": trial.suggest_int("n_units", 50, 500),
        "n_epochs": trial.suggest_int("n_epochs", 50, 500),
        "batch_size": trial.suggest_int("batch_size", 2, 128),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-1,
            log=True,
        ),
    }


def selection_monitor_name(args: argparse.Namespace) -> str:
    """Keras validation quantity used for early stopping and best-epoch selection."""
    return "val_rmse" if args.optuna_objective == "rmse" else "val_loss"


def fit_callbacks(
    patience: int,
    monitor: str,
) -> list[tf.keras.callbacks.Callback]:
    """Callbacks aligned with the quantity used to select the Optuna model.

    The model is always trained with the direction-aware custom loss.  Only the
    stopping/weight-selection criterion changes: val_rmse when Optuna minimizes
    RMSE, val_loss when Optuna minimizes the custom loss.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=int(patience),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def validation_score(
    node: NodeData,
    state: ScaleState,
    prediction_scaled: np.ndarray,
    loss_config: LossConfig,
    objective_name: str,
) -> float:
    if objective_name == "custom_loss":
        y_val_scaled = scale_y(node.y_val, state)
        values = direction_aware_loss_numpy(
            y_true=y_val_scaled,
            y_pred=prediction_scaled,
            config=loss_config,
        )
        return float(np.mean(values))

    prediction = inverse_y(prediction_scaled, state)
    return rmse(node.y_val, prediction)


def objective_factory(
    tuning_nodes: Sequence[NodeData],
    args: argparse.Namespace,
    fixed_loss_config: LossConfig,
):
    def objective(trial: optuna.Trial) -> float:
        params = tparams(trial)
        current_loss_config = trial_loss_config(trial, args, fixed_loss_config)
        monitor = selection_monitor_name(args)
        trial.set_user_attr("effective_loss_config", current_loss_config.to_dict())
        trial.set_user_attr("selection_monitor", monitor)
        scores: list[float] = []

        for node_index, node in enumerate(tuning_nodes):
            model: tf.keras.Model | None = None
            try:
                clear_tf()
                tf.keras.utils.set_random_seed(
                    args.seed + trial.number * 10000 + node_index
                )
                state = fit_scale(node.X_train, node.y_train)
                X_train = lstm_X(scale_X(node.X_train, state))
                y_train = scale_y(node.y_train, state)
                X_val = lstm_X(scale_X(node.X_val, state))
                y_val = scale_y(node.y_val, state)

                model = build_model(
                    n_features=X_train.shape[2],
                    n_units=params["n_units"],
                    learning_rate=params["learning_rate"],
                    loss_config=current_loss_config,
                    output_activation=args.output_activation,
                )
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=int(params["n_epochs"]),
                    batch_size=int(params["batch_size"]),
                    shuffle=False,
                    verbose=args.verbose_fit,
                    callbacks=fit_callbacks(
                        patience=args.patience,
                        monitor=monitor,
                    ),
                )
                prediction_scaled = model.predict(X_val, verbose=0).reshape(-1)
                score = validation_score(
                    node=node,
                    state=state,
                    prediction_scaled=prediction_scaled,
                    loss_config=current_loss_config,
                    objective_name=args.optuna_objective,
                )
                if not np.isfinite(score):
                    raise optuna.TrialPruned("Non-finite validation score")
                scores.append(score)

                trial.set_user_attr(
                    f"node_{node_index}_c{node.cluster}_fe{node.far_edge}_score",
                    float(score),
                )
                trial.set_user_attr(
                    f"node_{node_index}_epochs_run",
                    int(len(history.history.get("loss", []))),
                )
                partial_mean = float(np.mean(scores))
                trial.report(partial_mean, step=node_index)
                if args.enable_pruning and trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned at node {node_index}; mean score={partial_mean:.8f}"
                    )
            finally:
                if model is not None:
                    del model
                clear_tf()

        return float(np.mean(scores))

    return objective


def safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return text or "node"


def provisioning_diagnostics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(y_true, dtype=float)
    under = np.maximum(0.0, -error)
    over = np.maximum(0.0, error)
    return {
        "Underprediction_rate_pct": float(np.mean(error < 0.0) * 100.0),
        "Overprediction_rate_pct": float(np.mean(error > 0.0) * 100.0),
        "Mean_underprediction": float(np.mean(under)),
        "Mean_overprediction": float(np.mean(over)),
        "P95_underprediction": float(np.percentile(under, 95)),
        "P95_overprediction": float(np.percentile(over, 95)),
    }


def train_final_node(
    node: NodeData,
    best_params: dict[str, Any],
    args: argparse.Namespace,
    loss_config: LossConfig,
    model_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Phase 1: determine the best epoch from train/validation only.
    clear_tf()
    tf.keras.utils.set_random_seed(args.seed)
    select_state = fit_scale(node.X_train, node.y_train)
    X_train = lstm_X(scale_X(node.X_train, select_state))
    y_train = scale_y(node.y_train, select_state)
    X_val = lstm_X(scale_X(node.X_val, select_state))
    y_val = scale_y(node.y_val, select_state)

    selector = build_model(
        n_features=X_train.shape[2],
        n_units=best_params["n_units"],
        learning_rate=best_params["learning_rate"],
        loss_config=loss_config,
        output_activation=args.output_activation,
    )
    monitor = selection_monitor_name(args)
    history = selector.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(best_params["n_epochs"]),
        batch_size=int(best_params["batch_size"]),
        shuffle=False,
        verbose=args.verbose_fit,
        callbacks=fit_callbacks(
            patience=args.patience,
            monitor=monitor,
        ),
    )

    selection_values = np.asarray(history.history.get(monitor, []), dtype=float)
    validation_losses = np.asarray(history.history.get("val_loss", []), dtype=float)
    validation_rmses_scaled = np.asarray(
        history.history.get("val_rmse", []), dtype=float
    )

    if len(selection_values) and np.isfinite(selection_values).any():
        best_index = int(np.nanargmin(selection_values))
        best_epoch = best_index + 1
        best_validation_selection_value = float(selection_values[best_index])
    else:
        best_index = max(0, len(history.history.get("loss", [])) - 1)
        best_epoch = max(1, best_index + 1)
        best_validation_selection_value = float("nan")

    best_validation_custom_loss = (
        float(validation_losses[best_index])
        if best_index < len(validation_losses) and np.isfinite(validation_losses[best_index])
        else float("nan")
    )
    best_validation_rmse_scaled = (
        float(validation_rmses_scaled[best_index])
        if best_index < len(validation_rmses_scaled)
        and np.isfinite(validation_rmses_scaled[best_index])
        else float("nan")
    )
    best_validation_rmse_original = (
        best_validation_rmse_scaled * select_state.y_range
        if np.isfinite(best_validation_rmse_scaled)
        else float("nan")
    )
    min_validation_custom_loss = (
        float(np.nanmin(validation_losses))
        if len(validation_losses) and np.isfinite(validation_losses).any()
        else float("nan")
    )
    del selector
    clear_tf()

    # Phase 2: refit from scratch on train+validation for the chosen epoch count.
    X_fit_raw = np.concatenate([node.X_train, node.X_val], axis=0)
    y_fit_raw = np.concatenate([node.y_train, node.y_val], axis=0)
    final_state = fit_scale(X_fit_raw, y_fit_raw)
    X_fit = lstm_X(scale_X(X_fit_raw, final_state))
    y_fit = scale_y(y_fit_raw, final_state)
    X_test = lstm_X(scale_X(node.X_test, final_state))

    tf.keras.utils.set_random_seed(args.seed + 1)
    model = build_model(
        n_features=X_fit.shape[2],
        n_units=best_params["n_units"],
        learning_rate=best_params["learning_rate"],
        loss_config=loss_config,
        output_activation=args.output_activation,
    )
    model.fit(
        X_fit,
        y_fit,
        epochs=best_epoch,
        batch_size=int(best_params["batch_size"]),
        shuffle=False,
        verbose=args.verbose_fit,
        callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    )

    prediction_scaled = model.predict(X_test, verbose=0).reshape(-1)
    prediction = inverse_y(prediction_scaled, final_state).astype(float)
    if args.clip_negative_predictions:
        prediction = np.maximum(0.0, prediction)
        prediction_scaled = scale_y(prediction, final_state)

    y_true = node.y_test.astype(float)
    naive = node.lag1_test.astype(float)
    y_true_scaled = scale_y(y_true, final_state)
    naive_scaled = scale_y(naive, final_state)

    custom_loss_values = direction_aware_loss_numpy(
        y_true_scaled,
        prediction_scaled,
        loss_config,
    )
    naive_custom_loss_values = direction_aware_loss_numpy(
        y_true_scaled,
        naive_scaled,
        loss_config,
    )

    row: dict[str, Any] = {
        "cluster": node.cluster,
        "far_edge": node.far_edge,
        "model": "S3_LSTM_Optuna_DirectionAware",
        "event_aware": bool(args.event_features_file is not None),
        "train_rows": len(node.y_train),
        "validation_rows": len(node.y_val),
        "test_rows": len(node.y_test),
        "best_epoch": best_epoch,
        "selection_monitor": monitor,
        "best_validation_selection_value": best_validation_selection_value,
        "validation_custom_loss_normalized": best_validation_custom_loss,
        "min_validation_custom_loss_normalized": min_validation_custom_loss,
        "validation_rmse_scaled": best_validation_rmse_scaled,
        "validation_rmse_original": best_validation_rmse_original,
        "CustomLoss_normalized": float(np.mean(custom_loss_values)),
        "Naive_CustomLoss_normalized": float(np.mean(naive_custom_loss_values)),
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
    row["CustomLoss_improvement_vs_naive_pct"] = (
        100.0
        * (row["Naive_CustomLoss_normalized"] - row["CustomLoss_normalized"])
        / row["Naive_CustomLoss_normalized"]
        if row["Naive_CustomLoss_normalized"] > 1e-12
        else float("nan")
    )
    row.update(provisioning_diagnostics(y_true, prediction))

    for cap_level in CAP_LEVELS:
        suffix = int(cap_level * 100)
        row[f"Dpp@{suffix}"] = traffic_reduction(y_true, prediction, cap_level)
        row[f"Naive_Dpp@{suffix}"] = traffic_reduction(y_true, naive, cap_level)

    predictions = node.test_keys.copy()
    predictions.insert(0, "cluster", node.cluster)
    predictions["y_true"] = y_true
    predictions["yhat_custom_loss"] = prediction
    predictions["yhat_naive"] = naive
    predictions["signed_error_custom"] = prediction - y_true
    predictions["custom_loss_normalized"] = custom_loss_values

    if args.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
        stem = f"cluster_{node.cluster}_far_edge_{safe_component(node.far_edge)}"
        model_path = model_dir / f"{stem}.keras"
        scaler_path = model_dir / f"{stem}_scaler.npz"
        metadata_path = model_dir / f"{stem}_metadata.json"
        model.save(model_path)
        np.savez_compressed(
            scaler_path,
            x_min=final_state.x_min,
            x_range=final_state.x_range,
            y_min=np.asarray([final_state.y_min], dtype=np.float64),
            y_range=np.asarray([final_state.y_range], dtype=np.float64),
            feature_names=np.asarray(node.feature_names, dtype=str),
        )
        write_json(
            metadata_path,
            {
                "cluster": node.cluster,
                "far_edge": node.far_edge,
                "model_path": model_path.name,
                "scaler_path": scaler_path.name,
                "best_epoch": best_epoch,
                "selection_monitor": monitor,
                "best_validation_selection_value": best_validation_selection_value,
                "best_params": best_params,
                "loss_config": loss_config.to_dict(),
                "loss_tuning_enabled": bool(args.tune_loss_params),
                "output_activation": args.output_activation,
                "feature_names": list(node.feature_names),
                "metrics": row,
            },
        )

    del model
    clear_tf()
    return row, predictions


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def experiment_signature(
    args: argparse.Namespace,
    loss_config: LossConfig,
    tuning_nodes: Sequence[NodeData],
    events: EventFeatureTable | None,
) -> tuple[str, dict[str, Any]]:
    event_metadata: dict[str, Any] | None = None
    if events is not None:
        stat = events.source_path.stat()
        event_metadata = {
            "path": str(events.source_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "join_keys": list(events.join_keys),
            "features": list(events.source_feature_names),
        }

    payload = {
        "clusters": list(args.clusters or []),
        "n_lags": args.n_lags,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "min_train_rows": args.min_train_rows,
        "include_weekend_onehot": args.include_weekend_onehot,
        "optuna_objective": args.optuna_objective,
        "selection_monitor": selection_monitor_name(args),
        "selection_protocol": "objective_aligned_early_stopping_v3",
        "output_activation": args.output_activation,
        "loss_config": loss_config.to_dict(),
        "loss_tuning": loss_search_space(args),
        "event_features": event_metadata,
        "tuning_nodes": [
            {
                "cluster": node.cluster,
                "far_edge": node.far_edge,
                "train_rows": len(node.y_train),
                "validation_rows": len(node.y_val),
            }
            for node in tuning_nodes
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return signature, payload


def create_study(
    args: argparse.Namespace,
    signature: str,
    signature_payload: dict[str, Any],
) -> optuna.Study:
    storage_path = (args.output_dir / "optuna_study.sqlite3").resolve()
    storage = f"sqlite:///{storage_path.as_posix()}"

    if args.reset_study:
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage)
            LOGGER.info("Deleted previous study: %s", args.study_name)
        except KeyError:
            pass

    pruner: optuna.pruners.BasePruner
    if args.enable_pruning:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=min(10, max(5, args.n_trials // 10)),
            n_warmup_steps=1,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    previous = study.user_attrs.get("experiment_signature")
    if previous and previous != signature and study.trials:
        raise RuntimeError(
            "The existing Optuna study uses a different loss/data configuration. "
            "Use --reset-study or a different --study-name."
        )
    study.set_user_attr("experiment_signature", signature)
    study.set_user_attr("experiment_signature_payload", signature_payload)
    return study


def export_study(
    study: optuna.Study,
    output_dir: Path,
    args: argparse.Namespace,
    fixed_loss_config: LossConfig,
    events: EventFeatureTable | None,
) -> tuple[dict[str, Any], LossConfig]:
    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_dir / "study_trials.csv", index=False)

    best_all = dict(study.best_params)
    best_network = {
        key: best_all[key]
        for key in ("n_units", "n_epochs", "batch_size", "learning_rate")
    }
    best_loss_config = best_loss_config_from_study(
        study=study,
        args=args,
        fixed_config=fixed_loss_config,
    )
    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    payload = {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_validation_value": study.best_value,
        "optuna_objective": args.optuna_objective,
        "selection_monitor": selection_monitor_name(args),
        "selection_protocol": "objective_aligned_early_stopping_v3",
        "best_network_params": best_network,
        "best_trial_all_optuna_params": best_all,
        "network_search_space": {
            "n_units": [50, 500],
            "n_epochs": [50, 500],
            "batch_size": [2, 128],
            "learning_rate": [1e-5, 1e-1, "log"],
            "optimizer": "Adam",
            "new_trials_requested_this_run": int(args.n_trials),
            "completed_trials_in_study": int(completed_trials),
        },
        "training_loss": "DirectionAwareProvisioningLoss",
        "loss_tuning": loss_search_space(args),
        "fixed_loss_config_at_start": fixed_loss_config.to_dict(),
        "best_loss_config": best_loss_config.to_dict(),
        "event_aware": events is not None,
        "event_feature_source": str(events.source_path) if events is not None else None,
        "event_feature_columns": (
            list(events.source_feature_names) if events is not None else []
        ),
    }
    write_json(output_dir / "best_params.json", payload)
    return best_network, best_loss_config


def final_evaluation(
    nodes: Sequence[NodeData],
    best_params: dict[str, Any],
    args: argparse.Namespace,
    loss_config: LossConfig,
) -> None:
    metric_rows: list[dict[str, Any]] = []
    model_dir = args.output_dir / "models"
    predictions_path = args.output_dir / "final_predictions.csv"
    first_prediction_chunk = True

    if args.save_predictions and predictions_path.exists():
        predictions_path.unlink()

    for index, node in enumerate(nodes, start=1):
        LOGGER.info(
            "Final %d/%d: cluster=%s far_edge=%s",
            index,
            len(nodes),
            node.cluster,
            node.far_edge,
        )
        row, predictions = train_final_node(
            node=node,
            best_params=best_params,
            args=args,
            loss_config=loss_config,
            model_dir=model_dir,
        )
        metric_rows.append(row)

        if args.save_predictions:
            predictions.to_csv(
                predictions_path,
                mode="w" if first_prediction_chunk else "a",
                header=first_prediction_chunk,
                index=False,
            )
            first_prediction_chunk = False

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "final_metrics_by_node.csv", index=False)

    numeric_columns = [
        column
        for column in metrics.columns
        if column not in {"cluster", "far_edge", "model", "event_aware", "selection_monitor"}
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]

    overall: dict[str, Any] = {
        "model": "S3_LSTM_Optuna_DirectionAware",
        "nodes": int(len(metrics)),
        "event_aware": bool(args.event_features_file is not None),
    }
    for column in numeric_columns:
        values = pd.to_numeric(metrics[column], errors="coerce")
        overall[f"{column}_mean"] = float(values.mean())
        overall[f"{column}_std"] = float(values.std(ddof=1))
    pd.DataFrame([overall]).to_csv(args.output_dir / "final_summary.csv", index=False)

    cluster_summary = metrics.groupby("cluster", as_index=False)[numeric_columns].agg(
        ["mean", "std"]
    )
    cluster_summary.columns = [
        "cluster"
        if first == "cluster"
        else f"{first}_{second}"
        for first, second in cluster_summary.columns.to_flat_index()
    ]
    cluster_summary.to_csv(args.output_dir / "final_summary_by_cluster.csv", index=False)

    write_json(
        args.output_dir / "final_summary.json",
        {
            "summary": overall,
            "loss_config": loss_config.to_dict(),
            "best_params": best_params,
            "selection_monitor": selection_monitor_name(args),
            "selection_protocol": "objective_aligned_early_stopping_v3",
            "event_features_file": args.event_features_file,
        },
    )

    LOGGER.info("Final evaluation completed on %d nodes", len(metrics))
    LOGGER.info("Mean RMSE custom-loss LSTM: %.6f", overall["RMSE_mean"])
    LOGGER.info("Mean RMSE Naive: %.6f", overall["Naive_RMSE_mean"])
    LOGGER.info(
        "Mean normalized custom loss: %.6f",
        overall["CustomLoss_normalized_mean"],
    )
    for cap_level in CAP_LEVELS:
        suffix = int(cap_level * 100)
        LOGGER.info(
            "Mean Dpp@%d: %.4f",
            suffix,
            overall[f"Dpp@{suffix}_mean"],
        )


def run(args: argparse.Namespace) -> int:
    setup(args.seed)
    fixed_loss_config = make_loss_config(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_test = self_test_loss(fixed_loss_config)
    self_test.to_csv(args.output_dir / "loss_self_test.csv", index=False)
    write_loss_curve(args.output_dir, fixed_loss_config)
    write_json(args.output_dir / "loss_config.json", fixed_loss_config.to_dict())

    if args.n_trials == 1 and not args.self_test_only:
        LOGGER.warning(
            "n_trials=1: Optuna is not meaningfully optimizing; this run evaluates "
            "only one sampled network/loss configuration."
        )
    elif args.tune_loss_params and args.n_trials < 20 and not args.self_test_only:
        LOGGER.warning(
            "Loss-parameter tuning is enabled with only %d trials; treat this as a "
            "functional/quick run, not as a stable optimum.",
            args.n_trials,
        )

    if args.self_test_only:
        LOGGER.info("Self-test-only run completed")
        return 0

    events = load_event_features(args.event_features_file)
    if args.require_event_features and events is None:
        raise RuntimeError("Event-aware run requested but no event feature file was loaded")
    if events is None:
        LOGGER.warning(
            "No event feature file supplied: this run is traffic-only/no-event."
        )

    clusters = discover_clusters(args.data_dir, args.clusters)
    if not clusters:
        raise RuntimeError("No hourly_cluster_*.csv cache was found")
    args.clusters = clusters
    LOGGER.info("Selected clusters: %s", clusters)

    panels = load_panels(args.data_dir, clusters, events)
    nodes = collect_nodes(panels, args)
    tuning_nodes = choose_tuning_nodes(nodes, args.tuning_nodes_per_cluster)
    LOGGER.info(
        "Total nodes=%d | Optuna nodes=%d | new trials=%d | objective=%s | monitor=%s | tune_loss=%s",
        len(nodes),
        len(tuning_nodes),
        args.n_trials,
        args.optuna_objective,
        selection_monitor_name(args),
        args.tune_loss_params,
    )

    pd.DataFrame(
        [
            {
                "cluster": node.cluster,
                "far_edge": node.far_edge,
                "train_rows": len(node.y_train),
                "validation_rows": len(node.y_val),
                "test_rows": len(node.y_test),
                "n_features": len(node.feature_names),
            }
            for node in tuning_nodes
        ]
    ).to_csv(args.output_dir / "tuning_nodes.csv", index=False)

    signature, signature_payload = experiment_signature(
        args=args,
        loss_config=fixed_loss_config,
        tuning_nodes=tuning_nodes,
        events=events,
    )
    study = create_study(
        args=args,
        signature=signature,
        signature_payload=signature_payload,
    )
    study.optimize(
        objective_factory(
            tuning_nodes=tuning_nodes,
            args=args,
            fixed_loss_config=fixed_loss_config,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best_params, best_loss_config = export_study(
        study=study,
        output_dir=args.output_dir,
        args=args,
        fixed_loss_config=fixed_loss_config,
        events=events,
    )
    best_loss_self_test = self_test_loss(best_loss_config)
    best_loss_self_test.to_csv(
        args.output_dir / "best_loss_self_test.csv", index=False
    )
    write_loss_curve(
        args.output_dir,
        best_loss_config,
        filename="best_loss_curve.csv",
    )
    write_json(
        args.output_dir / "best_loss_config.json",
        best_loss_config.to_dict(),
    )
    LOGGER.info("Best validation value: %.8f", study.best_value)
    LOGGER.info("Best network params: %s", best_params)
    LOGGER.info("Best loss config: %s", best_loss_config.to_dict())

    write_json(
        args.output_dir / "run_config.json",
        {
            **vars(args),
            "fixed_loss_config": fixed_loss_config.to_dict(),
            "best_loss_config": best_loss_config.to_dict(),
            "loss_tuning": loss_search_space(args),
            "event_feature_join_keys": list(events.join_keys) if events else [],
            "event_feature_columns": list(events.source_feature_names) if events else [],
            "tensorflow_version": tf.__version__,
            "optuna_version": optuna.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "experiment_signature": signature,
        },
    )

    if not args.tune_only:
        final_evaluation(
            nodes=nodes,
            best_params=best_params,
            args=args,
            loss_config=best_loss_config,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.exception("Run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
