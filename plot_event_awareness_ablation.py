#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect heterogeneous experiment summaries and compare event-aware and no-event runs.

The collector accepts multi-row benchmark summaries as well as explicitly
labeled single-model summaries. ``AUTO`` preserves model names found in the
input CSV. The output includes separate comparison plots and a delta table for
models available in both modes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = [
    "LSTM opt_cst",
    "LSTM opt_std",
    "LSTM",
    "RandomForest",
    "Ridge",
    "Naive",
    "PatchTST-style",
]

ALIASES = {
    "s0_naive": "Naive",
    "naive": "Naive",
    "s1_ridge": "Ridge",
    "ridge": "Ridge",
    "s2_randomforest": "RandomForest",
    "randomforest": "RandomForest",
    "random_forest": "RandomForest",
    "s3_lstm": "LSTM",
    "lstm": "LSTM",
    "s3_lstm_optuna": "LSTM opt_std",
    "lstm_opt_std": "LSTM opt_std",
    "s3_lstm_optuna_directionaware": "LSTM opt_cst",
    "lstm_opt_cst": "LSTM opt_cst",
    "patchtst": "PatchTST-style",
    "patchtst_style": "PatchTST-style",
}


def canonical_model(name: str) -> str:
    text = str(name).strip()
    key = text.lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, text)


def parse_spec(text: str) -> tuple[str, Path]:
    if "::" in text:
        label, path = text.split("::", 1)
        return label.strip(), Path(path)
    return "AUTO", Path(text)


def _value(row: pd.Series, metric: str, stat: str) -> float:
    candidates = []
    if stat == "mean":
        candidates = [f"{metric}_mean", metric, f"mean_{metric}"]
    else:
        candidates = [f"{metric}_std", f"std_{metric}"]
    for c in candidates:
        if c in row.index:
            value = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return float("nan")


def load_summary(spec: str, mode: str) -> list[dict]:
    label, path = parse_spec(spec)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty summary: {path}")
    rows: list[dict] = []
    for _, row in df.iterrows():
        if label != "AUTO":
            model = label
        elif "model" in df.columns:
            model = canonical_model(row["model"])
        else:
            model = canonical_model(path.parent.name)
        item = {"mode": mode, "Model": canonical_model(model), "source": str(path)}
        for metric in ("MAE", "RMSE", "MAPE", "R2", "Dpp@70", "Dpp@80", "Dpp@90"):
            item[f"{metric}_mean"] = _value(row, metric, "mean")
            item[f"{metric}_std"] = _value(row, metric, "std")
        rows.append(item)
    return rows


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    rank = {name: i for i, name in enumerate(ORDER)}
    out = df.copy()
    out["__rank"] = out["Model"].map(rank).fillna(len(rank))
    return out.sort_values(["__rank", "Model"]).drop(columns="__rank").reset_index(drop=True)


def plot_summary(df: pd.DataFrame, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    df = ordered(df)
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    specs = [
        ("R2", "Mean R²", 1.0),
        ("MAPE", "Mean MAPE [%]", 1.0),
        ("RMSE", "Mean RMSE (×10⁶)", 1e6),
    ]
    for ax, (metric, subtitle, scale) in zip(axes, specs):
        means = df[f"{metric}_mean"].to_numpy(float) / scale
        stds = df[f"{metric}_std"].fillna(0.0).to_numpy(float) / scale
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Model"], rotation=25, ha="right")
        ax.set_title(subtitle)
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--no-event", action="append", default=[], metavar="LABEL::CSV")
    p.add_argument("--event-aware", action="append", default=[], metavar="LABEL::CSV")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/event_ablation"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_event and not args.event_aware:
        raise SystemExit("Provide at least one --no-event or --event-aware summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.no_event:
        rows.extend(load_summary(spec, "no_event"))
    for spec in args.event_aware:
        rows.extend(load_summary(spec, "event_aware"))
    combined = pd.DataFrame(rows)
    combined.to_csv(args.output_dir / "model_summary_all.csv", index=False)

    if args.no_event:
        no = ordered(combined.loc[combined["mode"] == "no_event"].copy())
        no.to_csv(args.output_dir / "model_summary_no_event.csv", index=False)
        plot_summary(no, args.output_dir / "comparison_no_event.pdf", "Model comparison (no-event features)")
    if args.event_aware:
        ev = ordered(combined.loc[combined["mode"] == "event_aware"].copy())
        ev.to_csv(args.output_dir / "model_summary_event_aware.csv", index=False)
        plot_summary(ev, args.output_dir / "comparison_event_aware.pdf", "Model comparison (event-aware features)")
    if args.no_event and args.event_aware:
        a = combined.loc[combined["mode"] == "no_event"].set_index("Model")
        b = combined.loc[combined["mode"] == "event_aware"].set_index("Model")
        common = sorted(set(a.index).intersection(b.index))
        delta_rows = []
        for model in common:
            row = {"Model": model}
            for metric in ("MAE", "RMSE", "MAPE", "R2", "Dpp@70", "Dpp@80", "Dpp@90"):
                nv = float(a.loc[model, f"{metric}_mean"])
                ev = float(b.loc[model, f"{metric}_mean"])
                row[f"{metric}_no_event"] = nv
                row[f"{metric}_event_aware"] = ev
                row[f"{metric}_delta_event_minus_noevent"] = ev - nv
                if metric in {"MAE", "RMSE", "MAPE"} and np.isfinite(nv) and abs(nv) > 1e-12:
                    row[f"{metric}_improvement_pct"] = 100.0 * (nv - ev) / nv
            delta_rows.append(row)
        pd.DataFrame(delta_rows).to_csv(args.output_dir / "event_awareness_delta.csv", index=False)
    print(f"Saved outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
