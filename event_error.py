#!/usr/bin/env python
# coding: utf-8

# In[5]:


########################à solo cluster 3
if False:
    import numpy as np
    import pandas as pd

    CSV_PATH = "prediction_clusto_3.csv"

    # --- load & sanitize ---
    df = pd.read_csv(CSV_PATH)

    # build timestamp from date+hour
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if pd.api.types.is_numeric_dtype(df["hour"]):
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    else:
        hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour
        df["timestamp"] = df["date"] + pd.to_timedelta(hh.fillna(0).astype(int), unit="h")

    # rename and coerce to numeric
    df = df.rename(columns={"far_edge":"node", "value":"y_true", "predicted":"y_pred"})
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    # drop rows with missing essentials
    before = len(df)
    df = df.dropna(subset=["timestamp","node","y_true","y_pred"]).copy()
    after = len(df)
    print(f"Dropped {before-after} rows with NaNs in timestamp/node/y_true/y_pred")

    # optional: remove negatives if not meaningful
    # df = df[(df["y_true"]>=0) & (df["y_pred"]>=0)].copy()

    df = df.sort_values(["node","timestamp"]).reset_index(drop=True)

    # --- helpers ---
    def mae(y, yhat): return float(np.mean(np.abs(y-yhat)))
    def mape(y, yhat):
        denom = np.maximum(1e-9, np.abs(y))
        return float(100*np.mean(np.abs((y-yhat)/denom)))

    def traffic_reduction(y, yhat, cap_percentile):
        y = np.asarray(y, float)
        yhat = np.asarray(yhat, float)
        # guard: if all-NaN or empty
        if y.size == 0 or np.all(~np.isfinite(y)): return np.nan
        C = np.nanpercentile(y, cap_percentile*100)
        if not np.isfinite(C): return np.nan
        overflow_fixed = np.nansum(np.maximum(0.0, y - C))
        C_dyn = np.maximum(C, yhat)  # simple proxy
        overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
        if overflow_fixed <= 1e-9:  # no overflow → no reduction definable
            return np.nan
        return float(100*(overflow_fixed - overflow_dyn)/overflow_fixed)

    # --- simulation params (as before) ---
    CANCEL_RATE = 0.30; FP_RATE = 0.10; SHIFT_PROB = 0.20; SHIFT_STEPS = 1
    SCALE_CANCEL = 0.6; SCALE_FP = 1.25
    CAP_LEVELS = [0.70,0.75,0.80,0.85,0.90,0.95]
    rng = np.random.default_rng(42)

    rows_detail, rows_summary = [], []
    bad_nodes = []

    for node, g in df.groupby("node", sort=False):
        g = g.copy()
        if len(g) < 24:  # too few points → skip
            bad_nodes.append((node, "too_few_points"))
            continue

        y_true = g["y_true"].to_numpy(float)
        y_pred_base = g["y_pred"].to_numpy(float)

        # if still any NaN after drop → skip node
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred_base).all():
            bad_nodes.append((node, "non_finite_values"))
            continue

        # --- build event_flag from top 5% peaks (fallback) ---
        k = max(1, int(0.05*len(g)))
        if k >= len(g):
            k = max(1, len(g)//20)  # ~5% safeguard
        thr = np.partition(y_true, -k)[-k]
        event_flag = (y_true >= thr)

        # contig runs
        grp_id = (pd.Series(event_flag).diff().fillna(0) != 0).cumsum().to_numpy()
        run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()==1]

        y_pred_corr = y_pred_base.copy()

        # 1) cancellations
        if run_ids:
            num_cancel = max(1, int(CANCEL_RATE*len(run_ids)))
            cancel_runs = set(rng.choice(run_ids, size=min(num_cancel,len(run_ids)), replace=False))
            mask_cancel = np.isin(grp_id, list(cancel_runs)) & event_flag
            y_pred_corr[mask_cancel] = np.maximum(0.0, y_pred_corr[mask_cancel]*SCALE_CANCEL)

        # 2) false positives
        non_event_idx = np.where(~event_flag)[0]
        num_fp = int(FP_RATE * event_flag.sum())
        if num_fp>0 and len(non_event_idx)>0:
            fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
            y_pred_corr[fp_idx] = y_pred_corr[fp_idx]*SCALE_FP

        # 3) shifts (only if enough points)
        if SHIFT_PROB>0 and SHIFT_STEPS>0 and len(run_ids)>0 and len(g)>SHIFT_STEPS:
            shift_runs = [rid for rid in run_ids if rng.random()<SHIFT_PROB]
            for rid in shift_runs:
                m = (grp_id==rid) & event_flag
                if not m.any(): continue
                direction = SHIFT_STEPS if rng.random()<0.5 else -SHIFT_STEPS
                m_shift = np.roll(m, direction)
                y_pred_corr[m_shift] = y_pred_corr[m_shift]*SCALE_FP
                y_pred_corr[m & ~m_shift] = y_pred_corr[m & ~m_shift]*SCALE_CANCEL

        # metrics X
        mae_base, mae_corr = mae(y_true, y_pred_base), mae(y_true, y_pred_corr)
        mape_base, mape_corr = mape(y_true, y_pred_base), mape(y_true, y_pred_corr)
        X_mae = 100*(mae_corr - mae_base)/max(mae_base,1e-9)
        X_mape = (mape_corr - mape_base)

        # metrics Y
        for lvl in CAP_LEVELS:
            r_base = traffic_reduction(y_true, y_pred_base, lvl)
            r_corr = traffic_reduction(y_true, y_pred_corr, lvl)
            rows_detail.append({"node":node,"C_level":lvl,
                                "reduction_base_%":r_base,
                                "reduction_corrupted_%":r_corr,
                                "delta_pp": None if (pd.isna(r_base) or pd.isna(r_corr)) else r_corr - r_base})

        rows_summary.append({"node":node,
                             "MAE_base":mae_base,"MAE_corr":mae_corr,"X_MAE_delta_%":X_mae,
                             "MAPE_base_%":mape_base,"MAPE_corr_%":mape_corr,"X_MAPE_delta_pp":X_mape})

    detail = pd.DataFrame(rows_detail)
    summary = pd.DataFrame(rows_summary)

    print("Skipped nodes (reason):", bad_nodes)

    # aggregate excluding NaNs
    X_mae_avg  = summary["X_MAE_delta_%"].replace([np.inf,-np.inf], np.nan).dropna().mean()
    X_mape_avg = summary["X_MAPE_delta_pp"].replace([np.inf,-np.inf], np.nan).dropna().mean()
    by_C = detail.groupby("C_level")["delta_pp"].mean(numeric_only=True)

    print("\n=== Forecasting impact (X) ===")
    print(f"ΔMAE avg: {X_mae_avg:.1f}%   ΔMAPE avg: {X_mape_avg:.1f} pp")
    print("\n=== Operational impact (Y) ===")
    print("Δ traffic reduction (pp) per C level (averaged over nodes):")
    print(by_C.round(2))
    print(f"\nOverall average Y across C levels and nodes: {detail['delta_pp'].dropna().mean():.1f} pp")


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob, os, re

# =================== CONFIG ===================
# Se usi 5 file separati tipo prediction_clusto_1.csv ... _5.csv lascia SINGLE_FILE_PATH=None
SINGLE_FILE_PATH = None  # e.g. "all_clusters.csv" se hai un file unico con colonna 'cluster'

# Parametri “errore eventi”
CANCEL_RATE = 0.30
FP_RATE     = 0.10
SHIFT_PROB  = 0.20
SHIFT_STEPS = 1
SCALE_CANCEL = 0.6
SCALE_FP     = 1.25

# Livelli di capacità C (percentili)
CAP_LEVELS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Stile
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 12
# ==============================================

rng = np.random.default_rng(42)

def load_cluster_csvs(single_file=None):
    """
    Ritorna dict: {cluster_id: DataFrame}
    Supporta:
      - 5 file: prediction_clusto_1.csv ... prediction_clusto_5.csv
      - 1 file con colonna 'cluster' (o 'clusto')
    """
    out = {}
    if single_file is None:
        paths = sorted(glob.glob("prediction_clusto_*.csv"))
        # Filtra solo quelli 1..5
        for p in paths:
            m = re.search(r'clusto[_-]?(\d+)\.csv$', p)
            if not m: 
                continue
            cid = int(m.group(1))
            if cid not in range(1,6):
                continue
            df = pd.read_csv(p)
            out[cid] = df
    else:
        df = pd.read_csv(single_file)
        # Prova a trovare colonna cluster
        ccol = None
        for c in df.columns:
            if re.search(r'^(cluster|clusto)$', c, re.I):
                ccol = c; break
        if ccol is None:
            raise ValueError("Single-file mode: expected a 'cluster' (or 'clusto') column.")
        for cid, g in df.groupby(ccol):
            try:
                cid_int = int(cid)
            except:
                continue
            if cid_int in range(1,6):
                out[cid_int] = g.copy()
    if not out:
        raise FileNotFoundError("No cluster data found. Provide prediction_clusto_1..5.csv or a single CSV with a 'cluster' column.")
    return out

def build_timestamp(df):
    # Richieste: date,hour,far_edge,value,predicted
    if "date" not in df.columns or "hour" not in df.columns:
        raise ValueError("CSV must contain 'date' and 'hour' columns.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if pd.api.types.is_numeric_dtype(df["hour"]):
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    else:
        hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
        df["timestamp"] = df["date"] + pd.to_timedelta(hh, unit="h")
    return df

def sanitize(df):
    # Rinominare/convertire
    if "far_edge" not in df.columns or "value" not in df.columns or "predicted" not in df.columns:
        raise ValueError("CSV must contain 'far_edge', 'value', 'predicted' columns.")
    df = df.rename(columns={"far_edge":"node", "value":"y_true", "predicted":"y_pred"})
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp","node","y_true","y_pred"]).copy()
    df = df.sort_values(["node","timestamp"]).reset_index(drop=True)
    return df

def mae(y, yhat): return float(np.mean(np.abs(y-yhat)))
def mape(y, yhat):
    denom = np.maximum(1e-9, np.abs(y))
    return float(100*np.mean(np.abs((y-yhat)/denom)))

def traffic_reduction(y, yhat, cap_percentile):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0 or np.all(~np.isfinite(y)): return np.nan
    C = np.nanpercentile(y, cap_percentile*100)
    if not np.isfinite(C): return np.nan
    overflow_fixed = np.nansum(np.maximum(0.0, y - C))
    C_dyn = np.maximum(C, yhat)  # proxy semplice
    overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
    if overflow_fixed <= 1e-9:
        return np.nan
    return float(100*(overflow_fixed - overflow_dyn)/overflow_fixed)

def simulate_event_noise_for_node(g, cancel_rate, fp_rate, shift_prob, shift_steps, scale_cancel, scale_fp):
    """Applica cancellazioni/FP/shift su previsioni; ritorna y_true, y_pred_base, y_pred_corr"""
    y_true = g["y_true"].to_numpy(float)
    y_pred_base = g["y_pred"].to_numpy(float)
    y_pred_corr = y_pred_base.copy()
    if len(g) < 24:
        return y_true, y_pred_base, y_pred_corr  # troppo corto: non alteriamo

    # Eventi = top 5% picchi per nodo
    k = max(1, int(0.05*len(g)))
    if k >= len(g): k = max(1, len(g)//20)
    thr = np.partition(y_true, -k)[-k]
    event_flag = (y_true >= thr)

    # run contigui
    grp_id = (pd.Series(event_flag).diff().fillna(0) != 0).cumsum().to_numpy()
    run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()==1]

    # 1) Cancellazioni
    if run_ids:
        num_cancel = max(1, int(cancel_rate*len(run_ids)))
        cancel_runs = set(rng.choice(run_ids, size=min(num_cancel, len(run_ids)), replace=False))
        mask_cancel = np.isin(grp_id, list(cancel_runs)) & event_flag
        y_pred_corr[mask_cancel] = np.maximum(0.0, y_pred_corr[mask_cancel]*scale_cancel)

    # 2) Falsi positivi
    non_event_idx = np.where(~event_flag)[0]
    num_fp = int(fp_rate * event_flag.sum())
    if num_fp>0 and len(non_event_idx)>0:
        fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
        y_pred_corr[fp_idx] = y_pred_corr[fp_idx]*scale_fp

    # 3) Shift
    if shift_prob>0 and shift_steps>0 and len(run_ids)>0 and len(g)>shift_steps:
        shift_runs = [rid for rid in run_ids if rng.random()<shift_prob]
        for rid in shift_runs:
            m = (grp_id==rid) & event_flag
            if not m.any(): continue
            direction = shift_steps if rng.random()<0.5 else -shift_steps
            m_shift = np.roll(m, direction)
            y_pred_corr[m_shift] = y_pred_corr[m_shift]*scale_fp
            y_pred_corr[m & ~m_shift] = y_pred_corr[m & ~m_shift]*scale_cancel

    return y_true, y_pred_base, y_pred_corr

def evaluate_cluster(df_cluster, cluster_id):
    """Esegue simulazione per tutti i nodi del cluster"""
    summary_rows = []
    detail_rows = []
    for node, g in df_cluster.groupby("node", sort=False):
        y_true, y_pred_base, y_pred_corr = simulate_event_noise_for_node(
            g, CANCEL_RATE, FP_RATE, SHIFT_PROB, SHIFT_STEPS, SCALE_CANCEL, SCALE_FP
        )
        mae_base, mae_corr = mae(y_true, y_pred_base), mae(y_true, y_pred_corr)
        mape_base, mape_corr = mape(y_true, y_pred_base), mape(y_true, y_pred_corr)
        X_mae = 100*(mae_corr - mae_base)/max(mae_base,1e-9)
        X_mape = (mape_corr - mape_base)

        summary_rows.append({
            "cluster": cluster_id, "node": node,
            "MAE_base": mae_base, "MAE_corr": mae_corr, "X_MAE_delta_%": X_mae,
            "MAPE_base_%": mape_base, "MAPE_corr_%": mape_corr, "X_MAPE_delta_pp": X_mape
        })

        for lvl in CAP_LEVELS:
            r_base = traffic_reduction(y_true, y_pred_base, lvl)
            r_corr = traffic_reduction(y_true, y_pred_corr, lvl)
            delta_pp = (np.nan if (pd.isna(r_base) or pd.isna(r_corr)) else r_corr - r_base)
            detail_rows.append({
                "cluster": cluster_id, "node": node, "C_level": lvl,
                "reduction_base_%": r_base, "reduction_corrupted_%": r_corr, "delta_pp": delta_pp
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)

# =================== MAIN ===================
clusters = load_cluster_csvs(SINGLE_FILE_PATH)

all_summary = []
all_detail  = []

for cid, raw in clusters.items():
    raw = build_timestamp(raw)
    raw = sanitize(raw)
    s, d = evaluate_cluster(raw, cid)
    all_summary.append(s); all_detail.append(d)

summary = pd.concat(all_summary, ignore_index=True)
detail  = pd.concat(all_detail,  ignore_index=True)

# Salva risultati
summary.to_csv("robustness_forecasting_by_cluster.csv", index=False)
detail.to_csv("robustness_traffic_reduction_by_cluster_and_C.csv", index=False)

# =================== PLOT DI CONFRONTO ===================
# 1) delta MAE% medio per cluster (X)
X_by_cluster = summary.groupby("cluster")["X_MAE_delta_%"].mean()

# 2) Δ traffic reduction (pp) medio per cluster (Y), per C e aggregato
Y_by_cluster_and_C = detail.groupby(["cluster","C_level"])["delta_pp"].mean()
Y_overall_by_cluster = detail.groupby("cluster")["delta_pp"].mean()

# --- figura 1: barre delta MAE% e delta traffic reduction complessivo ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# delta MAE%
ax[0].bar(X_by_cluster.index.astype(int), X_by_cluster.values)
ax[0].set_title(r"Forecasting impact $X$ (ΔMAE%)")
ax[0].set_xlabel("Cluster"); ax[0].set_ylabel("ΔMAE [%]")
ax[0].set_xticks(range(1,6))

# delta traffic reduction complessivo
ax[1].bar(Y_overall_by_cluster.index.astype(int), Y_overall_by_cluster.values)
ax[1].set_title(r"Operational impact $Y$ (Δ traffic reduction, pp)")
ax[1].set_xlabel("Cluster"); ax[1].set_ylabel("Δ reduction [pp]")
ax[1].set_xticks(range(1,6))

plt.tight_layout()
plt.savefig("robustness_overview_by_cluster.png", dpi=300, bbox_inches='tight')
plt.savefig("robustness_overview_by_cluster.svg", bbox_inches='tight')
plt.show()

# --- figura 2: linee per C_level, uno per cluster ---
import matplotlib.pyplot as plt
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
for cid in sorted(clusters.keys()):
    ys = [Y_by_cluster_and_C.loc[(cid, c)] if (cid, c) in Y_by_cluster_and_C.index else np.nan for c in CAP_LEVELS]
    ax2.plot(CAP_LEVELS, ys, marker='o', label=f"Cluster {cid}")
ax2.set_title(r"Operational impact $Y$ vs. $C$ level")
ax2.set_xlabel("Capacity level C (percentile)"); ax2.set_ylabel("Δ reduction [pp]")
ax2.set_xticks(CAP_LEVELS, [f"{int(c*100)}%" for c in CAP_LEVELS])
ax2.legend(ncol=2)
plt.tight_layout()
plt.savefig("robustness_Y_vs_C_by_cluster.png", dpi=300, bbox_inches='tight')
plt.savefig("robustness_Y_vs_C_by_cluster.svg", bbox_inches='tight')
plt.show()


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob, os, re

# =================== CONFIG ===================
# Se usi 5 file separati tipo prediction_clusto_1.csv ... _5.csv lascia SINGLE_FILE_PATH=None
SINGLE_FILE_PATH = None  # e.g. "all_clusters.csv" se hai un file unico con colonna 'cluster'

# Parametri “errore eventi”
CANCEL_RATE = 0.0
FP_RATE     = 0.0
SHIFT_PROB  = 0.10
SHIFT_STEPS = 3
SCALE_CANCEL = 0.6
SCALE_FP     = 1.1

# Livelli di capacità C (percentili)
CAP_LEVELS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Stile
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 12
# ==============================================

rng = np.random.default_rng(42)

def load_cluster_csvs(single_file=None):
    """
    Ritorna dict: {cluster_id: DataFrame}
    Supporta:
      - 5 file: prediction_clusto_1.csv ... prediction_clusto_5.csv
      - 1 file con colonna 'cluster' (o 'clusto')
    """
    out = {}
    if single_file is None:
        paths = sorted(glob.glob("prediction_clusto_*.csv"))
        # Filtra solo quelli 1..5
        for p in paths:
            m = re.search(r'clusto[_-]?(\d+)\.csv$', p)
            if not m: 
                continue
            cid = int(m.group(1))
            if cid not in range(1,6):
                continue
            df = pd.read_csv(p)
            out[cid] = df
    else:
        df = pd.read_csv(single_file)
        # Prova a trovare colonna cluster
        ccol = None
        for c in df.columns:
            if re.search(r'^(cluster|clusto)$', c, re.I):
                ccol = c; break
        if ccol is None:
            raise ValueError("Single-file mode: expected a 'cluster' (or 'clusto') column.")
        for cid, g in df.groupby(ccol):
            try:
                cid_int = int(cid)
            except:
                continue
            if cid_int in range(1,6):
                out[cid_int] = g.copy()
    if not out:
        raise FileNotFoundError("No cluster data found. Provide prediction_clusto_1..5.csv or a single CSV with a 'cluster' column.")
    return out

def build_timestamp(df):
    # Richieste: date,hour,far_edge,value,predicted
    if "date" not in df.columns or "hour" not in df.columns:
        raise ValueError("CSV must contain 'date' and 'hour' columns.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if pd.api.types.is_numeric_dtype(df["hour"]):
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    else:
        hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
        df["timestamp"] = df["date"] + pd.to_timedelta(hh, unit="h")
    return df

def sanitize(df):
    # Rinominare/convertire
    if "far_edge" not in df.columns or "value" not in df.columns or "predicted" not in df.columns:
        raise ValueError("CSV must contain 'far_edge', 'value', 'predicted' columns.")
    df = df.rename(columns={"far_edge":"node", "value":"y_true", "predicted":"y_pred"})
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp","node","y_true","y_pred"]).copy()
    df = df.sort_values(["node","timestamp"]).reset_index(drop=True)
    return df

def mae(y, yhat): return float(np.mean(np.abs(y-yhat)))
def mape(y, yhat):
    denom = np.maximum(1e-9, np.abs(y))
    return float(100*np.mean(np.abs((y-yhat)/denom)))

def traffic_reduction(y, yhat, cap_percentile):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0 or np.all(~np.isfinite(y)): return np.nan
    C = np.nanpercentile(y, cap_percentile*100)
    if not np.isfinite(C): return np.nan
    overflow_fixed = np.nansum(np.maximum(0.0, y - C))
    C_dyn = np.maximum(C, yhat)  # proxy semplice
    overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
    if overflow_fixed <= 1e-9:
        return np.nan
    return float(100*(overflow_fixed - overflow_dyn)/overflow_fixed)

def simulate_event_noise_for_node(g, cancel_rate, fp_rate, shift_prob, shift_steps, scale_cancel, scale_fp):
    """Applica cancellazioni/FP/shift su previsioni; ritorna y_true, y_pred_base, y_pred_corr"""
    y_true = g["y_true"].to_numpy(float)
    y_pred_base = g["y_pred"].to_numpy(float)
    y_pred_corr = y_pred_base.copy()
    if len(g) < 24:
        return y_true, y_pred_base, y_pred_corr  # troppo corto: non alteriamo

    # Eventi = top 5% picchi per nodo
    k = max(1, int(0.05*len(g)))
    if k >= len(g): k = max(1, len(g)//20)
    thr = np.partition(y_true, -k)[-k]
    event_flag = (y_true >= thr)

    # run contigui
    grp_id = (pd.Series(event_flag).diff().fillna(0) != 0).cumsum().to_numpy()
    run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()==1]

    # 1) Cancellazioni
    if run_ids:
        num_cancel = max(1, int(cancel_rate*len(run_ids)))
        cancel_runs = set(rng.choice(run_ids, size=min(num_cancel, len(run_ids)), replace=False))
        mask_cancel = np.isin(grp_id, list(cancel_runs)) & event_flag
        y_pred_corr[mask_cancel] = np.maximum(0.0, y_pred_corr[mask_cancel]*scale_cancel)

    # 2) Falsi positivi
    non_event_idx = np.where(~event_flag)[0]
    num_fp = int(fp_rate * event_flag.sum())
    if num_fp>0 and len(non_event_idx)>0:
        fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
        y_pred_corr[fp_idx] = y_pred_corr[fp_idx]*scale_fp

    # 3) Shift
    if shift_prob>0 and shift_steps>0 and len(run_ids)>0 and len(g)>shift_steps:
        shift_runs = [rid for rid in run_ids if rng.random()<shift_prob]
        for rid in shift_runs:
            m = (grp_id==rid) & event_flag
            if not m.any(): continue
            direction = shift_steps if rng.random()<0.5 else -shift_steps
            m_shift = np.roll(m, direction)
            y_pred_corr[m_shift] = y_pred_corr[m_shift]*scale_fp
            y_pred_corr[m & ~m_shift] = y_pred_corr[m & ~m_shift]*scale_cancel

    return y_true, y_pred_base, y_pred_corr

def evaluate_cluster(df_cluster, cluster_id):
    """Esegue simulazione per tutti i nodi del cluster"""
    summary_rows = []
    detail_rows = []
    for node, g in df_cluster.groupby("node", sort=False):
        y_true, y_pred_base, y_pred_corr = simulate_event_noise_for_node(
            g, CANCEL_RATE, FP_RATE, SHIFT_PROB, SHIFT_STEPS, SCALE_CANCEL, SCALE_FP
        )
        mae_base, mae_corr = mae(y_true, y_pred_base), mae(y_true, y_pred_corr)
        mape_base, mape_corr = mape(y_true, y_pred_base), mape(y_true, y_pred_corr)
        X_mae = 100*(mae_corr - mae_base)/max(mae_base,1e-9)
        X_mape = (mape_corr - mape_base)

        summary_rows.append({
            "cluster": cluster_id, "node": node,
            "MAE_base": mae_base, "MAE_corr": mae_corr, "X_MAE_delta_%": X_mae,
            "MAPE_base_%": mape_base, "MAPE_corr_%": mape_corr, "X_MAPE_delta_pp": X_mape
        })

        for lvl in CAP_LEVELS:
            r_base = traffic_reduction(y_true, y_pred_base, lvl)
            r_corr = traffic_reduction(y_true, y_pred_corr, lvl)
            delta_pp = (np.nan if (pd.isna(r_base) or pd.isna(r_corr)) else r_corr - r_base)
            detail_rows.append({
                "cluster": cluster_id, "node": node, "C_level": lvl,
                "reduction_base_%": r_base, "reduction_corrupted_%": r_corr, "delta_pp": delta_pp
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)

# =================== MAIN ===================
clusters = load_cluster_csvs(SINGLE_FILE_PATH)

all_summary = []
all_detail  = []

for cid, raw in clusters.items():
    raw = build_timestamp(raw)
    raw = sanitize(raw)
    s, d = evaluate_cluster(raw, cid)
    all_summary.append(s); all_detail.append(d)

summary = pd.concat(all_summary, ignore_index=True)
detail  = pd.concat(all_detail,  ignore_index=True)

# Salva risultati
summary.to_csv("robustness_forecasting_by_cluster.csv", index=False)
detail.to_csv("robustness_traffic_reduction_by_cluster_and_C.csv", index=False)

# =================== PLOT DI CONFRONTO ===================
# 1) delta MAE% medio per cluster (X)
X_by_cluster = summary.groupby("cluster")["X_MAE_delta_%"].mean()

# 2) Δ traffic reduction (pp) medio per cluster (Y), per C e aggregato
Y_by_cluster_and_C = detail.groupby(["cluster","C_level"])["delta_pp"].mean()
Y_overall_by_cluster = detail.groupby("cluster")["delta_pp"].mean()

# --- figura 1: barre delta MAE% e delta traffic reduction complessivo ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# delta MAE%
ax[0].bar(X_by_cluster.index.astype(int), X_by_cluster.values)
ax[0].set_title(r"Forecasting impact $X$ (ΔMAE%)")
ax[0].set_xlabel("Cluster"); ax[0].set_ylabel("ΔMAE [%]")
ax[0].set_xticks(range(1,6))

# delta traffic reduction complessivo
ax[1].bar(Y_overall_by_cluster.index.astype(int), Y_overall_by_cluster.values)
ax[1].set_title(r"Operational impact $Y$ (Δ traffic reduction, pp)")
ax[1].set_xlabel("Cluster"); ax[1].set_ylabel("Δ reduction [pp]")
ax[1].set_xticks(range(1,6))

plt.tight_layout()
plt.savefig("robustness_overview_by_cluster2.png", dpi=300, bbox_inches='tight')
plt.savefig("robustness_overview_by_cluster2.svg", bbox_inches='tight')
plt.show()

# --- figura 2: linee per C_level, uno per cluster ---
import matplotlib.pyplot as plt
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
for cid in sorted(clusters.keys()):
    ys = [Y_by_cluster_and_C.loc[(cid, c)] if (cid, c) in Y_by_cluster_and_C.index else np.nan for c in CAP_LEVELS]
    ax2.plot(CAP_LEVELS, ys, marker='o', label=f"Cluster {cid}")
ax2.set_title(r"Operational impact $Y$ vs. $C$ level")
ax2.set_xlabel("Capacity level C (percentile)"); ax2.set_ylabel("Δ reduction [pp]")
ax2.set_xticks(CAP_LEVELS, [f"{int(c*100)}%" for c in CAP_LEVELS])
ax2.legend(ncol=2)
plt.tight_layout()
plt.savefig("robustness_Y_vs_C_by_cluster2.png", dpi=300, bbox_inches='tight')
plt.savefig("robustness_Y_vs_C_by_cluster2.svg", bbox_inches='tight')
plt.show()


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import glob, os, re

# =================== I/O CONFIG ===================
SINGLE_FILE_PATH = None  # e.g., "all_clusters.csv" if you have a single file with 'cluster'
FILE_PATTERN = "prediction_clusto_*.csv"  # expects ..._1.csv ... _5.csv
# Columns expected in each CSV: date, hour, far_edge, value, predicted
# ==================================================

# Fixed perturbation parameters (kept constant across scenarios)
SHIFT_PROB  = 0.20
SHIFT_STEPS = 1
SCALE_CANCEL = 0.6
SCALE_FP     = 1.25

# 9 scenarios = 3x3 grid: CANCEL_RATE × FP_RATE (SHIFT_PROB fixed)
CANCEL_GRID = [0.10, 0.30, 0.50]
FP_GRID     = [0.05, 0.10, 0.20]
CAP_LEVELS  = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 12

rng = np.random.default_rng(42)

# ------------------ Data loading helpers ------------------
def load_cluster_csvs(single_file=None):
    out = {}
    if single_file is None:
        paths = sorted(glob.glob(FILE_PATTERN))
        for p in paths:
            m = re.search(r'clusto[_-]?(\d+)\.csv$', p)
            if not m: 
                continue
            cid = int(m.group(1))
            if cid not in range(1,6):
                continue
            out[cid] = pd.read_csv(p)
    else:
        df = pd.read_csv(single_file)
        ccol = None
        for c in df.columns:
            if re.search(r'^(cluster|clusto)$', c, re.I):
                ccol = c; break
        if ccol is None:
            raise ValueError("Single-file mode requires a 'cluster' (or 'clusto') column.")
        for cid, g in df.groupby(ccol):
            try:
                cid_int = int(cid)
            except:
                continue
            if cid_int in range(1,6):
                out[cid_int] = g.copy()
    if not out:
        raise FileNotFoundError("No cluster data found. Provide prediction_clusto_1..5.csv or a single CSV with 'cluster'.")
    return out

def build_timestamp(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if pd.api.types.is_numeric_dtype(df["hour"]):
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    else:
        hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
        df["timestamp"] = df["date"] + pd.to_timedelta(hh, unit="h")
    return df

def sanitize(df):
    need = {"far_edge","value","predicted","timestamp"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.rename(columns={"far_edge":"node", "value":"y_true", "predicted":"y_pred"})
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp","node","y_true","y_pred"]).copy()
    df = df.sort_values(["node","timestamp"]).reset_index(drop=True)
    return df

# ------------------ Metrics ------------------
def mae(y, yhat): return float(np.mean(np.abs(y-yhat)))
def mape(y, yhat):
    denom = np.maximum(1e-9, np.abs(y))
    return float(100*np.mean(np.abs((y-yhat)/denom)))

def traffic_reduction(y, yhat, cap_percentile):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0 or np.all(~np.isfinite(y)): return np.nan
    C = np.nanpercentile(y, cap_percentile*100)
    if not np.isfinite(C): return np.nan
    overflow_fixed = np.nansum(np.maximum(0.0, y - C))
    C_dyn = np.maximum(C, yhat)  # simple proxy
    overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
    if overflow_fixed <= 1e-9:
        return np.nan
    return float(100*(overflow_fixed - overflow_dyn)/overflow_fixed)

# ------------------ Simulation ------------------
def simulate_event_noise_for_node(g, cancel_rate, fp_rate):
    y_true = g["y_true"].to_numpy(float)
    y_pred_base = g["y_pred"].to_numpy(float)
    y_pred_corr = y_pred_base.copy()
    if len(g) < 24:
        return y_true, y_pred_base, y_pred_corr  # too short; leave unchanged

    # Event windows = top 5% demand peaks (per node)
    k = max(1, int(0.05*len(g)))
    if k >= len(g): k = max(1, len(g)//20)
    thr = np.partition(y_true, -k)[-k]
    event_flag = (y_true >= thr)

    grp_id = (pd.Series(event_flag).diff().fillna(0) != 0).cumsum().to_numpy()
    run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()==1]

    # 1) cancellations (FN)
    if run_ids:
        num_cancel = max(1, int(cancel_rate*len(run_ids)))
        cancel_runs = set(rng.choice(run_ids, size=min(num_cancel, len(run_ids)), replace=False))
        mask_cancel = np.isin(grp_id, list(cancel_runs)) & event_flag
        y_pred_corr[mask_cancel] = np.maximum(0.0, y_pred_corr[mask_cancel]*SCALE_CANCEL)

    # 2) false positives (FP)
    non_event_idx = np.where(~event_flag)[0]
    num_fp = int(FP_GRID[1] * event_flag.sum())  # default; overridden per scenario in driver if needed
    # (We'll pass fp_rate explicitly from the driver instead)
    num_fp = int(fp_rate * event_flag.sum())
    if num_fp>0 and len(non_event_idx)>0:
        fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
        y_pred_corr[fp_idx] = y_pred_corr[fp_idx]*SCALE_FP

    # 3) temporal shifts (kept fixed globally)
    if SHIFT_PROB>0 and SHIFT_STEPS>0 and len(run_ids)>0 and len(g)>SHIFT_STEPS:
        shift_runs = [rid for rid in run_ids if rng.random()<SHIFT_PROB]
        for rid in shift_runs:
            m = (grp_id==rid) & event_flag
            if not m.any(): continue
            direction = SHIFT_STEPS if rng.random()<0.5 else -SHIFT_STEPS
            m_shift = np.roll(m, direction)
            y_pred_corr[m_shift] = y_pred_corr[m_shift]*SCALE_FP
            y_pred_corr[m & ~m_shift] = y_pred_corr[m & ~m_shift]*SCALE_CANCEL

    return y_true, y_pred_base, y_pred_corr

def evaluate_cluster(df_cluster, cancel_rate, fp_rate):
    """Return average ΔMAE% and Δ traffic-reduction (pp) over nodes in this cluster."""
    X_mae_list, Y_list = [], []
    for node, g in df_cluster.groupby("node", sort=False):
        y_true, y_pred_base, y_pred_corr = simulate_event_noise_for_node(g, cancel_rate, fp_rate)
        mae_base, mae_corr = mae(y_true, y_pred_base), mae(y_true, y_pred_corr)
        if mae_base <= 1e-12:  # guard
            continue
        X_mae = 100*(mae_corr - mae_base)/mae_base
        # Y: average over capacity levels
        deltas = []
        for lvl in CAP_LEVELS:
            r_base = traffic_reduction(y_true, y_pred_base, lvl)
            r_corr = traffic_reduction(y_true, y_pred_corr, lvl)
            if pd.isna(r_base) or pd.isna(r_corr): 
                continue
            deltas.append(r_corr - r_base)
        if len(deltas)==0: 
            continue
        X_mae_list.append(X_mae)
        Y_list.append(np.mean(deltas))
    return (np.nan if not X_mae_list else float(np.mean(X_mae_list)),
            np.nan if not Y_list else float(np.mean(Y_list)))

# =================== MAIN ===================
clusters = load_cluster_csvs(SINGLE_FILE_PATH)
# preprocess
for cid in clusters.keys():
    clusters[cid] = sanitize(build_timestamp(clusters[cid]))

# Evaluate 9 scenarios on all clusters
results = []
for i_c, cr in enumerate(CANCEL_GRID):
    for i_f, fr in enumerate(FP_GRID):
        X_vals, Y_vals = [], []
        for cid, dfc in clusters.items():
            X_avg, Y_avg = evaluate_cluster(dfc, cancel_rate=cr, fp_rate=fr)
            X_vals.append(X_avg); Y_vals.append(Y_avg)
        results.append({
            "i_c": i_c, "i_f": i_f,
            "CANCEL_RATE": cr, "FP_RATE": fr,
            "X_MAE_delta_avg_pct": np.nanmean(X_vals),
            "Y_reduction_delta_avg_pp": np.nanmean(Y_vals)
        })

res = pd.DataFrame(results)
res.to_csv("robustness_grid_results.csv", index=False)
print(res)

# =================== 3D PLOTS ===================
# Prepare grids for plotting
Xc = np.array([r["i_c"] for _, r in res.iterrows()])
Yf = np.array([r["i_f"] for _, r in res.iterrows()])
Zm = np.array([r["X_MAE_delta_avg_pct"] for _, r in res.iterrows()])
Zy = np.array([r["Y_reduction_delta_avg_pp"] for _, r in res.iterrows()])

# Map indices to tick labels
cancel_labels = [f"{c:.2f}" for c in CANCEL_GRID]
fp_labels     = [f"{f:.2f}" for f in FP_GRID]

# --- 3D bar: ΔMAE% ---
fig = plt.figure(figsize=(7.5, 5.5))
ax = fig.add_subplot(111, projection='3d')
dx = dy = 0.4
ax.bar3d(Xc, Yf, np.zeros_like(Zm), dx, dy, Zm, shade=True)
ax.set_title(r"Forecasting impact $X$ (ΔMAE%) vs CANCEL and FP rates")
ax.set_xlabel("CANCEL_RATE"); ax.set_ylabel("FP_RATE"); ax.set_zlabel("ΔMAE [%]")
ax.set_xticks(range(len(CANCEL_GRID))); ax.set_xticklabels(cancel_labels)
ax.set_yticks(range(len(FP_GRID)));     ax.set_yticklabels(fp_labels)
plt.tight_layout(); plt.savefig("robustness_3D_MAE.pdf", dpi=300, bbox_inches='tight'); plt.show()

# --- 3D bar: Δ reduction (pp) ---
fig = plt.figure(figsize=(7.5, 5.5))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(Xc, Yf, np.zeros_like(Zy), dx, dy, Zy, shade=True)
ax.set_title(r"Operational impact $Y$ (Δ traffic reduction, pp) vs CANCEL and FP rates")
ax.set_xlabel("CANCEL_RATE"); ax.set_ylabel("FP_RATE"); ax.set_zlabel("Δ reduction [pp]")
ax.set_xticks(range(len(CANCEL_GRID))); ax.set_xticklabels(cancel_labels)
ax.set_yticks(range(len(FP_GRID)));     ax.set_yticklabels(fp_labels)
plt.tight_layout(); plt.savefig("robustness_3D_reduction.pdf", dpi=300, bbox_inches='tight'); plt.show()

print("Saved:")
print(" - robustness_grid_results.csv")
print(" - robustness_3D_MAE.pdf")
print(" - robustness_3D_reduction.pdf")


# In[2]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, glob
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ========= I/O =========
SINGLE_FILE_PATH = None          # es. "all_clusters.csv" se hai un unico file con colonna 'cluster'
FILE_PATTERN     = "prediction_clusto_*.csv"  # cerca ..._1.csv ... _5.csv
# Colonne attese: date, hour, far_edge, value, predicted

# ========= Parametri fissi =========
FP_RATE      = 0.10
SHIFT_PROB   = 0.20
SCALE_CANCEL = 0.6
SCALE_FP     = 1.25
CAP_LEVELS   = [0.70,0.75,0.80,0.85,0.90,0.95]

# 9 scenari (asse Y): CANCEL_RATE x SHIFT_STEPS
CANCEL_GRID  = [0.10, 0.30, 0.50]
SHIFT_GRID   = [0, 1, 2]

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica','DejaVu Sans']
rcParams['font.size'] = 12
rng = np.random.default_rng(42)

# ========= Loader =========
def load_cluster_csvs(single_file=None):
    out = {}
    if single_file is None:
        for p in sorted(glob.glob(FILE_PATTERN)):
            m = re.search(r'clusto[_-]?(\d+)\.csv$', p)
            if not m: continue
            cid = int(m.group(1))
            if cid not in range(1,6): continue
            out[cid] = pd.read_csv(p)
    else:
        df = pd.read_csv(single_file)
        ccol = None
        for c in df.columns:
            if re.search(r'^(cluster|clusto)$', c, re.I):
                ccol = c; break
        if ccol is None:
            raise ValueError("Single-file mode requires a 'cluster' (or 'clusto') column.")
        for cid, g in df.groupby(ccol):
            try: cid = int(cid)
            except: continue
            if cid in range(1,6): out[cid] = g.copy()
    if not out:
        raise FileNotFoundError("No cluster data found.")
    return out

def build_timestamp(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if pd.api.types.is_numeric_dtype(df["hour"]):
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    else:
        hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
        df["timestamp"] = df["date"] + pd.to_timedelta(hh, unit="h")
    return df

def sanitize(df):
    need = {"far_edge","value","predicted","timestamp"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.rename(columns={"far_edge":"node","value":"y_true","predicted":"y_pred"})
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp","node","y_true","y_pred"])
    return df.sort_values(["node","timestamp"]).reset_index(drop=True)

# ========= Metriche =========
def traffic_reduction(y, yhat, cap_percentile):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0 or np.all(~np.isfinite(y)): return np.nan
    C = np.nanpercentile(y, cap_percentile*100)
    if not np.isfinite(C): return np.nan
    overflow_fixed = np.nansum(np.maximum(0.0, y - C))
    C_dyn = np.maximum(C, yhat)  # proxy semplice
    overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
    if overflow_fixed <= 1e-9: return np.nan
    return float(100*(overflow_fixed - overflow_dyn)/overflow_fixed)

# ========= Simulazione rumore eventi =========
def simulate_event_noise_for_node(g, cancel_rate, fp_rate, shift_steps):
    y_true = g["y_true"].to_numpy(float)
    y_pred = g["y_pred"].to_numpy(float)
    y_corr = y_pred.copy()
    n = len(g)
    if n < 24:  # troppo corto → lascia invariato
        return y_true, y_pred, y_corr

    # Eventi = top 5% picchi per nodo
    k = max(1, n//20)
    thr = np.partition(y_true, -k)[-k]
    event_flag = (y_true >= thr)

    grp_id = (pd.Series(event_flag).diff().fillna(0) != 0).cumsum().to_numpy()
    run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()==1]

    # 1) Cancellazioni (FN)
    if run_ids:
        num_cancel = max(1, int(cancel_rate*len(run_ids)))
        sel = rng.choice(run_ids, size=min(num_cancel, len(run_ids)), replace=False)
        mask_cancel = np.isin(grp_id, sel) & event_flag
        y_corr[mask_cancel] = np.maximum(0.0, y_corr[mask_cancel]*SCALE_CANCEL)

    # 2) Falsi positivi (FP)
    non_event_idx = np.where(~event_flag)[0]
    num_fp = int(fp_rate * event_flag.sum())
    if num_fp>0 and len(non_event_idx)>0:
        fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
        y_corr[fp_idx] = y_corr[fp_idx]*SCALE_FP

    # 3) Shift temporali (±shift_steps) con prob SHIFT_PROB
    if SHIFT_PROB>0 and shift_steps>0 and len(run_ids)>0 and n>shift_steps:
        shift_runs = [rid for rid in run_ids if rng.random()<SHIFT_PROB]
        for rid in shift_runs:
            m = (grp_id==rid) & event_flag
            if not m.any(): continue
            direction = shift_steps if rng.random()<0.5 else -shift_steps
            m_shift = np.roll(m, direction)
            y_corr[m_shift] = y_corr[m_shift]*SCALE_FP
            y_corr[m & ~m_shift] = y_corr[m & ~m_shift]*SCALE_CANCEL

    return y_true, y_pred, y_corr

# ========= MAIN =========
clusters = load_cluster_csvs(SINGLE_FILE_PATH)
for cid in clusters:
    clusters[cid] = sanitize(build_timestamp(clusters[cid]))

# Per ogni scenario, calcola Y(C) medio su cluster e nodi
records = []
for i_c, cr in enumerate(CANCEL_GRID):
    for i_s, ss in enumerate(SHIFT_GRID):
        scenario_id = i_c*len(SHIFT_GRID) + i_s  # 0..8
        # accumulo per ciascun C
        Y_by_C_accum = {C: [] for C in CAP_LEVELS}
        for cid, dfc in clusters.items():
            for node, g in dfc.groupby("node", sort=False):
                y, yhat_base, yhat_corr = simulate_event_noise_for_node(
                    g, cancel_rate=cr, fp_rate=FP_RATE, shift_steps=ss
                )
                for C in CAP_LEVELS:
                    r_base = traffic_reduction(y, yhat_base, C)
                    r_corr = traffic_reduction(y, yhat_corr, C)
                    if not (np.isnan(r_base) or np.isnan(r_corr)):
                        Y_by_C_accum[C].append(r_corr - r_base)
        # media su tutti i nodi/cluster
        for C in CAP_LEVELS:
            vals = np.array(Y_by_C_accum[C], dtype=float)
            z = np.nan if vals.size==0 else float(np.nanmean(vals))
            records.append({"scenario": scenario_id, "C": C, "delta_pp": z,
                            "CANCEL_RATE": cr, "SHIFT_STEPS": ss})

res = pd.DataFrame(records).dropna(subset=["delta_pp"])
res.to_csv("robustness_Y_vs_C_lines3D.csv", index=False)
print(res.head())

# ========= PLOT 3D: linee Y(C) per scenario =========
fig = plt.figure(figsize=(8.5, 6))
ax = fig.add_subplot(111, projection='3d')

# Per ogni scenario, traccia la linea 3D: X=C, Y=scenario (costante), Z=delta_pp(C)
for sid, g in res.groupby("scenario"):
    X = g["C"].values
    Y = np.full_like(X, fill_value=sid, dtype=float)
    Z = g["delta_pp"].values
    # ordina per C per avere linee monotone sull'asse X
    idx = np.argsort(X); X, Y, Z = X[idx], Y[idx], Z[idx]
    ax.plot(X, Y, Z, linewidth=2)

# Asse Y: etichette scenari leggibili
ax.set_xlabel("Capacity level C")
ax.set_ylabel("Scenario index")
ax.set_zlabel("Δ reduction [pp]")
ax.set_xticks(CAP_LEVELS); ax.set_xticklabels([f"{int(c*100)}%" for c in CAP_LEVELS])

# legenda sintetica con mapping scenario→(CANCEL, SHIFT)
legend_lines = []
for sid, g in res.groupby("scenario"):
    cr = g["CANCEL_RATE"].iloc[0]; ss = int(g["SHIFT_STEPS"].iloc[0])
    legend_lines.append(f"{sid}: cancel={cr:.2f}, shift={ss}")
ax.set_title("Operational impact $Y$ vs. $C$ (3D lines)\n" +
             "Scenarios: CANCEL_RATE x SHIFT_STEPS; FP_RATE=0.10, SHIFT_PROB=0.20")

plt.tight_layout()
plt.savefig("robustness_3D_Y_vs_C_lines.pdf", dpi=300, bbox_inches='tight')
plt.show()

print("Saved:")
print(" - robustness_Y_vs_C_lines3D.csv")
print(" - robustness_3D_Y_vs_C_lines.pdf")


# In[1]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, glob
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ========= I/O =========
SINGLE_FILE_PATH = None                      # e.g., "all_clusters.csv" if single file with 'cluster'
FILE_PATTERN     = "prediction_clusto_*.csv" # expects ..._1.csv ... _5.csv
# Required columns per file: date, hour, far_edge, value, predicted

# ========= Fixed parameters =========
SHIFT_PROB   = 0.20
SCALE_CANCEL = 0.6
SCALE_FP     = 1.25
CAP_LEVELS   = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# 27 scenarios: CANCEL × FP × SHIFT
CANCEL_GRID = [0.10, 0.20, 0.30]
FP_GRID     = [0.05, 0.10, 0.20]
SHIFT_GRID  = [0, 1, 2]

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica','DejaVu Sans']
rcParams['font.size'] = 11
rng = np.random.default_rng(42)

# ========= Loaders =========
def load_cluster_csvs(single_file=None):
    out = {}
    if single_file is None:
        for p in sorted(glob.glob(FILE_PATTERN)):
            m = re.search(r'clusto[_-]?(\d+)\.csv$', p)
            if not m: continue
            cid = int(m.group(1))
            if cid in range(1,6):
                out[cid] = pd.read_csv(p)
    else:
        df = pd.read_csv(single_file)
        ccol = None
        for c in df.columns:
            if re.search(r'^(cluster|clusto)$', c, re.I):
                ccol = c; break
        if ccol is None:
            raise ValueError("Single-file mode requires a 'cluster' (or 'clusto') column.")
        for cid, g in df.groupby(ccol):
            try: cid = int(cid)
            except: continue
            if cid in range(1,6):
                out[cid] = g.copy()
    if not out:
        raise FileNotFoundError("No cluster data found.")
    return out

def build_timestamp(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if pd.api.types.is_numeric_dtype(df["hour"]):
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    else:
        hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
        df["timestamp"] = df["date"] + pd.to_timedelta(hh, unit="h")
    return df

def sanitize(df):
    need = {"far_edge","value","predicted","timestamp"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.rename(columns={"far_edge":"node","value":"y_true","predicted":"y_pred"})
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp","node","y_true","y_pred"])
    return df.sort_values(["node","timestamp"]).reset_index(drop=True)

# ========= Metric =========
def traffic_reduction(y, yhat, cap_percentile):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0 or np.all(~np.isfinite(y)): return np.nan
    C = np.nanpercentile(y, cap_percentile*100)
    if not np.isfinite(C): return np.nan
    overflow_fixed = np.nansum(np.maximum(0.0, y - C))
    C_dyn = np.maximum(C, yhat)  # simple proxy
    overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
    if overflow_fixed <= 1e-9: return np.nan
    return float(100*(overflow_fixed - overflow_dyn)/overflow_fixed)

# ========= Event-noise simulator =========
def simulate_event_noise_for_node(g, cancel_rate, fp_rate, shift_steps):
    y_true = g["y_true"].to_numpy(float)
    y_pred = g["y_pred"].to_numpy(float)
    y_corr = y_pred.copy()
    n = len(g)
    if n < 24:
        return y_true, y_pred, y_corr

    # Events = top 5% true-demand peaks per node
    k = max(1, n//20)
    thr = np.partition(y_true, -k)[-k]
    event_flag = (y_true >= thr)

    grp_id = (pd.Series(event_flag).diff().fillna(0) != 0).cumsum().to_numpy()
    run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()==1]

    # 1) Cancellations (FN)
    if run_ids:
        num_cancel = max(1, int(cancel_rate*len(run_ids)))
        sel = rng.choice(run_ids, size=min(num_cancel, len(run_ids)), replace=False)
        mask_cancel = np.isin(grp_id, sel) & event_flag
        y_corr[mask_cancel] = np.maximum(0.0, y_corr[mask_cancel]*SCALE_CANCEL)

    # 2) False positives (FP)
    non_event_idx = np.where(~event_flag)[0]
    num_fp = int(fp_rate * event_flag.sum())
    if num_fp>0 and len(non_event_idx)>0:
        fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
        y_corr[fp_idx] = y_corr[fp_idx]*SCALE_FP

    # 3) Temporal shifts (±shift_steps) with prob SHIFT_PROB
    if SHIFT_PROB>0 and shift_steps>0 and len(run_ids)>0 and n>shift_steps:
        shift_runs = [rid for rid in run_ids if rng.random()<SHIFT_PROB]
        for rid in shift_runs:
            m = (grp_id==rid) & event_flag
            if not m.any(): continue
            direction = shift_steps if rng.random()<0.5 else -shift_steps
            m_shift = np.roll(m, direction)
            y_corr[m_shift] = y_corr[m_shift]*SCALE_FP
            y_corr[m & ~m_shift] = y_corr[m & ~m_shift]*SCALE_CANCEL

    return y_true, y_pred, y_corr

# ========= MAIN =========
clusters = load_cluster_csvs(SINGLE_FILE_PATH)
for cid in clusters:
    clusters[cid] = sanitize(build_timestamp(clusters[cid]))

# Compute Y(C) for all 27 scenarios, per cluster
records = []
scenario_map = []  # scenario → params
sid = 0
for cr in CANCEL_GRID:
    for fr in FP_GRID:
        for ss in SHIFT_GRID:
            scenario_map.append({"scenario": sid, "CANCEL_RATE": cr, "FP_RATE": fr, "SHIFT_STEPS": ss})
            for cid, dfc in clusters.items():
                # accumulate across nodes, then average
                accum = {C: [] for C in CAP_LEVELS}
                for node, g in dfc.groupby("node", sort=False):
                    y, yhat_base, yhat_corr = simulate_event_noise_for_node(g, cr, fr, ss)
                    for C in CAP_LEVELS:
                        r_base = traffic_reduction(y, yhat_base, C)
                        r_corr = traffic_reduction(y, yhat_corr, C)
                        if not (np.isnan(r_base) or np.isnan(r_corr)):
                            accum[C].append(r_corr - r_base)
                for C in CAP_LEVELS:
                    vals = np.array(accum[C], float)
                    if vals.size == 0: continue
                    records.append({"cluster": cid, "scenario": sid, "C": C, "delta_pp": float(np.nanmean(vals))})
            sid += 1

res = pd.DataFrame(records)
scenario_df = pd.DataFrame(scenario_map)
res.to_csv("robustness_Y_vs_C_lines3D_27scenarios.csv", index=False)
scenario_df.to_csv("robustness_scenarios_27_map.csv", index=False)

print("Saved CSVs:")
print(" - robustness_Y_vs_C_lines3D_27scenarios.csv")
print(" - robustness_scenarios_27_map.csv")



# ========= 3D line plots: one figure per cluster =========
rcParams['font.size'] = 10
for cid in sorted(res["cluster"].unique()):
    gC = res[res["cluster"] == cid]
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    for sid in sorted(gC["scenario"].unique()):
        gi = gC[gC["scenario"] == sid].sort_values("C")
        X = gi["C"].values
        Y = np.full_like(X, sid, dtype=float)  # scenario index on Y
        Z = gi["delta_pp"].values
        ax.plot(X, Y, Z, linewidth=1.7)

    ax.set_title(f"Operational impact $Y$ vs. $C$ (Cluster {cid})\n27 scenarios: CANCEL×FP×SHIFT")
    ax.set_xlabel("Capacity level C"); ax.set_ylabel("Scenario index"); ax.set_zlabel("Δ reduction [pp]")
    ax.set_xticks(CAP_LEVELS); ax.set_xticklabels([f"{int(c*100)}%" for c in CAP_LEVELS])
    # Y ticks can be crowded; show every ~3rd
    y_ticks = list(range(0, res["scenario"].max()+1, 3))
    ax.set_yticks(y_ticks); ax.set_yticklabels([str(t) for t in y_ticks])

    plt.tight_layout()
    out = f"robustness_3D_Y_vs_C_lines_cluster{cid}.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")

print("Scenario mapping (first rows):")
print(scenario_df.head())
print("Use 'robustness_scenarios_27_map.csv' to see scenario → (CANCEL_RATE, FP_RATE, SHIFT_STEPS).")


# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch

# --- STYLE ---
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica','DejaVu Sans']
rcParams['font.size'] = 11

# --- INPUT: `res` must contain columns: cluster, scenario, C, delta_pp ---
# Example structure:
# res.head()
#    cluster  scenario     C  delta_pp
# 0        1         0  0.70    -12.34
# ...

# Axes domains
clusters = sorted(res["cluster"].unique())
Cs       = np.sort(res["C"].unique())
Scens    = np.sort(res["scenario"].unique())

# Colors per cluster (distinct, semi-transparent)
cluster_colors = {
    1: (0.30, 0.55, 0.95, 0.60),  # blue-ish, alpha 0.6
    2: (0.95, 0.45, 0.30, 0.60),  # red-ish
    3: (0.30, 0.80, 0.45, 0.60),  # green-ish
    4: (0.70, 0.50, 0.95, 0.60),  # purple-ish
    5: (0.95, 0.75, 0.25, 0.60),  # orange-ish
}

# Build figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# For legend
legend_patches = []

# Plot one colored surface per cluster
for cid in clusters:
    g = res[res["cluster"] == cid].copy()
    # pivot to (C x scenario) grid
    M = g.pivot(index="C", columns="scenario", values="delta_pp").reindex(index=Cs, columns=Scens)
    Z = M.values  # shape (len(Cs), len(Scens))

    # Create meshgrid (X=C on x-axis, Y=scenario on y-axis)
    X, Y = np.meshgrid(Cs, Scens, indexing='ij')  # shapes match Z

    # Plot surface (constant color per cluster)
    col = cluster_colors.get(cid, (0.5, 0.5, 0.5, 0.6))
    surf = ax.plot_surface(X, Y, Z, color=col, linewidth=0, antialiased=True, shade=False)
    # Usa linee: per ogni scenario disegna la curva C→Δpp
    #for j, scen in enumerate(Scens):
    #    ax.plot(Cs, np.full_like(Cs, scen, dtype=float), Z[:, j],
    #            color=col, linewidth=1.8)
    legend_patches.append(Patch(color=col, label=f"Cluster {cid}"))
    
    #break

# Axes labels & ticks
ax.set_xlabel("Capacity level C")
ax.set_ylabel("Scenario index")
ax.set_zlabel("Δ reduction [pp]")

ax.set_xticks(Cs)
ax.set_xticklabels([f"{int(c*100)}%" for c in Cs])
ax.set_yticks(Scens[::3])  # show every 3rd scenario to declutter

# Optional: set a helpful Z range (uncomment/tune if needed)
# ax.set_zlim(np.nanmin(res['delta_pp']) * 1.1, np.nanmax(res['delta_pp']) * 1.1)

# Legend and title
ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax.set_title("Operational impact $Y$ vs. $C$ and scenario (colored planes by cluster)")
ax.view_init(elev=15, azim=-40)

plt.tight_layout()
plt.savefig("robustness_3D_Y_vs_C_planes_all_clusters.pdf", dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:





# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob, re

# ============================
# CONFIG
# ============================
FILE_PATTERN = "prediction_clusto_*.csv"  # expects ..._1.csv ... _5.csv
C_FIXED = 0.80                             # "C set to handle 80% of hourly traffic"
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica','DejaVu Sans']
rcParams['font.size'] = 12
FIGSIZE = (8.2, 4.8)

# ============================
# I/O HELPERS
# ============================
def load_clusters(pattern=FILE_PATTERN):
    clusters = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r'clusto[_-]?(\d+)\.csv$', p)
        if not m:
            continue
        cid = int(m.group(1))
        df = pd.read_csv(p)
        # build timestamp
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if pd.api.types.is_numeric_dtype(df["hour"]):
            df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
        else:
            hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
            df["timestamp"] = df["date"] + pd.to_timedelta(hh, unit="h")
        # rename
        df = df.rename(columns={"far_edge":"node", "value":"y_true", "predicted":"y_pred"})
        # sanitize
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        df = df.dropna(subset=["timestamp","node","y_true","y_pred"]).copy()
        clusters[cid] = df.sort_values(["timestamp","node"]).reset_index(drop=True)
    if not clusters:
        raise FileNotFoundError("No files matching pattern. Provide prediction_clusto_1..5.csv.")
    return clusters

# ============================
# CORE METRICS
# ============================
def overflow_fixed(d, C):
    """Overflow with fixed capacity C (per-entity scalar or array aligned to d)."""
    return np.maximum(0.0, d - C)

def overflow_dynamic_capacity(d, C, yhat):
    """
    Dynamic capacity guided by predictions: per slot, provision up to max(C, yhat).
    Overflow computed against that dynamic provision.
    """
    C_dyn = np.maximum(C, yhat)
    return np.maximum(0.0, d - C_dyn)

def overflow_dynamic_traffic(d_vec, C_vec):
    """
    Dynamic traffic (pooling): with multiple entities, pooling overflow = max(0, sum(d) - sum(C)).
    If single entity, this equals overflow_fixed.
    """
    return max(0.0, float(np.sum(d_vec) - np.sum(C_vec)))

def overflow_combined(d_vec, C_vec, yhat_vec):
    """
    Combined: dynamic capacity (guided by predictions) + pooling (sum across entities).
    """
    C_dyn_vec = np.maximum(C_vec, yhat_vec)
    return max(0.0, float(np.sum(d_vec) - np.sum(C_dyn_vec)))

def daily_reduction_percent(time_index, overflow_baseline, overflow_strategy):
    """
    % reduction in overflow per day: (OF_fixed - OF_strategy) / OF_fixed * 100
    Returns daily stats: mean, min, max (over days).
    """
    # Group by date
    days = pd.to_datetime(time_index).normalize()
    df = pd.DataFrame({
        "day": days,
        "of_fixed": overflow_baseline,
        "of_strat": overflow_strategy
    })
    daily = df.groupby("day").sum()
    # guard: if baseline == 0 → NaN (no overflow that day)
    valid = daily["of_fixed"] > 1e-9
    red = np.full(len(daily), np.nan, dtype=float)
    red[valid.values] = (daily.loc[valid, "of_fixed"] - daily.loc[valid, "of_strat"]) / daily.loc[valid, "of_fixed"] * 100.0
    return float(np.nanmean(red)), float(np.nanmin(red)), float(np.nanmax(red))

# ============================
# LAYER AGGREGATION
# ============================
def evaluate_layers(df_cluster, C_fixed=C_FIXED):
    """
    Compute Figure-11-like bars for one cluster:
      - Layers: far edge (per-node, aggregated), near edge (sum across nodes), core (sum across near).
      - Three strategies: dyn. capacity, dyn. traffic, combined.
      - Returns dict with (mean, min, max) for each layer × strategy.
    """
    out = {}

    # ---------- FAR EDGE ----------
    # Per node baseline capacity: 80th percentile of its demand distribution
    # Compute per-slot overflow then sum across nodes
    fe = df_cluster.copy()
    # Baseline fixed capacity per node
    C_node = fe.groupby("node")["y_true"].quantile(C_fixed).to_dict()

    # Build aligned arrays per slot per node
    # For overflow sums across nodes, we'll pivot to time × node matrices
    piv_true = fe.pivot(index="timestamp", columns="node", values="y_true").sort_index()
    piv_pred = fe.pivot(index="timestamp", columns="node", values="y_pred").reindex(index=piv_true.index).sort_index()
    C_mat = np.array([[C_node[n] for n in piv_true.columns]] * len(piv_true))  # time × nodes
    D_mat = piv_true.to_numpy()
    P_mat = piv_pred.to_numpy()

    # Far-edge overflow series per time (sum over nodes)
    of_fixed_fe = np.sum(np.maximum(0.0, D_mat - C_mat), axis=1)
    of_dyncap_fe = np.sum(overflow_dynamic_capacity(D_mat, C_mat, P_mat), axis=1)
    # Dynamic traffic pooling at far-edge layer (same nodes)
    of_dyntraf_fe = np.array([overflow_dynamic_traffic(D_mat[t, :], C_mat[t, :]) for t in range(D_mat.shape[0])])
    # Combined: dynamic capacity + pooling
    of_comb_fe   = np.array([overflow_combined(D_mat[t, :], C_mat[t, :], P_mat[t, :]) for t in range(D_mat.shape[0])])

    t_idx = piv_true.index
    out[("far", "dyncap")] = daily_reduction_percent(t_idx, of_fixed_fe, of_dyncap_fe)
    out[("far", "dyntraf")] = daily_reduction_percent(t_idx, of_fixed_fe, of_dyntraf_fe)
    out[("far", "combined")] = daily_reduction_percent(t_idx, of_fixed_fe, of_comb_fe)

    # ---------- NEAR EDGE ----------
    # Aggregate all far-edge nodes into a single near-edge entity per time
    D_near = D_mat.sum(axis=1)
    P_near = P_mat.sum(axis=1)
    # Fixed capacity at near edge: 80th percentile of aggregated demand
    C_near = np.quantile(D_near, C_fixed)
    C_near_vec = np.full_like(D_near, C_near, dtype=float)

    of_fixed_ne = overflow_fixed(D_near, C_near_vec)
    of_dyncap_ne = overflow_dynamic_capacity(D_near, C_near_vec, P_near)
    # Dynamic traffic at near-edge is trivial with one entity → same as fixed
    of_dyntraf_ne = of_fixed_ne.copy()
    # Combined at near-edge: identical to dyncap (pooling has no effect with one entity)
    of_comb_ne = of_dyncap_ne.copy()

    out[("near", "dyncap")] = daily_reduction_percent(t_idx, of_fixed_ne, of_dyncap_ne)
    out[("near", "dyntraf")] = daily_reduction_percent(t_idx, of_fixed_ne, of_dyntraf_ne)
    out[("near", "combined")] = daily_reduction_percent(t_idx, of_fixed_ne, of_comb_ne)

    # ---------- CORE ----------
    # In questo proxy, il core vede la stessa aggregazione (un’unica entità)
    D_core = D_near.copy()
    P_core = P_near.copy()
    C_core = np.quantile(D_core, C_fixed)
    C_core_vec = np.full_like(D_core, C_core, dtype=float)

    of_fixed_co = overflow_fixed(D_core, C_core_vec)
    of_dyncap_co = overflow_dynamic_capacity(D_core, C_core_vec, P_core)
    of_dyntraf_co = of_fixed_co.copy()   # senza multi-entità, pooling irrilevante
    of_comb_co = of_dyncap_co.copy()

    out[("core", "dyncap")] = daily_reduction_percent(t_idx, of_fixed_co, of_dyncap_co)
    out[("core", "dyntraf")] = daily_reduction_percent(t_idx, of_fixed_co, of_dyntraf_co)
    out[("core", "combined")] = daily_reduction_percent(t_idx, of_fixed_co, of_comb_co)

    return out

# ============================
# RUN & PLOT
# ============================
clusters = load_clusters()
# Evaluate each cluster; then (optionally) average across clusters
records = []
for cid, dfc in clusters.items():
    res = evaluate_layers(dfc, C_fixed=C_FIXED)
    for (layer, strat), (m, mn, mx) in res.items():
        records.append({"cluster": cid, "layer": layer, "strategy": strat,
                        "mean": m, "min": mn, "max": mx})
res_df = pd.DataFrame(records)

# (Optional) average across clusters to emulate "global" bars
avg = (res_df
       .groupby(["layer","strategy"])
       .agg(mean=("mean","mean"), min=("min","mean"), max=("max","mean"))
       .reset_index())

# --- Plot: 3 groups (dyn. capacity / dyn. traffic / combined), each with far/near/core ---
order_strat = ["dyncap","dyntraf","combined"]
order_layer = ["far","near","core"]
labels_strat = {
    "dyncap": "dyn. capacity",
    "dyntraf": "dyn. traffic",
    "combined": "combined"
}
labels_layer = {"far":"far edge","near":"near edge","core":"core"}

# Choose what to plot: average across clusters (Figure-11 style)
df_plot = avg.copy()

# Build positions
group_x = np.arange(len(order_strat))
width = 0.22
offsets = {
    "far": -width, 
    "near": 0.0,
    "core": +width
}
colors = {"far":"#1f77b4","near":"#ff7f0e","core":"#2ca02c"}

try:
    plt.figure(figsize=FIGSIZE)
    for layer in order_layer:
        ys = []
        yerr_low, yerr_up = [], []
        xpos = []
        for i, strat in enumerate(order_strat):
            row = df_plot[(df_plot["layer"]==layer) & (df_plot["strategy"]==strat)]
            if row.empty:
                ys.append(np.nan); yerr_low.append(0); yerr_up.append(0); xpos.append(group_x[i] + offsets[layer])
                continue
            m, mn, mx = float(row["mean"]), float(row["min"]), float(row["max"])
            ys.append(m)
            # errorbars: min/max over days (averaged over clusters); use asymmetrical
            yerr_low.append(max(0.0, m - mn))
            yerr_up.append(max(0.0, mx - m))
            xpos.append(group_x[i] + offsets[layer])
        plt.bar(xpos, ys, width=width*0.95, color=colors[layer], label=labels_layer[layer],
                yerr=[yerr_low, yerr_up], capsize=3, alpha=0.9)

    plt.xticks(group_x, [labels_strat[s] for s in order_strat])
    plt.ylabel("Traffic reduction over fixed capacity [%]")
    plt.title(f"Optimization results at C={int(C_FIXED*100)}% (avg across clusters)")
    plt.legend(ncol=3, loc="upper right")
    plt.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig("figure11_like_barplot.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Print numeric summary (means)
    print("\nFigure-11-like summary (avg across clusters):")
    print(df_plot.replace({"layer":labels_layer, "strategy":labels_strat})
        .pivot(index="layer", columns="strategy", values="mean")
        .round(2).to_string())
except Exception as e:
    print("Error during plotting:", e)
    print("Dataframe for plotting:")

# In[5]:


import numpy as np, pandas as pd, re, glob, os
from numpy.random import default_rng

# ======== I/O ========
FILE_PATTERN = "prediction_clusto_*.csv"   # es. prediction_clusto_1.csv ... _5.csv
OUT_DIR = "csv_scenarios_modified"
os.makedirs(OUT_DIR, exist_ok=True)

# ======== Parametri perturbazioni ========
SHIFT_PROB   = 0.20
SCALE_CANCEL = 0.6    # riduci il valore se l’evento è "cancellato" (evento non accaduto)
SCALE_FP     = 1.25   # aumenta il valore su falsi positivi (evento imprevisto)

# 27 scenari: CANCEL × FP × SHIFT
CANCEL_GRID = [0.10, 0.20, 0.30]
FP_GRID     = [0.05, 0.10, 0.20]
SHIFT_GRID  = [0, 1, 2]

rng = default_rng(42)

# ======== Loader (non droppa NaN) ========
def load_clusters(pattern=FILE_PATTERN):
    clusters = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r'clusto[_-]?(\d+)\.csv$', p)
        if not m: 
            continue
        cid = int(m.group(1))
        df = pd.read_csv(p)
        # timestamp solo per ordinare (non cambiamo le colonne in output)
        ddate = pd.to_datetime(df["date"], errors="coerce")
        if pd.api.types.is_numeric_dtype(df["hour"]):
            ts = ddate + pd.to_timedelta(df["hour"], unit="h")
        else:
            hh = pd.to_datetime(df["hour"], format="%H:%M", errors="coerce").dt.hour
            ts = ddate + pd.to_timedelta(hh.fillna(0).astype("Int64"), unit="h")
        df["_timestamp_for_sort"] = ts

        need = {"date","hour","far_edge","value","predicted"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{p} missing columns: {miss}")
        clusters[cid] = df.copy()
    if not clusters:
        raise FileNotFoundError("No files found for pattern: " + pattern)
    return clusters

# ======== Simulazione errori evento: modifica SOLO 'value' ========
def simulate_on_group_values(g, cancel_rate, fp_rate, shift_steps):
    """
    g: DataFrame (un singolo far_edge), preserva righe/NaN originali.
    Ritorna array 'value_corr' (stessa lunghezza) con SOLO i punti numerici modificati.
    """
    g_sorted = g.sort_values("_timestamp_for_sort", kind="mergesort")
    idx_sorted = g_sorted.index.to_numpy()

    # copie numeriche per calcolo; dove non numerico -> NaN
    val = pd.to_numeric(g_sorted["value"], errors="coerce").to_numpy(dtype=float)
    val_corr = val.copy()
    n = len(g_sorted)
    if n < 4:
        return g["value"].to_numpy()  # troppo corto: ritorna com’è

    # Eventi = top 5% dei picchi di 'value' (storico) → finestre evento originali
    valid_val = np.isfinite(val)
    k = max(1, int(0.05 * np.sum(valid_val)))
    if k >= np.sum(valid_val):
        k = max(1, np.sum(valid_val) // 20)  # salvaguardia
    event_flag = np.zeros(n, dtype=bool)
    if np.sum(valid_val) > 0 and k > 0:
        thr = np.partition(val[valid_val], -k)[-k]
        event_flag[valid_val & (val >= thr)] = True

    # Gruppi contigui (sull’asse temporale)
    grp_id = (pd.Series(event_flag).astype(int).diff().fillna(0) != 0).cumsum().to_numpy()
    run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()]

    # 1) CANCELLATIONS (FN): evento NON accade → abbassa 'value' in quelle finestre
    if run_ids:
        num_cancel = max(1, int(cancel_rate * len(run_ids)))
        sel = rng.choice(run_ids, size=min(num_cancel, len(run_ids)), replace=False)
        m_cancel = np.isin(grp_id, sel) & event_flag
        m_valid = m_cancel & np.isfinite(val_corr)
        val_corr[m_valid] = np.maximum(0.0, val_corr[m_valid] * SCALE_CANCEL)

    # 2) FALSE POSITIVES (FP): evento inaspettato → aumenta 'value' in finestre non-evento
    non_event_idx = np.where(~event_flag)[0]
    num_fp = int(fp_rate * event_flag.sum())  # proporzionale all’estensione eventi originali
    if num_fp > 0 and len(non_event_idx) > 0:
        fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
        m_valid = np.isfinite(val_corr[fp_idx])
        val_corr[fp_idx[m_valid]] = val_corr[fp_idx[m_valid]] * SCALE_FP

    # 3) SHIFTS ±shift_steps: sposta parte dell’intensità evento
    if SHIFT_PROB > 0 and shift_steps > 0 and len(run_ids) > 0 and n > shift_steps:
        shift_runs = [rid for rid in run_ids if rng.random() < SHIFT_PROB]
        for rid in shift_runs:
            m = (grp_id == rid) & event_flag
            if not m.any():
                continue
            direction = shift_steps if rng.random() < 0.5 else -shift_steps
            m_shift = np.roll(m, direction)
            # “sposta” l’impatto: riduci dove c’era, aumenta dove va
            m_up = m_shift & np.isfinite(val_corr)
            val_corr[m_up] = val_corr[m_up] * SCALE_FP
            m_down = (m & ~m_shift) & np.isfinite(val_corr)
            val_corr[m_down] = np.maximum(0.0, val_corr[m_down] * SCALE_CANCEL)

    # ricostruisci nell’ordine originale (preserva valori non numerici/NaN)
    value_corr_full = g["value"].to_numpy()
    #for pos, idx in enumerate(idx_sorted):
    #    if np.isfinite(val[pos]):
    #        value_corr_full[np.where(g.index == idx)[0][0]] = val_corr[pos]
    return value_corr_full

# ======== MAIN: genera 5×27 CSV con value modificato (predicted invariato) ========
clusters = load_clusters()

scenario_map = []
sid = 0
for cr in CANCEL_GRID:
    for fr in FP_GRID:
        for ss in SHIFT_GRID:
            scenario_map.append({"scenario": sid, "CANCEL_RATE": cr, "FP_RATE": fr, "SHIFT_STEPS": ss})
            for cid, df in clusters.items():
                df_out = df.copy()  # preserva colonne e NaN
                # applica simulazione per nodo (far_edge) modificando SOLO 'value'
                new_value = []
                for node, g in df_out.groupby("far_edge", sort=False):
                    v_corr = simulate_on_group_values(g, cancel_rate=cr, fp_rate=fr, shift_steps=ss)
                    new_value.append(pd.Series(v_corr, index=g.index))
                new_value = pd.concat(new_value).sort_index()
                df_out["value"] = new_value        # <-- SOLO value cambiato
                # 'predicted' resta invariato
                cols = ["date","hour","far_edge","value","predicted"]
                df_out[cols].to_csv(os.path.join(OUT_DIR, f"prediction_clusto_{cid}_{sid:02d}.csv"), index=False, na_rep="NaN")
            sid += 1

# opzionale: mappa scenari → parametri
pd.DataFrame(scenario_map).to_csv(os.path.join( "prediction_clusto_scenarios_map.csv"), index=False)
print(f"Saved modified CSVs (value perturbed, predicted unchanged)")






