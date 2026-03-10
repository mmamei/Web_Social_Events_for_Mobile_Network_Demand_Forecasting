#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =====================================================
# Event-Aware Forecasting Robustness Case Study (NetMob)
# =====================================================

import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- CONFIGURATION ----------------
DATA_PATH = "prediction_clusto_*.csv"
CANCEL_GRID = [0.1, 0.2, 0.3]
FP_GRID     = [0.05, 0.10, 0.20]
SHIFT_GRID  = [0, 1, 2]
CAP_LEVELS  = [0.70, 0.80, 0.90]
SCALE_CANCEL, SCALE_FP, SHIFT_PROB = 0.6, 1.25, 0.2

rng = np.random.default_rng(42)

# ---------------- LOAD DATA ----------------
def load_clusters(pattern=DATA_PATH):
    files = sorted(glob.glob(pattern))
    clusters = {}
    for f in files:
        m = re.search(r'clusto[_-]?(\d+)\.csv', f)
        if not m: continue
        cid = int(m.group(1))
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
        df["y_true"] = pd.to_numeric(df["value"], errors="coerce")
        df["y_pred"] = pd.to_numeric(df["predicted"], errors="coerce")
        clusters[cid] = df.dropna(subset=["timestamp","y_true","y_pred"]).reset_index(drop=True)
    return clusters

clusters = load_clusters()

# ---------------- FUNCTIONS ----------------
def simulate_event_noise(y_true, y_pred, cancel_rate, fp_rate, shift_steps):
    y_corr = y_true.copy()
    n = len(y_true)
    if n < 24: return y_true, y_pred, y_corr
    k = max(1, n//20)
    thr = np.partition(y_true, -k)[-k]
    event_flag = (y_true >= thr)
    grp_id = (pd.Series(event_flag).diff().fillna(0) != 0).cumsum().to_numpy()
    run_ids = [rid for rid in np.unique(grp_id) if event_flag[grp_id==rid].max()==1]

    # cancellations
    if run_ids:
        num_cancel = max(1, int(cancel_rate*len(run_ids)))
        sel = rng.choice(run_ids, size=min(num_cancel, len(run_ids)), replace=False)
        mask = np.isin(grp_id, sel)
        y_corr[mask] = np.maximum(0.0, y_corr[mask]*SCALE_CANCEL)

    # false positives
    non_event_idx = np.where(~event_flag)[0]
    num_fp = int(fp_rate * event_flag.sum())
    if num_fp>0 and len(non_event_idx)>0:
        fp_idx = rng.choice(non_event_idx, size=min(num_fp, len(non_event_idx)), replace=False)
        y_corr[fp_idx] = y_corr[fp_idx]*SCALE_FP

    # shifts
    if SHIFT_PROB>0 and shift_steps>0 and len(run_ids)>0:
        for rid in run_ids:
            if rng.random()>SHIFT_PROB: continue
            m = (grp_id==rid) & event_flag
            direction = shift_steps if rng.random()<0.5 else -shift_steps
            m_shift = np.roll(m, direction)
            y_corr[m_shift] = y_corr[m_shift]*SCALE_FP
            y_corr[m & ~m_shift] = y_corr[m & ~m_shift]*SCALE_CANCEL
    return y_true, y_pred, y_corr

def traffic_reduction(y, yhat, cap_percentile):
    y, yhat = np.asarray(y), np.asarray(yhat)
    C = np.nanpercentile(y, cap_percentile*100)
    overflow_fixed = np.nansum(np.maximum(0, y - C))
    C_dyn = np.maximum(C, yhat)
    overflow_dyn = np.nansum(np.maximum(0, y - C_dyn))
    if overflow_fixed <= 1e-9: return np.nan
    return 100*(overflow_fixed - overflow_dyn)/overflow_fixed

# ---------------- SCENARIO LOOP ----------------
records = []
for cid, dfc in clusters.items():
    for CANC in CANCEL_GRID:
        for FP in FP_GRID:
            for SHIFT in SHIFT_GRID:
                all_mae, all_rmse, all_mape, all_delta = [], [], [], []
                for node, g in dfc.groupby("far_edge"):
                    y, yhat, ycorr = simulate_event_noise(
                        g["y_true"].to_numpy(float),
                        g["y_pred"].to_numpy(float),
                        CANC, FP, SHIFT
                    )
                    if len(y)==0: continue
                    mae = mean_absolute_error(y, ycorr)
                    rmse = np.sqrt(mean_squared_error(y, ycorr))
                    mape = np.mean(np.abs((y - ycorr)/np.maximum(y,1e-9)))*100
                    deltas = [traffic_reduction(y, ycorr, c) - traffic_reduction(y, yhat, c) for c in CAP_LEVELS]
                    all_mae.append(mae); all_rmse.append(rmse); all_mape.append(mape)
                    all_delta.extend([d for d in deltas if np.isfinite(d)])
                records.append({
                    "cluster": cid, "CANC": CANC, "FP": FP, "SHIFT": SHIFT,
                    "MAE": np.nanmean(all_mae),
                    "RMSE": np.nanmean(all_rmse),
                    "MAPE": np.nanmean(all_mape),
                    "Δpp": np.nanmean(all_delta)
                })

results = pd.DataFrame(records)
print(results.head())


# ---------------- SUMMARY BY SCENARIO SEVERITY ----------------
def severity(c,f,s): return c*10 + f*5 + s
results["severity"] = results.apply(lambda r: severity(r.CANC, r.FP, r.SHIFT), axis=1)
agg = results.groupby("severity")[["MAE","RMSE","MAPE","Δpp"]].mean().reset_index()

# ---------------- FIGURES ----------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(1,2, figsize=(11,4))

ax[0].plot(agg["severity"], agg["RMSE"]/1e6, 'o-', label="RMSE ×10⁶")
ax[0].plot(agg["severity"], agg["MAE"]/1e6, 's-', label="MAE ×10⁶")
ax[0].set_xlabel("Scenario severity (CANC + FP + SHIFT)")
ax[0].set_ylabel("Error (×10⁶)")
ax[0].set_title("Forecasting Error vs Event Perturbation")
ax[0].legend()

ax[1].plot(agg["severity"], agg["Δpp"], 'o-', color='tab:red')
ax[1].axhline(0, color='gray', lw=0.8)
ax[1].set_xlabel("Scenario severity (CANC + FP + SHIFT)")
ax[1].set_ylabel("Δ Traffic Reduction (pp)")
ax[1].set_title("Operational Impact (Traffic Reduction Loss)")

plt.tight_layout()
plt.show()

# ---------------- EXPORT ----------------
results.to_csv("results_event_robustness.csv", index=False)
print("Saved: results_event_robustness.csv")


# In[2]:


# JUPYTER NOTEBOOK CELL — Event-Agnostic Forecasting Benchmark (NetMob / Nantes)
# ----------------------------------------------------------------------------------
# Models: S0 Naive | S1 Ridge | S2 RandomForest | S3 LSTM (opzionale)
# Features: lag 1..24 + dummies ora/dow. Nessun evento/Google Trends.
# Output: outputs/forecast_summary.csv, outputs/forecast_results_by_node.csv, outputs/forecast_pernode_predictions.csv
# ----------------------------------------------------------------------------------

import os, glob, warnings, sys, platform
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# --- TF opzionale (per S3). Se non presente, S3 viene saltato.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

print(f"Python {sys.version.split()[0]} | NumPy {np.__version__} | pandas {pd.__version__} | TF: {TF_AVAILABLE}")

# ---------------- CONFIG ----------------
RAW_GLOB = "nantes_antenna_serv_*.csv"
SERVICES_CLUSTERS = "services_clusters.csv"
ANTENNAS_FILE = "nantes_antenna_clustering.csv"
SERVICE_CLUSTER_FILE = "service_clustering.csv"

SELECTED_CLUSTERS: List[int] = None    # es. [0,3] per urbano+suburbano; None = tutti
TEST_SIZE_FRACTION = 0.2                # 20% test (in coda temporale)
N_LAGS = 24
CAP_LEVELS = [0.70, 0.80, 0.90]

# ---------------- HELPERS ----------------
def ensure_services_clusters():
    if os.path.exists(SERVICES_CLUSTERS):
        print(f"[INFO] Found {SERVICES_CLUSTERS}")
        return pd.read_csv(SERVICES_CLUSTERS)
    raw_files = sorted(glob.glob(RAW_GLOB))
    if len(raw_files)==0:
        raise FileNotFoundError(
            f"Missing {SERVICES_CLUSTERS} and no raw {RAW_GLOB}. "
            f"Fornisci data/services_clusters.csv (creato con il tuo script v08)."
        )
    print(f"[BUILD] services_clusters from {len(raw_files)} raws")
    df = pd.concat([pd.read_csv(f) for f in raw_files], axis=0).drop_duplicates().reset_index(drop=True)
    clust = pd.read_csv(SERVICE_CLUSTER_FILE)
    df_ = pd.merge(df, clust, on='service', how='left')
    agg = df_.groupby(['date','labels','lon','lat']).aggregate({str(i):[np.sum] for i in range(96)}).reset_index()
    agg.columns = ['date','labels','lon','lat']+[str(i) for i in range(96)]
    os.makedirs(os.path.dirname(SERVICES_CLUSTERS), exist_ok=True)
    agg.to_csv(SERVICES_CLUSTERS, index=False)
    print(f"[OK] saved {SERVICES_CLUSTERS}")
    return agg

def long_hourly_panel(services_clusters: pd.DataFrame, antennas: pd.DataFrame, cluster_label: int) -> pd.DataFrame:
    df_x = services_clusters[services_clusters['labels']==cluster_label].copy()
    df_x = df_x.groupby(['date','lon','lat']).aggregate({str(i):[np.sum] for i in range(96)}).reset_index()
    df_x.columns = ['date','lon','lat']+[str(i) for i in range(96)]
    df_x = pd.merge(antennas[['lat','lon','far_edge']], df_x, on=['lat','lon'], how='inner')
    df_x_ = df_x.groupby(['lat','lon','far_edge','date']).aggregate({str(i):[np.sum] for i in range(96)}).reset_index()
    df_x_.columns = ['lat','lon','far_edge','date']+[str(i) for i in range(96)]
    df_long = pd.melt(df_x_, id_vars=['date','far_edge'], value_vars=[str(i) for i in range(96)],
                      var_name='slot', value_name='value')
    def slot_to_hour(s):
        try: return int(str(s).split(':')[0])
        except: return 0
    df_long['hour'] = df_long['slot'].map(slot_to_hour)
    df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
    hourly = df_long.groupby(['date','far_edge','hour'])['value'].sum().reset_index()
    return hourly.sort_values(['far_edge','date','hour']).reset_index(drop=True)[['date','hour','far_edge','value']]

def make_supervised(df_node: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
    g = df_node.copy()
    g['dow'] = g['date'].dt.weekday
    g = g.sort_values(['date','hour']).reset_index(drop=True)
    for k in range(1, n_lags+1):
        g[f'lag{k}'] = g['value'].shift(k)
    hour_dum = pd.get_dummies(g['hour'], prefix='h')
    dow_dum  = pd.get_dummies(g['dow'],  prefix='d')
    X = pd.concat([hour_dum, dow_dum, g[[f'lag{k}' for k in range(1, n_lags+1)]]], axis=1)
    y = g['value'].copy()
    out = pd.concat([g[['date','hour','far_edge']], y, X], axis=1)
    return out.iloc[n_lags:].reset_index(drop=True)

def split_time(df_super: pd.DataFrame, test_frac: float):
    n = len(df_super); n_test = max(1, int(round(n*test_frac))); n_train = n - n_test
    return df_super.iloc[:n_train].copy(), df_super.iloc[n_train:].copy()

def traffic_reduction(y: np.ndarray, yhat: np.ndarray, cap_percentile: float) -> float:
    y = np.asarray(y).astype(float); yhat = np.asarray(yhat).astype(float)
    C = np.nanpercentile(y, cap_percentile*100.0)
    overflow_fixed = np.nansum(np.maximum(0.0, y - C))
    C_dyn = np.maximum(C, yhat)
    overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
    return np.nan if overflow_fixed<=1e-12 else 100.0*(overflow_fixed-overflow_dyn)/overflow_fixed

def evaluate_node(df_node: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
    df_super = make_supervised(df_node, n_lags)
    train, test = split_time(df_super, TEST_SIZE_FRACTION)
    Xtr = train.drop(columns=['date','hour','far_edge','value']).values
    ytr = train['value'].values
    Xte = test.drop(columns=['date','hour','far_edge','value']).values
    yte = test['value'].values
    rows = []; preds = {}

    # S0
    yhat0 = test['lag1'].to_numpy()
    rows.append(dict(model='S0_Naive',
                     MAE=mean_absolute_error(yte,yhat0),
                     RMSE=mean_squared_error(yte,yhat0,squared=False),
                     MAPE=np.mean(np.abs((yte-yhat0)/np.maximum(yte,1e-9)))*100,
                     **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat0,c) for c in CAP_LEVELS}))
    preds['S0']=yhat0

    # S1 Ridge
    m1 = Ridge(alpha=1.0, fit_intercept=True, random_state=42).fit(Xtr,ytr)
    yhat1 = m1.predict(Xte)
    rows.append(dict(model='S1_Ridge',
                     MAE=mean_absolute_error(yte,yhat1),
                     RMSE=mean_squared_error(yte,yhat1,squared=False),
                     MAPE=np.mean(np.abs((yte-yhat1)/np.maximum(yte,1e-9)))*100,
                     **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat1,c) for c in CAP_LEVELS}))
    preds['S1']=yhat1

    # S2 RF
    m2 = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr,ytr)
    yhat2 = m2.predict(Xte)
    rows.append(dict(model='S2_RandomForest',
                     MAE=mean_absolute_error(yte,yhat2),
                     RMSE=mean_squared_error(yte,yhat2,squared=False),
                     MAPE=np.mean(np.abs((yte-yhat2)/np.maximum(yte,1e-9)))*100,
                     **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat2,c) for c in CAP_LEVELS}))
    preds['S2']=yhat2

    # S3 LSTM opzionale
    if TF_AVAILABLE:
        xmin = Xtr.min(axis=0); xmax = Xtr.max(axis=0); xmax = np.where(xmax-xmin==0, xmax+1, xmax)
        Xtr_mm = (Xtr-xmin)/(xmax-xmin); Xte_mm = (Xte-xmin)/(xmax-xmin)
        y_min, y_max = ytr.min(), ytr.max(); yrng = (y_max-y_min) if y_max>y_min else 1.0
        ytr_mm = (ytr-y_min)/yrng
        Xtr3 = Xtr_mm.reshape((Xtr_mm.shape[0],1,Xtr_mm.shape[1]))
        Xte3 = Xte_mm.reshape((Xte_mm.shape[0],1,Xte_mm.shape[1]))
        lstm = Sequential([LSTM(128, activation='tanh', input_shape=(1, Xtr_mm.shape[1])),
                           Dropout(0.1), Dense(1,'linear')])
        lstm.compile(optimizer='adam', loss='mae')
        es=EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        lstm.fit(Xtr3, ytr_mm, epochs=100, batch_size=16, verbose=0, callbacks=[es])
        yhat_mm = lstm.predict(Xte3, verbose=0).ravel()
        yhat3 = yhat_mm*yrng + y_min
        rows.append(dict(model='S3_LSTM',
                         MAE=mean_absolute_error(yte,yhat3),
                         RMSE=mean_squared_error(yte,yhat3,squared=False),
                         MAPE=np.mean(np.abs((yte-yhat3)/np.maximum(yte,1e-9)))*100,
                         **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat3,c) for c in CAP_LEVELS}))
        preds['S3']=yhat3
    return pd.DataFrame(rows), test[['date','hour','far_edge']].reset_index(drop=True), preds, yte

# ---------------- RUN ----------------

antennas = pd.read_csv(ANTENNAS_FILE)
services_clusters = ensure_services_clusters()
available_clusters = sorted(services_clusters['labels'].dropna().astype(int).unique().tolist())
print("[INFO] Clusters disponibili:", available_clusters)
if SELECTED_CLUSTERS is None:
    SELECTED_CLUSTERS = available_clusters


all_metrics = []
pernode = []
for lab in SELECTED_CLUSTERS:
    print(f"\n[CLUSTER {lab}] preparo pannello orario …")
    hourly = long_hourly_panel(services_clusters, antennas, lab)
    for fe in hourly['far_edge'].unique():
        df_node = hourly[hourly['far_edge']==fe].copy()
        if len(df_node) < (N_LAGS + 48):  # un minimo di storia
            continue
        m, keys, preds, ytrue = evaluate_node(df_node, N_LAGS)
        m.insert(0,'far_edge', fe); m.insert(0,'cluster', lab); all_metrics.append(m)
        out = keys.copy(); out['y_true']=ytrue
        for k,v in preds.items():
            out[f'yhat_{k}']=v
        out['cluster']=lab; pernode.append(out)

results = pd.concat(all_metrics, ignore_index=True)
summary = results.groupby('model').agg({
    'MAE':'mean','RMSE':'mean','MAPE':'mean',
    **{f'Dpp@{int(c*100)}':'mean' for c in CAP_LEVELS}
}).reset_index().sort_values('RMSE')

os.makedirs('outputs', exist_ok=True)
results.to_csv('outputs/forecast_results_by_node.csv', index=False)
summary.to_csv('outputs/forecast_summary.csv', index=False)
if len(pernode):
    pd.concat(pernode, ignore_index=True).to_csv('outputs/forecast_pernode_predictions.csv', index=False)

print("\n=== SUMMARY (mean over nodes/clusters) ===")
print(summary)

# --------- FIGURES ---------
fig, ax = plt.subplots(figsize=(8,4))
x = np.arange(len(summary))
ax.bar(x-0.2, summary['RMSE'], width=0.4, label='RMSE')
ax.bar(x+0.2, summary['MAE'],  width=0.4, label='MAE')
ax.set_xticks(x); ax.set_xticklabels(summary['model'], rotation=15)
ax.set_ylabel("Error"); ax.set_title("Forecasting error by model")
ax.legend(); plt.tight_layout()
plt.savefig('outputs/fig_errors_by_model.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8,4))
for c in CAP_LEVELS:
    col=f'Dpp@{int(c*100)}'; ax.plot(summary['model'], summary[col], marker='o', label=col)
ax.axhline(0, color='gray', lw=0.8)
ax.set_ylabel("Δ traffic reduction (pp)"); ax.set_title("Operational gain vs static capacity")
ax.legend(); plt.tight_layout()
plt.savefig('outputs/fig_operational_gain.pdf', bbox_inches='tight')
plt.show()

print("Saved CSV and PNG in ./outputs")


# In[9]:


# ==== CELL 1: Setup & common utils (run first) ===================================
import os, sys, glob, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# TensorFlow opzionale (solo per S3_LSTM). Se non disponibile, lo saltiamo.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

print("Versions:",
      f"Python={sys.version.split()[0]}",
      f"NumPy={np.__version__}",
      f"pandas={pd.__version__}",
      f"TF={TF_AVAILABLE}", sep=" | ")

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
RAW_GLOB = str(DATA_DIR / "nantes_antenna_serv_*.csv")
SERVICES_CLUSTERS = DATA_DIR / "services_clusters.csv"
ANTENNAS_FILE = Path("nantes_antenna_clustering.csv")
SERVICE_CLUSTER_FILE = Path("service_clustering.csv")

HOURLY_DIR = DATA_DIR / "hourly_panels"  # cache Parquet
HOURLY_DIR.mkdir(parents=True, exist_ok=True)

# Cluster da processare: None = tutti quelli presenti
SELECTED_CLUSTERS: List[int] = [0,1,2,3,4] #None     # es. [0, 3] per urbano+suburbano
TEST_SIZE_FRACTION = 0.20               # split temporale: ultimo 20% = test
N_LAGS = 24
CAP_LEVELS = [0.70, 0.80, 0.90]

# ---------------- Utilities: supervised data, split, metrica operativa ----------
def make_supervised(df_node: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
    """Costruisce X con lag1..lagN + dummies ora e giorno-settimana. Nessuna feature evento."""
    g = df_node.copy()
    g['dow'] = pd.to_datetime(g['date']).dt.weekday
    g = g.sort_values(['date','hour']).reset_index(drop=True)
    for k in range(1, n_lags+1):
        g[f'lag{k}'] = g['value'].shift(k)
    hour_dum = pd.get_dummies(g['hour'], prefix='h')
    dow_dum  = pd.get_dummies(g['dow'],  prefix='d')
    X = pd.concat([hour_dum, dow_dum, g[[f'lag{k}' for k in range(1, n_lags+1)]]], axis=1)
    y = g['value'].copy()
    out = pd.concat([g[['date','hour','far_edge']], y, X], axis=1)
    return out.iloc[n_lags:].reset_index(drop=True)

def split_time(df_super: pd.DataFrame, test_frac: float):
    n = len(df_super); n_test = max(1, int(round(n*test_frac))); n_train = n - n_test
    return df_super.iloc[:n_train].copy(), df_super.iloc[n_train:].copy()

def traffic_reduction(y: np.ndarray, yhat: np.ndarray, cap_percentile: float) -> float:
    """Δ riduzione overflow rispetto a capacità fissa al percentile dato (pp, positivo=meglio)."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    C = np.nanpercentile(y, cap_percentile*100.0)
    overflow_fixed = np.nansum(np.maximum(0.0, y - C))
    C_dyn = np.maximum(C, yhat)
    overflow_dyn = np.nansum(np.maximum(0.0, y - C_dyn))
    return np.nan if overflow_fixed <= 1e-12 else 100.0*(overflow_fixed - overflow_dyn)/overflow_fixed

def evaluate_node(df_node: pd.DataFrame, n_lags: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    """Allena e valuta S0..S3 su un singolo far_edge. Ritorna (metrics_df, chiavi_test, preds, ytest)."""
    df_super = make_supervised(df_node, n_lags)
    train, test = split_time(df_super, TEST_SIZE_FRACTION)
    Xtr = train.drop(columns=['date','hour','far_edge','value']).values
    ytr = train['value'].values
    Xte = test.drop(columns=['date','hour','far_edge','value']).values
    yte = test['value'].values

    rows, preds = [], {}

    # S0: Naive (persistence y_{t-1})
    yhat0 = test['lag1'].to_numpy()
    rows.append(dict(model='S0_Naive',
                     MAE=mean_absolute_error(yte,yhat0),
                     RMSE=mean_squared_error(yte,yhat0,squared=False),
                     MAPE=np.mean(np.abs((yte-yhat0)/np.maximum(yte,1e-9)))*100,
                     **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat0,c) for c in CAP_LEVELS}))
    preds['S0'] = yhat0

    # S1: Ridge(lags+dummies)
    ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=42).fit(Xtr,ytr)
    yhat1 = ridge.predict(Xte)
    rows.append(dict(model='S1_Ridge',
                     MAE=mean_absolute_error(yte,yhat1),
                     RMSE=mean_squared_error(yte,yhat1,squared=False),
                     MAPE=np.mean(np.abs((yte-yhat1)/np.maximum(yte,1e-9)))*100,
                     **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat1,c) for c in CAP_LEVELS}))
    preds['S1'] = yhat1

    # S2: RandomForest(lags+dummies)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr,ytr)
    yhat2 = rf.predict(Xte)
    rows.append(dict(model='S2_RandomForest',
                     MAE=mean_absolute_error(yte,yhat2),
                     RMSE=mean_squared_error(yte,yhat2,squared=False),
                     MAPE=np.mean(np.abs((yte-yhat2)/np.maximum(yte,1e-9)))*100,
                     **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat2,c) for c in CAP_LEVELS}))
    preds['S2'] = yhat2

    # S3: LSTM opzionale (solo se TF presente)
    if TF_AVAILABLE:
        xmin = Xtr.min(axis=0); xmax = Xtr.max(axis=0); xmax = np.where(xmax-xmin==0, xmax+1, xmax)
        Xtr_mm = (Xtr-xmin)/(xmax-xmin); Xte_mm = (Xte-xmin)/(xmax-xmin)
        y_min, y_max = ytr.min(), ytr.max(); yrng = (y_max-y_min) if y_max>y_min else 1.0
        ytr_mm = (ytr-y_min)/yrng
        Xtr3 = Xtr_mm.reshape((Xtr_mm.shape[0],1,Xtr_mm.shape[1]))
        Xte3 = Xte_mm.reshape((Xte_mm.shape[0],1,Xte_mm.shape[1]))
        lstm = Sequential([
            LSTM(128, activation='tanh', input_shape=(1, Xtr_mm.shape[1])),
            Dropout(0.1), Dense(1, activation='linear')
        ])
        lstm.compile(optimizer='adam', loss='mae')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        lstm.fit(Xtr3, ytr_mm, epochs=100, batch_size=16, verbose=0, callbacks=[es])
        yhat_mm = lstm.predict(Xte3, verbose=0).ravel()
        yhat3 = yhat_mm*yrng + y_min
        rows.append(dict(model='S3_LSTM',
                         MAE=mean_absolute_error(yte,yhat3),
                         RMSE=mean_squared_error(yte,yhat3,squared=False),
                         MAPE=np.mean(np.abs((yte-yhat3)/np.maximum(yte,1e-9)))*100,
                         **{f'Dpp@{int(c*100)}': traffic_reduction(yte,yhat3,c) for c in CAP_LEVELS}))
        preds['S3'] = yhat3

    metrics_df = pd.DataFrame(rows)
    keys_test = test[['date','hour','far_edge']].reset_index(drop=True)
    return metrics_df, keys_test, preds, yte


# In[5]:


# ==== CELL 2: Build-or-Load hourly panels per cluster ============================
from tqdm import tqdm

def ensure_services_clusters(services_clusters_path: Path) -> pd.DataFrame:
    """Carica data/services_clusters.csv se esiste; altrimenti prova a costruirlo dai raw."""
    if services_clusters_path.exists():
        print(f"[INFO] Found {services_clusters_path}")
        return pd.read_csv(services_clusters_path)
    raw_files = sorted(glob.glob(RAW_GLOB))
    if len(raw_files) == 0:
        raise FileNotFoundError(
            f"Missing {services_clusters_path} and no raw files under {RAW_GLOB}.\n"
            f"Fornisci data/services_clusters.csv (generato con il tuo script) oppure i CSV raw."
        )
    print(f"[BUILD] services_clusters from {len(raw_files)} raws")
    df = pd.concat([pd.read_csv(f) for f in raw_files], axis=0).drop_duplicates().reset_index(drop=True)
    clust = pd.read_csv(SERVICE_CLUSTER_FILE)
    df_ = pd.merge(df, clust, on='service', how='left')
    agg = df_.groupby(['date','labels','lon','lat']).aggregate({str(i):[np.sum] for i in range(96)}).reset_index()
    agg.columns = ['date','labels','lon','lat'] + [str(i) for i in range(96)]
    services_clusters_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(services_clusters_path, index=False)
    print(f"[OK] saved {services_clusters_path}")
    return agg

def build_hourly_for_cluster(cluster_label: int,
                             services_clusters: pd.DataFrame,
                             antennas: pd.DataFrame,
                             cache_dir: Path = HOURLY_DIR,
                             force_rebuild: bool = False) -> pd.DataFrame:
    """Costruisce o carica da cache csv il pannello orario per un service-cluster."""
    out_file = cache_dir / f"hourly_cluster_{cluster_label}.csv"
    if out_file.exists() and not force_rebuild:
        print(f"[CACHE] {out_file} found")
        return pd.read_csv(out_file)

    print(f"[BUILD] cluster {cluster_label} → hourly panel…")
    df_x = services_clusters[services_clusters['labels']==cluster_label].copy()
    df_x = df_x.groupby(['date','lon','lat']).aggregate({str(i): [np.sum] for i in range(96)}).reset_index()
    df_x.columns = ['date','lon','lat'] + [str(i) for i in range(96)]
    df_x = pd.merge(antennas[['lat','lon','far_edge']], df_x, on=['lat','lon'], how='inner')
    df_x_ = df_x.groupby(['lat','lon','far_edge','date']).aggregate({str(i): [np.sum] for i in range(96)}).reset_index()
    df_x_.columns = ['lat','lon','far_edge','date'] + [str(i) for i in range(96)]
    df_long = pd.melt(df_x_, id_vars=['date','far_edge'], value_vars=[str(i) for i in range(96)],
                      var_name='slot', value_name='value')

    # slot -> ora (se lo slot è già un numero 0..95, prendi int, altrimenti estrai ore dal pattern "H:MM:SS")
    if df_long['slot'].astype(str).str.fullmatch(r'\d+').all():
        df_long['hour'] = df_long['slot'].astype(int) // 4  # 4 slot da 15' per ora
    else:
        df_long['hour'] = df_long['slot'].astype(str).str.extract(r'^(\d+)').astype(int)

    df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
    hourly = df_long.groupby(['date','far_edge','hour'])['value'].sum().reset_index()
    hourly = hourly.sort_values(['far_edge','date','hour']).reset_index(drop=True)
    hourly.to_csv(out_file, index=False)
    print(f"[OK] saved {out_file}")
    return hourly

# ---- Esecuzione: build-or-load per i cluster selezionati ----
antennas = pd.read_csv(ANTENNAS_FILE)
services_clusters = ensure_services_clusters(SERVICES_CLUSTERS)

available_clusters = sorted(services_clusters['labels'].dropna().astype(int).unique().tolist())
print("[INFO] Service clusters disponibili:", available_clusters)

if SELECTED_CLUSTERS is None:
    SELECTED_CLUSTERS = available_clusters[:2]  # es.: primi due per iniziare più rapidamente

hourly_panels = {}
for cl in tqdm(SELECTED_CLUSTERS, desc="Build/Load hourly panels"):
    hourly_panels[cl] = build_hourly_for_cluster(cl, services_clusters, antennas, HOURLY_DIR, force_rebuild=False)

print("Cache pronta in:", HOURLY_DIR)


# In[6]:


# ==== CELL 3: Forecasting & export using cached hourly panels ====================
results_rows = []
pernode_preds = []

for lab, hourly in hourly_panels.items():
    print(f"\n[CLUSTER {lab}] forecasting on {hourly['far_edge'].nunique()} far_edge nodes …")
    for fe in hourly['far_edge'].unique():
        df_node = hourly[hourly['far_edge']==fe].copy()
        if len(df_node) < (N_LAGS + 48):  # minimo di storia
            continue
        m, keys, preds, ytrue = evaluate_node(df_node, N_LAGS)
        m.insert(0,'far_edge', fe); m.insert(0,'cluster', lab); results_rows.append(m)

        out = keys.copy(); out['y_true'] = ytrue
        for tag, arr in preds.items():
            out[f'yhat_{tag}'] = arr
        out['cluster'] = lab
        pernode_preds.append(out)

results = pd.concat(results_rows, ignore_index=True)
summary = results.groupby('model').agg({
    'MAE':'mean','RMSE':'mean','MAPE':'mean',
    **{f'Dpp@{int(c*100)}':'mean' for c in CAP_LEVELS}
}).reset_index().sort_values('RMSE')

# Export
OUTDIR = Path("outputs"); OUTDIR.mkdir(exist_ok=True)
results.to_csv(OUTDIR / "forecast_results_by_node.csv", index=False)
summary.to_csv(OUTDIR / "forecast_summary.csv", index=False)
if len(pernode_preds):
    pd.concat(pernode_preds, ignore_index=True).to_csv(OUTDIR / "forecast_pernode_predictions.csv", index=False)

print("\n=== SUMMARY (mean over nodes/clusters) ===")
print(summary)
print(f"\nSaved CSV in {OUTDIR}/")


# In[8]:


# ==== CELL 4: Plots & display ====================================================
#from caas_jupyter_tools import display_dataframe_to_user

# Carica il riassunto appena prodotto (utile se si rilancia solo questa cella)
summary = pd.read_csv("outputs/forecast_summary.csv")

# 1) Errori per modello
fig, ax = plt.subplots(figsize=(8,4))
x = np.arange(len(summary))
ax.bar(x-0.2, summary['RMSE'], width=0.4, label='RMSE')
ax.bar(x+0.2, summary['MAE'],  width=0.4, label='MAE')
ax.set_xticks(x); ax.set_xticklabels(summary['model'], rotation=15)
ax.set_ylabel("Error"); ax.set_title("Forecasting error by model (mean)")
ax.legend(); plt.tight_layout()
plt.savefig('outputs/fig_errors_by_model.pdf', bbox_inches='tight')
plt.show()

# 2) Δ traffic reduction (pp) vs capacità (per modello)
fig, ax = plt.subplots(figsize=(8,4))
for c in CAP_LEVELS:
    col=f'Dpp@{int(c*100)}'
    ax.plot(summary['model'], summary[col], marker='o', label=col)
ax.axhline(0, color='gray', lw=0.8)
ax.set_ylabel("Δ traffic reduction (pp)")
ax.set_title("Operational gain vs static capacity")
ax.legend(); plt.tight_layout()
plt.savefig('outputs/fig_operational_gain.pdf', bbox_inches='tight')
plt.show()

# Tabella riassuntiva interattiva
display_dataframe_to_user("Forecast summary (by model)", summary)
print("Figures saved to outputs/.")






