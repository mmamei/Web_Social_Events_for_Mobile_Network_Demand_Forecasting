#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from datetime import datetime, timedelta


NFClist = [ "far_edge", "near_edge", "core" ]
NFC = NFClist[1]


ERROR_SCENARIO_INDEX = 26


# In[202]:


t = pd.read_csv(f'prediction_clusto_2.csv')
t = t.loc[t['far_edge']==33,:]
t['date'] = t['date'].astype(str) +' '+ t['hour'].astype(str)
t['date'] = t['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H'))
t = t.sort_values(['date'])
t['rw'] = t['value'].shift(1)
#t = t.fillna(0)
t = t.dropna()
fig = plt.figure(figsize = (20, 8))
#plt.subplot(2,1,1)
plt.plot(t['date'],t['value'])
#plt.subplot(2,1,2)
plt.plot(t['date'],t['predicted'],c='orange')
plt.show()

rmse1 = ((t['value'] - t['predicted']) ** 2).mean() ** .5
print(round(rmse1,2),'rmse forecast')
rmse0 = ((t['value'] - t['rw']) ** 2).mean() ** .5
print(round(rmse0,2),'rmse rw')
print((rmse1-rmse0)/rmse1)


# In[214]:


# Carica originale
t = pd.read_csv("prediction_clusto_2.csv")
t = t.loc[t["far_edge"] == 33, :].copy()
t['date'] = t['date'].astype(str) +' '+ t['hour'].astype(str)
t['date'] = t['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H'))
t = t.sort_values("date")
t["rw"] = t["value"].shift(1)
t = t.dropna()

# Carica scenario perturbato (value modificato)
t2 = pd.read_csv(f"event_cancelled_prediction_clusto_2_{ERROR_SCENARIO_INDEX}.csv")
t2 = t2.loc[t2["far_edge"] == 33, :].copy()
t2['date'] = t2['date'].astype(str) +' '+ t2['hour'].astype(str)
t2['date'] = t2['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H'))
t2["rw"] = t2["value"].shift(1)
t2 = t2.sort_values("date")

# Filtro finestra temporale: 27–30 maggio
start, end = datetime(2019, 5, 27), datetime(2019, 5, 30)
mask1 = (t["date"] >= start) & (t["date"] <= end)
mask2 = (t2["date"] >= start) & (t2["date"] <= end)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})
ax.plot(t.loc[mask1, "date"], t.loc[mask1, "value"], label="Original value")
ax.plot(t.loc[mask1, "date"], t.loc[mask1, "predicted"], c="orange", label="Predicted")
ax.plot(t2.loc[mask2, "date"], t2.loc[mask2, "value"], c="red", linestyle="--", label="Perturbed value")
ax.legend(loc='lower right')
plt.savefig("orig_vs_pert_value.pdf")
plt.show()

# RMSE calcolati sul dataset originale
rmse1 = ((t2["value"] - t2["predicted"]) ** 2).mean() ** 0.5
rmse0 = ((t2["value"] - t2["rw"]) ** 2).mean() ** 0.5
print(round(rmse1, 2), "rmse forecast")
print(round(rmse0, 2), "rmse rw")
print((rmse1 - rmse0) / rmse1)

