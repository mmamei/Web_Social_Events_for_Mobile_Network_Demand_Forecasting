
# Web and Social Event Signals for AI-Driven Mobile Network Demand Forecasting

This is the repositoty to replicate the experiments of a paper we published in Open Research Europe (https://open-research-europe.ec.europa.eu).

It provides a single, consistent workflow for running traffic-only
and event-aware forecasting experiments. The code paths are shared across both
modes: omitting ``--event-features-file`` disables exogenous context, while
providing an aligned feature CSV enables it.

## Event feature sources

Three event-input workflows are included:

1. ``events_v1_csv_trends.py`` reads researcher-provided event CSV files and can
   optionally merge an archived Google Trends export.
2. ``events_v2_online_trends.py`` retrieves historical event information online
   and can optionally query Google Trends through ``pytrends``. Retrieved data
   are cached in the output directory.
3. ``events_v3_ai_trends.py`` generates synthetic event scenarios with a local
   language model. Synthetic records are explicitly tagged and are intended for
   robustness and what-if experiments rather than factual historical analysis.

``google_trends_features.py`` can also be used independently to add Google
Trends signals to any existing event-feature CSV.

## Common exogenous feature format

Feature CSV files are aligned to the hourly traffic timeline and contain
``date,hour`` plus optional ``cluster`` and ``far_edge`` keys. Every other column
must be numeric. Typical columns include:

- event activity and event count
- event intensity and geographical relevance
- lead/lag temporal proximity indicators
- event-type indicators
- public-holiday indicators
- normalized Google Trends signals

Missing event values are interpreted as absence of contextual information.

## Build scheduled-event features from local CSV files

```bash
python events_v1_csv_trends.py \
  --partite partite.csv \
  --champions champions.csv \
  --eu-league eu_league.csv \
  --start-date 2019-03-01 \
  --end-date 2019-05-31 \
  --output-dir outputs/events_v1
```

Add an archived Google Trends export:

```bash
python events_v1_csv_trends.py \
  --partite partite.csv \
  --champions champions.csv \
  --eu-league eu_league.csv \
  --start-date 2019-03-01 \
  --end-date 2019-05-31 \
  --google-trends-csv google_trends_nantes_2019.csv \
  --trends-fill ffill \
  --output-dir outputs/events_v1_trends
```

## Retrieve historical events online

```bash
python events_v2_online_trends.py \
  --season-code 1819 \
  --team Nantes \
  --start-date 2019-03-01 \
  --end-date 2019-05-31 \
  --output-dir outputs/events_v2
```

Optional Google Trends retrieval:

```bash
pip install pytrends
python events_v2_online_trends.py \
  --season-code 1819 \
  --team Nantes \
  --start-date 2019-03-01 \
  --end-date 2019-05-31 \
  --fetch-google-trends \
  --trends-keywords "FC Nantes" "Nantes football" \
  --trends-geo FR \
  --output-dir outputs/events_v2_trends
```

## Generate synthetic event scenarios

```bash
python events_v3_ai_trends.py \
  --model qwen2.5:7b \
  --start-date 2019-03-01 \
  --end-date 2019-05-31 \
  --output-dir outputs/events_v3
```

## Google Trends as a reusable stage

From an archived export:

```bash
python google_trends_features.py \
  --trends-csv google_trends_nantes_2019.csv \
  --start-date 2019-03-01 \
  --end-date 2019-05-31 \
  --base-event-features outputs/events_v1/v1_event_features_hourly.csv \
  --output outputs/events_v1/event_features_with_trends.csv
```

Google Trends signals are normalized to ``[0,1]`` and expanded to the hourly
timeline. Forward fill is the default to avoid interpolation from future values.

## Base benchmark: Naive, Ridge, Random Forest, LSTM

Traffic-only run:

```bash
python ml_exps_eventaware.py \
  --clusters 0 1 2 3 4 \
  --output-dir outputs/base_noevent
```

Event-aware run:

```bash
python ml_exps_eventaware.py \
  --clusters 0 1 2 3 4 \
  --event-features-file outputs/events_v1/v1_event_features_hourly.csv \
  --require-event-features \
  --output-dir outputs/base_event
```

Run both modes through the same pipeline:

```bash
python ml_exps_eventaware.py \
  --clusters 0 1 2 3 4 \
  --event-features-file outputs/events_v1/v1_event_features_hourly.csv \
  --compare-event-awareness \
  --output-dir outputs/base_ablation
```

## Standard LSTM Optuna search

Traffic-only:

```bash
python ml_exps_lstm_optuna_standard_eventaware.py \
  --clusters 0 1 2 3 4 \
  --n-trials 30 \
  --study-name optstd_noevent_30 \
  --output-dir outputs/optstd_noevent_30 \
  --reset-study
```

Event-aware:

```bash
python ml_exps_lstm_optuna_standard_eventaware.py \
  --clusters 0 1 2 3 4 \
  --event-features-file outputs/events_v1/v1_event_features_hourly.csv \
  --require-event-features \
  --n-trials 30 \
  --study-name optstd_event_30 \
  --output-dir outputs/optstd_event_30 \
  --reset-study
```

## Extended LSTM Optuna search

``ml_exps_lstm_optuna_v2_eventaware.py`` accepts the same event-feature options.
Omitting ``--event-features-file`` runs the traffic-only configuration.

## Direction-aware custom-loss LSTM

```bash
python ml_exps_lstm_optuna_custom_loss_v3_eventaware.py \
  --clusters 0 1 2 3 4 \
  --event-features-file outputs/events_v1/v1_event_features_hourly.csv \
  --require-event-features \
  --n-trials 30 \
  --tune-loss-params \
  --optuna-objective rmse \
  --study-name optcst_event_30 \
  --output-dir outputs/optcst_event_30 \
  --reset-study
```

Remove the event-feature arguments to run the corresponding traffic-only test.

## Aggregate event-aware and no-event results

```bash
python plot_event_awareness_ablation.py \
  --no-event AUTO::outputs/base_ablation/no_event/forecast_summary.csv \
  --no-event 'LSTM opt_std::outputs/optstd_noevent_30/final_summary.csv' \
  --no-event 'LSTM opt_cst::outputs/optcst_noevent_30/final_summary.csv' \
  --event-aware AUTO::outputs/base_ablation/event_aware/forecast_summary.csv \
  --event-aware 'LSTM opt_std::outputs/optstd_event_30/final_summary.csv' \
  --event-aware 'LSTM opt_cst::outputs/optcst_event_30/final_summary.csv' \
  --output-dir outputs/event_ablation
```

The collector writes separate summaries and plots for the two modes, plus an
``event_awareness_delta.csv`` table containing per-model changes.

## Robustness analysis for unreliable event information.

The script perturbs only the exogenous event-information layer. The observed
mobile-traffic series is never edited.

```bash
python event_input_robustness_factorial_fixed.py \
        --canonical-events outputs/events/events_canonical.csv \
        --baseline-features outputs/events/event_features_hourly.csv \
        --experiment-dir outputs/custom_event_1000 \
        --evaluate-models \
        --output-dir outputs/event_robustness
```

## Note

The released code includes the complete preprocessing pipeline, feature engineering, model definitions, optimization objectives, search spaces, and training procedures required to reproduce the experiments.
The specific best-performing hyperparameter configurations selected during the original optimization campaign are not distributed. They can be independently regenerated by rerunning the provided hyperparameter optimization procedures.
Reduced-trial runs are intended for functional validation and rapid experimentation and are not expected to recover the optimum obtained from the full search. Full optimization runs may be computationally intensive.
Numerical differences may occur across software versions, hardware platforms, and deep-learning backends because of stochastic optimization and backend-specific numerical behavior.

Quick validation:
    --n-trials 30

Full hyperparameter search:
    --n-trials 1000
