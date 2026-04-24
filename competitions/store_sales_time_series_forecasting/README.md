# Store Sales - Time Series Forecasting

Workspace for Kaggle competition:

- competition slug: `store-sales-time-series-forecasting`
- task: time-series regression (`sales`)
- expected submission: `id` + `sales`

## Layout

- `scripts/download_data.py`: downloads competition files via Kaggle CLI
- `scripts/train_model.py`: trains baseline forecasters and writes submission
- `models/baseline.py`: feature engineering + recursive forecasting utilities
- `tests/`: unit tests for downloader and baseline behavior

## Quick Start

```bash
python competitions/store_sales_time_series_forecasting/scripts/download_data.py --all-files
python competitions/store_sales_time_series_forecasting/scripts/train_model.py
```

The baseline uses lagged sales, rolling means, promotion lags, store metadata,
oil prices, a per-store weekly transactions profile, and holiday features, then
predicts the forecast horizon recursively.

## Modeling notes

- **Strategies**: a seasonal-naive baseline plus LightGBM and XGBoost are scored
  on a time-based holdout (the final `--holdout-days` train dates) and the best
  RMSLE wins. The boosters train on a log1p target with recent-row sample
  weighting and use early stopping against the most recent days of the training
  frame.
- **Holidays**: a holiday counts as active only when it actually occurs —
  `transferred=True` holidays are dropped and `Transfer` rows (where the day off
  lands) are counted instead.
- **Transactions**: the transactions file (training period only) is summarised
  into a `store_nbr × dayofweek` average and joined onto both train and test.
- **`--force-strategy`**: pins the deployed strategy while still reporting that
  strategy's own holdout metrics and predictions (it is added to the candidate
  set automatically when missing).
- **`--max-train-rows`**: caps training to the most recent rows by date.
