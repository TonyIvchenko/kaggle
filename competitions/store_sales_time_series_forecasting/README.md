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

The baseline uses lagged sales, rolling means, store metadata, oil prices, and holiday features, then predicts the forecast horizon recursively.
