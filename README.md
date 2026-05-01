# Kaggle

Workspace for KAggle experiments

## Files

- `requirements.txt`: shared dependencies
- `environment.yml`: conda environment with `requirements.txt`
- `Makefile`: setup/update helpers with Jupyter kernel registration
- `pyproject.toml`: pytest configuration (test paths, import path, warning filters)

## Setup

Create conda environment and register Jupyter kernel:

```bash
make setup
```

Creates conda environment named `kaggle` and installs a Jupyter kernel named `Python (kaggle)`.

## Update Environment

```bash
make update
```

Updates conda environment and refreshes Jupyter kernel registration

## Start Jupyter Lab

```bash
make lab
```

And select the `Python (kaggle)` kernel inside Jupyter

## Clean repo (delete temporary files)

```bash
make clean
```

Deletes files under `data/raw`, `data/processed` and `submissions` folders, deleted non-Python files under `models`

## Run tests

```bash
make test
```

Runs `pytest` in the `kaggle` conda environment. Pass a path or flags with
`args`, e.g. `make test args="competitions/store_sales_time_series_forecasting -q"`.
Test discovery, the import path, and warning filters are configured in
`pyproject.toml`, so plain `pytest` works too.

## Kaggle API Auth

If you want to use Kaggle API, add your credentials to `~/.kaggle/kaggle.json` and set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Notes

- Python is pinned to `3.11`
- Root dependencies are intentionally lightweight
- Competition workspaces live under `competitions/`
