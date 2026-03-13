# Kaggle

Workspace for KAggle experiments

## Files

- `requirements.txt`: shared dependencies
- `environment.yml`: conda environment with `requirements.txt`
- `Makefile`: setup/update helpers with Jupyter kernel registration

## Setup

Create conda environment and register Jupyter kernel:

```bash
make setup
```

Creates conda environment named `kaggle` and installs a Jupyter kernel named `Kaggle`.

## Update Environment

```bash
make update
```

Updates conda environment and refreshes Jupyter kernel registration

## Start Jupyter Lab

```bash
make lab
```

And select the `Kaggle` kernel inside Jupyter

## Kaggle API Auth

If you want to use Kaggle API, add your credentials to `~/.kaggle/kaggle.json` and set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Notes

- Python is pinned to `3.11`
- Root dependencies are intentionally lightweight
- Competition workspaces live under `competitions/`
