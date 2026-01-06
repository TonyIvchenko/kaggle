# Kaggle

Minimal workspace for Kaggle experiments, notebooks, and one-off modeling runs.

## Files

- `requirements.txt`: shared Python dependencies for the repo
- `environment.yml`: conda environment definition that installs `requirements.txt`
- `Makefile`: setup/update helpers plus Jupyter kernel registration

## Setup

Create the conda environment and register the notebook kernel:

```bash
make setup
```

This creates a conda environment named `kaggle` and installs a Jupyter kernel named `Python (kaggle)`.

## Update Environment

```bash
make update
```

This updates the conda environment and refreshes the Jupyter kernel registration.

## Start Jupyter Lab

```bash
make lab
```

Then select the `Python (kaggle)` kernel inside Jupyter.

## Kaggle API Auth

If you want to use the Kaggle API locally, place your `kaggle.json` credentials file at:

```bash
~/.kaggle/kaggle.json
```

and lock down permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Notes

- Python is pinned to `3.11`.
- Root dependencies are intentionally lightweight and notebook-oriented.
- Add competition-specific code wherever you want; a simple pattern is `competitions/<slug>/`.
