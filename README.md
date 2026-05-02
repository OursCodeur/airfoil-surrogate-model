# Airfoil Surrogate Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airfoil-surrogate-model.streamlit.app/)

End-to-end Streamlit app for exploring a surrogate-model workflow on the NASA /
UCI Airfoil Self-Noise dataset.

Live app: https://airfoil-surrogate-model.streamlit.app/

The app trains regression models to predict sound pressure level in dB from five
airfoil and wind-tunnel parameters, then compares standard validation against a
physical holdout and adds product-style trust signals.

## What It Shows

- Dataset ranges and feature meanings.
- Baseline model comparison:
  - linear regression
  - random forest
  - gradient boosting
- Regression metrics: MAE, RMSE, and R2.
- Error analysis through predicted-vs-actual plots, residual histograms, error
  by feature range, and worst predictions.
- Random train/test split versus a physical-region holdout.
- Interactive prediction with range checks, nearest-neighbor distance, and
  random-forest tree disagreement.
- Terminology lookup with external references for ML and validation vocabulary.

## Requirements

- Python 3.11 or newer.
- A shell with standard Python virtual environment support.
- Network access for first-time dependency installation.

The dataset is committed in `data/airfoil_self_noise.dat`, so the app does not
need network access at runtime. If the local file is missing, the app tries to
download the dataset from UCI.

This project was tested with Python 3.14.3 and Streamlit 1.57.0.

## Quick Start

From the project root:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

On Windows PowerShell, use:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

## App Navigation

- `1. Data`: dataset source, feature ranges, and raw sample rows.
- `2. Models`: baseline model comparison and error analysis.
- `3. Extrapolation`: random split versus high-region physical holdout.
- `4. Trust signal`: single prediction with confidence and validity signals.
- `5. Terminology`: vocabulary lookup with external resources.
- `6. Notes`: technical takeaways from the workflow.

## Project Structure

```text
.
├── app.py                       # Streamlit app and modeling workflow
├── terminology.py               # Glossary content and lookup resources
├── data/
│   └── airfoil_self_noise.dat   # Local copy of the UCI/NASA dataset
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Developer tooling
├── pyproject.toml               # Tool configuration
└── README.md
```

## Development Checks

Install the development tools:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

Run the checks:

```bash
.venv/bin/python -m compileall app.py terminology.py
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

Run a non-UI smoke test:

```bash
.venv/bin/python -c "import app; df = app.read_airfoil_data(); exp = app.run_experiment(df, 42, 0.2, 'angle_of_attack_deg', 0.8); print(app.metrics_table(exp['random_results']).round(3).to_string())"
```

Apply formatting:

```bash
.venv/bin/ruff format .
```

PyCharm users can also run the default inspection profile. The project excludes
`.venv`, `.matplotlib-cache`, and `__pycache__` from the module.

## Deployment

The app is ready for Streamlit Community Cloud:

- Repository: any public GitHub repository containing these files.
- Branch: `main`.
- Entrypoint file: `app.py`.
- Python version: select Python 3.12 or newer in Streamlit's advanced settings.
- Secrets: none.

Streamlit Community Cloud reads `requirements.txt` from the repository root and
runs `streamlit run app.py`.

## Troubleshooting

If Streamlit starts but the browser does not open automatically, copy the
printed local URL into the browser.

If dependencies are missing, confirm that commands are using the virtual
environment:

```bash
which python
which streamlit
```

Expected paths should point into `.venv`.

If Matplotlib cache warnings appear, confirm `.matplotlib-cache/` is writable.
The app sets Matplotlib's config directory before importing `matplotlib`.

If the model metrics change slightly after changing the random seed, that is
expected. The app exposes the random seed in the sidebar.
