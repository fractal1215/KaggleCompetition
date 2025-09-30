<!-- Copilot / AI agent guidance for the KaggleCompetition repo -->
# Repo overview (short)

This repository contains a Kaggle competition solution for real-estate demand prediction. Work is notebook-first (data exploration, preprocessing, modelling, submissions) with supporting scripts and an optional `src/` package for reusable code.

## Terminal Safety Rules

- Never delete, modify, or chmod:
  - `/bin`, `/sbin`, `/usr`, `/lib`, `/System`, `/boot`, `/efi`
- Never touch `/etc` (system configs) or `/dev`, `/proc`, `/sys` (devices & kernel)
- Do not run destructive commands:
  - `rm -rf /`, `chmod -R 777 /`, `shutdown`, `reboot`, `kill -9 1`
- Do not install/uninstall system-wide packages (`apt`, `yum`, `brew`, etc.)
- Do not modify networking or security configs (`/etc/hosts`, firewall, SSL certs)
- Allowed: only work inside project folders (e.g., `~/Python/KaggleCompetition/`), safe dirs (`/tmp`, `~/Downloads`), and virtual environments.

# Where to look first

- `notebooks/` — primary work. Cells contain the full data pipeline and model experiments (00...09). Use these for data flow, feature engineering, and modelling examples.
- `data/` — raw and processed CSVs. Key files: `data/processed_data.csv`, `data/train/*` (transaction tables) and `data/test.csv`.
- `output/` — generated submission CSVs and images. Check `output/submissions/` for past submission formats.
- `main.py` — trivial entry point (prints a message). The project is NOT primarily a CLI app; use the notebooks for runnable workflows.
- `pyproject.toml` — lists the primary Python dependencies (XGBoost, LightGBM, CatBoost, Optuna, Jupyter). Use this to install the environment.

# High-value facts an agent should know

- This is a notebook-first ML project. Changes often mean editing notebooks rather than a runnable package. Prefer creating small reusable modules under `src/` when refactoring repeated notebook code.
- Data is loaded from `data/`. Notebook cells assume relative paths from the repository root.
- Submissions are CSVs written to `output/submissions/` (see `output/Fourth_submission.csv` and `output/submissions/*` for expected column names and index handling).
- Experiments use Optuna and XGBoost/LightGBM/CatBoost. Look at `notebooks/04_Tuning_XGBoost_with_Optuna.ipynb` and `notebooks/05_Modelling_XGBoost.ipynb` for tuning patterns.

# Developer workflows (commands an agent should suggest or run)

- Create a virtualenv, install deps from `pyproject.toml` (poetry or pip):
  - Recommended (pip): create a venv and run `python -m pip install -r <your-reqs>` after translating `pyproject.toml` deps into a `requirements.txt` (this repo does not include one). Alternatively, use `pip install` directly for the packages in `pyproject.toml`.
- Run a single notebook cell interactively using JupyterLab: `jupyter lab` (notebooks expect an interactive kernel with the listed packages).
- To reproduce a submission end-to-end: run the sequence of notebooks in order (00 -> 09) or extract the key data-prep and model cells into a script under `src/` and run it from the repo root.

# Conventions and patterns

- Notebooks are numbered to indicate pipeline order; follow that order for reproducibility.
- Dataframes are saved/loaded with pandas CSVs. Avoid changing column names or indexes unless reflecting those changes in downstream notebooks and `output/submissions/` format.
- Keep heavy compute (model training/Optuna) behind a clear cell boundary and prefer adding a `%%time` or progress logging so runs are easy to inspect.
- When adding reusable functions, place them under `src/` and keep notebook cells minimal (import from `src`). `src/` is currently empty — adding small modules is encouraged.

# Editing patterns (concise examples for an AI agent)

- Add a small preprocessing helper:
  - Create `src/preprocessing.py` with functions like `load_data(path)` and `prepare_features(df)` and import from notebooks.
- Change a notebook parameter safely:
  - Add a top cell named "Parameters" that sets `SEED`, `N_ESTIMATORS`, `TRAIN_FRACTION` and use those variables in later cells.
- Produce a submission CSV:
  - Ensure final DataFrame has the expected columns by inspecting `output/submissions/First_submission.csv` for the right header and index.

# Integration points & external dependencies

- No external APIs or services are used. All data is local under `data/`.
- Primary Python packages are declared in `pyproject.toml`. Major ML libs: `xgboost`, `lightgbm`, `catboost`, `optuna`, `scikit-learn`.

# Things NOT to change without author signoff

- The `data/` CSV layout and column names — many notebooks assume exact names and indexes.
- Historical submission formats in `output/submissions/` — preserve examples when changing the submission generator.

# Quality gates for code changes

- For small edits (bugfixes or refactors): run affected notebook cells locally to confirm no NameError/KeyError and that the submission shape matches previous outputs.
- For new modules under `src/`: add a minimal unit test in `tests/` (if you change core logic). This repo currently lacks tests; add one small pytest test that imports your function and checks a simple case.

# When you need clarification

- If column names in `data/processed_data.csv` are unclear, open `notebooks/00_Dataloading.ipynb` to see how they are used.
- If an Optuna tuning job is long-running, prefer adding a small sample dataset or limiting trials for quick feedback.

# Contact / follow-up

If anything is ambiguous in the notebooks (hard-coded paths, magic constants), ask the repository owner to point to the canonical preprocessing cell or to approve changes to `data/` column names.

---
_If you'd like, I can also extract the main data-prep and model cells from the notebooks into small runnable scripts under `src/` and add a short `requirements.txt` for easier environment setup._
