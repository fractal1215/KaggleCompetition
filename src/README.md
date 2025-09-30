# src/

This directory contains reusable modules for the KaggleCompetition project. The goal is to modularize code that is frequently used across notebooks and scripts, improving maintainability and reusability.

## Layout

- **io.py**: Contains helper functions for handling data input/output operations, including constructing standardized paths, reading CSVs safely, and writing submission files.
- **preprocessing.py**: (To be implemented) Functions for loading raw tables, merging them into a master DataFrame, applying basic cleaning, and performing sample-based quick runs.
- **features.py**: (To be implemented) Functions for feature engineering such as creating time features, lag features, and sector statistics.
- **modeling.py**: (To be implemented) Functions for model training, cross-validation, and final model training (e.g., using XGBoost).
- **metrics.py**: (To be implemented) Custom evaluation metrics (e.g., custom competitive score) and diagnostic functions.
- **utils.py**: (Optional) Miscellaneous helper functions (e.g., seed setting, logging).

## Usage

Import the required modules in your notebooks or scripts. For example:

```python
from src.io import get_paths, read_csv, write_submission

# Define your root path
import pathlib
ROOT = pathlib.Path('.').resolve()
paths = get_paths(ROOT / 'data')

# Read sample submission
sample_df = read_csv(paths['sample_submission'])

# After model prediction (assume df_submission is a DataFrame with 'id' and 'amount')
write_submission(df_submission, paths['submission'])
```

## API (for `src/io.py`)

- `get_paths(data_root: Path) -> dict`: Returns a dictionary of standardized paths (e.g., 'train', 'test', 'sample_submission', 'processed', 'submission').
- `read_csv(path: Path, **kwargs) -> pd.DataFrame`: Safe wrapper around `pandas.read_csv` with basic error handling.
- `write_submission(df: pd.DataFrame, path: Path) -> None`: Writes the submission DataFrame to the given path and prints a confirmation message.
- `read_sample_submission(path: Path) -> pd.DataFrame`: Convenience function to load the sample submission file.

Other modules will follow similar patterns with clear contracts that allow easy unit testing.
