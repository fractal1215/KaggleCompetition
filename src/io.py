import pandas as pd
from pathlib import Path


def get_paths(data_root: Path) -> dict:
    """
    Constructs a dictionary of standardized paths based on the provided data_root.
    
    Returns keys:
      - 'train': Path to training folder
      - 'test': Path to the test CSV file
      - 'sample_submission': Path to the sample submission CSV
      - 'processed': Path to the processed data CSV (if applicable)
      - 'submission': Path to the output submission CSV
    """
    paths = {
        'train': data_root / 'train',
        'test': data_root / 'test.csv',
        'sample_submission': data_root / 'sample_submission.csv',
        'processed': data_root / 'processed_data.csv',
        'submission': data_root / 'submission.csv'
    }
    return paths


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Safe wrapper for pd.read_csv with basic error handling.
    """
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except Exception as e:
        raise Exception(f"Error reading CSV file at {path}: {e}")


def write_submission(df: pd.DataFrame, path: Path) -> None:
    """
    Writes the submission DataFrame to the given path. Expects df to have at least columns 'id' and 'amount'.
    """
    try:
        df.to_csv(path, index=False)
        print(f"Submission successfully written to: {path}")
    except Exception as e:
        raise Exception(f"Failed to write submission to {path}: {e}")


def read_sample_submission(path: Path) -> pd.DataFrame:
    """
    Convenience function to read the sample submission CSV.
    """
    return read_csv(path)
