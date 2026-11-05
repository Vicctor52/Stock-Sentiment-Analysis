from pathlib import Path

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        The loaded data.
    """
    return pd.read_parquet(path)
