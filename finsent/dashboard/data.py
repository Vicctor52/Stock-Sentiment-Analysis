from pathlib import Path

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
