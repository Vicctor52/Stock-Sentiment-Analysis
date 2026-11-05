import pandas as pd


def assign_sessions(
    frame: pd.DataFrame,
    timestamp_col: str = "published_utc",
) -> pd.DataFrame:
    raise NotImplementedError
