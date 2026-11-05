import pandas as pd


def to_utc(series: pd.Series) -> pd.Series:
    raise NotImplementedError


def assign_session_dates(
    frame: pd.DataFrame,
    timestamp_col: str = "published_utc",
) -> pd.DataFrame:
    raise NotImplementedError
