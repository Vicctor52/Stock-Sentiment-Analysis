import pandas as pd


def to_utc(series: pd.Series) -> pd.Series:
    """Convert timestamp values to timezone-aware UTC.

    Args:
        series: Timestamp-like values to convert.

    Returns:
        A timezone-aware series in UTC.
    """
    raise NotImplementedError


def assign_session_dates(
    frame: pd.DataFrame,
    timestamp_col: str = "published_utc",
) -> pd.DataFrame:
    """Assign the corresponding NYSE session date to each row.

    Args:
        frame: Frame containing a timestamp column.
        timestamp_col: Name of the timestamp column to map.

    Returns:
        The frame with an added ``session_date`` column.
    """
    raise NotImplementedError
