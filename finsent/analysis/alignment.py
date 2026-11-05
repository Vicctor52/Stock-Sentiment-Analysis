import pandas as pd


def assign_sessions(
    frame: pd.DataFrame,
    timestamp_col: str = "published_utc",
) -> pd.DataFrame:
    """Map timestamps to their corresponding market session dates.

    Args:
        frame: Frame containing the timestamp column.
        timestamp_col: Name of the timestamp column to map.

    Returns:
        The frame with an added ``session_date`` column.
    """
    raise NotImplementedError
