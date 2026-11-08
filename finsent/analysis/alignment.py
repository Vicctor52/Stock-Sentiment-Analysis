from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def _get_calendar(name: str):
    """Return a cached exchange calendar."""
    return xcals.get_calendar(name)


def assign_sessions(
    frame: pd.DataFrame,
    timestamp_col: str = "published_utc",
    calendar_name: str = "NYSE",
) -> pd.DataFrame:
    """Map timestamps to their corresponding market session dates.

    News published after the market close is assigned to the next session.
    Timestamps on weekends or holidays are also moved to the next session.

    Args:
        frame: Frame containing the timestamp column.
        timestamp_col: Name of the timestamp column to map.
        calendar_name: Exchange calendar name, such as ``"NYSE"``.

    Returns:
        The frame with an added ``session_date`` column.
    """
    if timestamp_col not in frame.columns:
        raise KeyError(f"Column '{timestamp_col}' not found in frame.")

    calendar = _get_calendar(calendar_name)
    timestamps = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    sessions: list[pd.Timestamp | None] = []

    for value in timestamps:
        if pd.isna(value):
            sessions.append(None)
            continue

        timestamp = pd.Timestamp(value)
        date = timestamp.tz_localize(None).normalize()

        try:
            session = calendar.date_to_session(date, direction="next")
            if timestamp >= calendar.session_close(session):
                session = calendar.next_session(session)
        except ValueError:
            logger.warning("Timestamp %s is outside the calendar range.", timestamp)
            sessions.append(None)
            continue

        sessions.append(pd.Timestamp(session).normalize())

    result = frame.copy()
    result["session_date"] = pd.to_datetime(pd.Series(sessions)).to_numpy()
    return result
