import pandas as pd

from finsent.analysis.alignment import assign_sessions


def _assign(*timestamps: str) -> pd.DataFrame:
    frame = pd.DataFrame({"published_utc": pd.to_datetime(list(timestamps), utc=True)})
    return assign_sessions(frame)


def test_assign_sessions_during_market_hours() -> None:
    result = _assign("2024-01-02T15:00:00Z")

    assert result["session_date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_assign_sessions_before_open_uses_same_session() -> None:
    result = _assign("2024-01-02T13:00:00Z")

    assert result["session_date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_assign_sessions_after_close_uses_next_session() -> None:
    result = _assign("2024-01-02T21:30:00Z")

    assert result["session_date"].iloc[0] == pd.Timestamp("2024-01-03")


def test_assign_sessions_at_close_uses_next_session() -> None:
    result = _assign("2024-01-02T21:00:00Z")

    assert result["session_date"].iloc[0] == pd.Timestamp("2024-01-03")


def test_assign_sessions_weekend_uses_next_session() -> None:
    result = _assign("2024-01-06T15:00:00Z")

    assert result["session_date"].iloc[0] == pd.Timestamp("2024-01-08")


def test_assign_sessions_holiday_uses_next_session() -> None:
    result = _assign("2024-01-15T15:00:00Z")

    assert result["session_date"].iloc[0] == pd.Timestamp("2024-01-16")


def test_assign_sessions_handles_multiple_rows_and_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "published_utc": pd.to_datetime(
                ["2024-01-02T15:00:00Z", None, "2024-01-06T15:00:00Z"],
                utc=True,
            )
        }
    )

    result = assign_sessions(frame)

    assert result["session_date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert pd.isna(result["session_date"].iloc[1])
    assert result["session_date"].iloc[2] == pd.Timestamp("2024-01-08")
