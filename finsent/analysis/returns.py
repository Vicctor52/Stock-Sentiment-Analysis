import numpy as np
import pandas as pd

from finsent.storage.schemas import RETURNS_COLUMNS


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    """Return the first available candidate column or raise a descriptive error."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Missing {label} column; expected one of {candidates}.")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns and forward returns from prices.

    Args:
        prices: Frame with ticker, date, and adjusted close columns.

    Returns:
        A long frame with one row per ticker and session, containing the
        contemporaneous log return and cumulative forward returns.
    """
    if prices.empty:
        return pd.DataFrame(columns=RETURNS_COLUMNS)

    date_col = _resolve_column(prices, ("session_date", "date"), "date")
    price_col = _resolve_column(prices, ("adj_close", "close"), "price")

    frame = prices[["ticker", date_col, price_col]].copy()
    frame = frame.rename(columns={date_col: "session_date", price_col: "adj_close"})
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "session_date", "adj_close"])
    frame = frame[frame["adj_close"] > 0]
    frame = frame.sort_values(["ticker", "session_date"]).reset_index(drop=True)

    grouped = frame.groupby("ticker", sort=False)["adj_close"]
    frame["log_return"] = np.log(frame["adj_close"] / grouped.shift(1))
    frame["fwd_return_1"] = np.log(grouped.shift(-1) / frame["adj_close"])
    frame["fwd_return_2"] = np.log(grouped.shift(-2) / frame["adj_close"])

    for column in ("log_return", "fwd_return_1", "fwd_return_2"):
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    return frame.reindex(columns=RETURNS_COLUMNS)
