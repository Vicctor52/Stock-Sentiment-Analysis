import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price frame.

    Args:
        prices: Frame with ticker, date, and adjusted close columns.

    Returns:
        A frame with one log-return column per ticker.
    """
    raise NotImplementedError
