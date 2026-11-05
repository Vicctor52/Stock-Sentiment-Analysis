import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


def fetch_prices(
    tickers: list[str],
    period: str = "12mo",
    interval: str = "1d",
) -> pd.DataFrame:
    raise NotImplementedError
