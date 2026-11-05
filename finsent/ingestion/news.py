import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


def fetch_news(tickers: list[str], page_size: int = 100) -> pd.DataFrame:
    """Fetch recent news articles for the given tickers.

    Args:
        tickers: Ticker symbols or search terms to query.
        page_size: Maximum number of articles to request per query.

    Returns:
        A normalized frame of news articles.
    """
    raise NotImplementedError
