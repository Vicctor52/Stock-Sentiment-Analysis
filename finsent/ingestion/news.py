import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


def fetch_news(tickers: list[str], page_size: int = 100) -> pd.DataFrame:
    raise NotImplementedError
